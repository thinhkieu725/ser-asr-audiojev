"""Phase 3: relabeling, soft-target training, checkpoint averaging, initialization from an adapter."""
import copy
import json

import numpy as np
import pytest
import soundfile as sf
import torch

from sersys.average import average_adapters, last_checkpoints, merged_deltas
from sersys.config import load_config
from sersys.losses import load_soft_targets
from sersys.model import build_trainable, load_trained, save_trainable
from sersys.relabel import DROP, FLIP, KEEP, relabel
from sersys.relabel import run as relabel_run
from sersys.scoring import build_choice_batch, forward_canonical, label_token_ids, random_perms


# --------------------------------------------------------------------------- relabeling

def test_relabel_rules():
    probs = np.array([[0.7, 0.2, 0.1],     # argmax == y=0           -> keep
                      [0.95, 0.03, 0.02],  # argmax 0 != y=1, 0.95   -> flip to 0
                      [0.6, 0.3, 0.1],     # argmax 0 != y=2, 0.6    -> drop
                      [0.05, 0.05, 0.9]])  # argmax 2 != y=1, 0.9    -> flip (>= tau)
    labels = np.array([0, 1, 2, 1])
    actions, targets = relabel(labels, probs, alpha=0.5, tau_flip=0.9)
    assert actions.tolist() == [KEEP, FLIP, DROP, FLIP]
    np.testing.assert_allclose(targets[0], 0.5 * np.array([1, 0, 0]) + 0.5 * probs[0])
    np.testing.assert_allclose(targets[1], probs[1])
    np.testing.assert_allclose(targets[2], 0)
    np.testing.assert_allclose(targets[[0, 1, 3]].sum(1), 1)
    assert targets[[0, 1, 3]].argmax(1).tolist() == [0, 0, 2]


def test_relabel_run_outputs(tmp_path, space):
    classes = space.classes
    rng = np.random.default_rng(0)
    n = 200
    labels = rng.integers(0, 6, n)
    labels[:30] = 6                                        # the only 'surprise' clips; the teacher rejects them
    logits = rng.normal(size=(n, 7))
    logits[np.arange(n), labels] += 3.0
    logits[:30, 6] -= 10.0
    rows = [{"id": f"u{i}", "audio_path": f"/nowhere/u{i}.wav", "label": classes[y], "text": "t"}
            for i, y in enumerate(labels)]
    manifest = tmp_path / "pool.jsonl"
    manifest.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    order = rng.permutation(n)                             # predictions in a different order
    np.savez(tmp_path / "preds.npz", ids=np.array([f"u{i}" for i in order]), labels=labels[order],
             logits=logits[order], temperature=np.float64(1.5))
    out_m, out_t = tmp_path / "train_round1.jsonl", tmp_path / "soft_targets_round1.npz"
    stats = relabel_run(manifest, tmp_path / "preds.npz", 1, classes, out_manifest=out_m, out_targets=out_t)

    assert stats["temperature"] == 1.5
    new = [json.loads(line) for line in out_m.read_text(encoding="utf-8").splitlines()]
    assert all(set(r) == {"id", "audio_path", "label", "text"} for r in new)
    o = stats["overall"]
    assert len(new) == o["keep"] + o["flip"] and o["keep"] + o["flip"] + o["drop"] == n
    targets = load_soft_targets(out_t, [r["id"] for r in new], 7)  # the loader's id check passes
    assert [classes[k] for k in targets.argmax(1)] == [r["label"] for r in new]
    np.testing.assert_allclose(targets.sum(1), 1, atol=1e-5)
    assert any("surprise" in w for w in stats["warnings"])
    assert (tmp_path / "relabel_round1_stats.json").is_file()


def test_soft_kl_requires_targets():
    with pytest.raises(ValueError, match="soft_targets"):
        load_config(overrides=["loss.type=soft_kl"])


# --------------------------------------------------------------------------- averaging

def _trained_ckpts(tiny_thinker, tmp_path, k=3):
    cfg = load_config(overrides=["lora.r=4", "lora.alpha=8", "audio_encoder.unfreeze_top_k=1",
                                 "optim.grad_checkpointing=false"])
    model = build_trainable(copy.deepcopy(tiny_thinker), cfg)
    dirs = []
    torch.manual_seed(1)
    for i in range(k):
        with torch.no_grad():
            for _, p in model.named_parameters():
                if p.requires_grad:
                    p.add_(0.05 * torch.randn_like(p))
        d = tmp_path / f"ckpt-step{(i + 1) * 5}"
        save_trainable(model, d)
        dirs.append(d)
    return dirs


def _manual_average(tiny_thinker, ckpts):
    """Base model with mean_i(s B_i A_i) added to each weight and mean audio tensors."""
    from safetensors.torch import load_file
    model = copy.deepcopy(tiny_thinker)
    params = dict(model.named_parameters())
    deltas = [merged_deltas(c) for c in ckpts]
    with torch.no_grad():
        for m in deltas[0]:
            name = m.removeprefix("base_model.model.") + ".weight"
            params[name] += sum(d[m] for d in deltas) / len(deltas)
        audio = [load_file(str(c / "audio_encoder.safetensors")) for c in ckpts]
        for n in audio[0]:
            params[n.removeprefix("base_model.model.")].copy_(sum(a[n] for a in audio) / len(audio))
    return model.eval()


def test_last_checkpoints_numeric_order(tmp_path):
    for s in (5, 10, 15, 100):
        (tmp_path / f"ckpt-step{s}").mkdir()
    assert [p.name for p in last_checkpoints(tmp_path, 3)] == ["ckpt-step10", "ckpt-step15", "ckpt-step100"]
    with pytest.raises(ValueError):
        last_checkpoints(tmp_path, 5)


@pytest.mark.parametrize("svd_rank", [None, 12])
def test_average_equals_mean_of_merged_deltas(svd_rank, tiny_thinker, processor, space, synthetic_audios, tmp_path):
    ckpts = _trained_ckpts(tiny_thinker, tmp_path)
    out = tmp_path / "avg"
    info = average_adapters(ckpts, out, svd_rank=svd_rank)
    assert info["r"] == (12 if svd_rank else 4 * 3)
    # The averaged adapter's merged delta is the mean of the merged deltas (not of A and B).
    avg = merged_deltas(out)
    per = [merged_deltas(c) for c in ckpts]
    for m in avg:
        torch.testing.assert_close(avg[m], sum(d[m] for d in per) / 3, atol=1e-5, rtol=1e-4)
    # ... and it loads as a normal adapter giving the same outputs as the merged average.
    label_ids = label_token_ids(processor.tokenizer, space.num_classes)
    perms = random_perms(len(synthetic_audios), 7, torch.Generator().manual_seed(0))
    b = build_choice_batch(processor, synthetic_audios, space.question, space.descriptions, perms, label_ids)
    with torch.no_grad():
        ours = forward_canonical(load_trained(copy.deepcopy(tiny_thinker), out), b)
        ref = forward_canonical(_manual_average(tiny_thinker, ckpts), b)
    torch.testing.assert_close(ours, ref, atol=1e-4, rtol=1e-4)


def test_average_differs_from_averaging_factors(tiny_thinker, tmp_path):
    """Sanity: mean(B) @ mean(A) is not the mean of the products (why 3.4 merges first)."""
    from safetensors.torch import load_file
    ckpts = _trained_ckpts(tiny_thinker, tmp_path)
    states = [load_file(str(c / "adapter_model.safetensors")) for c in ckpts]
    m = next(k[: -len(".lora_A.weight")] for k in states[0] if k.endswith(".lora_A.weight"))
    naive = 2.0 * (sum(s[f"{m}.lora_B.weight"] for s in states) / 3) @ (sum(s[f"{m}.lora_A.weight"] for s in states) / 3)
    average_adapters(ckpts, tmp_path / "avg")
    assert not torch.allclose(naive, merged_deltas(tmp_path / "avg")[m], atol=1e-6)


# --------------------------------------------------------------------------- self-training driver

def test_selftrain_driver_rounds_and_stop(tmp_path, space, monkeypatch):
    """Round logic of scripts/p3_selftrain.py with predict/train stubbed: round 1 gains 2
    points (continue, teacher = round 1), round 2 gains 0.2 points (stop)."""
    import importlib.util
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("p3_selftrain", root / "scripts" / "p3_selftrain.py")
    drv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(drv)

    rng = np.random.default_rng(0)
    n = 40
    pool = tmp_path / "pool.jsonl"
    pool.write_text("\n".join(json.dumps({"id": f"u{i}", "audio_path": "/x.wav", "label": space.classes[i % 7],
                                          "text": "t"}) for i in range(n)), encoding="utf-8")
    teacher = tmp_path / "runs" / "p2_best" / "1"
    teacher.mkdir(parents=True)
    (teacher / "results.json").write_text(json.dumps({"best_dev_ua": 0.50}))
    uas = {"p3_round1": 0.52, "p3_round2": 0.522}
    calls = []

    def fake_sh(cmd):
        calls.append(cmd)
        if "sersys.predict" in cmd:
            out = Path(cmd[cmd.index("--out") + 1])
            out.mkdir(parents=True, exist_ok=True)
            logits = rng.normal(size=(n, 7))
            logits[np.arange(n), np.arange(n) % 7] += 4
            np.savez(out / "preds.npz", ids=np.array([f"u{i}" for i in range(n)]), logits=logits,
                     labels=np.arange(n) % 7, temperature=np.float64(1.0))
        else:
            exp = Path(cmd[cmd.index("--config") + 1]).stem
            run = Path(f"runs/{exp}/1")
            run.mkdir(parents=True, exist_ok=True)
            (run / "results.json").write_text(json.dumps({"best_dev_ua": uas[exp]}))

    monkeypatch.setattr(drv, "sh", fake_sh)
    monkeypatch.setattr(drv, "ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)  # restores the cwd after the driver's os.chdir
    for d in ("configs/exp", "data/manifests", "data/cache"):
        (tmp_path / d).mkdir(parents=True, exist_ok=True)
    for f in ("configs/base.yaml", "configs/labels.yaml"):
        (tmp_path / f).write_text((root / f).read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "configs/exp/p2_best.yaml").write_text("exp_name: p2_best\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["p3_selftrain.py", "--teacher_run", str(teacher),
                                      "--base_config", "configs/exp/p2_best.yaml", "--pool", str(pool)])
    drv.main()

    summary = json.loads((tmp_path / "runs/p3_selftrain_summary.json").read_text())
    assert [r["round"] for r in summary["rounds"]] == [1, 2]
    assert summary["best"]["dev_ua"] == 0.522
    predicts = [c for c in calls if "sersys.predict" in c]
    assert str(teacher / "best") in predicts[0] and str(Path("runs/p3_round1/1") / "best") in predicts[1]
    cfg = load_config(str(tmp_path / "configs/exp/p3_round2.yaml"), base_path=str(tmp_path / "configs/base.yaml"))
    assert cfg["loss"]["type"] == "soft_kl" and cfg["data"]["train_manifest"].endswith("train_round2.jsonl")
    assert cfg["eval"]["keep_last"] == 5


# --------------------------------------------------------------------------- init from adapter (3.5)

def test_init_from_adapter(tiny_thinker, processor, space, synthetic_audios, tmp_path):
    ckpts = _trained_ckpts(tiny_thinker, tmp_path, k=1)
    cfg = load_config(overrides=[f"init.adapter={ckpts[0]}", "audio_encoder.unfreeze_top_k=1",
                                 "optim.grad_checkpointing=false"])
    model = build_trainable(copy.deepcopy(tiny_thinker), cfg).eval()
    assert any(p.requires_grad for n, p in model.named_parameters() if "lora_" in n)
    label_ids = label_token_ids(processor.tokenizer, space.num_classes)
    perms = random_perms(len(synthetic_audios), 7, torch.Generator().manual_seed(0))
    b = build_choice_batch(processor, synthetic_audios, space.question, space.descriptions, perms, label_ids)
    with torch.no_grad():
        torch.testing.assert_close(forward_canonical(model, b),
                                   forward_canonical(load_trained(copy.deepcopy(tiny_thinker), ckpts[0]), b),
                                   atol=1e-5, rtol=1e-5)
    bad = load_config(overrides=[f"init.adapter={ckpts[0]}", "audio_encoder.unfreeze_top_k=0"])
    with pytest.raises(ValueError, match="frozen layers"):
        build_trainable(copy.deepcopy(tiny_thinker), bad)


# --------------------------------------------------------------------------- soft-label training loop

def test_soft_kl_training_on_relabeled_round(tiny_thinker, processor, space, tmp_path, monkeypatch):
    """relabel output -> train with loss.type=soft_kl (as p3_round{r}.yaml does)."""
    import sersys.train as train_mod

    rng = np.random.default_rng(0)
    for split, n in (("pool", 12), ("dev", 5)):
        rows = []
        for i in range(n):
            path = tmp_path / f"{split}{i}.wav"
            sf.write(path, (0.1 * rng.standard_normal(16000 + 800 * i)).astype(np.float32), 16000)
            rows.append({"id": f"{split}{i}", "audio_path": str(path), "label": space.classes[i % 7], "text": "a b"})
        (tmp_path / f"{split}.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    logits = rng.normal(size=(12, 7))
    logits[np.arange(12), np.arange(12) % 7] += 2.0
    np.savez(tmp_path / "preds.npz", ids=np.array([f"pool{i}" for i in range(12)]), logits=logits,
             labels=np.arange(12) % 7, temperature=np.float64(1.0))
    relabel_run(tmp_path / "pool.jsonl", tmp_path / "preds.npz", 1, space.classes, tau_flip=0.5,
                out_manifest=tmp_path / "train_round1.jsonl", out_targets=tmp_path / "st1.npz")

    monkeypatch.setattr(train_mod, "load_thinker", lambda *a, **k: (copy.deepcopy(tiny_thinker), processor, tmp_path))
    run_dir = tmp_path / "run"
    train_mod.main([
        "--config", "configs/exp/p1_e1_frozen_smoothce.yaml", "--cpu", "--run_dir", str(run_dir),
        "--max_steps", "6", "--set", f"data.train_manifest={tmp_path / 'train_round1.jsonl'}",
        f"data.dev_manifest={tmp_path / 'dev.jsonl'}", f"data.durations_cache={tmp_path / 'dur.json'}",
        "data.num_workers=0", "loss.type=soft_kl", f"loss.soft_targets={tmp_path / 'st1.npz'}",
        "optim.micro_batch=2", "optim.grad_accum=1", "optim.eval_batch=4", "optim.grad_checkpointing=false",
        "eval.every_steps=2", "eval.keep_last=3", "lora.r=4", "lora.alpha=8",
    ])
    assert json.loads((run_dir / "results.json").read_text())["steps"] == 6
    assert len(last_checkpoints(run_dir, 3)) == 3
    info = average_adapters(last_checkpoints(run_dir, 3), run_dir / "avg3")
    assert info["r"] == 12 and (run_dir / "avg3" / "adapter_model.safetensors").is_file()

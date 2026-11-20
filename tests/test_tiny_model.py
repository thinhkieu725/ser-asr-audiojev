"""End-to-end checks on a tiny, randomly initialized Qwen2.5-Omni thinker (CPU, float32).

These use the real AudioJev processor and config layout, so they cover the code paths
of the 3B model: LoRA targeting, audio unfreezing, the lm_head readout hook, padding
invariance of batched scoring, gradient flow with checkpointing, adapter save/load, and
the manifest -> loader -> predict pipeline. They do not touch the real weights.
"""
import copy
import json

import numpy as np
import pytest
import soundfile as sf
import torch

from sersys.config import load_config
from sersys.model import build_trainable, load_trained, save_trainable
from sersys.scoring import (MODEL_INPUT_KEYS, build_choice_batch, forward_canonical, forward_choice,
                            label_token_ids, random_perms)


@pytest.fixture(scope="module")
def train_cfg():
    return load_config(overrides=["lora.r=4", "lora.alpha=8", "lora.dropout=0.0",
                                  "audio_encoder.unfreeze_top_k=1"])


def _batch(processor, space, audios, seed=0):
    label_ids = label_token_ids(processor.tokenizer, space.num_classes)
    perms = random_perms(len(audios), space.num_classes, torch.Generator().manual_seed(seed))
    return build_choice_batch(processor, audios, space.question, space.descriptions, perms, label_ids)


def test_readout_hook_matches_full_logits(tiny_thinker, processor, space, synthetic_audios):
    b = _batch(processor, space, synthetic_audios)
    with torch.no_grad():
        hooked = forward_choice(tiny_thinker, b)
        full = tiny_thinker(**{k: b[k] for k in MODEL_INPUT_KEYS}, use_cache=False).logits
    rows = torch.arange(len(synthetic_audios))
    expected = full[rows, b["readout_pos"]][:, b["label_ids"]]
    torch.testing.assert_close(hooked, expected, atol=1e-5, rtol=1e-5)


def test_batched_scoring_equals_single(tiny_thinker, processor, space, synthetic_audios):
    """Padding (text and audio) must not change any row's scores."""
    b = _batch(processor, space, synthetic_audios, seed=3)
    label_ids = b["label_ids"].tolist()
    with torch.no_grad():
        batched = forward_canonical(tiny_thinker, b)
        for i, audio in enumerate(synthetic_audios):
            s = build_choice_batch(processor, [audio], space.question, space.descriptions,
                                   b["perm"][i:i + 1], label_ids)
            torch.testing.assert_close(batched[i:i + 1], forward_canonical(tiny_thinker, s), atol=1e-4, rtol=1e-4)


def test_train_step_save_load(tiny_thinker, train_cfg, processor, space, synthetic_audios, tmp_path):
    from sersys.losses import per_sample_loss
    from sersys.model import param_groups

    base = copy.deepcopy(tiny_thinker)
    model = build_trainable(copy.deepcopy(tiny_thinker), train_cfg)
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    lora = [n for n in names if "lora_" in n]
    audio = [n for n in names if "audio_tower" in n]
    assert lora and all(".model.layers." in n and "audio_tower" not in n and "visual" not in n for n in lora)
    assert len(lora) == 2 * 7 * 2  # 2 layers x 7 target modules x (A, B)
    assert audio and all("audio_tower.layers.2." in n or "ln_post" in n or ".proj." in n for n in audio)
    assert set(names) == set(lora) | set(audio)

    model.train()
    opt = torch.optim.AdamW(param_groups(model, train_cfg), lr=1e-2)
    labels = torch.tensor([0, 1, 2, 3])
    for _ in range(2):
        b = _batch(processor, space, synthetic_audios)
        loss = per_sample_loss(forward_canonical(model, b), labels, "smoothed_ce", 0.1).mean()
        loss.backward()
        grads = {n: p.grad for n, p in model.named_parameters() if p.requires_grad}
        assert any(g is not None and g.abs().sum() > 0 for n, g in grads.items() if "lora_B" in n)
        assert any(g is not None and g.abs().sum() > 0 for n, g in grads.items() if "audio_tower" in n)
        opt.step()
        opt.zero_grad()

    model.eval()
    b = _batch(processor, space, synthetic_audios, seed=5)
    with torch.no_grad():
        trained = forward_canonical(model, b)
        untouched = forward_canonical(base, b)
    assert not torch.allclose(trained, untouched)

    save_trainable(model, tmp_path / "ckpt")
    assert (tmp_path / "ckpt" / "adapter_model.safetensors").is_file()
    assert (tmp_path / "ckpt" / "audio_encoder.safetensors").is_file()
    reloaded = load_trained(copy.deepcopy(base), tmp_path / "ckpt")
    with torch.no_grad():
        torch.testing.assert_close(forward_canonical(reloaded, b), trained, atol=1e-5, rtol=1e-5)


def test_manifest_to_predictions(tiny_thinker, processor, space, synthetic_audios, tmp_path):
    from sersys.data import SERDataset, load_durations, read_manifest
    from sersys.predict import make_eval_loader, predict_logits

    rows = []
    for i, audio in enumerate(synthetic_audios):
        path = tmp_path / f"u{i}.wav"
        sf.write(path, np.repeat(audio[:, None], 2, 1), 22050 if i == 1 else 16000)  # stereo, mixed rates
        rows.append({"id": f"u{i}", "audio_path": str(path), "label": space.classes[i], "text": "xin chào"})
    manifest = tmp_path / "dev.jsonl"
    manifest.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")

    records = read_manifest(manifest, space.classes)
    durations = load_durations(records, tmp_path / "durations.json")
    assert (tmp_path / "durations.json").is_file() and durations["u0"] == pytest.approx(2.0)
    ds = SERDataset(records, space.classes, durations, max_seconds=3.0)
    assert len(ds[1]["wave"]) == 3 * 16000  # center-cropped
    label_ids = label_token_ids(processor.tokenizer, space.num_classes)
    loader = make_eval_loader(ds, processor, space, label_ids, k=3, batch_size=3, num_workers=0)
    preds = predict_logits(tiny_thinker, loader, torch.device("cpu"))
    assert preds["logits"].shape == (4, 7)
    assert sorted(preds["ids"].tolist()) == ["u0", "u1", "u2", "u3"]
    np.testing.assert_allclose(np.exp(preds["logits"]).sum(1) <= 1.0 + 1e-5, True)

    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({**rows[0], "label": "bored"}), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown label"):
        read_manifest(bad, space.classes)


def test_train_loop_end_to_end(tiny_thinker, processor, space, synthetic_audios, tmp_path, monkeypatch):
    """sersys.train.main on CPU with the tiny thinker: steps, evals, checkpoints, calibration."""
    import sersys.train as train_mod

    rng = np.random.default_rng(0)
    for split, n in (("train", 12), ("dev", 7)):
        rows = []
        for i in range(n):
            path = tmp_path / f"{split}{i}.wav"
            sf.write(path, (0.1 * rng.standard_normal(16000 + 1000 * i)).astype(np.float32), 16000)
            rows.append({"id": f"{split}{i}", "audio_path": str(path), "label": space.classes[i % 7], "text": "a"})
        (tmp_path / f"{split}.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    monkeypatch.setattr(train_mod, "load_thinker",
                        lambda *a, **k: (copy.deepcopy(tiny_thinker), processor, tmp_path))
    run_dir = tmp_path / "run"
    train_mod.main([
        "--config", "configs/exp/p1_e4_smallloss_0.1.yaml", "--cpu", "--run_dir", str(run_dir),
        "--max_steps", "4", "--set",
        f"data.train_manifest={tmp_path / 'train.jsonl'}", f"data.dev_manifest={tmp_path / 'dev.jsonl'}",
        f"data.durations_cache={tmp_path / 'dur.json'}", "data.num_workers=0",
        "optim.micro_batch=3", "optim.grad_accum=2", "optim.eval_batch=4", "optim.grad_checkpointing=false",
        "eval.every_steps=2", "eval.keep_last=1", "lora.r=4", "lora.alpha=8",
        "loss.small_loss.warmup_epochs=0", "audio_encoder.unfreeze_top_k=1",
    ])
    results = json.loads((run_dir / "results.json").read_text())
    assert results["steps"] == 4 and [h["step"] for h in results["history"]] == [2, 4]
    assert (run_dir / "best" / "adapter_model.safetensors").is_file()
    assert (run_dir / "best" / "temperature.json").is_file()
    assert sorted(p.name for p in run_dir.glob("ckpt-step*")) == ["ckpt-step4"]
    for name in ("config.yaml", "git_hash.txt", "pip_freeze.txt", "model_revision.txt", "train.log"):
        assert (run_dir / name).is_file()

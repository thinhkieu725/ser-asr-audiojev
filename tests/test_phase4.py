"""Phase 4: test evaluation (CIs, once-only guard), tables, export + validation, offline inference.

Uses the tiny random thinker (CPU, float32) in place of AudioJev by patching
sersys.model.load_thinker, plus a fake trained run directory.
"""
import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
import yaml

import sersys.model
from sersys.config import load_config
from sersys.metrics import bootstrap_ci, reliability_bins
from sersys.model import build_trainable, load_trained, save_trainable

ROOT = Path(__file__).resolve().parents[1]


def _write_split(tmp_path, space, split, n, rng, long_idx=()):
    rows = []
    for i in range(n):
        path = tmp_path / "wav" / f"{split}{i}.wav"
        path.parent.mkdir(exist_ok=True)
        secs = 2.5 if i in long_idx else 0.6 + 0.1 * i
        sf.write(path, (0.1 * rng.standard_normal(int(16000 * secs))).astype(np.float32), 16000)
        rows.append({"id": f"{split}{i}", "audio_path": str(path), "label": space.classes[i % 7], "text": "x"})
    p = tmp_path / f"{split}.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return p


def _make_run(tmp_path, space, tiny_thinker, processor, name="final", seed=1, perturb_seed=0):
    from sersys.data import SERDataset, load_durations, read_manifest
    from sersys.predict import make_eval_loader, predict_logits
    from sersys.scoring import label_token_ids

    cfg = load_config(overrides=[
        f"data.dev_manifest={tmp_path / 'dev.jsonl'}", f"data.test_manifest={tmp_path / 'test.jsonl'}",
        f"data.durations_cache={tmp_path / 'dur.json'}", "data.num_workers=0", "data.max_seconds=2.0",
        "lora.r=4", "lora.alpha=8", "audio_encoder.unfreeze_top_k=1", "optim.grad_checkpointing=false",
        "optim.eval_batch=3", f"exp_name={name}", f"seed={seed}", "ser.tta_permutations=2"])
    run_dir = tmp_path / "runs" / name / str(seed)
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    model = build_trainable(copy.deepcopy(tiny_thinker), cfg)
    torch.manual_seed(perturb_seed)
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.add_(0.05 * torch.randn_like(p))
    best = run_dir / "best"
    save_trainable(model, best)
    (best / "temperature.json").write_text(json.dumps({"temperature": 1.3}))
    (best / "dev_metrics.json").write_text(json.dumps({"ua": 0.5, "wa": 0.5, "macro_f1": 0.4, "nll": 1.0, "ece": 0.1}))
    recs = read_manifest(cfg["data"]["dev_manifest"], space.classes)
    ds = SERDataset(recs, space.classes, load_durations(recs, cfg["data"]["durations_cache"]), max_seconds=2.0)
    loader = make_eval_loader(ds, processor, space, label_token_ids(processor.tokenizer, 7), 2, 3, 0)
    preds = predict_logits(load_trained(copy.deepcopy(tiny_thinker), best), loader, torch.device("cpu"))
    np.savez(best / "dev_preds.npz", **preds)
    return run_dir, cfg


@pytest.fixture
def env(tmp_path, space, tiny_thinker, processor, monkeypatch):
    rng = np.random.default_rng(0)
    _write_split(tmp_path, space, "dev", 6, rng, long_idx=(5,))
    _write_split(tmp_path, space, "test", 9, rng, long_idx=(8,))
    monkeypatch.setattr(sersys.model, "load_thinker",
                        lambda *a, **k: (copy.deepcopy(tiny_thinker), processor, tmp_path))
    run_dir, cfg = _make_run(tmp_path, space, tiny_thinker, processor)
    return tmp_path, run_dir, cfg


# --------------------------------------------------------------------------- metrics helpers

def test_bootstrap_ci_brackets_estimate():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 7, 500)
    logits = rng.normal(size=(500, 7))
    logits[np.arange(500), y] += 1.5
    p = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    ci = bootstrap_ci([p], y, 7, n_boot=200)
    wa = (p.argmax(1) == y).mean()
    assert ci["wa"][0] < wa < ci["wa"][1] and ci["ua"][1] - ci["ua"][0] < 0.2
    two = bootstrap_ci([p, p], y, 7, n_boot=200)  # identical seeds -> same CI as one seed
    assert np.allclose(two["ua"], ci["ua"])
    bins = reliability_bins(p, y, 15)
    assert sum(b["count"] for b in bins) == 500


# --------------------------------------------------------------------------- export + inference

def test_export_reload_and_validate(env, space):
    from sersys.export import export, validate_export
    from sersys.infer import SERPredictor

    tmp_path, run_dir, cfg = env
    out = tmp_path / "exports" / "final"
    sc = export(run_dir, "best", out, dtype="float32", device="cpu")
    assert sc["temperature"] == 1.3 and sc["tta_permutations"] == 2 and sc["classes"] == space.classes
    for f in ("config.json", "model.safetensors", "ser_config.json", "preprocessor_config.json"):
        assert (out / f).is_file(), f
    report = validate_export(out, run_dir / "best" / "dev_preds.npz", cfg["data"]["dev_manifest"], device="cpu")
    assert report["passed"] and report["argmax_agreement"] == 1.0 and report["max_abs_dprob"] < 1e-4
    assert json.loads((out / "ser_config.json").read_text())["validation"]["passed"]

    ser = SERPredictor.from_pretrained(out, device="cpu")
    assert ser.k == 2 and ser.temperature == 1.3
    # a corrupted export must fail validation
    with torch.no_grad():
        ser.model.lm_head.weight.mul_(1.5)
    bad = validate_export(out, run_dir / "best" / "dev_preds.npz", cfg["data"]["dev_manifest"], predictor=ser)
    assert not bad["passed"]


def test_predictor_from_run_inputs(env, space):
    from sersys.infer import SERPredictor

    tmp_path, run_dir, _ = env
    ser = SERPredictor.from_run(run_dir, "best", device="cpu")
    files = sorted((tmp_path / "wav").glob("test*.wav"))
    wave, sr = sf.read(files[0], dtype="float32")
    stereo44 = np.repeat(np.interp(np.linspace(0, len(wave), int(len(wave) * 44100 / 16000)),
                                   np.arange(len(wave)), wave)[:, None], 2, 1)
    out = ser.predict([files[0], (stereo44, 44100), wave, str(files[-1])], batch_size=3)
    assert len(out) == 4
    for r in out:
        assert r["label"] in space.classes and abs(sum(r["probs"].values()) - 1) < 1e-5
        assert r["confidence"] == max(r["probs"].values())
    assert out[0]["probs"] == pytest.approx(out[2]["probs"], abs=1e-6)       # path == raw waveform
    assert out[3]["cropped"] and not out[0]["cropped"]                        # test8 is 2.5 s > 2.0 s
    single = ser.predict_file(files[0])
    assert single["probs"] == pytest.approx(out[0]["probs"], abs=1e-5)        # batching/order invariant
    zs = SERPredictor.from_run(run_dir, "none", device="cpu")
    assert zs.temperature == 1.0


def test_infer_cli(env, space, capsys):
    from sersys.export import export
    from sersys.infer import main as infer_main

    tmp_path, run_dir, cfg = env
    export(run_dir, "best", tmp_path / "exp", dtype="float32", device="cpu")
    out = tmp_path / "preds.jsonl"
    infer_main(["--model", str(tmp_path / "exp"), str(tmp_path / "wav"), "--manifest",
                cfg["data"]["test_manifest"], "--out", str(out), "--device", "cpu", "--batch", "4"])
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 15 + 9                                    # directory (15 wavs) + manifest (9)
    assert {"id", "audio_path", "label", "confidence", "probs", "cropped"} <= set(rows[0])
    assert sum("gold" in r for r in rows) == 9


# --------------------------------------------------------------------------- test evaluation + tables

def test_evaluate_once_and_tables(env, space, tiny_thinker, processor):
    from sersys.evaluate import main as eval_main

    tmp_path, run_dir, _ = env
    eval_main(["--run_dir", str(run_dir), "--adapter", "best", "--device", "cpu", "--n_boot", "100"])
    m = json.loads((run_dir / "test_metrics.json").read_text())
    assert m["n"] == 9 and m["temperature"] == 1.3 and m["tta_permutations"] == 2
    for k in ("ua", "wa", "macro_f1", "ece"):
        assert m["ci95"][k][0] <= m["ci95"][k][1]
    assert (run_dir / "test_reliability.png").is_file() and (run_dir / "test" / "preds.npz").is_file()
    assert len((run_dir / "test_access.log").read_text().splitlines()) == 1
    with pytest.raises(SystemExit):                                # test set is touched once
        eval_main(["--run_dir", str(run_dir), "--device", "cpu"])

    # a second seed and a zero-shot row, then the tables
    run2, _ = _make_run(tmp_path, space, tiny_thinker, processor, seed=2, perturb_seed=1)
    eval_main(["--run_dir", str(run2), "--device", "cpu", "--n_boot", "100"])
    zs = tmp_path / "runs" / "zeroshot_en" / "1"
    eval_main(["--run_dir", str(zs), "--zeroshot", "--config", str(run_dir / "config.yaml"),
               "--fit_temperature_on_dev", "--device", "cpu", "--n_boot", "100"])
    assert "fitted on dev" in json.loads((zs / "test_metrics.json").read_text())["temperature_source"]

    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_tables.py"), "--runs", str(tmp_path / "runs"),
                    "--out", str(tmp_path / "tables"), "--n_boot", "50", "--exps", "zeroshot_en", "final"],
                   check=True, capture_output=True)
    md = (tmp_path / "tables" / "test_results.md").read_text(encoding="utf-8")
    assert "| zeroshot_en | 1 | 9 |" in md and "| final | 2 | 9 |" in md and "±" in md
    tex = (tmp_path / "tables" / "test_results.tex").read_text(encoding="utf-8")
    assert r"\begin{tabular}" in tex and r"zeroshot\_en" in tex

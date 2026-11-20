"""Export a trained run as a standalone model for offline inference (PLAN 4.2).

Merges the LoRA weights (from best/ or an averaged adapter such as avg5/) and any
unfrozen audio-encoder layers into the AudioJev thinker. It then saves the merged model
(save_pretrained, BF16 by default), the processor, and ser_config.json: labels,
candidate descriptions, question, T, tta_permutations, AudioJev revision and git hash.
Load the result with sersys.infer.SERPredictor.from_pretrained.

Validation (on by default): the exported model re-scores the dev clips behind the run's
dev_preds.npz and must agree with them, with argmax agreement 100% and max |dp| < 1e-3
(calibrated probabilities). If BF16 rounding of the merged weights breaks this, re-export
with --dtype float32.

    python -m sersys.export --run_dir runs/final/1 --adapter avg5 --out exports/final
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

from .infer import SER_CONFIG, SERPredictor, build_ser_config

log = logging.getLogger(__name__)


def resolve_adapter(run_dir: Path, adapter: str) -> Path:
    path = run_dir / adapter if (run_dir / adapter).is_dir() else Path(adapter)
    if not (path / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(f"no adapter in {path}")
    return path


def export(run_dir: str | Path, adapter: str, out_dir: str | Path, dtype: str = "bfloat16",
           device: str = "cuda:0") -> dict:
    from .config import load_config
    from .model import load_thinker, load_trained

    run_dir, out_dir = Path(run_dir), Path(out_dir)
    cfg = load_config(str(run_dir / "config.yaml"))
    adapter_dir = resolve_adapter(run_dir, adapter)
    ser_config = build_ser_config(cfg, adapter_dir, run_dir)
    # Keep every thinker module so the export reloads without randomly initialized weights.
    thinker, processor, _ = load_thinker({**cfg["model"], "drop_modules": []}, device, attn_impl="sdpa")
    model = load_trained(thinker, adapter_dir)
    merged = model.merge_and_unload(safe_merge=True)
    merged.to(getattr(torch, dtype))
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(out_dir), safe_serialization=True)
    processor.save_pretrained(str(out_dir))
    ser_config["dtype"] = dtype
    (out_dir / SER_CONFIG).write_text(json.dumps(ser_config, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("exported %s (%s) -> %s", adapter_dir, dtype, out_dir)
    del merged, model, thinker
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return ser_config


def validate_export(out_dir: str | Path, reference_preds: str | Path, dev_manifest: str | Path,
                    device: str = "cuda:0", tol: float = 1e-3, limit: int | None = None,
                    batch_size: int = 8, predictor: SERPredictor | None = None) -> dict:
    """Compare the exported model's calibrated dev probabilities with the run's dev_preds.npz."""
    from .relabel import softmax

    out_dir = Path(out_dir)
    ser = predictor or SERPredictor.from_pretrained(out_dir, device=device)
    ref = np.load(reference_preds)
    ids = [str(i) for i in ref["ids"]][:limit]
    paths = {}
    with open(dev_manifest, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rec = json.loads(line)
                paths[rec["id"]] = rec["audio_path"]
    missing = [i for i in ids if i not in paths]
    if missing:
        raise KeyError(f"{len(missing)} reference ids not in {dev_manifest}, e.g. {missing[:3]}")
    ref_probs = softmax(ref["logits"][: len(ids)], ser.temperature)
    results = ser.predict([paths[i] for i in ids], batch_size=batch_size)
    probs = np.array([[r["probs"][c] for c in ser.space.classes] for r in results])
    diff = float(np.abs(probs - ref_probs).max())
    agree = float((probs.argmax(1) == ref_probs.argmax(1)).mean())
    report = {"n": len(ids), "max_abs_dprob": diff, "argmax_agreement": agree, "tolerance": tol,
              "passed": bool(agree == 1.0 and diff < tol), "reference": str(reference_preds)}
    (out_dir / "validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    cfg_path = out_dir / SER_CONFIG
    ser_config = json.loads(cfg_path.read_text(encoding="utf-8"))
    ser_config["validation"] = report
    cfg_path.write_text(json.dumps(ser_config, indent=2, ensure_ascii=False), encoding="utf-8")
    (log.info if report["passed"] else log.error)(
        "export validation %s: n=%d, argmax agreement %.4f, max |dp| %.2e (tol %.0e)",
        "PASSED" if report["passed"] else "FAILED", len(ids), agree, diff, tol)
    return report


def main(argv=None):
    from .config import load_config

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--adapter", default="best", help="best | avg5 | <path to an adapter dir>")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--no_validate", action="store_true")
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--limit", type=int, default=None, help="validate on the first N dev clips only")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    export(args.run_dir, args.adapter, args.out, args.dtype, args.device)
    if args.no_validate:
        return
    adapter_dir = resolve_adapter(Path(args.run_dir), args.adapter)
    ref = adapter_dir / "dev_preds.npz"
    if not ref.is_file():
        log.error("%s missing; run `python -m sersys.average ... --refit` for averaged adapters", ref)
        sys.exit(2)
    cfg = load_config(str(Path(args.run_dir) / "config.yaml"))
    report = validate_export(args.out, ref, cfg["data"]["dev_manifest"], args.device, args.tol, args.limit)
    if not report["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()

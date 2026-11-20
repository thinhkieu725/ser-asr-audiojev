"""Final test-set evaluation of one run (PLAN 4.1).

    python -m sersys.evaluate --run_dir runs/final/1 --adapter avg5
    python -m sersys.evaluate --run_dir runs/zeroshot_en/1 --zeroshot --config configs/base.yaml \
        --fit_temperature_on_dev

Writes into the run directory:
  <split>_metrics.json   UA / WA / macro-F1 / NLL / ECE at the dev-fitted T, ECE at T = 1,
                         95% bootstrap CIs (1,000 utterance resamples), per-class recall,
                         confusion matrix, reliability bins
  <split>/preds.npz      ids, labels, logits, temperature (used by scripts/make_tables.py)
  <split>/preds.jsonl    per-utterance calibrated probabilities
  <split>_reliability.png
  test_access.log        one line per test evaluation

The test set is touched once: if test_metrics.json exists, the run is refused unless --force.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

log = logging.getLogger(__name__)


def main(argv=None):
    from .calibration import fit_temperature
    from .config import load_config
    from .data import SERDataset, load_durations, read_manifest
    from .metrics import bootstrap_ci, format_confusion, plot_reliability, reliability_bins, ser_metrics
    from .model import load_thinker, load_trained
    from .predict import make_eval_loader, predict_logits, save_predictions
    from .prompts import load_label_space
    from .relabel import softmax
    from .scoring import label_token_ids

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--adapter", default="best", help="best | avg5 | <path>")
    ap.add_argument("--zeroshot", action="store_true", help="no adapter (zero-shot AudioJev row)")
    ap.add_argument("--config", default=None, help="config for --zeroshot (default <run_dir>/config.yaml or base)")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--split", default="test", choices=["test", "dev"])
    ap.add_argument("--tta", type=int, default=None)
    ap.add_argument("--fit_temperature_on_dev", action="store_true",
                    help="fit T on dev here (for runs without temperature.json, e.g. zero-shot)")
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--force", action="store_true", help="re-evaluate although <split>_metrics.json exists")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    run_dir = Path(args.run_dir)
    out_json = run_dir / f"{args.split}_metrics.json"
    if out_json.exists() and not args.force:
        log.error("%s exists: the %s set is evaluated once per run (use --force to override)", out_json, args.split)
        sys.exit(1)
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = args.config or (run_dir / "config.yaml" if (run_dir / "config.yaml").is_file() else None)
    cfg = load_config(str(cfg_path) if cfg_path else None, args.set)
    if args.zeroshot and not (run_dir / "config.yaml").is_file():
        from .config import dump_run_info
        dump_run_info(cfg, run_dir)
    adapter_dir = None
    if not args.zeroshot:
        adapter_dir = run_dir / args.adapter if (run_dir / args.adapter).is_dir() else Path(args.adapter)
    k = args.tta or cfg["ser"]["tta_permutations"]
    space = load_label_space(cfg["labels"], cfg["ser"]["prompt_lang"], cfg["ser"].get("question"))
    C = space.num_classes

    thinker, processor, _ = load_thinker(cfg["model"], args.device)
    model = load_trained(thinker, adapter_dir) if adapter_dir else thinker.eval().requires_grad_(False)
    label_ids = label_token_ids(processor.tokenizer, C)
    view_builder = None
    if cfg["asr"]["transcript_in_ser"]:
        from .asr import TranscriptViewBuilder, make_prompter
        view_builder = TranscriptViewBuilder(processor, space, label_ids, make_prompter(processor, cfg),
                                             cfg["asr"]["generation"], cfg["asr"]["transcript_template"],
                                             sample_rate=cfg["data"]["sample_rate"])

    def predict_split(manifest):
        records = read_manifest(manifest, space.classes)
        durations = load_durations(records, cfg["data"]["durations_cache"])
        ds = SERDataset(records, space.classes, durations, cfg["data"]["sample_rate"],
                        cfg["data"]["max_seconds"], name=Path(manifest).name)
        loader = make_eval_loader(ds, processor, space, label_ids, k, cfg["optim"]["eval_batch"],
                                  cfg["data"]["num_workers"], cfg["data"]["bucket_by_length"],
                                  build_views=view_builder is None)
        return predict_logits(model, loader, torch.device(args.device), view_builder)

    # Temperature: fitted on dev only (stored with the adapter, or fitted here).
    temperature, t_source = 1.0, "none (T = 1)"
    if args.fit_temperature_on_dev:
        dev = predict_split(cfg["data"]["dev_manifest"])
        temperature, t_source = fit_temperature(dev["logits"], dev["labels"]), f"fitted on dev, K={k}"
    elif adapter_dir is not None and (adapter_dir / "temperature.json").is_file():
        temperature = json.loads((adapter_dir / "temperature.json").read_text())["temperature"]
        t_source = str(adapter_dir / "temperature.json")

    manifest = cfg["data"][f"{args.split}_manifest"]
    if args.split == "test":
        with (run_dir / "test_access.log").open("a", encoding="utf-8") as fh:
            fh.write(f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} adapter={adapter_dir} "
                     f"K={k} force={args.force}\n")
    preds = predict_split(manifest)
    m = save_predictions(preds, run_dir / args.split, space.classes, temperature, cfg["eval"]["ece_bins"])
    probs = softmax(preds["logits"], temperature)
    m.update(
        split=args.split, manifest=str(manifest), exp_name=cfg.get("exp_name"), seed=cfg.get("seed"),
        adapter=str(adapter_dir) if adapter_dir else "zero-shot", tta_permutations=k,
        temperature_source=t_source,
        ece_uncalibrated=ser_metrics(preds["logits"], preds["labels"], C, cfg["eval"]["ece_bins"])["ece"],
        ci95=bootstrap_ci([probs], preds["labels"], C, args.n_boot, cfg["eval"]["ece_bins"]),
        n_boot=args.n_boot, classes=space.classes,
        reliability=reliability_bins(probs, preds["labels"], cfg["eval"]["ece_bins"]),
    )
    out_json.write_text(json.dumps(m, indent=2), encoding="utf-8")
    plot_reliability(m["reliability"], run_dir / f"{args.split}_reliability.png",
                     f"{cfg.get('exp_name')} seed {cfg.get('seed')} ({args.split}, T={temperature:.2f})")
    ci = m["ci95"]
    log.info("%s: UA %.4f [%.4f, %.4f]  WA %.4f  F1 %.4f  ECE %.4f (T=1: %.4f)", args.split, m["ua"], *ci["ua"],
             m["wa"], m["macro_f1"], m["ece"], m["ece_uncalibrated"])
    print(format_confusion(m["confusion"], space.classes))


if __name__ == "__main__":
    main()

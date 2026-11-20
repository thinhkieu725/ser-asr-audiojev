"""Batched SER inference with test-time permutation averaging (Phase 1.9).

For K = ser.tta_permutations, each clip is scored with K cyclic shifts of the reference
candidate order. Each result is mapped back to canonical class order, and the K
log-probability vectors are averaged. With K = 1 the output is the zero-shot AudioJev
protocol: the reference order, softmax over label-token logits.

Zero-shot dev pass (the "step 0" row):
    python -m sersys.predict --split dev --out runs/zeroshot/dev
Fine-tuned adapter, with K = 3 and its fitted temperature:
    python -m sersys.predict --config runs/<exp>/1/config.yaml --adapter runs/<exp>/1/best \
        --split dev --tta 3 --out runs/<exp>/1/best/dev_tta3
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

from .config import load_config
from .data import LengthBucketBatchSampler, SERDataset, load_durations, read_manifest
from .metrics import format_confusion, ser_metrics
from .model import load_thinker, load_trained, needs_autocast
from .prompts import LabelSpace, load_label_space
from .scoring import ChoiceCollator, forward_canonical, label_token_ids, tta_perms

log = logging.getLogger(__name__)


def make_eval_loader(dataset: SERDataset, processor, space: LabelSpace, label_ids: list[int], k: int,
                     batch_size: int, num_workers: int = 2, bucket_by_length: bool = True,
                     build_views: bool = True):
    sampler = LengthBucketBatchSampler(dataset.durations, batch_size, shuffle=False,
                                       bucket_by_length=bucket_by_length)
    collate = ChoiceCollator(processor, space, label_ids, mode="fixed",
                             perms=tta_perms(space.reference_order, k), sample_rate=dataset.sample_rate,
                             build_views=build_views)
    return torch.utils.data.DataLoader(dataset, batch_sampler=sampler, collate_fn=collate,
                                       num_workers=num_workers, pin_memory=True)


@torch.no_grad()
def predict_logits(model, loader, device, view_builder=None) -> dict:
    """Returns {'ids', 'labels', 'logits'}; logits = mean over views of canonical log-probs.

    `view_builder(model, batch, device)` builds the views when the loader defers them (E2.4).
    """
    was_training = model.training
    model.eval()
    amp = (torch.autocast("cuda", dtype=torch.bfloat16) if needs_autocast(model)
           else contextlib.nullcontext())
    ids, labels, out = [], [], []
    try:
        for batch in loader:
            views = []
            for view in batch["views"] if view_builder is None else view_builder(model, batch, device):
                with amp:
                    logits = forward_canonical(model, view, device)
                views.append(torch.log_softmax(logits.float(), dim=-1))
            out.append(torch.stack(views).mean(0).cpu())
            ids += batch["ids"]
            labels.append(batch["labels"])
    finally:
        model.train(was_training)
    return {"ids": np.array(ids), "labels": torch.cat(labels).numpy(), "logits": torch.cat(out).numpy()}


def save_predictions(preds: dict, out_dir: Path, classes: list[str], temperature: float = 1.0,
                     n_bins: int = 15) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "preds.npz", ids=preds["ids"], labels=preds["labels"], logits=preds["logits"],
             temperature=np.float64(temperature))
    probs = torch.softmax(torch.from_numpy(preds["logits"]).double() / temperature, dim=-1).numpy()
    with (out_dir / "preds.jsonl").open("w", encoding="utf-8") as fh:
        for i, y, p in zip(preds["ids"], preds["labels"], probs):
            fh.write(json.dumps({"id": str(i), "label": classes[y], "pred": classes[int(p.argmax())],
                                 "probs": {c: round(float(v), 6) for c, v in zip(classes, p)}},
                                ensure_ascii=False) + "\n")
    metrics = ser_metrics(preds["logits"], preds["labels"], len(classes), n_bins, temperature)
    metrics["temperature"] = temperature
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="experiment yaml (or a run's config.yaml)")
    ap.add_argument("--set", nargs="*", default=[], help="overrides, key.path=value")
    ap.add_argument("--adapter", default=None, help="checkpoint dir from training; omit for zero-shot")
    ap.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    ap.add_argument("--manifest", default=None, help="explicit manifest (overrides --split)")
    ap.add_argument("--tta", type=int, default=None, help="permutations K (default ser.tta_permutations)")
    ap.add_argument("--temperature", default=None,
                    help="float, or a temperature.json; default: <adapter>/temperature.json if present")
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="score only the first N utterances")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = load_config(args.config, args.set)  # a run's config.yaml is a full config: merges cleanly
    space = load_label_space(cfg["labels"], cfg["ser"]["prompt_lang"], cfg["ser"].get("question"))
    manifest = args.manifest or cfg["data"][f"{args.split}_manifest"]
    records = read_manifest(manifest, space.classes)[: args.limit]
    durations = load_durations(records, cfg["data"]["durations_cache"])
    dataset = SERDataset(records, space.classes, durations, cfg["data"]["sample_rate"],
                         cfg["data"]["max_seconds"], train=False, name=Path(manifest).name)

    thinker, processor, _ = load_thinker(cfg["model"], args.device)
    model = load_trained(thinker, args.adapter) if args.adapter else thinker.eval().requires_grad_(False)
    label_ids = label_token_ids(processor.tokenizer, space.num_classes)
    k = args.tta or cfg["ser"]["tta_permutations"]
    view_builder = None
    if cfg["asr"]["transcript_in_ser"]:  # E2.4: classify with the model's own transcript
        from .asr import TranscriptViewBuilder, make_prompter
        view_builder = TranscriptViewBuilder(processor, space, label_ids, make_prompter(processor, cfg),
                                             cfg["asr"]["generation"], cfg["asr"]["transcript_template"],
                                             sample_rate=cfg["data"]["sample_rate"])
    loader = make_eval_loader(dataset, processor, space, label_ids, k,
                              args.batch or cfg["optim"]["eval_batch"], cfg["data"]["num_workers"],
                              cfg["data"]["bucket_by_length"], build_views=view_builder is None)

    temperature = 1.0
    t_src = args.temperature or (str(Path(args.adapter) / "temperature.json") if args.adapter else None)
    if t_src:
        if Path(t_src).is_file():
            temperature = json.loads(Path(t_src).read_text())["temperature"]
        elif not t_src.endswith(".json"):
            temperature = float(t_src)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    preds = predict_logits(model, loader, torch.device(args.device), view_builder)
    elapsed = time.perf_counter() - t0
    metrics = save_predictions(preds, Path(args.out), space.classes, temperature, cfg["eval"]["ece_bins"])
    metrics.update(tta_permutations=k, seconds=elapsed, clips_per_second=len(records) / elapsed,
                   peak_mem_gb=torch.cuda.max_memory_allocated() / 2**30, adapter=args.adapter,
                   manifest=str(manifest))
    (Path(args.out) / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    log.info("UA %.4f  WA %.4f  F1 %.4f  NLL %.4f  ECE %.4f  (T=%.3f, K=%d, %.1f clips/s, %.1f GB)",
             metrics["ua"], metrics["wa"], metrics["macro_f1"], metrics["nll"], metrics["ece"],
             temperature, k, metrics["clips_per_second"], metrics["peak_mem_gb"])
    print(format_confusion(metrics["confusion"], space.classes))


if __name__ == "__main__":
    main()

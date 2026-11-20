"""Self-training relabeling (PLAN 3.2).

For each pool utterance with original (pseudo) label y and calibrated teacher probabilities p:

* keep:  argmax p == y                      -> target = alpha * onehot(y) + (1 - alpha) * p
* flip:  argmax p != y and max p >= tau     -> target = p
* drop:  otherwise

Outputs (the manifests stay at four fields):
* data/manifests/train_round{r}.jsonl   kept + flipped utterances, `label` = argmax(target)
* data/cache/soft_targets_round{r}.npz  `ids` and `targets` (N, C) in canonical class order
* data/cache/relabel_round{r}_stats.json  per-class kept / flipped / dropped shares and drift

    python -m sersys.relabel --round 1 --preds runs/p2_best/1/pool/preds.npz \
        --manifest data/manifests/train_pseudo.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

KEEP, FLIP, DROP = 0, 1, 2
ACTIONS = ("keep", "flip", "drop")


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64) / temperature
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def relabel(labels: np.ndarray, probs: np.ndarray, alpha: float = 0.5, tau_flip: float = 0.9):
    """Returns (actions (N,), targets (N, C)); targets of dropped rows are zero."""
    labels = np.asarray(labels)
    probs = np.asarray(probs, dtype=np.float64)
    n, c = probs.shape
    pred = probs.argmax(1)
    conf = probs.max(1)
    actions = np.full(n, DROP, dtype=np.int64)
    actions[pred == labels] = KEEP
    actions[(pred != labels) & (conf >= tau_flip)] = FLIP
    targets = np.zeros_like(probs)
    keep = actions == KEEP
    targets[keep] = alpha * np.eye(c)[labels[keep]] + (1 - alpha) * probs[keep]
    targets[actions == FLIP] = probs[actions == FLIP]
    return actions, targets


def relabel_stats(labels: np.ndarray, actions: np.ndarray, new_labels: np.ndarray, classes: list[str],
                  shrink_warn: float = 0.5) -> dict:
    """Per original class: kept / flipped-out / dropped shares; per class: size before -> after."""
    stats, warnings = {"per_class": {}, "overall": {}}, []
    for k, name in enumerate(classes):
        m = labels == k
        n = int(m.sum())
        after = int((new_labels[actions != DROP] == k).sum())
        row = {"n_before": n, "n_after": after,
               "kept": int((actions[m] == KEEP).sum()), "flipped_out": int((actions[m] == FLIP).sum()),
               "dropped": int((actions[m] == DROP).sum()),
               "flipped_in": int(((actions == FLIP) & (new_labels == k) & (labels != k)).sum()),
               "ratio_after_before": after / n if n else float("nan")}
        for a in ("kept", "flipped_out", "dropped"):
            row[f"{a}_share"] = row[a] / n if n else float("nan")
        stats["per_class"][name] = row
        if n and row["ratio_after_before"] < shrink_warn:
            warnings.append(f"class {name!r} shrinks to {row['ratio_after_before']:.0%} of its size; "
                            f"the teacher may be reinforcing its own errors")
    total = len(labels)
    stats["overall"] = {a: int((actions == i).sum()) for i, a in enumerate(ACTIONS)}
    stats["overall"].update({f"{a}_share": stats["overall"][a] / total for a in ACTIONS}, n=total)
    stats["warnings"] = warnings
    return stats


def format_stats(stats: dict) -> str:
    lines = [f"{'class':<10}{'before':>8}{'after':>8}{'ratio':>7}{'kept':>8}{'flip-out':>9}{'flip-in':>8}{'drop':>8}"]
    for name, r in stats["per_class"].items():
        lines.append(f"{name:<10}{r['n_before']:>8}{r['n_after']:>8}{r['ratio_after_before']:>7.2f}"
                     f"{r['kept_share']:>8.1%}{r['flipped_out_share']:>9.1%}{r['flipped_in']:>8}"
                     f"{r['dropped_share']:>8.1%}")
    o = stats["overall"]
    lines.append(f"overall: kept {o['keep_share']:.1%}  flipped {o['flip_share']:.1%}  dropped {o['drop_share']:.1%}"
                 f"  (n={o['n']})")
    return "\n".join(lines)


def run(manifest: str | Path, preds: str | Path, round_idx: int, classes: list[str], alpha: float = 0.5,
        tau_flip: float = 0.9, temperature: float | None = None, out_manifest: str | Path | None = None,
        out_targets: str | Path | None = None, out_stats: str | Path | None = None,
        shrink_warn: float = 0.5) -> dict:
    from .data import read_manifest

    records = read_manifest(manifest, classes, check_audio=False)
    data = np.load(preds)
    if temperature is None:
        temperature = float(data["temperature"]) if "temperature" in data.files else 1.0
    row = {str(i): j for j, i in enumerate(data["ids"])}
    missing = [r["id"] for r in records if r["id"] not in row]
    if missing:
        raise KeyError(f"{len(missing)} manifest ids have no prediction in {preds}, e.g. {missing[:3]}")
    idx = np.array([row[r["id"]] for r in records])
    probs = softmax(data["logits"][idx], temperature)
    if probs.shape[1] != len(classes):
        raise ValueError(f"predictions have {probs.shape[1]} classes, expected {len(classes)}")
    labels = np.array([classes.index(r["label"]) for r in records])
    actions, targets = relabel(labels, probs, alpha, tau_flip)
    new_labels = targets.argmax(1)
    stats = relabel_stats(labels, actions, new_labels, classes, shrink_warn)
    stats.update(round=round_idx, alpha=alpha, tau_flip=tau_flip, temperature=temperature,
                 source_manifest=str(manifest), preds=str(preds))

    out_manifest = Path(out_manifest or f"data/manifests/train_round{round_idx}.jsonl")
    out_targets = Path(out_targets or f"data/cache/soft_targets_round{round_idx}.npz")
    out_stats = Path(out_stats or out_targets.with_name(f"relabel_round{round_idx}_stats.json"))
    for p in (out_manifest, out_targets, out_stats):
        p.parent.mkdir(parents=True, exist_ok=True)
    kept = np.flatnonzero(actions != DROP)
    with out_manifest.open("w", encoding="utf-8") as fh:
        for j in kept:
            rec = dict(records[j], label=classes[new_labels[j]])
            fh.write(json.dumps({k: rec[k] for k in ("id", "audio_path", "label", "text")}, ensure_ascii=False) + "\n")
    np.savez(out_targets, ids=np.array([records[j]["id"] for j in kept]), targets=targets[kept].astype(np.float32),
             classes=np.array(classes))
    stats.update(out_manifest=str(out_manifest), out_targets=str(out_targets))
    out_stats.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    log.info("round %d relabeling (T=%.3f, alpha=%.2f, tau_flip=%.2f):\n%s", round_idx, temperature, alpha,
             tau_flip, format_stats(stats))
    for w in stats["warnings"]:
        log.warning(w)
    return stats


def main():
    from .config import load_config
    from .prompts import load_label_space

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--preds", required=True, help="preds.npz from sersys.predict over the pool")
    ap.add_argument("--manifest", default=None, help="original pool (default data.train_manifest)")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--tau_flip", type=float, default=0.9)
    ap.add_argument("--temperature", type=float, default=None, help="default: the one stored in preds.npz")
    ap.add_argument("--out_manifest", default=None)
    ap.add_argument("--out_targets", default=None)
    ap.add_argument("--shrink_warn", type=float, default=0.5)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config()
    classes = load_label_space(cfg["labels"], "en").classes
    run(args.manifest or cfg["data"]["train_manifest"], args.preds, args.round, classes, args.alpha,
        args.tau_flip, args.temperature, args.out_manifest, args.out_targets, shrink_warn=args.shrink_warn)


if __name__ == "__main__":
    main()

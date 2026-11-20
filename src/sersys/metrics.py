"""SER metrics (UA, WA, macro-F1, NLL, ECE, confusion) and ASR metrics (CER, WER)."""

from __future__ import annotations

import re
import unicodedata

import numpy as np
from scipy.special import log_softmax

_SPACE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """ASR target/scoring normalization (PLAN 2.1): Unicode NFC, lowercase, punctuation ->
    space, collapse whitespace. Vietnamese diacritics are kept (NFC composes them)."""
    text = unicodedata.normalize("NFC", text).lower()
    text = "".join(" " if unicodedata.category(ch).startswith("P") else ch for ch in text)
    return _SPACE.sub(" ", text).strip()


def asr_metrics(refs: list[str], hyps: list[str]) -> dict:
    """CER and syllable-level WER (Vietnamese words are space-separated syllables) after
    normalization. Pairs whose normalized reference is empty are skipped and counted."""
    import jiwer

    pairs = [(normalize_text(r), normalize_text(h)) for r, h in zip(refs, hyps)]
    kept = [(r, h) for r, h in pairs if r]
    if not kept:
        return {"n": 0, "skipped_empty_ref": len(pairs), "cer": float("nan"), "wer": float("nan")}
    r, h = zip(*kept)
    return {"n": len(kept), "skipped_empty_ref": len(pairs) - len(kept),
            "cer": float(jiwer.cer(list(r), list(h))), "wer": float(jiwer.wer(list(r), list(h)))}


def confusion(labels: np.ndarray, preds: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (labels, preds), 1)
    return cm


def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Top-label expected calibration error with equal-width confidence bins."""
    conf = probs.max(1)
    correct = (probs.argmax(1) == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def ser_metrics(logits: np.ndarray, labels: np.ndarray, num_classes: int, n_bins: int = 15,
                temperature: float = 1.0) -> dict:
    """All SER metrics from canonical-order logits (N, C) and integer labels (N,)."""
    logits = np.asarray(logits, dtype=np.float64) / temperature
    labels = np.asarray(labels)
    logp = log_softmax(logits, axis=1)
    probs = np.exp(logp)
    preds = probs.argmax(1)
    cm = confusion(labels, preds, num_classes)
    support = cm.sum(1)
    recall = np.divide(cm.diagonal(), support, out=np.zeros(num_classes), where=support > 0)
    predicted = cm.sum(0)
    precision = np.divide(cm.diagonal(), predicted, out=np.zeros(num_classes), where=predicted > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(num_classes),
                   where=(precision + recall) > 0)
    present = support > 0
    return {
        "n": int(len(labels)),
        "ua": float(recall[present].mean()),
        "wa": float((preds == labels).mean()),
        "macro_f1": float(f1[present].mean()),
        "nll": float(-logp[np.arange(len(labels)), labels].mean()),
        "ece": ece(probs, labels, n_bins),
        "per_class_recall": recall.tolist(),
        "confusion": cm.tolist(),
    }


def format_confusion(cm, classes: list[str]) -> str:
    w = max(9, max(len(c) for c in classes) + 1)
    lines = ["true\\pred".ljust(w) + "".join(c[:w - 1].rjust(w) for c in classes)]
    for c, row in zip(classes, cm):
        lines.append(c.ljust(w) + "".join(str(v).rjust(w) for v in row))
    return "\n".join(lines)


# --------------------------------------------------------------------------- Phase 4 helpers

def _fast_metrics(probs: np.ndarray, labels: np.ndarray, num_classes: int, n_bins: int) -> dict:
    preds = probs.argmax(1)
    cm = confusion(labels, preds, num_classes)
    support, predicted, tp = cm.sum(1), cm.sum(0), cm.diagonal()
    present = support > 0
    recall = np.divide(tp, support, out=np.zeros(num_classes), where=present)
    precision = np.divide(tp, predicted, out=np.zeros(num_classes), where=predicted > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(num_classes),
                   where=(precision + recall) > 0)
    return {"ua": recall[present].mean(), "wa": (preds == labels).mean(), "macro_f1": f1[present].mean(),
            "ece": ece(probs, labels, n_bins)}


def bootstrap_ci(probs_per_seed: list[np.ndarray], labels: np.ndarray, num_classes: int, n_boot: int = 1000,
                 n_bins: int = 15, alpha: float = 0.05, seed: int = 0) -> dict:
    """95% percentile bootstrap CI over utterances for UA, WA, macro-F1 and ECE.

    With several seeds' predictions on the same utterances, each resample is scored for
    every seed and the seed-mean is recorded: the CI of the mean-over-seeds metric."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    n = len(labels)
    samples = {k: [] for k in ("ua", "wa", "macro_f1", "ece")}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        ms = [_fast_metrics(p[idx], labels[idx], num_classes, n_bins) for p in probs_per_seed]
        for k in samples:
            samples[k].append(np.mean([m[k] for m in ms]))
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {k: [float(np.percentile(v, lo)), float(np.percentile(v, hi))] for k, v in samples.items()}


def reliability_bins(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> list[dict]:
    """Per confidence bin: count, mean confidence, accuracy (for the reliability diagram)."""
    conf, correct = probs.max(1), (probs.argmax(1) == labels)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    out = []
    for b in range(n_bins):
        m = bins == b
        out.append({"lo": float(edges[b]), "hi": float(edges[b + 1]), "count": int(m.sum()),
                    "confidence": float(conf[m].mean()) if m.any() else None,
                    "accuracy": float(correct[m].mean()) if m.any() else None})
    return out


def plot_reliability(bins: list[dict], path, title: str = "") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    width = bins[0]["hi"] - bins[0]["lo"]
    filled = [b for b in bins if b["count"]]
    ax.bar([b["lo"] for b in filled], [b["accuracy"] for b in filled], width=width, align="edge",
           color="#4C72B0", edgecolor="white", label="accuracy")
    ax.bar([b["lo"] for b in filled], [b["confidence"] - b["accuracy"] for b in filled], width=width,
           bottom=[b["accuracy"] for b in filled], align="edge", color="#DD8452", alpha=0.5, label="gap")
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    ax.set_xlim(0, 1), ax.set_ylim(0, 1)
    ax.set_xlabel("confidence"), ax.set_ylabel("accuracy")
    ax.set_title(title, fontsize=9)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

"""SER losses on canonical-order logits (B, C), returned per sample."""

from __future__ import annotations

import logging
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)


def class_weights(counts: np.ndarray, mode: str) -> torch.Tensor | None:
    """inv_sqrt_freq: w_k ∝ 1/sqrt(n_k), normalized to mean 1. Other modes: no loss weights."""
    if mode != "inv_sqrt_freq":
        return None
    counts = np.asarray(counts, dtype=np.float64)
    present = counts > 0
    if not present.all():
        # Can happen after Phase 3 relabeling. The weight of an empty class is never used.
        log.warning("classes with zero training samples: %s", np.flatnonzero(~present).tolist())
    w = np.zeros_like(counts)
    w[present] = 1.0 / np.sqrt(counts[present])
    w[present] /= w[present].mean()
    return torch.tensor(w, dtype=torch.float32)


def per_sample_loss(logits: torch.Tensor, labels: torch.Tensor, loss_type: str, smoothing: float = 0.0,
                    soft_targets: torch.Tensor | None = None) -> torch.Tensor:
    """(B,) loss. ce / smoothed_ce use hard labels; soft_kl uses (B, C) target distributions."""
    logits = logits.float()
    if loss_type == "ce":
        return F.cross_entropy(logits, labels, reduction="none")
    if loss_type == "smoothed_ce":
        return F.cross_entropy(logits, labels, reduction="none", label_smoothing=smoothing)
    if loss_type == "soft_kl":
        if soft_targets is None:
            raise ValueError("soft_kl needs soft targets")
        logp = F.log_softmax(logits, dim=-1)
        t = soft_targets.to(logp)
        return (t * (torch.log(t.clamp_min(1e-12)) - logp)).sum(-1)
    raise ValueError(f"unknown loss type {loss_type!r}")


class SmallLossFilter:
    """Per-class small-loss selection for noisy pseudo-labels (Phase 1.5).

    A micro-batch holds about one sample per class, so the per-class cut-off cannot be
    taken inside a batch. Instead, each class keeps a buffer of its recent per-sample
    losses, and a sample is left out of the backward pass when its loss is above that
    class's (1 - drop_frac) quantile. The filter is off until `warmup_epochs` have passed
    and until a class has `min_history` losses in its buffer.
    """

    def __init__(self, num_classes: int, drop_frac: float, warmup_epochs: float,
                 buffer_size: int = 2048, min_history: int = 64):
        self.num_classes = num_classes
        self.drop_frac = drop_frac
        self.warmup_epochs = warmup_epochs
        self.min_history = min_history
        self.buffers = [deque(maxlen=buffer_size) for _ in range(num_classes)]
        self.seen = np.zeros(num_classes, dtype=np.int64)
        self.dropped = np.zeros(num_classes, dtype=np.int64)

    @property
    def enabled(self) -> bool:
        return self.drop_frac > 0

    def __call__(self, losses: torch.Tensor, labels: torch.Tensor, epoch: float) -> torch.Tensor:
        """Returns a (B,) float keep-mask. Records the losses either way."""
        keep = torch.ones_like(losses, dtype=torch.float32)
        if not self.enabled:
            return keep
        values = losses.detach().float().cpu().numpy()
        ys = labels.cpu().numpy()
        active = epoch >= self.warmup_epochs
        for i, (v, y) in enumerate(zip(values, ys)):
            buf = self.buffers[y]
            if active and len(buf) >= self.min_history:
                if v > np.quantile(np.fromiter(buf, dtype=np.float64), 1.0 - self.drop_frac):
                    keep[i] = 0.0
                    self.dropped[y] += 1
            if active:
                self.seen[y] += 1
            buf.append(float(v))
        return keep.to(losses.device)

    def stats_and_reset(self) -> dict[int, float]:
        """Fraction dropped per class since the last call (only counted after warm-up)."""
        frac = {k: float(self.dropped[k] / self.seen[k]) if self.seen[k] else 0.0
                for k in range(self.num_classes)}
        self.seen[:] = 0
        self.dropped[:] = 0
        return frac


def reduce_loss(losses: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor | None,
                keep: torch.Tensor) -> torch.Tensor:
    """Weighted mean over kept samples: sum(w_y * keep * l) / sum(w_y * keep)."""
    w = keep if weights is None else keep * weights.to(losses.device)[labels]
    denom = w.sum()
    if denom <= 0:
        return losses.sum() * 0.0  # keep the graph alive; contributes nothing
    return (w * losses).sum() / denom


def load_soft_targets(path: str | Path, ids: list[str], num_classes: int) -> np.ndarray:
    """(N, C) targets aligned with `ids`, from an npz with 'ids' and 'targets'. Phase 3."""
    data = np.load(path, allow_pickle=False)
    table = {str(i): row for i, row in zip(data["ids"], data["targets"])}
    missing = [i for i in ids if i not in table]
    if missing:
        raise KeyError(f"{len(missing)} manifest ids have no soft target in {path}, e.g. {missing[:3]}")
    targets = np.stack([table[i] for i in ids]).astype(np.float32)
    if targets.shape[1] != num_classes:
        raise ValueError(f"soft targets have {targets.shape[1]} classes, expected {num_classes}")
    return targets

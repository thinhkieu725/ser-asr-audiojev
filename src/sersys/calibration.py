"""Temperature scaling: one scalar T fitted on dev logits by minimizing NLL (LBFGS on log T).

CLI:  python -m sersys.calibration --preds runs/<exp>/<seed>/best/dev_preds.npz
      writes temperature.json next to the predictions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .metrics import ser_metrics


def fit_temperature(logits: np.ndarray, labels: np.ndarray, max_iter: int = 200) -> float:
    x = torch.as_tensor(np.asarray(logits), dtype=torch.float64)
    y = torch.as_tensor(np.asarray(labels), dtype=torch.long)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(x / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())


def calibrate(logits: np.ndarray, labels: np.ndarray, num_classes: int, n_bins: int = 15) -> dict:
    t = fit_temperature(logits, labels)
    before = ser_metrics(logits, labels, num_classes, n_bins)
    after = ser_metrics(logits, labels, num_classes, n_bins, temperature=t)
    return {"temperature": t,
            "before": {k: before[k] for k in ("ua", "nll", "ece")},
            "after": {k: after[k] for k in ("ua", "nll", "ece")}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds", required=True, help="npz with 'logits' (N, C) and 'labels' (N,)")
    ap.add_argument("--bins", type=int, default=15)
    args = ap.parse_args()
    data = np.load(args.preds)
    result = calibrate(data["logits"], data["labels"], data["logits"].shape[1], args.bins)
    out = Path(args.preds).with_name("temperature.json")
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

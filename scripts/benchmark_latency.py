"""Offline inference latency (PLAN 4.3): p50 / p95 per clip for 5, 10 and 30 s clips at
tta_permutations K in {1, 3}, one clip at a time (batch size 1), on synthetic audio.

    python scripts/benchmark_latency.py --model exports/final --out exports/final/latency.json
    python scripts/benchmark_latency.py --run runs/final/1 --adapter avg5
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sersys.infer import SERPredictor  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--model")
    src.add_argument("--run")
    ap.add_argument("--adapter", default="best")
    ap.add_argument("--seconds", type=float, nargs="*", default=[5, 10, 30])
    ap.add_argument("--tta", type=int, nargs="*", default=[1, 3])
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ser = (SERPredictor.from_pretrained(args.model, device=args.device) if args.model
           else SERPredictor.from_run(args.run, args.adapter, device=args.device))
    rng = np.random.default_rng(0)
    cuda = torch.device(args.device).type == "cuda"
    results = []
    for k in args.tta:
        ser.k = k
        from sersys.scoring import tta_perms
        ser.perms = tta_perms(ser.space.reference_order, k)
        for sec in args.seconds:
            wave = (0.05 * rng.standard_normal(int(sec * ser.sample_rate))).astype(np.float32)
            times = []
            for i in range(args.warmup + args.repeats):
                if cuda:
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                ser.predict([wave], batch_size=1)
                if cuda:
                    torch.cuda.synchronize()
                if i >= args.warmup:
                    times.append(1000 * (time.perf_counter() - t0))
            row = {"tta": k, "seconds": sec, "p50_ms": float(np.percentile(times, 50)),
                   "p95_ms": float(np.percentile(times, 95)), "mean_ms": float(np.mean(times)),
                   "repeats": args.repeats}
            results.append(row)
            print(f"K={k}  {sec:>4.0f}s clip: p50 {row['p50_ms']:7.1f} ms   p95 {row['p95_ms']:7.1f} ms")
    info = {"device": torch.cuda.get_device_name(0) if cuda else "cpu",
            "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30 if cuda else None, "results": results}
    if args.out:
        Path(args.out).write_text(json.dumps(info, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

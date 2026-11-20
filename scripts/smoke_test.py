"""PLAN 1.1 environment smoke test.

Loads AudioJev through audiojev_inference, runs the 7-class Choice on the first N dev
clips, and reports predictions, peak GPU memory and per-clip latency.

    python scripts/smoke_test.py --manifest /path/to/dev_gold.jsonl --n 5

The full zero-shot dev pass (the "step 0" row) is sersys.predict; see scripts/zeroshot_dev.sh.
"""
import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sersys.config import load_config  # noqa: E402
from sersys.data import read_manifest  # noqa: E402
from sersys.prompts import load_label_space  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--lang", default="en")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    cfg = load_config()
    space = load_label_space(cfg["labels"], args.lang)
    records = read_manifest(args.manifest or cfg["data"]["dev_manifest"], space.classes)[: args.n]

    from audiojev_inference import AudioJev, Choice
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    model = AudioJev(device=args.device, revision=cfg["model"]["revision"])
    print(f"load: {time.perf_counter() - t0:.1f}s, weights {torch.cuda.memory_allocated() / 2**30:.2f} GB")
    question = {"emo": Choice(instructions=space.question, criteria=space.criteria())}

    model.system_one(state=records[0]["audio_path"], questions=question)  # warm-up
    torch.cuda.synchronize()
    for rec in records:
        t0 = time.perf_counter()
        ans = model.system_one(state=rec["audio_path"], questions=question)["answers"]["emo"]
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        top = sorted(ans["probabilities"].items(), key=lambda kv: -kv[1])[:3]
        print(f"{rec['id']}: gold={rec['label']:<9} pred={ans['choice']:<9} {dt * 1000:6.0f} ms  "
              + "  ".join(f"{k}={v:.3f}" for k, v in top))
    print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")


if __name__ == "__main__":
    main()

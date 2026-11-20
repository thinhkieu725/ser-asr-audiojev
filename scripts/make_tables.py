"""Collect runs/*/*/<split>_metrics.json into Markdown and LaTeX tables (PLAN 4.1).

Per experiment (runs/<exp>/<seed>/): mean ± std over seeds of UA, WA, macro-F1 and ECE
(percent), and a 95% bootstrap CI of the seed-mean. Each of the --n_boot resamples of
test utterances is scored for every seed, using each seed's <split>/preds.npz and dev-fitted T.
Also writes a per-class recall table.

    python scripts/make_tables.py                       # all experiments with test results
    python scripts/make_tables.py --exps zeroshot_en p1_best p2_best final --out tables
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sersys.metrics import bootstrap_ci  # noqa: E402
from sersys.relabel import softmax  # noqa: E402

METRICS = [("ua", "UA"), ("wa", "WA"), ("macro_f1", "Macro-F1"), ("ece", "ECE")]


def collect(runs: Path, split: str, exps: list[str] | None, n_boot: int) -> list[dict]:
    found = {}
    for f in sorted(runs.glob(f"*/*/{split}_metrics.json")):
        found.setdefault(f.parent.parent.name, []).append(f.parent)
    names = exps or sorted(found)
    rows = []
    for exp in names:
        if exp not in found:
            print(f"warning: no {split} results for {exp}", file=sys.stderr)
            continue
        seeds = sorted(found[exp], key=lambda p: p.name)
        ms = [json.loads((s / f"{split}_metrics.json").read_text()) for s in seeds]
        classes = ms[0]["classes"]
        row = {"exp": exp, "seeds": [s.name for s in seeds], "classes": classes}
        for key, _ in METRICS:
            vals = np.array([m[key] for m in ms]) * 100
            row[key] = (vals.mean(), vals.std(ddof=1) if len(vals) > 1 else 0.0)
        row["recall"] = np.mean([m["per_class_recall"] for m in ms], axis=0) * 100
        # CI of the seed-mean on a shared resample of utterances (aligned by id)
        preds = [np.load(s / split / "preds.npz") for s in seeds]
        common = sorted(set.intersection(*[set(map(str, p["ids"])) for p in preds]))
        probs = []
        for p, m in zip(preds, ms):
            pos = {str(i): j for j, i in enumerate(p["ids"])}
            idx = [pos[i] for i in common]
            probs.append(softmax(p["logits"][idx], m["temperature"]))
            labels = p["labels"][idx]
        ci = bootstrap_ci(probs, labels, len(classes), n_boot)
        row["ci"] = {k: (100 * v[0], 100 * v[1]) for k, v in ci.items()}
        row["n"] = len(common)
        rows.append(row)
    return rows


def cell(row, key, latex=False):
    mean, std = row[key]
    lo, hi = row["ci"][key]
    pm = r"$\pm$" if latex else "±"
    s = f"{mean:.1f}"
    if len(row["seeds"]) > 1:
        s += f" {pm} {std:.1f}"
    return s + f" [{lo:.1f}, {hi:.1f}]"


def markdown(rows, split) -> str:
    out = [f"## {split} results (%, mean ± std over seeds [95% bootstrap CI])", "",
           "| Experiment | Seeds | n | " + " | ".join(n for _, n in METRICS) + " |",
           "|---|---|---|" + "---|" * len(METRICS)]
    for r in rows:
        out.append(f"| {r['exp']} | {len(r['seeds'])} | {r['n']} | " + " | ".join(cell(r, k) for k, _ in METRICS) + " |")
    if rows:
        classes = rows[0]["classes"]
        out += ["", "## Per-class recall (%, mean over seeds)", "",
                "| Experiment | " + " | ".join(classes) + " |", "|---|" + "---|" * len(classes)]
        for r in rows:
            out.append(f"| {r['exp']} | " + " | ".join(f"{v:.1f}" for v in r["recall"]) + " |")
    return "\n".join(out) + "\n"


def latex(rows, split) -> str:
    esc = lambda s: s.replace("_", r"\_")  # noqa: E731
    out = [r"\begin{table}[t]", r"\centering", r"\small",
           r"\begin{tabular}{l" + "c" * len(METRICS) + "}", r"\toprule",
           "Experiment & " + " & ".join(n for _, n in METRICS) + r" \\", r"\midrule"]
    for r in rows:
        out.append(esc(r["exp"]) + " & " + " & ".join(cell(r, k, latex=True) for k, _ in METRICS) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}",
            rf"\caption{{{split.capitalize()} results (\%): mean $\pm$ std over seeds, 95\% bootstrap CI in brackets.}}",
            r"\end{table}"]
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--split", default="test")
    ap.add_argument("--exps", nargs="*", default=None, help="rows, in order (default: all, sorted)")
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--out", default="tables")
    args = ap.parse_args()
    rows = collect(Path(args.runs), args.split, args.exps, args.n_boot)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.split}_results.md").write_text(markdown(rows, args.split), encoding="utf-8")
    (out / f"{args.split}_results.tex").write_text(latex(rows, args.split), encoding="utf-8")
    print(markdown(rows, args.split))


if __name__ == "__main__":
    main()

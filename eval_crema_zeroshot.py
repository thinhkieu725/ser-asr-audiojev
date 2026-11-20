"""Zero-shot AudioJev SER over all CREMA-D wavs (prompt from zeroshot_audioJEV_ser.py).

Loads the model once, writes one JSON line per file (resumable), then prints
UA and confusion matrices.
"""
import json
import sys
from pathlib import Path

import numpy as np

DATA = Path("data/Crema")
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "results/crema_zeroshot.jsonl")
CODE2EMO = {"ANG": "angry", "DIS": "disgust", "FEA": "fear",
            "HAP": "happy", "NEU": "neutral", "SAD": "sad"}
EMO = {
    "neutral": "The speaker sounds calm and emotionally neutral",
    "happy":   "The speaker sounds happy or joyful",
    "sad":     "The speaker sounds sad",
    "angry":   "The speaker sounds angry",
    "fear":    "The speaker sounds fearful or anxious",
    "disgust": "The speaker sounds disgusted",
    "surprise": "The speaker sounds surprised",
}


def run():
    from audiojev_inference import AudioJev, Choice
    files = sorted(DATA.glob("*.wav"))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if OUT.exists():
        done = {json.loads(l)["file"] for l in OUT.open(encoding="utf-8") if l.strip()}
    todo = [f for f in files if f.name not in done]
    print(f"{len(files)} files, {len(done)} done, {len(todo)} to go", flush=True)
    if not todo:
        return
    model = AudioJev(device="cuda:0")
    question = {"emo": Choice(
        instructions="Judging by tone of voice, which emotion does the speaker express?",
        criteria=EMO)}
    with OUT.open("a", encoding="utf-8") as fh:
        for i, f in enumerate(todo, 1):
            rec = {"file": f.name, "label": CODE2EMO[f.stem.split("_")[2]]}
            try:
                ans = model.system_one(state=str(f), questions=question)["answers"]["emo"]
                rec.update(pred=ans["choice"], probs=ans["probabilities"])
            except Exception as exc:  # keep going; report failures at the end
                rec.update(pred=None, error=repr(exc))
            fh.write(json.dumps(rec) + "\n")
            if i % 200 == 0:
                fh.flush()
                print(f"{i}/{len(todo)}", flush=True)


def report():
    recs = [json.loads(l) for l in OUT.open(encoding="utf-8") if l.strip()]
    errs = [r for r in recs if r.get("pred") is None]
    recs = [r for r in recs if r.get("pred") is not None]
    true_cls = list(CODE2EMO.values())
    pred_cls = true_cls + ["surprise"]

    def show(name, key, cols):
        cm = np.zeros((len(true_cls), len(cols)), dtype=int)
        for r in recs:
            cm[true_cls.index(r["label"]), cols.index(key(r))] += 1
        recall = cm[:, :len(true_cls)].diagonal() / cm.sum(1)
        print(f"\n== {name} ==  n={cm.sum()}  UA={recall.mean():.4f}  "
              f"WA={cm.diagonal().sum() / cm.sum():.4f}")
        w = 9
        print("true\\pred".ljust(w) + "".join(c[:8].rjust(w) for c in cols) + "   recall")
        for t, row, rc in zip(true_cls, cm, recall):
            print(t.ljust(w) + "".join(str(v).rjust(w) for v in row) + f"   {rc:.3f}")

    show("7-way prompt as written (surprise counted wrong)", lambda r: r["pred"], pred_cls)
    show("restricted to 6 CREMA classes (argmax over 6 probs)",
         lambda r: max(true_cls, key=lambda c: r["probs"][c]), true_cls)
    print(f"\nfailed files: {len(errs)}")
    for r in errs[:10]:
        print(" ", r["file"], r["error"])


if __name__ == "__main__":
    run()
    report()

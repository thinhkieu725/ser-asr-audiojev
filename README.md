# sersys — Vietnamese Speech Emotion Recognition on AudioJev

This repository fine-tunes **AudioJev** (`shlv/AudioJev`, a Qwen2.5-Omni-3B model) to
recognise **7 emotions in Vietnamese speech**. It also uses speech recognition (ASR) as a
helper task. It follows [PLAN.md](PLAN.md), which explains the reasons behind each step.

You will:

1. prepare data, server and environment (**Stage 0**),
2. fine-tune for emotion recognition only (**Stage 1**),
3. add ASR as an auxiliary task (**Stage 2**),
4. improve on the noisy pseudo-labels with self-training and checkpoint averaging (**Stage 3**),
5. evaluate once on the test set, build the paper tables, export the model (**Stage 4**),
6. run **inference** with the final model.

Every stage ends with a decision: you pick the best configuration on the **dev** set and
freeze it as a config file, and the next stage builds on that file.

| Stage | Main command | You get | You decide / freeze |
|---|---|---|---|
| 0 | `bash scripts/setup_env.sh` | environment, data paths set | — |
| 1 | `scripts/smoke.sh`, `scripts/p1_grid.sh` | SER models + zero-shot baseline | `configs/exp/p1_best.yaml` |
| 2 | `scripts/p2_grid.sh` | SER + ASR models, ASR error rates | `configs/exp/p2_best.yaml` |
| 3 | `scripts/p3_selftrain.py`, `sersys.average` | self-trained, averaged models | `configs/exp/final.yaml` |
| 4 | `scripts/p4_final.sh` | test results, tables, exported model | — |

---

## Stage 0 — Prepare

### 0.1 Server

- **GPU:** one card with **40–48 GB** that supports BF16 (A100, A6000, L40S, H100, …).
- **Disk:**
  - ~15 GB for the model in the Hugging Face cache
  - ~1–2 GB per training run (adapters, logs, predictions)
  - ~8 GB per exported model
- **Linux** with conda. Run long jobs inside `tmux` or `screen`.

### 0.2 Data

You need three **JSON Lines** files, with one utterance per line:

| File | What it is |
|---|---|
| `train_pseudo.jsonl` | the large pseudo-labeled training pool (> 50 h) |
| `dev_gold.jsonl` | human-labeled dev set: used for **all** model choices |
| `test_gold.jsonl` | human-labeled test set: used **once**, at Stage 4 |

Each line has **exactly these four fields**:

```json
{"id": "spk012_utt0345", "audio_path": "/data/vi_ser/wav/spk012_utt0345.wav", "label": "angry", "text": "transcript in Vietnamese"}
```

Check before you start:

- [ ] `id` is unique across **all three** files.
- [ ] `audio_path` is an **absolute** path the server can read. Any format `soundfile` reads
      works (wav, flac, ogg, …), at any sample rate, mono or stereo.
- [ ] `label` is one of the class names in [configs/labels.yaml](configs/labels.yaml):
      `neutral, happy, sad, angry, fear, disgust, surprise`. Rename either side if your
      dataset uses different names.
- [ ] `text` is the transcript. Stage 2 uses it as the ASR target.
- [ ] The splits are **speaker-disjoint**. The code cannot check this, because there is no
      speaker field.

The loader stops with a clear error on a missing field, an unknown label, a duplicate id or
a missing audio file.

### 0.3 Labels and prompts

Open [configs/labels.yaml](configs/labels.yaml):

- [ ] The class list matches your labels. **Never change the order** once you have started
      training.
- [ ] Ask a native speaker to review the Vietnamese candidate descriptions and question
      (`vi`). They are used when `ser.prompt_lang: vi`.

### 0.4 Environment

```bash
git clone <this repo> && cd ser-asr
bash scripts/setup_env.sh
conda activate ser-audiojev
```

`setup_env.sh` does the following:
- creates the conda env `ser-audiojev`
- installs PyTorch with CUDA. Edit the `cu124` index URL in the script to match the
  server's driver; `nvidia-smi` shows the CUDA version.
- installs `requirements.txt`, including the `audiojev_inference` library
- installs this package with `pip install -e .`
- tries to build `flash-attn`. If that fails, nothing breaks: the code falls back to `sdpa`.

### 0.5 Point the code at your data and GPU

1. **[scripts/env.sh](scripts/env.sh)** (every script sources it):
   ```bash
   export CUDA_VISIBLE_DEVICES=0                  # which GPU to use
   export HF_HOME=/big/disk/hf_home               # where the model is downloaded
   ```
2. **[configs/base.yaml](configs/base.yaml)**, the `data:` block:
   ```yaml
   train_manifest: /data/vi_ser/manifests/train_pseudo.jsonl
   dev_manifest:   /data/vi_ser/manifests/dev_gold.jsonl
   test_manifest:  /data/vi_ser/manifests/test_gold.jsonl
   ```
3. **The model revision** is pinned in `configs/base.yaml` (`model.revision`). Keep it, so
   results stay reproducible. The model is downloaded automatically on first use.

**You are ready when** `conda activate ser-audiojev && python -c "import sersys"` works and
the three manifest paths exist.

---

## Stage 1 — Emotion recognition only

### 1.1 Smoke test and parity gate (must pass before any training)

```bash
bash scripts/smoke.sh /data/vi_ser/manifests/dev_gold.jsonl
```

It runs three things:

1. **AudioJev on 5 dev clips.** It prints gold vs. predicted label, latency, and peak GPU
   memory (expect roughly 9–12 GB).
2. **The CPU unit tests**, which should all pass.
3. **The parity test**, which checks that this repo's scoring reproduces the official
   AudioJev library on 50 dev clips. Look for:
   ```
   batch=1: max |dp| = 3.1e-05 over 50 clips; argmax agreement 1.000   <- must be < 1e-3
   batch=8: max |dp| = ...
   ```

**If the `batch=1` parity check fails, stop.** Everything after it depends on the scoring
being identical to the library. If only `batch=8` is slightly above 1e-3, it is BF16 noise
from batching. You can check that with `SER_PARITY_BATCH_TOL=5e-3` and continue.

### 1.2 Zero-shot baseline ("step 0" row)

```bash
bash scripts/zeroshot_dev.sh
```

This scores the untouched AudioJev on dev with English and Vietnamese prompts, at K = 1 and
K = 3 candidate orders.

**Observe:** `runs/zeroshot_en/dev_tta1/metrics.json` → `ua`, `macro_f1`, `ece`. These are
the numbers every fine-tuned model must beat.

### 1.3 Training experiments

```bash
bash scripts/p1_grid.sh a      # E1.1 (English prompt) and E1.2 (Vietnamese prompt)
```

Each run writes to `runs/<experiment>/<seed>/`. While it trains:

```bash
tail -f runs/p1_e1_frozen_smoothce/1/train.log
tensorboard --logdir runs        # then open http://<server>:6006
```

What to look at:

| Where | What healthy looks like |
|---|---|
| `train.log` start | trainable parameter count, class counts, `0/N clips longer than 30 s` (should be near 0) |
| `train/loss` | goes down over the first few hundred steps |
| `[eval step N] UA …` every 500 steps | dev UA rises above the zero-shot UA; a confusion matrix is printed |
| `dev/ece` | stays small, and is fixed further by temperature scaling at the end |
| end of log | `temperature 1.4: dev ECE 0.09 -> 0.03`, `done: best dev UA …` |

**Results:**
- `runs/<exp>/1/results.json`: `best_dev_ua` and the full eval history
- `runs/<exp>/1/best/`: the best adapter, its dev predictions and `temperature.json`

**Out of GPU memory?** Halve the micro-batch and double accumulation, which keeps the
effective batch at 48:
```bash
scripts/train.sh configs/exp/p1_e1_frozen_smoothce.yaml 1 --set optim.micro_batch=4 optim.grad_accum=12
```

Then run the rest of the grid. **Before each stage**, copy the winning settings so far into
the next configs; each config file says what to copy.

```bash
bash scripts/p1_grid.sh b      # E1.3 plain CE, E1.4 small-loss filtering (0.1, 0.2)
bash scripts/p1_grid.sh c      # E1.5/E1.6 unfreeze the top 4 / 8 audio-encoder blocks
bash scripts/p1_grid.sh d      # E1.7 no candidate shuffling, E1.8 LoRA rank 16 / 64
bash scripts/p1_grid.sh seeds p1_e1_frozen_smoothce p1_e4_smallloss_0.1   # top 2 configs, seeds 2-3
bash scripts/eval_tta.sh runs/<best_exp>/1                                 # dev metrics at K = 1, 3, 7
```

To compare runs, put the `best_dev_ua` values from `results.json` side by side.

**Decide:** write the winning settings into **`configs/exp/p1_best.yaml`**. It currently
points at E1.1 as a placeholder.

✅ Done when: the parity test passed, the best config beats zero-shot on dev UA, and its
dev ECE after temperature scaling is no worse than zero-shot's.

---

## Stage 2 — Add ASR as an auxiliary task

```bash
bash scripts/eval_asr.sh zeroshot          # how well the base model transcribes Vietnamese
bash scripts/p2_grid.sh a                  # λ = 0 (control), 0.1, 0.3, 0.3 → 0.05
```

Each training step now also runs a transcription pass, weighted by λ. What to look at:

| Where | What to check |
|---|---|
| `grad_components/ser` vs `grad_components/asr_weighted` (first 300 steps) | neither is consistently 10× the other; if ASR dominates, λ is too high |
| `train/loss_asr` | goes down |
| `train/asr_targets_truncated` | ~0; if not, transcripts are longer than 448 tokens |
| `[asr eval step N] CER … WER …` (every 3rd eval) | CER drops below the zero-shot CER |
| `runs/<exp>/1/asr/dev_hyps_step*.jsonl` | read some reference/hypothesis pairs; they should look like the reference transcripts |
| `dev/ua` | the number that matters: does any λ > 0 beat λ = 0? |

Then run seeds 2–3 for λ = 0 and the best λ:

```bash
bash scripts/p2_grid.sh seeds p2_e0_lambda0 p2_e2_lambda0.3
bash scripts/p2_grid.sh e4     # optional: transcribe-then-classify (slower; keep only if clearly better)
```

**Decide:** point **`configs/exp/p2_best.yaml`** at the winner. If no λ > 0 beats λ = 0 by
more than the seed-to-seed spread, carry λ = 0 forward and report the negative result.

---

## Stage 3 — Self-training and checkpoint averaging

### 3.1 Self-training rounds

Pass the run directory of your Stage 2 winner, e.g. `runs/p2_e2_lambda0.3/1`:

```bash
python scripts/p3_selftrain.py --teacher_run runs/p2_e2_lambda0.3/1 \
                               --base_config configs/exp/p2_best.yaml
```

Each round:
1. scores the **whole pool** with the current best model. This takes a while; use
   `--tta 1` to halve the time.
2. relabels each training clip: keep its label, flip it to a confident prediction, or drop it.
3. retrains from the original AudioJev weights on the relabeled data.

It stops after 2 rounds, or as soon as a round gains less than 0.5 UA points. You can
restart it at any time; finished steps are skipped.

**Observe the relabeling table** printed each round:

```
class       before   after  ratio    kept flip-out flip-in    drop
neutral       9120    8710   0.96   81.2%     6.1%     410   12.7%
...
overall: kept 78.4%  flipped 5.2%  dropped 16.4%
```

- A **WARNING `class 'x' shrinks to 35%`** means the model may be reinforcing its own
  mistakes for that class. Look at that class's dev recall before trusting the round.
- `runs/p3_selftrain_summary.json` lists dev UA per round and the best run.

### 3.2 Checkpoint averaging

```bash
python -m sersys.average --run_dir runs/p3_round1/1 --last 5 --refit
```

**Observe:** `averaged model: dev UA … | T … ECE …`. Compare
`runs/p3_round1/1/avg5/dev_metrics.json` with `runs/p3_round1/1/best/dev_metrics.json`
and keep whichever is better.

### 3.3 Optional: fine-tune on extra gold data

Only do this if you have **additional** human-labeled training clips, separate from dev
and test. Fill in the placeholders in
[configs/exp/p3_gold_ft.yaml](configs/exp/p3_gold_ft.yaml), then run:

```bash
scripts/train.sh configs/exp/p3_gold_ft.yaml 1
```

**Decide:** write the final choice into **`configs/exp/final.yaml`**: which round,
averaging or not, gold fine-tuning or not. **After this, change nothing.**

---

## Stage 4 — Final evaluation and export

```bash
bash scripts/p4_final.sh train                 # final.yaml with seeds 1, 2, 3
# if final.yaml uses averaging, average each seed first:
for s in 1 2 3; do python -m sersys.average --run_dir runs/final/$s --last 5 --refit; done
bash scripts/p4_final.sh test avg5             # or: test best
```

**The test set is evaluated once per run.** A second attempt is refused, and every access
is logged in `runs/final/<seed>/test_access.log`. Per seed you get:

- `test_metrics.json`: UA, WA, macro-F1, ECE with 95% bootstrap intervals, per-class
  recall, confusion matrix
- `test_reliability.png`: the calibration plot (bars close to the diagonal = well calibrated)

Paper tables, with every ablation row scored the same way:

```bash
# edit the run list under "ablations" in scripts/p4_final.sh first
bash scripts/p4_final.sh ablations
bash scripts/p4_final.sh tables                # -> tables/test_results.md and .tex
```

Rows look like `61.2 ± 0.8 [59.0, 63.1]`: mean ± std over seeds, then the 95% confidence
interval.

Export one seed as a standalone model:

```bash
bash scripts/p4_final.sh export 1 avg5         # -> exports/final/
```

**Observe:**
- `export validation PASSED: argmax agreement 1.0000, max |dp| …`. The exported model must
  reproduce the run's dev predictions. If it fails, re-export with
  `python -m sersys.export … --dtype float32`.
- Latency is printed and saved to `exports/final/latency.json`, e.g.
  `K=1  10s clip: p50 85 ms  p95 97 ms`.

Before sharing the exported weights, check the **Qwen Research License** (inherited from
AudioJev); see PLAN.md 4.4.

---

## Inference with the final model

Everything runs offline in Python. You need the conda environment and the `exports/final/`
folder; the manifests and training runs are not needed.

### From Python

```python
from sersys.infer import SERPredictor

ser = SERPredictor.from_pretrained("exports/final", device="cuda:0")

ser.predict_file("clip.wav")
# {'label': 'angry', 'confidence': 0.83,
#  'probs': {'neutral': 0.04, 'happy': 0.01, 'sad': 0.03, 'angry': 0.83, ...},
#  'cropped': False}

results = ser.predict(["a.wav", "b.flac", (waveform, 44100)], batch_size=8)
```

- **Inputs:** file paths, `(waveform, sample_rate)` tuples (any rate, mono or stereo), or
  16 kHz mono numpy arrays.
- **`confidence`** is the calibrated probability of the predicted label. Use a threshold
  on it, e.g. send clips under 0.5 to a human.
- **`cropped: True`** means the clip was longer than 30 s and only its middle 30 s was used.
- **Speed vs. robustness:** `SERPredictor.from_pretrained(..., tta_permutations=1)` is the
  fastest. The exported value (see `ser_config.json`) averages over several candidate
  orders and is a little more robust.

### From the command line

```bash
# files and/or folders (searched recursively)
python -m sersys.infer --model exports/final recordings/ extra.wav --out predictions.jsonl

# a JSONL list with "audio_path" (and optionally "id", "label")
python -m sersys.infer --model exports/final --manifest my_clips.jsonl --out predictions.jsonl
```

Each output line is `{"id", "audio_path", "label", "confidence", "probs", "cropped"}`. If
the manifest has gold labels, UA / WA / F1 / ECE and a confusion matrix are printed as well.

### Without exporting

To use a training run directly (base AudioJev + adapter):

```python
ser = SERPredictor.from_run("runs/final/1", adapter="avg5")    # or "best"; "none" = zero-shot
```

```bash
python -m sersys.infer --run runs/final/1 --adapter avg5 recordings/ --out predictions.jsonl
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `CUDA out of memory` in training | `--set optim.micro_batch=4 optim.grad_accum=12` (Stage 2: `2` and `24`) |
| `flash_attn not installed; falling back to sdpa` | harmless; training is a little slower |
| `unknown label 'xyz'` | the manifest label is not in `configs/labels.yaml` |
| `N audio files missing` | `audio_path` is wrong, or not absolute, on this server |
| parity test fails at batch = 1 | do not train. Check `transformers` (4.52–4.57) and the pinned `model.revision` |
| `test_metrics.json exists … evaluated once` | intended. Use `--force` only if the first run was broken, and say so in the paper |
| export validation FAILED | re-export with `--dtype float32` |
| `transcript_in_ser … not supported` at export | E2.4 models cannot be exported; pick a non-E2.4 final config |

## Reference

### What a run directory contains

```
runs/<experiment>/<seed>/
├── config.yaml, git_hash.txt, pip_freeze.txt, model_revision.txt   # reproducibility
├── train.log, tb/                                                   # logs, TensorBoard
├── ckpt-step*/                                                      # last checkpoints (for averaging)
├── best/            adapter + dev_preds.npz + temperature.json      # best dev-UA checkpoint
├── avg5/            (after sersys.average) averaged adapter
├── asr/             (Stage 2) dev transcripts
├── results.json     best dev UA, eval history
└── test_metrics.json, test/, test_reliability.png                   # (Stage 4)
```

### Code

| Module | Role |
|---|---|
| `src/sersys/data.py` | manifest checks, duration cache, audio loading, cropping, length bucketing |
| `src/sersys/prompts.py`, `scoring.py` | the AudioJev multiple-choice prompt and its scoring |
| `src/sersys/model.py` | model loading, LoRA, audio-encoder unfreezing, checkpoints |
| `src/sersys/losses.py`, `metrics.py`, `calibration.py` | losses, metrics (UA, F1, ECE, CER, WER), temperature |
| `src/sersys/train.py`, `predict.py` | training loop, batched dev/test prediction |
| `src/sersys/asr.py` | Stage 2 ASR targets, loss, decoding, evaluation |
| `src/sersys/relabel.py`, `average.py` | Stage 3 relabeling, checkpoint averaging |
| `src/sersys/evaluate.py`, `export.py`, `infer.py` | Stage 4 test evaluation, export, inference |
| `configs/` | `base.yaml` defaults, `labels.yaml`, `exp/` one file per experiment |
| `scripts/` | one script per stage |

Configs can inherit from each other (`inherit: configs/exp/p1_best.yaml`), and any value
can be overridden on the command line, e.g. `--set optim.lr=5e-5`.

### Tests

```bash
pytest -m "not gpu"                                   # CPU, ~30 s, uses a tiny random model
SER_DEV_MANIFEST=/path/dev_gold.jsonl pytest -m gpu   # parity with the AudioJev library (GPU)
```

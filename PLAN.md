# PLAN.md — Vietnamese SER on AudioJev with auxiliary ASR

Implementation plan for a speech emotion recognition (SER) system fine-tuned from the
AudioJev checkpoint (`shlv/AudioJev`, fine-tuned from Qwen2.5-Omni-3B), with ASR as an
auxiliary task. Target: one Linux server with a single 40–48 GB GPU (BF16-capable).

- **Primary objective:** 7-class SER on Vietnamese speech. Headline metric: unweighted
  accuracy (UA, mean per-class recall) and macro-F1 on the human-labeled (gold) set.
- **Secondary objective:** calibrated class probabilities (ECE).
- **Auxiliary:** ASR, used to shape representations; CER/WER reported, not optimized.
- **End goals:** a research paper (ablations, seeds, confidence intervals) and an
  exported model with an offline Python inference API/CLI. No web service or
  deployment is in scope.

Phases:

1. Single-task SER fine-tuning of AudioJev
2. ASR as an auxiliary task
3. Beyond the pseudo-label ceiling (self-training, checkpoint averaging)
4. Final evaluation, export and offline inference

---

## 0. Prerequisites and conventions (applies to all phases)

### 0.1 Inputs assumed to exist

The data audit is done and produced three manifests (JSON Lines, one utterance per line):

| File | Content |
|---|---|
| `data/manifests/train_pseudo.jsonl` | Pseudo-labeled training pool (>50 h) |
| `data/manifests/dev_gold.jsonl` | Gold dev split (~30% of gold), used for all model selection |
| `data/manifests/test_gold.jsonl` | Gold test split (~70% of gold), touched **once**, in Phase 4 |

Splits are made upstream and are assumed to be **speaker-disjoint**. The manifests carry
no speaker field, so this cannot be checked here. Manifest schema: exactly four fields.

```json
{"id": "spk012_utt0345", "audio_path": "/data/vi_ser/wav/spk012_utt0345.wav",
 "label": "angry", "text": "transcription in Vietnamese"}
```

| Field | Type | Notes |
|---|---|---|
| `id` | str | Unique across all manifests |
| `audio_path` | str | Absolute path; any format `soundfile` reads |
| `label` | str | One of the classes in `configs/labels.yaml` (hard label) |
| `text` | str | Transcript (ASR target in Phase 2) |

The loader validates the four fields and fails on unknown labels or missing audio.
Anything derived (clip duration, soft targets from Phase 3) lives in sidecar files keyed
by `id` under `data/cache/` and never goes back into the manifests. The class order is
fixed by `configs/labels.yaml` and never changes.

### 0.2 Repository layout

```
ser-audiojev/
├── configs/
│   ├── labels.yaml            # canonical class order + candidate descriptions (EN and VI)
│   ├── base.yaml              # shared defaults
│   └── exp/                   # one YAML per experiment, overriding base.yaml
├── data/manifests/            # 4-field manifests, read-only
├── data/cache/                # durations.json, soft_targets_round{r}.npz (keyed by id)
├── src/sersys/
│   ├── data.py                # manifest reader, duration cache, audio loading, cropping, bucketing
│   ├── config.py              # base.yaml + experiment yaml (`inherit:`) + --set overrides
│   ├── prompts.py             # SER choice prompt + ASR prompt builders
│   ├── scoring.py             # reproduction of AudioJev candidate scoring (Phase 1.3)
│   ├── model.py               # load thinker, LoRA, freezing, param groups
│   ├── losses.py              # CE / smoothed CE / soft-label KL / ASR CE
│   ├── metrics.py             # UA, WA, macro-F1, ECE, NLL, CER, WER
│   ├── calibration.py         # temperature scaling
│   ├── train.py               # training loop (accelerate)
│   ├── predict.py             # batched inference, permutation averaging
│   ├── asr.py                 # ASR targets, loss, greedy decoding, CER/WER (Phase 2)
│   ├── relabel.py             # self-training relabeling (Phase 3)
│   ├── average.py             # checkpoint averaging (Phase 3)
│   ├── evaluate.py            # one-time test evaluation, bootstrap CIs, reliability (Phase 4)
│   ├── export.py              # merge LoRA, package model, parity validation (Phase 4)
│   └── infer.py               # offline inference API + CLI (Phase 4)
├── scripts/                   # thin shell wrappers per experiment, make_tables.py, benchmark_latency.py
├── tests/
│   ├── test_scoring_parity.py
│   ├── test_permutation.py
│   └── test_collate.py
└── runs/<exp_name>/<seed>/    # config dump, git hash, logs, adapters, predictions
```

### 0.3 Environment

```bash
conda create -n ser-audiojev python=3.11 -y && conda activate ser-audiojev
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124   # match the server's driver
pip install "transformers>=4.52" peft accelerate soundfile librosa jiwer scikit-learn \
            pyyaml tensorboard matplotlib
pip install flash-attn --no-build-isolation      # optional, if it builds on the server
pip install -e git+https://github.com/SihanLv/AudioJev-Inference#egg=audiojev_inference
```

Shared-server hygiene:

- Pin the GPU: `export CUDA_VISIBLE_DEVICES=<id>`. Run in `tmux`, log to `runs/`.
- Put `HF_HOME` on a large disk. Pin the AudioJev model **revision hash** in `base.yaml`
  so a model update on the Hub cannot silently change results.
- Every run writes `config.yaml`, `git_hash.txt`, `pip_freeze.txt`, and the AudioJev
  revision into its run directory.

### 0.4 Base config (excerpt)

```yaml
# configs/base.yaml
model:
  name: shlv/AudioJev
  revision: <pin-commit-hash>
  dtype: bfloat16
  attn_impl: flash_attention_2          # fall back to sdpa
  drop_modules: [visual, talker, token2wav]   # if present in the checkpoint
data:
  sample_rate: 16000
  max_seconds: 30
  bucket_by_length: true
ser:
  prompt_lang: en                        # en | vi
  question: "Judging by the tone of voice, which emotion does the speaker express?"
  permute_candidates: true
  tta_permutations: 1                    # Phase 1.9
lora:
  r: 32
  alpha: 64
  dropout: 0.05
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
  restrict_to: llm                       # never apply LoRA to the audio tower
audio_encoder:
  unfreeze_top_k: 0                      # 0 = frozen
  lr: 1.0e-5
optim:
  lr: 1.0e-4
  weight_decay: 0.0
  warmup_ratio: 0.03
  scheduler: cosine
  micro_batch: 8
  grad_accum: 6                          # effective batch 48
  epochs: 3
  grad_checkpointing: true
  max_grad_norm: 1.0
loss:
  type: smoothed_ce                      # ce | smoothed_ce  (soft_kl: Phase 3 only)
  smoothing: 0.1
  class_weighting: inv_sqrt_freq         # none | inv_sqrt_freq | balanced_sampler
  soft_targets: null                     # Phase 3: path to data/cache/soft_targets_round{r}.npz
  small_loss:                            # Phase 1.5, noisy-label filtering
    drop_frac: 0.0                       # 0 = off
    warmup_epochs: 1
eval:
  every_steps: 500
  select_metric: dev_ua
  early_stop_patience: 4
seed: 1
```

---

## Phase 1 — Single-task SER fine-tuning of AudioJev

**Goal:** the best SER-only model fine-tuned from AudioJev, with probabilities that are
still calibrated.

### 1.1 Environment smoke test

- Load AudioJev through `audiojev_inference.AudioJev` and run one 7-class `Choice` on
  five dev clips.
- Record peak memory (`torch.cuda.max_memory_allocated`) and latency.
- Run the zero-shot pass over the whole dev set. This becomes the "step 0" row of every
  results table.

### 1.2 Data pipeline (`data.py`, `prompts.py`)

- Read the manifest, then load audio with `soundfile`, convert to mono, and resample to 16 kHz.
- **Duration cache:** on first use, read each clip's duration with `soundfile.info` and
  write `data/cache/durations.json` (`id → seconds`). Later runs reuse it. The
  bucketing, the cropping log and the corpus-hours sanity check all read from it.
- **Cropping:** for clips over 30 s, take a random 30 s crop in training and a center crop
  at eval. Log how many clips are affected. It should be close to none.
- **Length-bucketed sampler:** sort into buckets by duration and shuffle within buckets.
  This keeps padding low.
- **Class balance:** use either a `WeightedRandomSampler` (`balanced_sampler`) or loss
  weights (`inv_sqrt_freq`). Not both.
- `prompts.py` builds the SER choice prompt from `labels.yaml`. Candidate descriptions
  exist in EN and VI, and `prompt_lang` selects one.
- `test_collate.py` checks that a batch round-trips: shapes, attention masks, and
  candidate order.

### 1.3 Reproduce AudioJev's candidate scoring (`scoring.py`): critical path

The inference repo does not document how candidate probabilities are computed, and no
training code is released. Reproduce it from the `audiojev_inference` source:

1. Read `AudioJev.system_one` and the `Choice` handling. Write down:
   - the exact chat template / prompt string, including where audio, the question and the
     candidates are inserted, and how candidates are enumerated (letters, indices, keys);
   - which forward-pass outputs are read (logits at which positions, which vocabulary
     IDs), and how they become a distribution (softmax over candidate tokens, any
     temperature, any order-calibration step).
2. Implement:
   ```python
   def build_choice_batch(processor, audios, question, cand_texts, perms) -> BatchEncoding
   def candidate_logits(model_out, batch) -> torch.Tensor   # (B, C), in the permuted order
   ```
3. **Parity test** (`tests/test_scoring_parity.py`): on 50 dev clips, `eval()` mode,
   BF16, the probabilities from `softmax(candidate_logits)` must match the library's
   output with `max |Δp| < 1e-3`.
   **Do not start training until this passes.** If the library has a post-hoc
   calibration step outside the forward pass, reproduce it at inference only, and train on
   the raw logits.

### 1.4 Candidate permutation (`test_permutation.py`)

AudioJev was trained with permuted candidates. Keep doing that, so the model doesn't learn
a positional prior.

```python
# per sample i: perm[i] is a permutation of range(C); candidates shown in order perm[i]
logits_perm = candidate_logits(out, batch)            # (B, C), position j = class perm[i][j]
inv = torch.argsort(perm, dim=1)                      # (B, C)
logits = torch.gather(logits_perm, 1, inv)            # back to canonical class order
```

The unit test feeds a known permutation and checks that `logits[:, k]` always corresponds
to class `k`.

### 1.5 Losses (`losses.py`)

All losses are computed on the canonical-order `logits` (B, 7).

| `loss.type` | Target | Use when |
|---|---|---|
| `ce` | hard label | sanity baseline |
| `smoothed_ce` | hard label, smoothing ε | default in Phases 1–2 |
| `soft_kl` | soft targets from `loss.soft_targets` (sidecar `.npz`, keyed by `id`) | Phase 3 only, where relabeling produces the targets |

The manifests carry hard labels only, so Phases 1–2 handle label noise without labeler
confidences:

- **Class weighting:** `w_k ∝ 1/sqrt(n_k)`, normalized to mean 1.
- **Small-loss filtering** (`small_loss.drop_frac`): after `warmup_epochs`, compute each
  sample's loss in the forward pass and, within each class, leave the `drop_frac`
  highest-loss samples out of the backward pass. Mislabeled clips tend to keep a high
  loss, so this drops many of them. Do it per class so rare classes are not emptied.
  Log how many samples are dropped per class.
- Setting `loss.type: soft_kl` without `loss.soft_targets` is a config error.

### 1.6 Model wrapper (`model.py`)

- Load only the components scoring needs (the thinker: audio encoder + LLM), in BF16.
  Delete any vision / speech-output modules listed in `drop_modules` if the checkpoint has
  them, and confirm that nothing in the scoring path references them.
- Turn on `gradient_checkpointing_enable()` and `enable_input_require_grads()` (needed
  for LoRA with checkpointing).
- Apply LoRA with `peft` to the LLM linear layers only. Pass `target_modules` as a regex
  that matches LLM blocks and excludes the audio tower.
- Audio encoder:
  - `unfreeze_top_k = 0`: fully frozen.
  - `unfreeze_top_k = k`: set `requires_grad=True` on the last `k` transformer blocks
    plus the final layer norm / projector, with a separate param group at `audio_encoder.lr`.
- Log trainable parameter counts per group at startup.

**Memory budget (to check in 1.1):** about 9 GB of weights plus activations. An audio
token stream at roughly 25 Hz is under 1k tokens per 30 s clip. Start with
`micro_batch: 8`, and halve it if you hit OOM. 40–48 GB should be enough with gradient
checkpointing.

### 1.7 Training loop (`train.py`)

- Use a plain `accelerate` loop rather than the HF `Trainer`, since the loss and batch
  structure are custom.
- AdamW, cosine schedule with warmup, BF16 autocast, gradient clipping 1.0.
- Every `eval.every_steps`, evaluate on the gold dev set: UA, WA, macro-F1, NLL, ECE (15
  bins), and the per-class confusion matrix. Log to TensorBoard.
- **Select checkpoints on dev UA.** Save only the LoRA adapters and unfrozen audio
  layers, keeping the best and the last 3 (Phase 3 averages them).
- Early stopping with patience 4 evals.
- Run command:
  ```bash
  accelerate launch --num_processes 1 -m sersys.train \
      --config configs/exp/p1_e1_frozen_smoothce.yaml --seed 1
  ```

### 1.8 Temperature scaling (`calibration.py`)

- Fit one scalar `T` on dev-set logits by minimizing NLL (LBFGS on `log T`).
- Report ECE before and after, and save `T` alongside the adapter.
- Fitting `T` does not change argmax, so UA is unaffected.

### 1.9 Test-time permutation averaging (`predict.py`)

- `tta_permutations = K`: run K cyclic shifts of the candidate order, un-permute, and
  average the log-probabilities.
- Report K ∈ {1, 3, 7}. Each extra permutation is roughly one more forward pass, so the
  latency trade-off is part of choosing the exported `tta_permutations`.

### 1.10 Experiment grid (dev set, seed 1; the top two configurations re-run with seeds 2–3)

| ID | Audio encoder | Loss | Prompt | Notes |
|---|---|---|---|---|
| E1.1 | frozen | smoothed_ce | en | default |
| E1.2 | frozen | smoothed_ce | vi | prompt language |
| E1.3 | frozen | ce | best lang | noise-handling ablation |
| E1.4 | frozen | best | best | small-loss filtering, `drop_frac` ∈ {0.1, 0.2} |
| E1.5 | top-4 unfrozen | best | best | |
| E1.6 | top-8 unfrozen | best | best | |
| E1.7 | best | best | best | `permute_candidates: false` (ablation) |
| E1.8 | best | best | best | LoRA r ∈ {16, 64} |

### Phase 1 exit criteria

- [ ] The parity test passes.
- [ ] The best configuration beats zero-shot AudioJev **and** the conventional
      audio-only baseline on dev UA. If it doesn't beat the conventional baseline, stop
      and rethink before Phase 2.
- [ ] Dev ECE after temperature scaling is no worse than zero-shot AudioJev.
- [ ] Best config frozen as `configs/exp/p1_best.yaml`.

---

## Phase 2 — ASR as an auxiliary task

**Goal:** check whether a shared ASR objective improves SER, and pick λ.

### 2.1 ASR samples (`prompts.py`)

- Build an ASR prompt with the same audio, a fixed instruction (e.g. "Transcribe the
  Vietnamese speech."), and the target `text` + EOS.
- Set labels to `-100` on all prompt and audio positions, so the loss covers transcript
  tokens only.
- **Text normalization for targets and scoring** (`metrics.py`): Unicode NFC, lowercase,
  strip punctuation, collapse whitespace. Keep the Vietnamese diacritics.
- If the transcripts are ASR output rather than human-written (known from how the data
  was made; the manifest does not record it), note it in the paper: the auxiliary task
  then learns from noisy targets.

### 2.2 Joint step

- Each micro-batch runs **two forward passes on the same audio**: the SER choice prompt
  and the ASR prompt.
- `L = L_SER + λ · L_ASR`, with one backward pass.
- Running the audio encoder twice is acceptable to start with. Cache the audio-encoder
  output and feed it to both passes only if profiling shows the encoder dominates step
  time.
- Halve `micro_batch` (and double `grad_accum`) to keep the effective batch at 48.

### 2.3 ASR loss normalization

- Take the token cross-entropy averaged **per sample** (mean over that sample's
  transcript tokens), then average over the batch.
- This stops long transcripts from outweighing the single SER decision.
- Log `L_SER`, `L_ASR` and the gradient norm of each component separately for the first
  few hundred steps, to check that neither dominates.

### 2.4 ASR evaluation

- Greedy decoding on dev, `max_new_tokens` scaled to clip duration.
- Report CER and syllable-level WER (`jiwer`) after normalization.
- Run this every 2–3 SER evals, since it is slower than SER scoring.

### 2.5 Experiment grid (starting from `p1_best`)

| ID | λ | Schedule | Notes |
|---|---|---|---|
| E2.0 | 0 | — | = Phase 1 best (control) |
| E2.1 | 0.1 | constant | |
| E2.2 | 0.3 | constant | |
| E2.3 | 0.3 → 0.05 | linear decay | ASR shapes features early, SER fine-tunes late |
| E2.4 | best | best | **optional** transcribe-then-classify: SER prompt also gets the transcript (reference in training with p = 0.5, own ASR hypothesis otherwise; ASR hypothesis at eval) |

E2.4 adds an autoregressive decode at inference. Keep it only if its dev-UA gain is clearly
above seed variance.

### Phase 2 exit criteria

- [ ] Dev UA for each λ reported (3 seeds for the best λ and for λ = 0).
- [ ] Selected λ frozen as `configs/exp/p2_best.yaml`. If no λ > 0 beats λ = 0 beyond
      seed variance, carry λ = 0 forward and report the negative result.

---

## Phase 3 — Beyond the pseudo-label ceiling (optional, observe phase 2 result first)

**Goal:** squeeze more SER accuracy out of the same data.

### 3.1 Pool inference (`predict.py`)

- Run `p2_best` (with its `T` and `tta_permutations = 3`) over `train_pseudo.jsonl`.
- Write `runs/<exp>/pool_preds.jsonl` with calibrated probabilities per utterance.

### 3.2 Relabeling (`relabel.py`)

For each training utterance with original label `y` and model probabilities `p`:

- **keep:** `argmax p == y`, so the target is `α·onehot(y) + (1−α)·p` (soft, α = 0.5);
- **flip:** `argmax p != y` and `max p ≥ τ_flip` (e.g. 0.9), so the target is `p`;
- **drop:** otherwise.

Outputs, which keep the manifests at four fields:

- `data/manifests/train_round{r}.jsonl`: the kept and flipped utterances, with `label`
  set to `argmax` of the new target. This drives class weighting and the sampler.
- `data/cache/soft_targets_round{r}.npz`: `ids` and a `(N, 7)` target array in canonical
  class order.

Train with `loss.type: soft_kl` and `loss.soft_targets` pointing to the `.npz`. The
loader checks that every manifest `id` has a target row.

Log the new class distribution and the share of kept, flipped and dropped utterances, per
class. A class that shrinks a lot is a sign the model is reinforcing its own errors.

### 3.3 Retraining

- Re-initialize from the **AudioJev checkpoint** each round. Don't continue from the
  previous round's model, since that amplifies its own errors.
- Same config as `p2_best`, with only the data changed.
- At most 2 rounds. Stop when the dev-UA gain is under 0.5 points.

### 3.4 Checkpoint averaging (`average.py`)

- Average the last 3–5 checkpoints of the best run.
- Average the **merged weight deltas** `ΔW = B·A·(α/r)` per layer, not the A and B
  matrices separately. (The average of the products is not the product of the averages.)
- Also average any unfrozen audio-encoder weights directly.
- Re-fit `T` on dev after averaging.

### 3.5 Optional: gold fine-tuning stage

If extra human-labeled **training** clips become available (separate from dev and
test): do a short final stage on them only, with LR = 0.1× the main LR, 1–2 epochs, and
early stopping on dev UA.

### Phase 3 exit criteria

- [ ] Dev UA reported for each round and with/without averaging.
- [ ] Final configuration frozen as `configs/exp/final.yaml`. **No further changes
      after this point.**

---

## Phase 4 — Final evaluation, export and offline inference

Scope: offline evaluation and inference in Python only. No web UI, HTTP service or
deployment.

### 4.1 Test-set evaluation (paper) (`evaluate.py`, `scripts/make_tables.py`)

- Train `final.yaml` with seeds 1–3, then fit `T` on dev for each seed (done by `train.py`;
  `average.py --refit` for an averaged adapter).
- Evaluate each seed on `test_gold.jsonl`, **once**. `evaluate.py` refuses to overwrite an
  existing `test_metrics.json` without `--force` and appends every test access to
  `test_access.log`.
- Report mean ± std across seeds, plus a 95% bootstrap CI (1,000 resamples of
  utterances) for UA, WA, macro-F1 and ECE. Per run: the CI of that run. In the tables: the
  CI of the seed-mean (each resample is scored for every seed).
- Per-class recall, the confusion matrix, ECE at T = 1 and at the dev-fitted T, and a
  reliability diagram (`test_reliability.png`, bins stored in `test_metrics.json`).
- Run the same test evaluation for every ablation row (zero-shot, E1.x best, E2 λ = 0
  vs best λ, Phase 3 variants) so the paper table uses one consistent protocol. For
  zero-shot, `T` is fitted on dev in the same run (`--fit_temperature_on_dev`).
- `scripts/make_tables.py` collects the `runs/*/*/test_metrics.json` files (and the
  per-seed `test/preds.npz`) into LaTeX and Markdown tables.

### 4.2 Export (`export.py`)

- Merge the LoRA weights (or averaged deltas) into the base model, plus any unfrozen
  audio layers.
- Save with `save_pretrained` in BF16 (`--dtype float32` if BF16 rounding fails the
  validation below), with the processor and:
  - `ser_config.json`: labels, candidate descriptions, prompt, `T`, `tta_permutations`,
    `max_seconds`, AudioJev revision, git hash, source run, dev metrics, validation result;
  - the SER-only scoring code path (`infer.py`; no ASR generation at inference, so an
    E2.4 transcribe-then-classify configuration is not exportable).
- Validation: the exported model reproduces the run's dev predictions exactly (argmax
  agreement 100%, `max |Δp| < 1e-3` on calibrated probabilities). The result is written to
  `validation.json`, and the CLI exits non-zero on failure.

### 4.3 Offline inference (`infer.py`)

- Python API: `SERPredictor.from_pretrained(export_dir)` for an exported model, or
  `SERPredictor.from_run(run_dir, adapter="best" | "avg5" | "none")` for a run without
  exporting. `predict([...])` takes file paths, `(waveform, sample_rate)` tuples or 16 kHz
  arrays, batches them by length, and returns `{label, probs, confidence, cropped}` per
  clip, in input order.
- CLI: `python -m sersys.infer --model exports/final <files/dirs> [--manifest m.jsonl]
  --out preds.jsonl`. If a manifest carries gold labels, UA/WA/F1/ECE are printed too.
- Probabilities are calibrated with `T` and averaged over `tta_permutations`;
  `confidence = max p` (after `T`), so low-confidence clips can be routed to a human.
  Clips longer than `max_seconds` are center-cropped and flagged.
- Latency benchmark (`scripts/benchmark_latency.py`): p50 and p95 per clip, batch size 1,
  for 5 s, 10 s and 30 s clips at `tta_permutations` ∈ {1, 3}.

### 4.4 Licensing note

- Model use falls under the **Qwen Research License** (inherited by AudioJev). Confirm
  with the supervisor that research use and any sharing of the exported weights are
  allowed.
- If they aren't, evaluate the best Apache-licensed alternative in the same protocol and
  report both in the paper.

### Phase 4 exit criteria

- [ ] Test results with seeds and CIs, ablation tables generated by script.
- [ ] Exported model passes the parity check against its training run.
- [ ] Offline inference latency measured. License cleared for the intended use.

---

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Scoring cannot be reproduced exactly (Phase 1.3) | Parity test gates training. If the library uses hidden post-processing, train on raw logits and apply it at inference only |
| Pseudo-labels are systematically biased | Gold dev drives all selection; Phase 3 tracks per-class drift; report the pseudo-vs-gold confusion |
| ASR loss swamps SER | Per-sample normalization, λ grid, per-component gradient-norm logging |
| Weak Vietnamese ASR from the base model | Acceptable: ASR is auxiliary. Report CER for transparency |
| Overfitting to the dev set through many experiments | Test touched once; seeds; bootstrap CIs; final config frozen before the test run |
| Hub model updated mid-project | Revision hash pinned in config and logged per run |
| License restricts use or sharing of the weights | Phase 4.4 check early; Apache-licensed fallback evaluated in the same protocol |

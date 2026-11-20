"""PLAN 1.3 gate: our scoring reproduces audiojev_inference. Do not train until this passes.

Needs a CUDA GPU, the AudioJev weights and the dev manifest:
    SER_DEV_MANIFEST=/path/to/dev_gold.jsonl pytest -m gpu tests/test_scoring_parity.py -s

The library's probabilities are computed first and the library is freed. Then our own
thinker is loaded through sersys.model.load_thinker (attn=sdpa, as the library uses) and
scores the same clips (a) one at a time, which must match to max |dp| < 1e-3, and
(b) in padded batches of 8. The batched tolerance is SER_PARITY_BATCH_TOL (default 1e-3);
BF16 batched matmuls can differ slightly from batch size 1, so (b) is reported separately.
"""
import gc
import os
from pathlib import Path

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.gpu

N = int(os.environ.get("SER_PARITY_N", "50"))
BATCH_TOL = float(os.environ.get("SER_PARITY_BATCH_TOL", "1e-3"))


@pytest.fixture(scope="module")
def setup(cfg):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    manifest = os.environ.get("SER_DEV_MANIFEST", cfg["data"]["dev_manifest"])
    if not Path(manifest).is_file():
        pytest.skip(f"dev manifest not found: {manifest}")
    from sersys.data import read_manifest
    from sersys.prompts import load_label_space

    space = load_label_space(cfg["labels"], "en")
    records = read_manifest(manifest, space.classes)[:N]

    from audiojev_inference import AudioJev, Choice
    lib = AudioJev(device="cuda:0", revision=cfg["model"]["revision"])
    question = {"emo": Choice(instructions=space.question, criteria=space.criteria())}
    lib_probs = []
    for rec in records:
        probs = lib.system_one(state=rec["audio_path"], questions=question)["answers"]["emo"]["probabilities"]
        lib_probs.append([probs[c] for c in space.classes])
    del lib
    gc.collect()
    torch.cuda.empty_cache()

    from sersys.model import load_thinker
    thinker, processor, _ = load_thinker(cfg["model"], "cuda:0", attn_impl="sdpa")
    thinker.eval().requires_grad_(False)
    return space, records, np.array(lib_probs), thinker, processor


def _ours(setup, batch_size):
    from sersys.data import MIN_SAMPLES, load_audio
    from sersys.scoring import build_choice_batch, forward_canonical, label_token_ids

    space, records, _, thinker, processor = setup
    label_ids = label_token_ids(processor.tokenizer, space.num_classes)
    out = []
    for s in range(0, len(records), batch_size):
        audios = [load_audio(r["audio_path"]) for r in records[s:s + batch_size]]
        audios = [np.pad(a, (0, max(0, MIN_SAMPLES - len(a)))) for a in audios]
        perms = torch.tensor([space.reference_order] * len(audios))
        batch = build_choice_batch(processor, audios, space.question, space.descriptions, perms, label_ids)
        with torch.inference_mode():
            logits = forward_canonical(thinker, batch, torch.device("cuda:0"))
        out.append(torch.softmax(logits, -1).cpu().numpy())
    return np.concatenate(out)


def test_parity_single(setup):
    lib = setup[2]
    ours = _ours(setup, 1)
    diff = np.abs(ours - lib).max()
    print(f"\nbatch=1: max |dp| = {diff:.2e} over {len(lib)} clips; "
          f"argmax agreement {np.mean(ours.argmax(1) == lib.argmax(1)):.3f}")
    assert diff < 1e-3


def test_parity_batched(setup):
    lib = setup[2]
    ours = _ours(setup, 8)
    diff = np.abs(ours - lib).max()
    print(f"\nbatch=8: max |dp| = {diff:.2e}; argmax agreement {np.mean(ours.argmax(1) == lib.argmax(1)):.3f}")
    assert diff < BATCH_TOL

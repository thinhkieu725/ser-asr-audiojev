"""Phase 2: ASR prompt/targets, normalization, per-sample ASR loss, greedy decoding, joint training.

Model-dependent tests use the tiny random thinker from conftest (CPU, float32).
"""
import copy
import json
import unicodedata

import numpy as np
import pytest
import soundfile as sf
import torch
import torch.nn.functional as F

from sersys.asr import (IGNORE, ASRPrompter, TranscriptViewBuilder, asr_loss, max_new_tokens, transcribe)
from sersys.config import load_config
from sersys.metrics import asr_metrics, normalize_text
from sersys.scoring import MODEL_INPUT_KEYS, label_token_ids, tta_perms

TEXTS = ["Xin chào, thế giới!", "Tôi rất VUI   hôm nay.", "", "Anh ấy (đang) giận dữ..."]
GEN = {"tokens_per_second": 3, "min_new_tokens": 2, "max_new_tokens": 20}


@pytest.fixture(scope="module")
def prompter(processor):
    return ASRPrompter(processor, "Transcribe the Vietnamese speech.", normalize=True, max_target_tokens=64)


# --------------------------------------------------------------------------- text / metrics

def test_normalize_text_keeps_diacritics():
    nfd = unicodedata.normalize("NFD", "Tiếng Việt")
    assert normalize_text(nfd) == "tiếng việt"
    assert normalize_text(nfd) == unicodedata.normalize("NFC", "tiếng việt")
    assert normalize_text("  Xin chào,   thế giới!! ") == "xin chào thế giới"
    assert normalize_text("Anh ấy (đang) giận...") == "anh ấy đang giận"


def test_asr_metrics_syllable_wer_and_cer():
    m = asr_metrics(["xin chào thế giới", "", "một hai"], ["xin chao thế giới", "abc", "một hai"])
    assert m["n"] == 2 and m["skipped_empty_ref"] == 1
    assert m["wer"] == pytest.approx(1 / 6)    # one of six syllables wrong
    assert 0 < m["cer"] < m["wer"]
    assert asr_metrics(["một"], [""])["wer"] == 1.0


# --------------------------------------------------------------------------- batches

def test_train_batch_labels_cover_only_target(prompter, processor, synthetic_audios):
    skip = torch.tensor([False, False, False, True])
    b = prompter.train_batch(synthetic_audios, TEXTS, skip)
    tok = processor.tokenizer
    pb = prompter.prompt_batch(synthetic_audios)
    for i, text in enumerate(TEXTS):
        lab = b["labels"][i]
        tgt = lab[lab != IGNORE]
        n_real = int(b["attention_mask"][i].sum())
        prompt_len = int(pb["attention_mask"][i].sum())
        # prompt tokens are identical to the generation prompt (which is left-padded)
        assert torch.equal(b["input_ids"][i, :prompt_len], pb["input_ids"][i, -prompt_len:])
        assert (lab[:prompt_len] == IGNORE).all() and (lab[n_real:] == IGNORE).all()
        if i in (2, 3):  # empty reference / skipped (cropped) row: no loss
            assert len(tgt) == 0
            continue
        assert tgt[-1] == prompter.eos_id
        assert tok.decode(tgt[:-1]) == normalize_text(text)
        assert torch.equal(b["input_ids"][i, prompt_len:n_real], tgt)


def test_target_truncation(processor):
    p = ASRPrompter(processor, "Transcribe.", max_target_tokens=5)
    ids, cut = p.target_ids("một hai ba bốn năm sáu bảy tám chín mười")
    assert cut and len(ids) == 5 and p.eos_id not in ids
    ids, cut = p.target_ids("một")
    assert not cut and ids[-1] == p.eos_id


def test_max_new_tokens_scales_with_duration():
    cfg = {"tokens_per_second": 15, "min_new_tokens": 16, "max_new_tokens": 512}
    assert max_new_tokens(2.0, cfg) == 46
    assert max_new_tokens(30.0, cfg) == 466
    assert max_new_tokens(100.0, cfg) == 512


# --------------------------------------------------------------------------- tiny model

def test_asr_loss_matches_full_logits(tiny_thinker, prompter, synthetic_audios):
    b = prompter.train_batch(synthetic_audios, TEXTS)
    with torch.no_grad():
        per_sample, valid = asr_loss(tiny_thinker, b, torch.device("cpu"))
        logits = tiny_thinker(**{k: b[k] for k in MODEL_INPUT_KEYS}, use_cache=False).logits
    assert valid.tolist() == [True, True, False, True]
    for i in range(len(TEXTS)):
        lab = b["labels"][i, 1:]
        m = lab != IGNORE
        if not m.any():
            continue
        expected = F.cross_entropy(logits[i, :-1][m], lab[m])  # mean over this sample's tokens
        torch.testing.assert_close(per_sample[i], expected, atol=1e-4, rtol=1e-4)


def test_asr_loss_has_gradient(tiny_thinker, prompter, synthetic_audios):
    model = copy.deepcopy(tiny_thinker).train()
    per_sample, valid = asr_loss(model, prompter.train_batch(synthetic_audios, TEXTS), torch.device("cpu"))
    per_sample[valid].mean().backward()
    assert model.model.layers[0].self_attn.q_proj.weight.grad.abs().sum() > 0


def test_batched_greedy_equals_single(tiny_thinker, prompter, synthetic_audios):
    """Left-padded batched generation must give each clip the same greedy output as alone."""
    seconds = [len(a) / 16000 for a in synthetic_audios]
    batched = transcribe(tiny_thinker, prompter, synthetic_audios, seconds, torch.device("cpu"), GEN)
    assert len(batched) == len(synthetic_audios) and all(isinstance(t, str) for t in batched)
    for i, audio in enumerate(synthetic_audios):
        single = transcribe(tiny_thinker, prompter, [audio], [seconds[i]], torch.device("cpu"), GEN)
        assert single[0] == batched[i]
    budget = max_new_tokens(seconds[2], GEN)
    assert len(prompter.tokenizer.encode(batched[2], add_special_tokens=False)) <= budget + 2


def test_transcribe_with_lora(tiny_thinker, prompter, synthetic_audios):
    from sersys.model import build_trainable
    cfg = load_config(overrides=["lora.r=4", "lora.alpha=8", "optim.grad_checkpointing=true"])
    model = build_trainable(copy.deepcopy(tiny_thinker), cfg).train()
    out = transcribe(model, prompter, synthetic_audios[:2], [2.0, 5.0], torch.device("cpu"), GEN)
    assert len(out) == 2 and model.training  # training mode restored


def test_transcript_view_builder(tiny_thinker, prompter, processor, space, synthetic_audios):
    label_ids = label_token_ids(processor.tokenizer, space.num_classes)
    vb = TranscriptViewBuilder(processor, space, label_ids, prompter, GEN, 'Transcript: "{transcript}"',
                               ref_prob=1.0)
    batch = {"waves": synthetic_audios, "texts": TEXTS, "seconds": [2.0, 5.0, 0.5, 3.0],
             "cropped": torch.tensor([False, True, False, False]),
             "perm_sets": [p.unsqueeze(0).expand(4, -1) for p in tta_perms(space.reference_order, 2)]}
    views = vb(tiny_thinker, batch, torch.device("cpu"), train=True)
    assert len(views) == 2
    tok = processor.tokenizer
    text0 = tok.decode(views[0]["input_ids"][0][views[0]["attention_mask"][0].bool()])
    assert f'Transcript: "{normalize_text(TEXTS[0])}"' in text0          # reference used (p = 1)
    text1 = tok.decode(views[0]["input_ids"][1][views[0]["attention_mask"][1].bool()])
    assert f'Transcript: "{normalize_text(TEXTS[1])}"' not in text1      # cropped -> own hypothesis


# --------------------------------------------------------------------------- config + training loop

def test_phase2_config_inheritance():
    cfg = load_config("configs/exp/p2_e3_lambda0.3to0.05.yaml")
    assert cfg["exp_name"] == "p2_e3_lambda0.3to0.05"
    assert cfg["asr"]["weight"] == 0.3 and cfg["asr"]["weight_end"] == 0.05
    assert cfg["optim"]["micro_batch"] * cfg["optim"]["grad_accum"] == 48
    p1 = load_config("configs/exp/p1_best.yaml")
    assert cfg["loss"] == p1["loss"] and cfg["lora"] == p1["lora"]


def _write_split(tmp_path, space, split, n, rng):
    rows = []
    for i in range(n):
        path = tmp_path / f"{split}{i}.wav"
        sf.write(path, (0.1 * rng.standard_normal(16000 + 1500 * i)).astype(np.float32), 16000)
        rows.append({"id": f"{split}{i}", "audio_path": str(path), "label": space.classes[i % 7],
                     "text": f"câu số {i} của tập {split}"})
    (tmp_path / f"{split}.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows),
                                             encoding="utf-8")


@pytest.mark.parametrize("exp", ["p2_e3_lambda0.3to0.05", "p2_e4_transcribe_then_classify"])
def test_joint_train_loop(exp, tiny_thinker, processor, space, tmp_path, monkeypatch):
    import sersys.train as train_mod

    rng = np.random.default_rng(0)
    _write_split(tmp_path, space, "train", 10, rng)
    _write_split(tmp_path, space, "dev", 5, rng)
    monkeypatch.setattr(train_mod, "load_thinker", lambda *a, **k: (copy.deepcopy(tiny_thinker), processor, tmp_path))
    run_dir = tmp_path / "run"
    train_mod.main([
        "--config", f"configs/exp/{exp}.yaml", "--cpu", "--run_dir", str(run_dir), "--max_steps", "10",
        "--set", f"data.train_manifest={tmp_path / 'train.jsonl'}", f"data.dev_manifest={tmp_path / 'dev.jsonl'}",
        f"data.durations_cache={tmp_path / 'dur.json'}", "data.num_workers=0", "data.max_seconds=1.2",
        "optim.micro_batch=2", "optim.grad_accum=2", "optim.eval_batch=3", "optim.grad_checkpointing=false",
        "eval.every_steps=5", "optim.epochs=4", "lora.r=4", "lora.alpha=8", "asr.eval_every=2", "asr.eval_batch=3",
        "asr.generation.max_new_tokens=6", "asr.grad_log_steps=10",
    ])
    results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    assert results["steps"] == 10
    hist = results["history"]
    assert [h["step"] for h in hist] == [5, 10]
    assert "asr" not in hist[0] and "asr" in hist[1]          # every 2nd SER eval, and the final one
    assert hist[1]["asr"]["skipped_cropped"] == 2             # dev clips > 1.2 s are not scored
    assert list((run_dir / "asr").glob("dev_hyps_step10.jsonl"))
    log = (run_dir / "train.log").read_text(encoding="utf-8")
    assert "asr eval step 10" in log
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    tags = EventAccumulator(str(run_dir / "tb")).Reload().Tags()["scalars"]
    for tag in ("train/loss_asr", "train/asr_lambda", "grad_components/ser", "grad_components/asr_weighted",
                "dev_asr/cer", "dev_asr/wer"):
        assert tag in tags, tag
    assert (run_dir / "best" / "temperature.json").is_file()

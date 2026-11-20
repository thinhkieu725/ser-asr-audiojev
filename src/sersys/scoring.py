"""Reproduction of AudioJev's candidate scoring, batched and differentiable.

audiojev_inference.backend.OmniBackend.predict_many does, per clip and question:

1. Build the chat text (prompts.choice_conversation) and call the Qwen2.5-Omni processor
   with `audio_kwargs={"sampling_rate": 16000, "max_length": _padded_length(...)}`. The
   audio is padded one STFT window past its end instead of to 300 s, and the log-mel
   features are then cut to the valid frames.
2. Run the thinker with `use_cache=False`. A forward pre-hook on `lm_head` keeps only
   the last position's hidden state.
3. Read the logits of the label tokens "0".."{C-1}" (one vocabulary id each) and take
   `softmax` in float32. Nothing else: no temperature, no order calibration.

This module does the same for a padded batch. Each row's readout position is its own
last real token, so the result does not depend on the padding side. Rows can show the
candidates in different orders (`perms`), and `unpermute` maps the result back to the
canonical class order.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch
from audiojev_inference.backend import _padded_length

from .prompts import LABELS, LabelSpace, choice_conversation, question_with_transcript

MODEL_INPUT_KEYS = ("input_ids", "attention_mask", "input_features", "feature_attention_mask")


def label_token_ids(tokenizer, num_classes: int) -> list[int]:
    ids = [tokenizer.encode(label, add_special_tokens=False) for label in LABELS[:num_classes]]
    if any(len(x) != 1 for x in ids) or len({x[0] for x in ids}) != num_classes:
        raise ValueError("tokenizer does not have distinct single-token option labels")
    return [x[0] for x in ids]


# --------------------------------------------------------------------------- permutations

def tta_perms(reference: list[int], k: int) -> torch.Tensor:
    """K cyclic shifts of the reference order, evenly spaced. Row 0 is the reference order."""
    c = len(reference)
    if not 1 <= k <= c:
        raise ValueError(f"tta_permutations must be in [1, {c}]")
    shifts = sorted({round(i * c / k) % c for i in range(k)})
    ref = torch.tensor(reference, dtype=torch.long)
    return torch.stack([torch.roll(ref, -s) for s in shifts])


def random_perms(batch_size: int, num_classes: int, generator: torch.Generator | None = None) -> torch.Tensor:
    return torch.stack([torch.randperm(num_classes, generator=generator) for _ in range(batch_size)])


def unpermute(logits_perm: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    """(B, C) with column j = class perm[i, j]  ->  (B, C) with column k = class k."""
    inv = torch.argsort(perm.to(logits_perm.device), dim=1)
    return torch.gather(logits_perm, 1, inv)


# --------------------------------------------------------------------------- batch building

def encode_audio_prompts(processor, texts: list[str], audios, sample_rate: int = 16000) -> dict:
    """Processor call shared by the SER and ASR prompts, with AudioJev's audio padding.

    Each clip is padded one STFT window past the longest clip (not to 300 s), and the log-mel
    features are cut to the valid frames, as in audiojev_inference.backend.predict_many.
    """
    fe = processor.feature_extractor
    max_length = max(_padded_length(len(a), fe) for a in audios)
    # audio_kwargs is consumed (emptied) by the processor, so build it per call.
    enc = processor(text=list(texts), audio=list(audios), return_tensors="pt", padding=True,
                    use_audio_in_video=False,
                    audio_kwargs={"sampling_rate": sample_rate, "max_length": max_length})
    frames = int(enc["feature_attention_mask"].sum(-1).max())
    batch = {k: enc[k] for k in MODEL_INPUT_KEYS}
    batch["input_features"] = batch["input_features"][:, :, :frames]
    batch["feature_attention_mask"] = batch["feature_attention_mask"][:, :frames]
    return batch


def build_choice_batch(processor, audios, question: str, cand_texts: list[str],
                       perms: torch.Tensor, label_ids: list[int], sample_rate: int = 16000,
                       transcripts: list[str] | None = None,
                       transcript_template: str = 'Transcript: "{transcript}"') -> dict:
    """Processor inputs for B clips. Row i lists the candidates in order cand_texts[perms[i]].

    `cand_texts` are the descriptions in canonical class order. With `transcripts` (E2.4),
    row i's question is followed by its transcript. The returned dict holds the model
    inputs plus `readout_pos` (B,), `perm` (B, C) and `label_ids` (C,).
    """
    perms = torch.as_tensor(perms, dtype=torch.long).contiguous()  # expanded views break pin_memory
    if perms.shape != (len(audios), len(cand_texts)):
        raise ValueError(f"perms shape {tuple(perms.shape)} != ({len(audios)}, {len(cand_texts)})")
    texts, cache = [], {}
    for i, row in enumerate(perms.tolist()):
        q = question if transcripts is None else question_with_transcript(question, transcripts[i],
                                                                           transcript_template)
        key = (tuple(row), q)
        if key not in cache:
            conv = choice_conversation(q, [cand_texts[k] for k in row])
            cache[key] = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        texts.append(cache[key])
    batch = encode_audio_prompts(processor, texts, audios, sample_rate)
    am = batch["attention_mask"]
    # Index of each row's last real token, valid for left or right padding.
    batch["readout_pos"] = am.shape[1] - 1 - torch.argmax(am.flip(1), dim=1)
    batch["perm"] = perms
    batch["label_ids"] = torch.tensor(label_ids, dtype=torch.long)
    return batch


class ChoiceCollator:
    """DataLoader collate_fn: list of dataset items -> {'views': [batch, ...], labels, ids, ...}.

    mode="random": one view, a fresh random candidate order per sample (training).
    mode="fixed":  one view per row of `perms` (K, C), the same order for every sample
                   (reference order, or K cyclic shifts for test-time averaging).
    asr:           an asr.ASRPrompter; adds the ASR training batch under 'asr' (PLAN 2.2).
    build_views:   False defers the SER views to the main process ('perm_sets' are still
                   returned), for E2.4, where the SER prompt needs the model's transcript.
    """

    def __init__(self, processor, space: LabelSpace, label_ids: list[int], mode: str = "fixed",
                 perms: torch.Tensor | None = None, sample_rate: int = 16000, asr=None,
                 build_views: bool = True):
        if mode not in ("random", "fixed"):
            raise ValueError(mode)
        self.processor = processor
        self.space = space
        self.label_ids = label_ids
        self.mode = mode
        self.perms = perms if perms is not None else torch.tensor([space.reference_order])
        self.sample_rate = sample_rate
        self.asr = asr
        self.build_views = build_views

    def __call__(self, items: list[dict]) -> dict:
        audios = [it["wave"] for it in items]
        b, c = len(items), self.space.num_classes
        if self.mode == "random":
            perm_sets = [random_perms(b, c)]
        else:
            perm_sets = [p.unsqueeze(0).expand(b, -1).contiguous() for p in self.perms]
        out = {"labels": torch.tensor([it["label"] for it in items], dtype=torch.long),
               "index": torch.tensor([it["index"] for it in items], dtype=torch.long),
               "ids": [it["id"] for it in items],
               "texts": [it.get("text", "") for it in items],
               "cropped": torch.tensor([bool(it.get("cropped", False)) for it in items]),
               "seconds": [it.get("seconds", len(it["wave"]) / self.sample_rate) for it in items],
               "perm_sets": perm_sets}
        if self.build_views:
            out["views"] = [build_choice_batch(self.processor, audios, self.space.question,
                                               self.space.descriptions, p, self.label_ids, self.sample_rate)
                            for p in perm_sets]
        else:
            out["waves"] = audios
        if self.asr is not None:
            out["asr"] = self.asr.train_batch(audios, out["texts"], out["cropped"])
        return out


# --------------------------------------------------------------------------- forward

def get_lm_head(model):
    model = getattr(model, "module", model)  # DDP
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    return base.lm_head


@contextmanager
def readout_at(lm_head, positions: torch.Tensor):
    """Make lm_head project only hidden[b, positions[b]] -> logits of shape (B, 1, V)."""

    def pre_hook(_module, args):
        hidden = args[0]
        rows = torch.arange(hidden.shape[0], device=hidden.device)
        return (hidden[rows, positions.to(hidden.device)].unsqueeze(1),) + tuple(args[1:])

    handle = lm_head.register_forward_pre_hook(pre_hook)
    try:
        yield
    finally:
        handle.remove()


def candidate_logits(model_out, batch: dict) -> torch.Tensor:
    """(B, C) float32 logits of the option-label tokens, in each row's displayed order."""
    logits = model_out.logits[:, -1].float()
    return logits[:, batch["label_ids"].to(logits.device)]


def forward_choice(model, batch: dict, device=None) -> torch.Tensor:
    """Run the thinker on a choice batch; returns permuted-order candidate logits (B, C)."""
    device = device or batch["input_ids"].device
    inputs = {k: batch[k].to(device) for k in MODEL_INPUT_KEYS}
    with readout_at(get_lm_head(model), batch["readout_pos"]):
        out = model(**inputs, use_cache=False, return_dict=True)
    return candidate_logits(out, batch)


def forward_canonical(model, batch: dict, device=None) -> torch.Tensor:
    """Candidate logits mapped back to canonical class order (B, C)."""
    return unpermute(forward_choice(model, batch, device), batch["perm"])

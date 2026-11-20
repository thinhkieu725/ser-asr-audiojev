"""Auxiliary ASR (Phase 2): prompt/target batches, per-sample ASR loss, greedy transcription,
CER/WER evaluation, and the E2.4 transcribe-then-classify view builder.

CLI (transcribe a manifest and score it):
    python -m sersys.asr --config runs/<exp>/1/config.yaml --adapter runs/<exp>/1/best \
        --split dev --out runs/<exp>/1/best/dev_asr
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .metrics import asr_metrics, normalize_text
from .prompts import asr_conversation
from .scoring import MODEL_INPUT_KEYS, build_choice_batch, encode_audio_prompts

log = logging.getLogger(__name__)

IGNORE = -100
EOS_TOKEN = "<|im_end|>"  # end of the assistant turn in the Qwen chat template


class ASRPrompter:
    """Builds ASR training batches (prompt + target + EOS, loss on target only) and
    left-padded prompt-only batches for generation. Picklable for DataLoader workers."""

    def __init__(self, processor, instruction: str, normalize: bool = True, max_target_tokens: int = 448,
                 sample_rate: int = 16000):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.prompt_text = processor.apply_chat_template(asr_conversation(instruction),
                                                         add_generation_prompt=True, tokenize=False)
        self.eos_id = self.tokenizer.convert_tokens_to_ids(EOS_TOKEN)
        self.pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.eos_id
        self.normalize = normalize
        self.max_target_tokens = max_target_tokens
        self.sample_rate = sample_rate

    def target_text(self, text: str) -> str:
        return normalize_text(text) if self.normalize else text.strip()

    def target_ids(self, text: str) -> tuple[list[int], bool]:
        """Token ids of the target + EOS. A truncated target gets no EOS (it did not end)."""
        ids = self.tokenizer.encode(self.target_text(text), add_special_tokens=False)
        if len(ids) >= self.max_target_tokens:
            return ids[: self.max_target_tokens], True
        return ids + [self.eos_id], False

    def _prompts(self, audios) -> tuple[dict, list[torch.Tensor]]:
        enc = encode_audio_prompts(self.processor, [self.prompt_text] * len(audios), audios, self.sample_rate)
        rows = [ids[m.bool()] for ids, m in zip(enc["input_ids"], enc["attention_mask"])]
        return enc, rows

    def train_batch(self, audios, texts: list[str], skip: torch.Tensor | None = None) -> dict:
        """Right-padded prompt+target sequences. `labels` is IGNORE on prompt, audio and padding
        positions. Rows in `skip` (e.g. cropped clips) or with an empty target get no loss."""
        enc, prompts = self._prompts(audios)
        seqs, labels, truncated = [], [], 0
        for i, (p, text) in enumerate(zip(prompts, texts)):
            tgt, cut = self.target_ids(text)
            truncated += cut
            if (skip is not None and bool(skip[i])) or not self.target_text(text):
                tgt = []
            t = torch.tensor(tgt, dtype=torch.long)
            seqs.append(torch.cat([p, t]))
            labels.append(torch.cat([torch.full_like(p, IGNORE), t]))
        length = max(len(s) for s in seqs)
        b = len(seqs)
        input_ids = torch.full((b, length), self.pad_id, dtype=torch.long)
        attention = torch.zeros((b, length), dtype=torch.long)
        label_t = torch.full((b, length), IGNORE, dtype=torch.long)
        for i, (s, lab) in enumerate(zip(seqs, labels)):
            input_ids[i, : len(s)] = s
            attention[i, : len(s)] = 1
            label_t[i, : len(s)] = lab
        return {"input_ids": input_ids, "attention_mask": attention, "labels": label_t,
                "input_features": enc["input_features"], "feature_attention_mask": enc["feature_attention_mask"],
                "n_truncated": truncated}

    def prompt_batch(self, audios) -> dict:
        """Left-padded prompts (no target) for batched generation."""
        enc, prompts = self._prompts(audios)
        length = max(len(p) for p in prompts)
        input_ids = torch.full((len(prompts), length), self.pad_id, dtype=torch.long)
        attention = torch.zeros((len(prompts), length), dtype=torch.long)
        for i, p in enumerate(prompts):
            input_ids[i, length - len(p):] = p
            attention[i, length - len(p):] = 1
        return {"input_ids": input_ids, "attention_mask": attention,
                "input_features": enc["input_features"], "feature_attention_mask": enc["feature_attention_mask"]}


# --------------------------------------------------------------------------- loss

def _lm_head(model):
    model = getattr(model, "module", model)
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    return base.lm_head


@contextlib.contextmanager
def _project_only(lm_head, mask: torch.Tensor):
    """lm_head projects only hidden states where `mask` (B, L) is True -> logits (1, N, V)."""

    def pre_hook(_module, args):
        hidden = args[0]
        return (hidden[mask.to(hidden.device)].unsqueeze(0),) + tuple(args[1:])

    handle = lm_head.register_forward_pre_hook(pre_hook)
    try:
        yield
    finally:
        handle.remove()


def asr_loss(model, batch: dict, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-sample ASR loss (B,) = mean token CE over that sample's target tokens (PLAN 2.3),
    and a (B,) bool mask of samples that have targets. Only target positions are projected
    through lm_head, so the (B, L, V) logits matrix is never built."""
    labels = batch["labels"].to(device)
    tgt_mask = labels[:, 1:] != IGNORE                      # token t+1 is a target
    pred_mask = F.pad(tgt_mask, (0, 1), value=False)        # ... predicted from position t
    inputs = {k: batch[k].to(device) for k in MODEL_INPUT_KEYS}
    with _project_only(_lm_head(model), pred_mask):
        out = model(**inputs, use_cache=False, return_dict=True)
    b = labels.shape[0]
    if not bool(tgt_mask.any()):
        zero = out.logits.sum() * 0.0
        return zero.expand(b), torch.zeros(b, dtype=torch.bool, device=device)
    tok_loss = F.cross_entropy(out.logits[0].float(), labels[:, 1:][tgt_mask], reduction="none")
    rows = tgt_mask.nonzero()[:, 0]
    counts = torch.bincount(rows, minlength=b)
    sums = torch.zeros(b, device=tok_loss.device, dtype=tok_loss.dtype).index_add(0, rows, tok_loss)
    return sums / counts.clamp_min(1), counts > 0


# --------------------------------------------------------------------------- generation

def max_new_tokens(seconds: float, gen_cfg: dict) -> int:
    """Decoding budget scaled to clip duration (PLAN 2.4)."""
    n = math.ceil(seconds * gen_cfg["tokens_per_second"]) + gen_cfg["min_new_tokens"]
    return min(n, gen_cfg["max_new_tokens"])


@torch.no_grad()
def transcribe(model, prompter: ASRPrompter, audios, seconds: list[float], device, gen_cfg: dict) -> list[str]:
    """Greedy transcription of a batch of clips."""
    from .model import needs_autocast

    if not len(audios):
        return []
    batch = prompter.prompt_batch(audios)
    budgets = [max_new_tokens(s, gen_cfg) for s in seconds]
    inputs = {k: v.to(device) for k, v in batch.items()}
    was_training = model.training
    model.eval()
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if needs_autocast(model) else contextlib.nullcontext()
    try:
        with amp:
            out = getattr(model, "module", model).generate(
                **inputs, max_new_tokens=max(budgets), do_sample=False, num_beams=1, use_cache=True,
                eos_token_id=prompter.eos_id, pad_token_id=prompter.pad_id)
    finally:
        model.train(was_training)
    new = out[:, inputs["input_ids"].shape[1]:].cpu()
    texts = []
    for row, budget in zip(new.tolist(), budgets):
        row = row[:budget]
        if prompter.eos_id in row:
            row = row[: row.index(prompter.eos_id)]
        texts.append(prompter.tokenizer.decode(row, skip_special_tokens=True).strip())
    return texts


def evaluate_asr(model, loader, prompter: ASRPrompter, device, gen_cfg: dict, max_batches: int | None = None) -> dict:
    """Greedy decoding over a loader built with ChoiceCollator(build_views=False).
    Center-cropped clips (> max_seconds) are decoded but not scored."""
    refs, hyps, ids, n_cropped = [], [], [], 0
    t0 = time.perf_counter()
    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        out = transcribe(model, prompter, batch["waves"], batch["seconds"], device, gen_cfg)
        for i, hyp in enumerate(out):
            if bool(batch["cropped"][i]):
                n_cropped += 1
                continue
            refs.append(batch["texts"][i])
            hyps.append(hyp)
            ids.append(batch["ids"][i])
    m = asr_metrics(refs, hyps)
    m.update(skipped_cropped=n_cropped, seconds=time.perf_counter() - t0)
    m["examples"] = [{"id": i, "ref": normalize_text(r), "hyp": normalize_text(h)}
                     for i, r, h in zip(ids, refs, hyps)]
    return m


# --------------------------------------------------------------------------- E2.4

class TranscriptViewBuilder:
    """Builds SER views whose question includes a transcript (E2.4 transcribe-then-classify).

    Eval: the model's own ASR hypothesis for every clip. Training: the normalized reference
    with probability `ref_prob`, the hypothesis otherwise (always the hypothesis for cropped
    clips, whose reference no longer matches the audio).
    """

    def __init__(self, processor, space, label_ids, prompter: ASRPrompter, gen_cfg: dict, template: str,
                 ref_prob: float = 0.5, sample_rate: int = 16000):
        self.processor, self.space, self.label_ids = processor, space, label_ids
        self.prompter, self.gen_cfg, self.template = prompter, gen_cfg, template
        self.ref_prob, self.sample_rate = ref_prob, sample_rate

    def __call__(self, model, batch: dict, device, train: bool = False) -> list[dict]:
        waves, texts = batch["waves"], batch["texts"]
        n = len(waves)
        if train:
            use_ref = (torch.rand(n) < self.ref_prob) & ~batch["cropped"]
            use_ref &= torch.tensor([bool(normalize_text(t)) for t in texts])
        else:
            use_ref = torch.zeros(n, dtype=torch.bool)
        need = [i for i in range(n) if not use_ref[i]]
        hyps = transcribe(model, self.prompter, [waves[i] for i in need], [batch["seconds"][i] for i in need],
                          device, self.gen_cfg)
        transcripts = [normalize_text(t) for t in texts]
        for i, h in zip(need, hyps):
            transcripts[i] = h
        return [build_choice_batch(self.processor, waves, self.space.question, self.space.descriptions, p,
                                   self.label_ids, self.sample_rate, transcripts=transcripts,
                                   transcript_template=self.template)
                for p in batch["perm_sets"]]


def make_prompter(processor, cfg: dict) -> ASRPrompter:
    a = cfg["asr"]
    return ASRPrompter(processor, a["instruction"], a["normalize_targets"], a["max_target_tokens"],
                       cfg["data"]["sample_rate"])


def asr_active(cfg: dict) -> bool:
    a = cfg["asr"]
    return a["weight"] > 0 or (a.get("weight_end") or 0) > 0


def asr_eval_enabled(cfg: dict) -> bool:
    flag = cfg["asr"].get("eval", "auto")
    if flag == "auto":
        return asr_active(cfg) or cfg["asr"]["transcript_in_ser"]
    return bool(flag)


# --------------------------------------------------------------------------- CLI

def main():
    from .config import load_config
    from .data import SERDataset, load_durations, read_manifest
    from .model import load_thinker, load_trained
    from .predict import make_eval_loader
    from .prompts import load_label_space
    from .scoring import label_token_ids

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = load_config(args.config, args.set)
    space = load_label_space(cfg["labels"], cfg["ser"]["prompt_lang"], cfg["ser"].get("question"))
    manifest = args.manifest or cfg["data"][f"{args.split}_manifest"]
    records = read_manifest(manifest, space.classes)[: args.limit]
    durations = load_durations(records, cfg["data"]["durations_cache"])
    ds = SERDataset(records, space.classes, durations, cfg["data"]["sample_rate"], cfg["data"]["max_seconds"],
                    name=Path(manifest).name)
    thinker, processor, _ = load_thinker(cfg["model"], args.device)
    model = load_trained(thinker, args.adapter) if args.adapter else thinker.eval().requires_grad_(False)
    prompter = make_prompter(processor, cfg)
    loader = make_eval_loader(ds, processor, space, label_token_ids(processor.tokenizer, space.num_classes), 1,
                              args.batch or cfg["asr"]["eval_batch"], cfg["data"]["num_workers"], build_views=False)
    m = evaluate_asr(model, loader, prompter, torch.device(args.device), cfg["asr"]["generation"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "hyps.jsonl").open("w", encoding="utf-8") as fh:
        for ex in m.pop("examples"):
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
    (out / "asr_metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
    log.info("CER %.4f  WER %.4f  (n=%d, %d cropped skipped, %.0fs)", m["cer"], m["wer"], m["n"],
             m["skipped_cropped"], m["seconds"])


if __name__ == "__main__":
    main()

"""Offline SER inference (PLAN 4.3).

Python:

    from sersys.infer import SERPredictor

    ser = SERPredictor.from_pretrained("exports/final")          # an exported model (sersys.export)
    # or, without exporting: base AudioJev + a trained adapter
    ser = SERPredictor.from_run("runs/final/1", adapter="best")

    ser.predict_file("clip.wav")
    # {'label': 'angry', 'confidence': 0.83, 'probs': {'neutral': 0.04, ...}, 'cropped': False}
    ser.predict(["a.wav", "b.flac", (waveform, 44100)], batch_size=8)

Command line (files, directories and/or a JSONL manifest with `audio_path`):

    python -m sersys.infer --model exports/final clips/ extra.wav --out preds.jsonl
    python -m sersys.infer --run runs/final/1 --manifest data/manifests/dev_gold.jsonl --out dev.jsonl

If every manifest row has a known `label`, UA / WA / macro-F1 / ECE are printed as well.

Probabilities are calibrated with the run's temperature T and averaged over
`tta_permutations` cyclic shifts of the candidate order (both stored in ser_config.json).
`confidence` = max probability. Clips longer than `max_seconds` are center-cropped.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .data import MIN_SAMPLES, crop, load_audio
from .prompts import LabelSpace
from .scoring import build_choice_batch, forward_canonical, label_token_ids, tta_perms

log = logging.getLogger(__name__)

SER_CONFIG = "ser_config.json"
FORMAT_VERSION = 1
AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3", ".m4a", ".opus", ".aiff", ".aif"}


# --------------------------------------------------------------------------- ser_config

def build_ser_config(cfg: dict, adapter_dir: Path | None, run_dir: Path | None = None) -> dict:
    """Everything needed to rebuild the SER prompt and post-processing without the repo configs."""
    from .prompts import load_label_space

    if cfg["asr"]["transcript_in_ser"]:
        raise NotImplementedError("transcript_in_ser (E2.4) needs ASR decoding at inference; the offline "
                                  "SER path supports only the SER-only prompt")
    space = load_label_space(cfg["labels"], cfg["ser"]["prompt_lang"], cfg["ser"].get("question"))
    temperature, dev = 1.0, None
    if adapter_dir is not None:
        t_file = Path(adapter_dir) / "temperature.json"
        if t_file.is_file():
            temperature = json.loads(t_file.read_text())["temperature"]
        else:
            log.warning("no temperature.json in %s; using T = 1", adapter_dir)
        m_file = Path(adapter_dir) / "dev_metrics.json"
        if m_file.is_file():
            m = json.loads(m_file.read_text())
            dev = {k: m.get(k) for k in ("ua", "wa", "macro_f1", "nll", "ece")}
    try:
        git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        git = None
    if run_dir is not None and (Path(run_dir) / "git_hash.txt").is_file():
        git = (Path(run_dir) / "git_hash.txt").read_text().strip()
    return {
        "format_version": FORMAT_VERSION,
        "classes": space.classes,
        "descriptions": space.descriptions,
        "question": space.question,
        "prompt_lang": cfg["ser"]["prompt_lang"],
        "temperature": temperature,
        "tta_permutations": cfg["ser"]["tta_permutations"],
        "sample_rate": cfg["data"]["sample_rate"],
        "max_seconds": cfg["data"]["max_seconds"],
        "base_model": cfg["model"]["name"],
        "model_revision": cfg["model"].get("revision"),
        "git_hash": git,
        "source_run": str(run_dir) if run_dir else None,
        "source_adapter": str(adapter_dir) if adapter_dir else None,
        "dev_metrics": dev,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


# --------------------------------------------------------------------------- predictor

class SERPredictor:
    """Batched, calibrated SER on raw audio. Thread-unsafe; one instance per GPU."""

    def __init__(self, model, processor, ser_config: dict, device: str | torch.device = "cuda:0",
                 tta_permutations: int | None = None, temperature: float | None = None):
        self.model = model.eval()
        self.model.requires_grad_(False)
        self.processor = processor
        self.config = ser_config
        self.device = torch.device(device)
        self.space = LabelSpace(ser_config["classes"], ser_config["descriptions"], ser_config["question"])
        self.label_ids = label_token_ids(processor.tokenizer, self.space.num_classes)
        self.k = tta_permutations or ser_config["tta_permutations"]
        self.perms = tta_perms(self.space.reference_order, self.k)
        self.temperature = temperature if temperature is not None else ser_config["temperature"]
        self.sample_rate = ser_config["sample_rate"]
        self.max_len = int(ser_config["max_seconds"] * self.sample_rate)
        from .model import needs_autocast
        self._amp = needs_autocast(model)  # fp32 audio layers of an un-exported run need BF16 autocast

    # ---- construction
    @classmethod
    def from_pretrained(cls, export_dir: str | Path, device: str = "cuda:0", dtype: str | None = None,
                        attn_impl: str = "sdpa", **kwargs) -> "SERPredictor":
        """Load a model written by sersys.export (merged weights + processor + ser_config.json)."""
        from transformers import Qwen2_5OmniProcessor
        from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
            Qwen2_5OmniThinkerForConditionalGeneration)

        from .model import resolve_attn_impl

        export_dir = Path(export_dir)
        ser_config = json.loads((export_dir / SER_CONFIG).read_text(encoding="utf-8"))
        if ser_config.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"unsupported ser_config format {ser_config.get('format_version')}")
        dev = torch.device(device)
        torch_dtype = getattr(torch, dtype or ser_config.get("dtype", "bfloat16"))
        model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            export_dir, dtype=torch_dtype, attn_implementation=resolve_attn_impl(attn_impl),
            device_map={"": dev.index if dev.type == "cuda" else "cpu"}, low_cpu_mem_usage=True)
        model.visual = None  # never used for audio-only scoring
        processor = Qwen2_5OmniProcessor.from_pretrained(export_dir, use_fast=False)
        return cls(model, processor, ser_config, device, **kwargs)

    @classmethod
    def from_run(cls, run_dir: str | Path, adapter: str = "best", device: str = "cuda:0",
                 **kwargs) -> "SERPredictor":
        """Base AudioJev + a trained adapter from runs/<exp>/<seed>/<adapter>, no export needed.
        `adapter="none"` gives zero-shot AudioJev with the run's prompt."""
        from .config import load_config
        from .model import load_thinker, load_trained

        run_dir = Path(run_dir)
        cfg = load_config(str(run_dir / "config.yaml"))
        adapter_dir = None if adapter == "none" else (run_dir / adapter if (run_dir / adapter).is_dir() else Path(adapter))
        thinker, processor, _ = load_thinker(cfg["model"], device)
        model = load_trained(thinker, adapter_dir) if adapter_dir else thinker
        return cls(model, processor, build_ser_config(cfg, adapter_dir, run_dir), device, **kwargs)

    # ---- audio
    def _prepare(self, item) -> tuple[np.ndarray, bool]:
        if isinstance(item, (str, Path)):
            wave = load_audio(item, self.sample_rate)
        elif isinstance(item, tuple):
            wave, sr = item
            wave = np.asarray(wave, dtype=np.float32)
            if wave.ndim == 2:  # (samples, channels) or (channels, samples)
                wave = wave.mean(axis=1 if wave.shape[0] > wave.shape[1] else 0)
            if sr != self.sample_rate:
                from scipy.signal import resample_poly
                g = math.gcd(int(sr), self.sample_rate)
                wave = resample_poly(wave, self.sample_rate // g, int(sr) // g).astype(np.float32)
        else:
            wave = np.asarray(item, dtype=np.float32)  # assumed mono at self.sample_rate
        cropped = len(wave) > self.max_len
        return crop(wave, self.max_len, train=False), cropped

    # ---- inference
    @torch.no_grad()
    def predict_logprobs(self, waves: list[np.ndarray]) -> np.ndarray:
        """(B, C) calibrated log-probabilities, canonical class order."""
        amp = torch.autocast("cuda", dtype=torch.bfloat16) if self._amp else contextlib.nullcontext()
        views = []
        for p in self.perms:
            batch = build_choice_batch(self.processor, waves, self.space.question, self.space.descriptions,
                                       p.unsqueeze(0).expand(len(waves), -1), self.label_ids, self.sample_rate)
            with amp:
                views.append(torch.log_softmax(forward_canonical(self.model, batch, self.device).float(), -1))
        logits = torch.stack(views).mean(0) / self.temperature
        return torch.log_softmax(logits, -1).cpu().numpy()

    def predict(self, inputs, batch_size: int = 8) -> list[dict]:
        """inputs: paths, (waveform, sample_rate) tuples, or mono float arrays at 16 kHz."""
        prepared = [self._prepare(x) for x in inputs]
        order = sorted(range(len(prepared)), key=lambda i: len(prepared[i][0]))  # less padding
        out: list[dict | None] = [None] * len(prepared)
        for s in range(0, len(order), batch_size):
            idx = order[s:s + batch_size]
            logp = self.predict_logprobs([prepared[i][0] for i in idx])
            for i, row in zip(idx, logp):
                probs = np.exp(row)
                k = int(probs.argmax())
                out[i] = {"label": self.space.classes[k], "confidence": float(probs[k]),
                          "probs": {c: float(p) for c, p in zip(self.space.classes, probs)},
                          "cropped": prepared[i][1]}
        return out

    def predict_file(self, path: str | Path) -> dict:
        return self.predict([path], batch_size=1)[0]


# --------------------------------------------------------------------------- CLI

def collect_inputs(paths: list[str], manifest: str | None) -> list[dict]:
    items = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            items += [{"id": str(f.relative_to(p)), "audio_path": str(f)}
                      for f in sorted(p.rglob("*")) if f.suffix.lower() in AUDIO_EXTENSIONS]
        elif p.is_file():
            items.append({"id": p.name, "audio_path": str(p)})
        else:
            raise FileNotFoundError(p)
    if manifest:
        with open(manifest, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rec = json.loads(line)
                    items.append({"id": rec.get("id", rec["audio_path"]), "audio_path": rec["audio_path"],
                                  **({"label": rec["label"]} if "label" in rec else {})})
    return items


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--model", help="exported model dir (sersys.export)")
    src.add_argument("--run", help="training run dir; uses --adapter inside it")
    ap.add_argument("--adapter", default="best", help="with --run: best | avg5 | <path> | none (zero-shot)")
    ap.add_argument("inputs", nargs="*", help="audio files and/or directories")
    ap.add_argument("--manifest", default=None, help="JSONL with audio_path (and optionally id, label)")
    ap.add_argument("--out", default=None, help="JSONL output (default: print)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--tta", type=int, default=None, help="override tta_permutations")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    items = collect_inputs(args.inputs, args.manifest)
    if not items:
        ap.error("no audio inputs")
    if args.model:
        ser = SERPredictor.from_pretrained(args.model, device=args.device, tta_permutations=args.tta)
    else:
        ser = SERPredictor.from_run(args.run, args.adapter, device=args.device, tta_permutations=args.tta)
    t0 = time.perf_counter()
    results = ser.predict([it["audio_path"] for it in items], batch_size=args.batch)
    elapsed = time.perf_counter() - t0
    fh = open(args.out, "w", encoding="utf-8") if args.out else sys.stdout
    try:
        for it, res in zip(items, results):
            fh.write(json.dumps({"id": it["id"], "audio_path": it["audio_path"], **res,
                                 **({"gold": it["label"]} if "label" in it else {})}, ensure_ascii=False) + "\n")
    finally:
        if args.out:
            fh.close()
    log.info("%d clips in %.1fs (%.1f clips/s), K=%d, T=%.3f", len(items), elapsed, len(items) / elapsed,
             ser.k, ser.temperature)
    if items and all(it.get("label") in ser.space.classes for it in items):
        from .metrics import format_confusion, ser_metrics
        logp = np.log(np.array([[r["probs"][c] for c in ser.space.classes] for r in results]).clip(1e-12))
        labels = np.array([ser.space.classes.index(it["label"]) for it in items])
        m = ser_metrics(logp, labels, ser.space.num_classes)
        log.info("UA %.4f  WA %.4f  F1 %.4f  NLL %.4f  ECE %.4f", m["ua"], m["wa"], m["macro_f1"], m["nll"], m["ece"])
        print(format_confusion(m["confusion"], ser.space.classes), file=sys.stderr)


if __name__ == "__main__":
    main()

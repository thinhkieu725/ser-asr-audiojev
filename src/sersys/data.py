"""Manifest reader, duration cache, audio loading, cropping and length bucketing."""

from __future__ import annotations

import json
import logging
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

log = logging.getLogger(__name__)

MANIFEST_FIELDS = {"id", "audio_path", "label", "text"}
MIN_SAMPLES = 400  # AudioJev rejects clips shorter than one STFT window


# --------------------------------------------------------------------------- manifests

def read_manifest(path: str | Path, classes: list[str], check_audio: bool = True) -> list[dict]:
    """Read a 4-field JSONL manifest. Fails on bad fields, unknown labels, missing audio."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"manifest not found: {path}")
    known = set(classes)
    records, seen, missing = [], set(), []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            rec = json.loads(line)
            if set(rec) != MANIFEST_FIELDS:
                raise ValueError(f"{path}:{lineno}: expected fields {sorted(MANIFEST_FIELDS)}, "
                                 f"got {sorted(rec)}")
            if rec["label"] not in known:
                raise ValueError(f"{path}:{lineno}: unknown label {rec['label']!r}")
            if rec["id"] in seen:
                raise ValueError(f"{path}:{lineno}: duplicate id {rec['id']!r}")
            seen.add(rec["id"])
            if check_audio and not Path(rec["audio_path"]).is_file():
                missing.append(rec["audio_path"])
            records.append(rec)
    if missing:
        raise FileNotFoundError(f"{len(missing)} audio files missing in {path}, e.g. {missing[:3]}")
    log.info("%s: %d utterances, labels %s", path.name, len(records),
             dict(sorted(Counter(r["label"] for r in records).items())))
    return records


def check_disjoint(*manifests: list[dict]) -> None:
    """Ids must be unique across all manifests."""
    seen: set[str] = set()
    for recs in manifests:
        ids = {r["id"] for r in recs}
        overlap = seen & ids
        if overlap:
            raise ValueError(f"{len(overlap)} ids appear in more than one manifest, e.g. "
                             f"{sorted(overlap)[:3]}")
        seen |= ids


# --------------------------------------------------------------------------- durations

def load_durations(records: list[dict], cache_path: str | Path) -> dict[str, float]:
    """id -> seconds, computed with soundfile.info on first use and cached as JSON."""
    cache_path = Path(cache_path)
    cache: dict[str, float] = {}
    if cache_path.is_file():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    todo = [r for r in records if r["id"] not in cache]
    if todo:
        log.info("reading durations of %d clips", len(todo))
        for rec in todo:
            info = sf.info(rec["audio_path"])
            cache[rec["id"]] = info.frames / info.samplerate
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(cache), encoding="utf-8")
        tmp.replace(cache_path)
    durations = {r["id"]: cache[r["id"]] for r in records}
    log.info("corpus: %d clips, %.2f h", len(durations), sum(durations.values()) / 3600)
    return durations


# --------------------------------------------------------------------------- audio

def load_audio(path: str | Path, sample_rate: int = 16000) -> np.ndarray:
    """Mono float32 waveform at `sample_rate`. Same decoding/resampling as audiojev_inference."""
    samples, rate = sf.read(str(path), dtype="float32", always_2d=True)
    samples = samples.mean(axis=1)
    if rate != sample_rate:
        divisor = math.gcd(rate, sample_rate)
        samples = resample_poly(samples, sample_rate // divisor, rate // divisor)
    samples = np.ascontiguousarray(samples, dtype=np.float32)
    if not np.isfinite(samples).all():
        raise ValueError(f"nonfinite samples in {path}")
    return samples


def crop(samples: np.ndarray, max_len: int, train: bool, rng: random.Random | None = None) -> np.ndarray:
    """Random crop in training, center crop at eval; pad clips shorter than MIN_SAMPLES."""
    if len(samples) < MIN_SAMPLES:
        samples = np.pad(samples, (0, MIN_SAMPLES - len(samples)))
    if len(samples) <= max_len:
        return samples
    if train:
        start = (rng or random).randint(0, len(samples) - max_len)
    else:
        start = (len(samples) - max_len) // 2
    return samples[start:start + max_len]


class SERDataset(torch.utils.data.Dataset):
    """Yields {'id', 'wave', 'label'} with `label` as the canonical class index."""

    def __init__(self, records: list[dict], classes: list[str], durations: dict[str, float],
                 sample_rate: int = 16000, max_seconds: float = 30.0, train: bool = False,
                 name: str = ""):
        self.records = records
        self.class_index = {c: i for i, c in enumerate(classes)}
        self.durations = [durations[r["id"]] for r in records]
        self.sample_rate = sample_rate
        self.max_len = int(max_seconds * sample_rate)
        self.train = train
        long = sum(d > max_seconds for d in self.durations)
        log.info("%s: %d/%d clips longer than %.1f s will be %s-cropped", name or "dataset", long,
                 len(records), max_seconds, "randomly" if train else "center")

    def __len__(self) -> int:
        return len(self.records)

    @property
    def labels(self) -> list[int]:
        return [self.class_index[r["label"]] for r in self.records]

    def __getitem__(self, i: int) -> dict:
        rec = self.records[i]
        wave = load_audio(rec["audio_path"], self.sample_rate)
        cropped = len(wave) > self.max_len  # a cropped clip no longer matches its transcript
        wave = crop(wave, self.max_len, self.train)
        return {"id": rec["id"], "wave": wave, "label": self.class_index[rec["label"]], "index": i,
                "text": rec["text"], "cropped": cropped, "seconds": len(wave) / self.sample_rate}


# --------------------------------------------------------------------------- samplers

class LengthBucketBatchSampler(torch.utils.data.Sampler):
    """Batches of similar-duration clips.

    Training: draw an epoch of indices (uniformly shuffled, or by class-balanced weights
    with replacement), cut it into chunks of `bucket_batches` batches, sort each chunk by
    duration, split into batches, and shuffle the batch order. Eval: one global sort,
    fixed order.
    """

    def __init__(self, durations: list[float], batch_size: int, shuffle: bool, seed: int = 0,
                 bucket_by_length: bool = True, sample_weights: list[float] | None = None,
                 bucket_batches: int = 100):
        self.durations = np.asarray(durations)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.bucket_by_length = bucket_by_length
        self.weights = None if sample_weights is None else torch.as_tensor(sample_weights, dtype=torch.double)
        self.bucket_batches = bucket_batches
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return math.ceil(len(self.durations) / self.batch_size)

    def _batches(self) -> list[list[int]]:
        n = len(self.durations)
        if not self.shuffle:
            order = np.argsort(self.durations, kind="stable") if self.bucket_by_length else np.arange(n)
            return [order[i:i + self.batch_size].tolist() for i in range(0, n, self.batch_size)]
        gen = torch.Generator().manual_seed(self.seed * 100003 + self.epoch)
        if self.weights is not None:
            order = torch.multinomial(self.weights, n, replacement=True, generator=gen).numpy()
        else:
            order = torch.randperm(n, generator=gen).numpy()
        if not self.bucket_by_length:
            return [order[i:i + self.batch_size].tolist() for i in range(0, n, self.batch_size)]
        chunk = self.batch_size * self.bucket_batches
        batches = []
        for start in range(0, n, chunk):
            part = order[start:start + chunk]
            part = part[np.argsort(self.durations[part], kind="stable")]
            batches += [part[i:i + self.batch_size].tolist() for i in range(0, len(part), self.batch_size)]
        perm = torch.randperm(len(batches), generator=gen).tolist()
        return [batches[i] for i in perm]

    def __iter__(self):
        yield from self._batches()


def balanced_sample_weights(labels: list[int], num_classes: int) -> list[float]:
    """Per-sample weights so every class is drawn equally often (balanced_sampler)."""
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    return [1.0 / counts[y] for y in labels]


def class_counts(labels: list[int], num_classes: int) -> np.ndarray:
    return np.bincount(labels, minlength=num_classes)

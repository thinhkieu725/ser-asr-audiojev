"""Checkpoint averaging (PLAN 3.4).

Averages the *merged* LoRA deltas of the last k checkpoints, dW = mean_i s * B_i A_i
(s = alpha / r), not A and B separately. This average is represented exactly as one
LoRA adapter of rank k*r:

    A' = [A_1; ...; A_k]   (stacked rows)      B' = [B_1, ..., B_k] / k   (stacked columns)
    alpha' = alpha * k, r' = r * k   ->  scaling s' = s,   s' * B' A' = mean_i s * B_i A_i

so the result is a normal PEFT adapter that load_trained / predict / export accept as is.
`--svd_rank` optionally compresses each averaged delta to a lower rank via SVD.
Unfrozen audio-encoder tensors are averaged directly. With `--refit`, the averaged model
is evaluated on dev and the temperature is re-fitted.

    python -m sersys.average --run_dir runs/p3_round1/1 --last 5 --out runs/p3_round1/1/avg5 --refit
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from .model import AUDIO_STATE_FILE

log = logging.getLogger(__name__)
ADAPTER_FILE = "adapter_model.safetensors"
ADAPTER_CONFIG = "adapter_config.json"


def last_checkpoints(run_dir: str | Path, k: int) -> list[Path]:
    ckpts = sorted(Path(run_dir).glob("ckpt-step*"), key=lambda p: int(re.sub(r"\D", "", p.name)))
    if len(ckpts) < k:
        raise ValueError(f"{run_dir} has {len(ckpts)} checkpoints, {k} requested "
                         f"(raise eval.keep_last when training)")
    return ckpts[-k:]


def _lora_modules(state: dict) -> list[str]:
    return sorted(k[: -len(".lora_A.weight")] for k in state if k.endswith(".lora_A.weight"))


def merged_deltas(ckpt: Path) -> dict[str, torch.Tensor]:
    """{module: s * B @ A} for one adapter (float32)."""
    cfg = json.loads((ckpt / ADAPTER_CONFIG).read_text())
    if cfg.get("use_rslora"):
        raise NotImplementedError("rsLoRA scaling is not supported")
    s = cfg["lora_alpha"] / cfg["r"]
    state = load_file(str(ckpt / ADAPTER_FILE))
    return {m: s * state[f"{m}.lora_B.weight"].float() @ state[f"{m}.lora_A.weight"].float()
            for m in _lora_modules(state)}


def average_adapters(ckpts: list[Path], out_dir: Path, svd_rank: int | None = None) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    k = len(ckpts)
    cfgs = [json.loads((c / ADAPTER_CONFIG).read_text()) for c in ckpts]
    states = [load_file(str(c / ADAPTER_FILE)) for c in ckpts]
    modules = _lora_modules(states[0])
    for c, st in zip(ckpts, states):
        if _lora_modules(st) != modules:
            raise ValueError(f"{c} has different LoRA modules")
    for key in ("lora_alpha", "r", "target_modules"):
        if any(cf[key] != cfgs[0][key] for cf in cfgs):
            raise ValueError(f"checkpoints differ in adapter {key}")
    if any(k_ for k_ in states[0] if "lora_" not in k_):
        raise ValueError("adapter has non-LoRA tensors (modules_to_save/bias); not supported")
    r, alpha = cfgs[0]["r"], cfgs[0]["lora_alpha"]
    s = alpha / r
    new = {}
    if svd_rank is None:
        for m in modules:
            new[f"{m}.lora_A.weight"] = torch.cat([st[f"{m}.lora_A.weight"].float() for st in states], 0)
            new[f"{m}.lora_B.weight"] = torch.cat([st[f"{m}.lora_B.weight"].float() for st in states], 1) / k
        new_r, new_alpha = r * k, alpha * k
    else:
        new_r, new_alpha = svd_rank, svd_rank  # scaling 1: the factors carry the full delta
        for m in modules:
            delta = sum(s * st[f"{m}.lora_B.weight"].float() @ st[f"{m}.lora_A.weight"].float() for st in states) / k
            u, sv, vh = torch.linalg.svd(delta, full_matrices=False)
            root = sv[:svd_rank].sqrt()
            new[f"{m}.lora_B.weight"] = (u[:, :svd_rank] * root).contiguous()
            new[f"{m}.lora_A.weight"] = (root[:, None] * vh[:svd_rank]).contiguous()
    save_file({n: t.contiguous() for n, t in new.items()}, str(out_dir / ADAPTER_FILE))
    out_cfg = dict(cfgs[0], r=new_r, lora_alpha=new_alpha)
    (out_dir / ADAPTER_CONFIG).write_text(json.dumps(out_cfg, indent=2), encoding="utf-8")

    audio_files = [c / AUDIO_STATE_FILE for c in ckpts]
    present = [f.is_file() for f in audio_files]
    if any(present):
        if not all(present):
            raise ValueError("only some checkpoints have audio-encoder tensors")
        audio = [load_file(str(f)) for f in audio_files]
        avg = {n: (sum(a[n].float() for a in audio) / k).to(audio[0][n].dtype) for n in audio[0]}
        save_file(avg, str(out_dir / AUDIO_STATE_FILE))
    info = {"checkpoints": [str(c) for c in ckpts], "r": new_r, "lora_alpha": new_alpha,
            "svd_rank": svd_rank, "audio_tensors_averaged": bool(any(present))}
    (out_dir / "average_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    log.info("averaged %d checkpoints -> %s (rank %d)", k, out_dir, new_r)
    return info


def refit(run_dir: Path, out_dir: Path, device: str = "cuda:0") -> dict:
    """Dev predictions of the averaged adapter, temperature re-fit, metrics (PLAN 3.4)."""
    import numpy as np

    from .calibration import calibrate
    from .config import load_config
    from .data import SERDataset, load_durations, read_manifest
    from .model import load_thinker, load_trained
    from .predict import make_eval_loader, predict_logits, save_predictions
    from .prompts import load_label_space
    from .scoring import label_token_ids

    cfg = load_config(str(run_dir / "config.yaml"))
    space = load_label_space(cfg["labels"], cfg["ser"]["prompt_lang"], cfg["ser"].get("question"))
    records = read_manifest(cfg["data"]["dev_manifest"], space.classes)
    durations = load_durations(records, cfg["data"]["durations_cache"])
    ds = SERDataset(records, space.classes, durations, cfg["data"]["sample_rate"], cfg["data"]["max_seconds"],
                    name="dev")
    thinker, processor, _ = load_thinker(cfg["model"], device)
    model = load_trained(thinker, out_dir)
    label_ids = label_token_ids(processor.tokenizer, space.num_classes)
    view_builder = None
    if cfg["asr"]["transcript_in_ser"]:
        from .asr import TranscriptViewBuilder, make_prompter
        view_builder = TranscriptViewBuilder(processor, space, label_ids, make_prompter(processor, cfg),
                                             cfg["asr"]["generation"], cfg["asr"]["transcript_template"],
                                             sample_rate=cfg["data"]["sample_rate"])
    loader = make_eval_loader(ds, processor, space, label_ids, cfg["ser"]["tta_permutations"],
                              cfg["optim"]["eval_batch"], cfg["data"]["num_workers"], cfg["data"]["bucket_by_length"],
                              build_views=view_builder is None)
    preds = predict_logits(model, loader, torch.device(device), view_builder)
    np.savez(out_dir / "dev_preds.npz", **preds)
    cal = calibrate(preds["logits"], preds["labels"], space.num_classes, cfg["eval"]["ece_bins"])
    (out_dir / "temperature.json").write_text(json.dumps(cal, indent=2), encoding="utf-8")
    m = save_predictions(preds, out_dir / "dev", space.classes, cal["temperature"], cfg["eval"]["ece_bins"])
    (out_dir / "dev_metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
    log.info("averaged model: dev UA %.4f WA %.4f F1 %.4f | T %.3f ECE %.4f -> %.4f", m["ua"], m["wa"],
             m["macro_f1"], cal["temperature"], cal["before"]["ece"], cal["after"]["ece"])
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--last", type=int, default=5, help="average the last k ckpt-step* checkpoints")
    ap.add_argument("--ckpts", nargs="*", default=None, help="explicit checkpoint dirs (overrides --last)")
    ap.add_argument("--out", default=None, help="default <run_dir>/avg<k>")
    ap.add_argument("--svd_rank", type=int, default=None, help="compress the averaged deltas to this rank")
    ap.add_argument("--refit", action="store_true", help="evaluate on dev and re-fit the temperature")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run_dir = Path(args.run_dir)
    ckpts = [Path(c) for c in args.ckpts] if args.ckpts else last_checkpoints(run_dir, args.last)
    out = Path(args.out or run_dir / f"avg{len(ckpts)}")
    if out.exists():
        shutil.rmtree(out)
    average_adapters(ckpts, out, args.svd_rank)
    if args.refit:
        refit(run_dir, out, args.device)


if __name__ == "__main__":
    main()

"""Phase 1 SER fine-tuning of the AudioJev thinker (LoRA on the LLM, optional audio blocks).

    accelerate launch --num_processes 1 -m sersys.train \
        --config configs/exp/p1_e1_frozen_smoothce.yaml --seed 1

Run directory: runs/<exp_name>/<seed>/ with config.yaml, git_hash.txt, pip_freeze.txt,
model_revision.txt, train.log, tb/ (TensorBoard), ckpt-step*/ (last `eval.keep_last`),
best/ (adapter + dev_preds.npz + temperature.json), and results.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .asr import (TranscriptViewBuilder, asr_active, asr_eval_enabled, asr_loss, evaluate_asr,
                  make_prompter)
from .calibration import calibrate
from .config import dump_run_info, load_config
from .data import (LengthBucketBatchSampler, SERDataset, balanced_sample_weights, check_disjoint,
                   class_counts, load_durations, read_manifest)
from .losses import SmallLossFilter, class_weights, load_soft_targets, per_sample_loss, reduce_loss
from .metrics import format_confusion, ser_metrics
from .model import build_trainable, load_thinker, param_groups, save_trainable
from .predict import make_eval_loader, predict_logits
from .prompts import load_label_space
from .scoring import ChoiceCollator, forward_canonical, label_token_ids

log = logging.getLogger("sersys.train")


def setup_logging(run_dir: Path) -> None:
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(run_dir / "train.log", encoding="utf-8")):
        handler.setFormatter(fmt)
        root.addHandler(handler)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--set", nargs="*", default=[], help="overrides, key.path=value")
    ap.add_argument("--run_dir", default=None)
    ap.add_argument("--eval_at_start", action="store_true", help="evaluate dev before the first step")
    ap.add_argument("--max_steps", type=int, default=None, help="debug: stop after N optimizer steps")
    ap.add_argument("--limit_train", type=int, default=None, help="debug: first N train utterances")
    ap.add_argument("--limit_dev", type=int, default=None, help="debug: first N dev utterances")
    ap.add_argument("--cpu", action="store_true", help="debug: run on CPU")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from torch.utils.tensorboard import SummaryWriter
    from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

    cfg = load_config(args.config, args.set)
    if args.seed is not None:
        cfg["seed"] = args.seed
    seed = cfg["seed"]
    run_dir = Path(args.run_dir or f"runs/{cfg['exp_name']}/{seed}")
    run_dir.mkdir(parents=True, exist_ok=True)

    accelerator = Accelerator(mixed_precision="bf16", cpu=args.cpu)
    if accelerator.num_processes != 1:
        raise SystemExit("this loop is written for a single GPU (--num_processes 1)")
    device = accelerator.device
    setup_logging(run_dir)
    set_seed(seed)
    dump_run_info(cfg, run_dir)
    log.info("run dir %s", run_dir)

    # ---------------------------------------------------------------- data
    dcfg, ocfg, lcfg, ecfg = cfg["data"], cfg["optim"], cfg["loss"], cfg["eval"]
    space = load_label_space(cfg["labels"], cfg["ser"]["prompt_lang"], cfg["ser"].get("question"))
    C = space.num_classes
    train_recs = read_manifest(dcfg["train_manifest"], space.classes)[: args.limit_train]
    dev_recs = read_manifest(dcfg["dev_manifest"], space.classes)[: args.limit_dev]
    check_disjoint(train_recs, dev_recs)
    durations = load_durations(train_recs + dev_recs, dcfg["durations_cache"])
    train_ds = SERDataset(train_recs, space.classes, durations, dcfg["sample_rate"], dcfg["max_seconds"],
                          train=True, name="train")
    dev_ds = SERDataset(dev_recs, space.classes, durations, dcfg["sample_rate"], dcfg["max_seconds"],
                        train=False, name="dev")
    counts = class_counts(train_ds.labels, C)
    log.info("train class counts: %s", dict(zip(space.classes, counts.tolist())))
    weights = class_weights(counts, lcfg["class_weighting"])
    if weights is not None:
        log.info("class loss weights: %s", dict(zip(space.classes, np.round(weights.numpy(), 3).tolist())))
    soft_targets = None
    if lcfg["type"] == "soft_kl":
        soft_targets = torch.from_numpy(load_soft_targets(lcfg["soft_targets"], [r["id"] for r in train_recs], C))

    # ---------------------------------------------------------------- model
    thinker, processor, model_dir = load_thinker(cfg["model"], device)
    label_ids = label_token_ids(processor.tokenizer, C)
    model = build_trainable(thinker, cfg)
    model.train()

    sampler = LengthBucketBatchSampler(
        train_ds.durations, ocfg["micro_batch"], shuffle=True, seed=seed,
        bucket_by_length=dcfg["bucket_by_length"],
        sample_weights=(balanced_sample_weights(train_ds.labels, C)
                        if lcfg["class_weighting"] == "balanced_sampler" else None))
    # Phase 2: auxiliary ASR and (E2.4) transcript-conditioned SER
    acfg = cfg["asr"]
    use_asr = asr_active(cfg)
    prompter = make_prompter(processor, cfg) if (use_asr or asr_eval_enabled(cfg)) else None
    view_builder = None
    if acfg["transcript_in_ser"]:
        view_builder = TranscriptViewBuilder(processor, space, label_ids, prompter, acfg["generation"],
                                             acfg["transcript_template"], acfg["transcript_ref_prob"],
                                             dcfg["sample_rate"])
    log.info("asr: lambda %s%s, transcript_in_ser=%s, asr eval=%s", acfg["weight"],
             f" -> {acfg['weight_end']}" if acfg.get("weight_end") is not None else "",
             acfg["transcript_in_ser"], asr_eval_enabled(cfg))

    collate = ChoiceCollator(processor, space, label_ids,
                             mode="random" if cfg["ser"]["permute_candidates"] else "fixed",
                             sample_rate=dcfg["sample_rate"], asr=prompter if use_asr else None,
                             build_views=view_builder is None)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_sampler=sampler, collate_fn=collate, num_workers=dcfg["num_workers"],
        pin_memory=True, persistent_workers=dcfg["num_workers"] > 0)
    dev_loader = make_eval_loader(dev_ds, processor, space, label_ids, cfg["ser"]["tta_permutations"],
                                  ocfg["eval_batch"], dcfg["num_workers"], dcfg["bucket_by_length"],
                                  build_views=view_builder is None)
    dev_asr_loader = (make_eval_loader(dev_ds, processor, space, label_ids, 1, acfg["eval_batch"],
                                       dcfg["num_workers"], dcfg["bucket_by_length"], build_views=False)
                      if asr_eval_enabled(cfg) else None)

    groups = param_groups(model, cfg)
    trainable = [p for g in groups for p in g["params"]]
    optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8)
    accum = ocfg["grad_accum"]
    steps_per_epoch = math.ceil(len(sampler) / accum)
    total_steps = steps_per_epoch * ocfg["epochs"]
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    warmup = int(ocfg["warmup_ratio"] * total_steps)
    make_sched = {"cosine": get_cosine_schedule_with_warmup,
                  "linear": get_linear_schedule_with_warmup}[ocfg["scheduler"]]
    scheduler = make_sched(optimizer, warmup, total_steps)
    log.info("%d micro-batches/epoch, %d optimizer steps/epoch, %d total, %d warmup",
             len(sampler), steps_per_epoch, total_steps, warmup)

    small_loss = SmallLossFilter(C, lcfg["small_loss"]["drop_frac"], lcfg["small_loss"]["warmup_epochs"],
                                 lcfg["small_loss"].get("buffer_size", 2048))
    tb = SummaryWriter(str(run_dir / "tb"))

    # ---------------------------------------------------------------- eval / checkpoint
    state = {"best": -1.0, "best_step": -1, "bad_evals": 0, "history": [], "n_evals": 0}
    ckpts: list[Path] = []

    def evaluate_asr_dev(step: int) -> dict:
        m = evaluate_asr(model, dev_asr_loader, prompter, device, acfg["generation"])
        examples = m.pop("examples")
        tb.add_scalar("dev_asr/cer", m["cer"], step)
        tb.add_scalar("dev_asr/wer", m["wer"], step)
        tb.add_text("dev_asr/examples",
                    "\n\n".join(f"REF: {e['ref']}  \nHYP: {e['hyp']}" for e in examples[:5]), step)
        (run_dir / "asr").mkdir(exist_ok=True)
        with (run_dir / "asr" / f"dev_hyps_step{step}.jsonl").open("w", encoding="utf-8") as fh:
            for e in examples:
                fh.write(json.dumps(e, ensure_ascii=False) + "\n")
        log.info("[asr eval step %d] CER %.4f WER %.4f (n=%d, %d cropped skipped, %.0fs)", step, m["cer"],
                 m["wer"], m["n"], m["skipped_cropped"], m["seconds"])
        return m

    def evaluate(step: int, final: bool = False) -> bool:
        """Evaluate on dev, checkpoint, update early stopping. Returns True to stop."""
        t0 = time.perf_counter()
        preds = predict_logits(model, dev_loader, device, view_builder)
        m = ser_metrics(preds["logits"], preds["labels"], C, ecfg["ece_bins"])
        m.update(step=step, eval_seconds=time.perf_counter() - t0,
                 peak_mem_gb=torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0)
        for key in ("ua", "wa", "macro_f1", "nll", "ece"):
            tb.add_scalar(f"dev/{key}", m[key], step)
        for c, r in zip(space.classes, m["per_class_recall"]):
            tb.add_scalar(f"dev_recall/{c}", r, step)
        tb.add_text("dev/confusion", "```\n" + format_confusion(m["confusion"], space.classes) + "\n```", step)
        log.info("[eval step %d] UA %.4f WA %.4f F1 %.4f NLL %.4f ECE %.4f (%.0fs, peak %.1f GB)\n%s",
                 step, m["ua"], m["wa"], m["macro_f1"], m["nll"], m["ece"], m["eval_seconds"],
                 m["peak_mem_gb"], format_confusion(m["confusion"], space.classes))
        state["n_evals"] += 1
        if dev_asr_loader is not None and (final or step == 0 or state["n_evals"] % acfg["eval_every"] == 0):
            m["asr"] = evaluate_asr_dev(step)
        state["history"].append(m)
        if step == 0:
            return False
        ckpt = run_dir / f"ckpt-step{step}"
        save_trainable(model, ckpt)
        (ckpt / "dev_metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
        ckpts.append(ckpt)
        while len(ckpts) > ecfg["keep_last"]:
            shutil.rmtree(ckpts.pop(0), ignore_errors=True)
        if m["ua"] > state["best"]:
            state.update(best=m["ua"], best_step=step, bad_evals=0)
            best = run_dir / "best"
            shutil.rmtree(best, ignore_errors=True)
            save_trainable(model, best)
            np.savez(best / "dev_preds.npz", **preds)
            (best / "dev_metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
            log.info("new best dev UA %.4f at step %d", m["ua"], step)
        else:
            state["bad_evals"] += 1
            log.info("no improvement (%d/%d)", state["bad_evals"], ecfg["early_stop_patience"])
        return state["bad_evals"] >= ecfg["early_stop_patience"]

    if args.eval_at_start:
        evaluate(0)

    # ---------------------------------------------------------------- loop
    def asr_weight(step: int) -> float:
        if acfg.get("weight_end") is None:
            return acfg["weight"]
        return acfg["weight"] + (acfg["weight_end"] - acfg["weight"]) * min(1.0, step / max(1, total_steps))

    def grad_snapshot():
        return [None if p.grad is None else p.grad.detach().clone() for p in trainable]

    def grad_increment_norm(before) -> float:
        sq = 0.0
        for p, b in zip(trainable, before):
            if p.grad is not None:
                d = p.grad if b is None else p.grad - b
                sq += float(d.float().pow(2).sum())
        return math.sqrt(sq) * accum  # per micro-batch, undoing the 1/accum scaling

    step, micro, stop = 0, 0, False
    zero_running = {"loss": 0.0, "kept": 0.0, "n": 0, "asr": 0.0, "asr_n": 0,
                    "g_ser": 0.0, "g_asr": 0.0, "g_n": 0, "asr_truncated": 0}
    running = dict(zero_running)
    t_start = time.perf_counter()
    for epoch in range(ocfg["epochs"]):
        sampler.set_epoch(epoch)
        n_batches = len(sampler)
        for i, batch in enumerate(train_loader):
            lam = asr_weight(step)
            log_components = use_asr and lam > 0 and step < acfg["grad_log_steps"]
            if view_builder is not None:  # E2.4: transcripts need the model, so views are built here
                view = view_builder(model, batch, device, train=True)[0]
            else:
                view = batch["views"][0]
            labels = batch["labels"].to(device)
            # SER pass. The SER and ASR passes are separate graphs, so backpropagating each in
            # turn gives the same gradient as one backward of L_SER + lambda * L_ASR, with only
            # one pass's activations alive at a time.
            before = grad_snapshot() if log_components else None
            with accelerator.autocast():
                logits = forward_canonical(model, view, device)
            targets = soft_targets[batch["index"]].to(device) if soft_targets is not None else None
            losses = per_sample_loss(logits, labels, lcfg["type"], lcfg.get("smoothing", 0.0), targets)
            keep = small_loss(losses, labels, epoch + i / n_batches)
            loss = reduce_loss(losses, labels, weights, keep)
            accelerator.backward(loss / accum)
            running["loss"] += float(losses.detach().mean())
            running["kept"] += float(keep.mean())
            running["n"] += 1
            # ASR pass (PLAN 2.2/2.3): per-sample mean token CE, averaged over samples with targets
            if use_asr and lam > 0:
                if log_components:
                    running["g_ser"] += grad_increment_norm(before)
                    before = grad_snapshot()
                with accelerator.autocast():
                    asr_losses, has_target = asr_loss(model, batch["asr"], device)
                if bool(has_target.any()):
                    l_asr = asr_losses[has_target].mean()
                    accelerator.backward(lam * l_asr / accum)
                    running["asr"] += float(l_asr.detach())
                    running["asr_n"] += 1
                if log_components:
                    running["g_asr"] += grad_increment_norm(before)
                    running["g_n"] += 1
                running["asr_truncated"] += batch["asr"]["n_truncated"]
            micro += 1
            if micro % accum:
                continue
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, ocfg["max_grad_norm"])
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step % 10 == 0:
                n = running["n"]
                tb.add_scalar("train/loss", running["loss"] / n, step)
                tb.add_scalar("train/kept_frac", running["kept"] / n, step)
                tb.add_scalar("train/grad_norm", float(grad_norm), step)
                if use_asr:
                    tb.add_scalar("train/asr_lambda", lam, step)
                    if running["asr_n"]:
                        tb.add_scalar("train/loss_asr", running["asr"] / running["asr_n"], step)
                    tb.add_scalar("train/asr_targets_truncated", running["asr_truncated"], step)
                    if running["g_n"]:
                        tb.add_scalar("grad_components/ser", running["g_ser"] / running["g_n"], step)
                        tb.add_scalar("grad_components/asr_weighted", running["g_asr"] / running["g_n"], step)
                for g in optimizer.param_groups:
                    tb.add_scalar(f"lr/{g['name']}", g["lr"], step)
                if small_loss.enabled:
                    for k, frac in small_loss.stats_and_reset().items():
                        tb.add_scalar(f"small_loss_dropped/{space.classes[k]}", frac, step)
                if step % 50 == 0:
                    rate = step / (time.perf_counter() - t_start)
                    asr_txt = ""
                    if use_asr and running["asr_n"]:
                        asr_txt = f" | asr {running['asr'] / running['asr_n']:.4f} (lambda {lam:.3f})"
                        if running["g_n"]:
                            asr_txt += (f" | gnorm ser {running['g_ser'] / running['g_n']:.3f}"
                                        f" asr*lambda {running['g_asr'] / running['g_n']:.3f}")
                    log.info("epoch %d step %d/%d loss %.4f kept %.3f gnorm %.3f lr %.2e (%.2f steps/s)%s",
                             epoch, step, total_steps, running["loss"] / n, running["kept"] / n,
                             float(grad_norm), optimizer.param_groups[0]["lr"], rate, asr_txt)
                running = dict(zero_running)
            if step % ecfg["every_steps"] == 0 and evaluate(step):
                log.info("early stopping at step %d", step)
                stop = True
            if stop or step >= total_steps:
                stop = True
                break
        if stop:
            break
    if not state["history"] or state["history"][-1]["step"] != step:
        evaluate(step, final=True)  # final checkpoint
    elif dev_asr_loader is not None and "asr" not in state["history"][-1]:
        state["history"][-1]["asr"] = evaluate_asr_dev(step)

    # ---------------------------------------------------------------- calibration (1.8)
    best = run_dir / "best"
    dev = np.load(best / "dev_preds.npz")
    cal = calibrate(dev["logits"], dev["labels"], C, ecfg["ece_bins"])
    (best / "temperature.json").write_text(json.dumps(cal, indent=2), encoding="utf-8")
    log.info("temperature %.3f: dev ECE %.4f -> %.4f, NLL %.4f -> %.4f", cal["temperature"],
             cal["before"]["ece"], cal["after"]["ece"], cal["before"]["nll"], cal["after"]["nll"])

    results = {"exp_name": cfg["exp_name"], "seed": seed, "best_step": state["best_step"],
               "best_dev_ua": state["best"], "steps": step, "temperature": cal,
               "model_dir": str(model_dir), "history": state["history"]}
    (run_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    tb.close()
    log.info("done: best dev UA %.4f at step %d", state["best"], state["best_step"])


if __name__ == "__main__":
    main()

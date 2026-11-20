"""Config loading: base.yaml <- experiment yaml <- `--set key.path=value` overrides."""

from __future__ import annotations

import copy
import subprocess
import sys
from pathlib import Path

import yaml

BASE_CONFIG = "configs/base.yaml"


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def apply_overrides(cfg: dict, overrides: list[str] | None) -> dict:
    """Apply `a.b.c=value` strings; values are parsed as YAML (numbers, lists, null...)."""
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must look like key.path=value: {item!r}")
        path, raw = item.split("=", 1)
        node = cfg
        keys = path.split(".")
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        node[keys[-1]] = yaml.safe_load(raw)
    return cfg


def _load_exp(path: str | Path, depth: int = 0) -> dict:
    """An experiment yaml, resolving `inherit: <other yaml>` chains (parent first)."""
    if depth > 10:
        raise ValueError("config inheritance too deep (cycle?)")
    exp = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    parent = exp.pop("inherit", None)
    return deep_merge(_load_exp(parent, depth + 1), exp) if parent else exp


def load_config(exp_path: str | None = None, overrides: list[str] | None = None,
                base_path: str = BASE_CONFIG) -> dict:
    cfg = yaml.safe_load(Path(base_path).read_text(encoding="utf-8"))
    if exp_path:
        cfg = deep_merge(cfg, _load_exp(exp_path))
    cfg = apply_overrides(cfg, overrides)
    validate(cfg)
    return cfg


def validate(cfg: dict) -> None:
    loss = cfg["loss"]
    if loss["type"] not in ("ce", "smoothed_ce", "soft_kl"):
        raise ValueError(f"unknown loss.type {loss['type']!r}")
    if loss["type"] == "soft_kl" and not loss.get("soft_targets"):
        raise ValueError("loss.type=soft_kl requires loss.soft_targets")
    if loss["class_weighting"] not in ("none", "inv_sqrt_freq", "balanced_sampler"):
        raise ValueError(f"unknown loss.class_weighting {loss['class_weighting']!r}")
    if not 0.0 <= loss["small_loss"]["drop_frac"] < 1.0:
        raise ValueError("loss.small_loss.drop_frac must be in [0, 1)")
    if cfg["ser"]["prompt_lang"] not in ("en", "vi"):
        raise ValueError("ser.prompt_lang must be en or vi")
    if cfg["lora"].get("restrict_to", "llm") != "llm":
        raise ValueError("only lora.restrict_to=llm is supported")
    if cfg["ser"]["tta_permutations"] < 1:
        raise ValueError("ser.tta_permutations must be >= 1")
    asr = cfg["asr"]
    if asr["weight"] < 0 or (asr.get("weight_end") or 0) < 0:
        raise ValueError("asr.weight / asr.weight_end must be >= 0")
    if asr.get("eval", "auto") not in ("auto", True, False):
        raise ValueError("asr.eval must be auto, true or false")
    if asr["transcript_in_ser"] and "{transcript}" not in asr["transcript_template"]:
        raise ValueError("asr.transcript_template needs a {transcript} field")


def dump_run_info(cfg: dict, run_dir: Path) -> None:
    """Write config.yaml, git_hash.txt, pip_freeze.txt and the model revision."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
                                         encoding="utf-8")
    try:
        git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout
        git = git.strip() + (" (dirty)" if dirty.strip() else "")
    except (OSError, subprocess.CalledProcessError):
        git = "unknown (not a git repository)"
    (run_dir / "git_hash.txt").write_text(git + "\n", encoding="utf-8")
    try:
        freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True,
                                text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        freeze = "pip freeze failed\n"
    (run_dir / "pip_freeze.txt").write_text(freeze, encoding="utf-8")
    (run_dir / "model_revision.txt").write_text(
        f"{cfg['model']['name']}@{cfg['model'].get('revision')}\n", encoding="utf-8")

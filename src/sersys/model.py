"""Load the AudioJev thinker, apply LoRA to the LLM, (un)freeze the audio tower."""

from __future__ import annotations

import logging
import re
from collections import OrderedDict
from pathlib import Path

import torch

log = logging.getLogger(__name__)

AUDIO_STATE_FILE = "audio_encoder.safetensors"


def resolve_attn_impl(name: str) -> str:
    if name == "flash_attention_2":
        try:
            import flash_attn  # noqa: F401
        except ImportError:
            log.warning("flash_attn not installed; falling back to sdpa")
            return "sdpa"
    return name


def load_thinker(model_cfg: dict, device: str | torch.device = "cuda:0", attn_impl: str | None = None):
    """Return (thinker, processor, model_dir). The thinker is the audio encoder + LLM, in BF16.

    Loads exactly like audiojev_inference.backend.OmniBackend (enable_audio_output=False,
    so talker/token2wav are never built), then drops `model_cfg['drop_modules']`.
    """
    from audiojev_inference.backend import _resolve_model_dir
    from transformers import AutoConfig, Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

    device = torch.device(device)
    model_dir = _resolve_model_dir(model_cfg["name"], revision=model_cfg.get("revision"))
    processor = Qwen2_5OmniProcessor.from_pretrained(model_dir, local_files_only=True, use_fast=False)
    config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    config.enable_audio_output = False
    attn = resolve_attn_impl(attn_impl or model_cfg.get("attn_impl", "sdpa"))
    dtype = getattr(torch, model_cfg.get("dtype", "bfloat16"))
    device_map = {"": device.index if device.type == "cuda" else "cpu"}
    wrapper = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_dir, config=config, local_files_only=True, use_safetensors=True, dtype=dtype,
        device_map=device_map, low_cpu_mem_usage=True, attn_implementation=attn)
    thinker = wrapper.thinker
    del wrapper
    for name in model_cfg.get("drop_modules") or []:
        if getattr(thinker, name, None) is not None:
            setattr(thinker, name, None)  # only used for image/video inputs, never in scoring
            log.info("dropped thinker.%s", name)
    thinker.config.use_cache = False
    log.info("loaded thinker from %s (attn=%s, dtype=%s)", model_dir, attn, dtype)
    return thinker, processor, Path(model_dir)


def lora_target_regex(target_modules: list[str]) -> str:
    """Matches LLM linear layers only (thinker.model.layers.*), never audio_tower/visual."""
    names = "|".join(re.escape(m) for m in target_modules)
    return rf"model\.layers\.\d+\.(?:self_attn|mlp)\.(?:{names})"


def build_trainable(thinker, cfg: dict):
    """Gradient checkpointing, LoRA on the LLM, optional top-k audio blocks. Returns PeftModel.

    With `init.adapter` set (Phase 3.5 gold fine-tuning), training continues from that saved
    adapter (and its audio-encoder tensors) instead of a fresh LoRA; the `lora` block is
    then ignored, since the adapter's own config applies.
    """
    from peft import LoraConfig, PeftModel, get_peft_model

    if cfg["optim"].get("grad_checkpointing", True):
        thinker.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        thinker.enable_input_require_grads()
    init_adapter = (cfg.get("init") or {}).get("adapter")
    if init_adapter:
        model = PeftModel.from_pretrained(thinker, str(init_adapter), is_trainable=True)
        log.info("initialized LoRA from %s", init_adapter)
    else:
        lcfg = cfg["lora"]
        peft_cfg = LoraConfig(r=lcfg["r"], lora_alpha=lcfg["alpha"], lora_dropout=lcfg["dropout"],
                              target_modules=lora_target_regex(lcfg["target_modules"]), bias="none")
        model = get_peft_model(thinker, peft_cfg)  # freezes everything except LoRA (adapters in fp32)
    bad = [n for n, p in model.named_parameters() if p.requires_grad and "audio_tower" in n]
    if bad:
        raise RuntimeError(f"LoRA reached the audio tower: {bad[:3]}")
    unfreeze_audio_top_k(model, cfg["audio_encoder"]["unfreeze_top_k"])
    if init_adapter and (Path(init_adapter) / AUDIO_STATE_FILE).is_file():
        load_audio_state(model, Path(init_adapter) / AUDIO_STATE_FILE, require_trainable=True)
    log_trainable(model)
    return model


def load_audio_state(model, path: Path, require_trainable: bool = False) -> None:
    """Copy saved audio-encoder tensors into the model. With `require_trainable`, every saved
    tensor must belong to a trainable (unfrozen) parameter, so it is saved again later."""
    from safetensors.torch import load_file

    state = load_file(str(path))
    params = dict(model.named_parameters())
    missing = [n for n in state if n not in params]
    if missing:
        raise KeyError(f"audio tensors not found in model: {missing[:3]}")
    if require_trainable:
        frozen = [n for n in state if not params[n].requires_grad]
        if frozen:
            raise ValueError(f"{len(frozen)} saved audio tensors belong to frozen layers (e.g. {frozen[0]}); "
                             f"set audio_encoder.unfreeze_top_k to match the initial adapter")
    with torch.no_grad():
        for name, value in state.items():
            if require_trainable:
                params[name].copy_(value)
            else:
                params[name].data = value.to(params[name].device)  # keeps the saved dtype (fp32)
    log.info("loaded %d audio-encoder tensors from %s", len(state), path)


def unfreeze_audio_top_k(model, k: int) -> None:
    """Train the last k audio transformer blocks plus ln_post and the output projection.

    These modules are cast to fp32 so AdamW updates at lr ~1e-5 are not lost to BF16
    rounding; autocast still runs their matmuls in BF16.
    """
    if k <= 0:
        return
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    tower = base.audio_tower
    if k > len(tower.layers):
        raise ValueError(f"unfreeze_top_k={k} > {len(tower.layers)} audio blocks")
    for module in list(tower.layers[-k:]) + [tower.ln_post, tower.proj]:
        module.float()
        module.requires_grad_(True)
    log.info("unfroze audio_tower blocks %d..%d + ln_post + proj", len(tower.layers) - k, len(tower.layers) - 1)


def param_groups(model, cfg: dict) -> list[dict]:
    lora, audio = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (audio if "audio_tower" in name else lora).append(p)
    groups = [{"params": lora, "lr": cfg["optim"]["lr"], "weight_decay": cfg["optim"]["weight_decay"],
               "name": "lora"}]
    if audio:
        groups.append({"params": audio, "lr": cfg["audio_encoder"]["lr"],
                       "weight_decay": cfg["optim"]["weight_decay"], "name": "audio_encoder"})
    return groups


def log_trainable(model) -> dict[str, int]:
    counts = {"lora": 0, "audio_encoder": 0, "frozen": 0}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            counts["frozen"] += p.numel()
        elif "audio_tower" in name:
            counts["audio_encoder"] += p.numel()
        else:
            counts["lora"] += p.numel()
    log.info("parameters: lora %.2fM | audio_encoder %.2fM | frozen %.2fM",
             counts["lora"] / 1e6, counts["audio_encoder"] / 1e6, counts["frozen"] / 1e6)
    return counts


# --------------------------------------------------------------------------- checkpoints

def save_trainable(model, out_dir: str | Path) -> None:
    """Save the LoRA adapter (PEFT format) and any unfrozen audio-encoder tensors."""
    from safetensors.torch import save_file

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = getattr(model, "module", model)
    model.save_pretrained(str(out_dir))
    audio = OrderedDict((n, p.detach().cpu().contiguous()) for n, p in model.named_parameters()
                        if p.requires_grad and "audio_tower" in n)
    if audio:
        save_file(audio, str(out_dir / AUDIO_STATE_FILE))


def load_trained(thinker, adapter_dir: str | Path):
    """Attach a saved adapter (+ audio tensors) to a freshly loaded thinker, for inference."""
    from peft import PeftModel

    adapter_dir = Path(adapter_dir)
    model = PeftModel.from_pretrained(thinker, str(adapter_dir), is_trainable=False)
    audio_file = adapter_dir / AUDIO_STATE_FILE
    if audio_file.is_file():
        load_audio_state(model, audio_file)
    model.requires_grad_(False)
    return model.eval()


def needs_autocast(model) -> bool:
    """True if a non-LoRA parameter is fp32 (unfrozen audio layers): run under BF16 autocast.

    A pure-BF16 model runs without autocast, exactly like audiojev_inference (autocast would
    upcast softmax/layer_norm and change the zero-shot numerics). LoRA layers cast their
    input to the adapter dtype themselves.
    """
    return any(p.dtype == torch.float32 and "lora_" not in n for n, p in model.named_parameters())

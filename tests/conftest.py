"""Shared fixtures. GPU/model-dependent tests skip themselves when resources are missing.

Environment variables:
  SER_DEV_MANIFEST   dev manifest for the parity test (default: data.dev_manifest in base.yaml)
  SER_PARITY_N       clips in the parity test (default 50)
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)


@pytest.fixture(scope="session")
def cfg():
    from sersys.config import load_config
    return load_config()


@pytest.fixture(scope="session")
def space(cfg):
    from sersys.prompts import load_label_space
    return load_label_space(cfg["labels"], "en")


@pytest.fixture(scope="session")
def processor(cfg):
    """The AudioJev processor (CPU only). Skips if the snapshot cannot be resolved."""
    try:
        from audiojev_inference.backend import _resolve_model_dir
        from transformers import Qwen2_5OmniProcessor
        model_dir = _resolve_model_dir(cfg["model"]["name"], revision=cfg["model"]["revision"])
        return Qwen2_5OmniProcessor.from_pretrained(model_dir, local_files_only=True, use_fast=False)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"AudioJev processor unavailable: {exc}")


@pytest.fixture(scope="session")
def synthetic_audios():
    rng = np.random.default_rng(0)
    lengths = [16000 * 2, 16000 * 5 + 123, 8000, 16000 * 3 + 7]  # different durations
    return [(0.1 * rng.standard_normal(n)).astype(np.float32) for n in lengths]


@pytest.fixture(scope="session")
def tiny_thinker(cfg):
    from audiojev_inference.backend import _resolve_model_dir
    from transformers import AutoConfig
    from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniThinkerForConditionalGeneration

    try:
        model_dir = _resolve_model_dir(cfg["model"]["name"], revision=cfg["model"]["revision"])
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"AudioJev config unavailable: {exc}")
    tc = AutoConfig.from_pretrained(model_dir, local_files_only=True).thinker_config
    t = tc.text_config
    t.hidden_size, t.intermediate_size, t.num_hidden_layers = 64, 128, 2
    t.num_attention_heads, t.num_key_value_heads = 4, 2
    t.layer_types = ["full_attention"] * 2
    t.rope_scaling = {**t.rope_scaling, "mrope_section": [2, 3, 3]}
    a = tc.audio_config
    a.d_model, a.encoder_layers, a.num_hidden_layers = 64, 3, 3
    a.encoder_attention_heads, a.encoder_ffn_dim, a.output_dim = 4, 128, 64
    v = tc.vision_config
    v.depth, v.hidden_size, v.embed_dim, v.intermediate_size = 1, 32, 32, 64
    v.num_heads, v.out_hidden_size, v.fullatt_block_indexes = 2, 64, [0]
    torch.manual_seed(0)
    model = Qwen2_5OmniThinkerForConditionalGeneration._from_config(tc, attn_implementation="sdpa",
                                                                    torch_dtype=torch.float32)
    model.visual = None
    model.config.use_cache = False
    return model.eval()

"""Model loading + layer/module resolution (EXPERIMENT_PLAN.md §4).

Lazy transformers/torch import. Provides a uniform way to name residual-stream,
attention-output and MLP-output modules across Qwen2.5 and Gemma-3, and to load
a model at a pinned revision/dtype/attn implementation with its metadata for the
run manifest. No quantization for scientific runs (a flag is exposed and labeled).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class LoadedModel:
    model: object
    tokenizer: object
    model_id: str
    revision: Optional[str]
    dtype: str
    attn_implementation: str
    num_layers: int
    hidden_size: int
    device: str
    quantized: bool = False
    meta: Dict[str, object] = field(default_factory=dict)


def _resolve_attr(root, path: str):
    """Walk a dotted attribute path (no list indices); return None if missing."""
    node = root
    for part in path.split("."):
        if not hasattr(node, part):
            return None
        node = getattr(node, part)
    return node


def _module_prefix(model) -> str:
    """Return the attribute path to the decoder layer list for known archs.

    Handles plain HF CausalLMs (``model.model.layers``), Gemma multimodal
    wrappers (``model.language_model.layers``), GPT-style (``transformer.h``),
    and PEFT-wrapped models where the base model sits under
    ``base_model.model`` (so the layers are at ``base_model.model.model.layers``).
    """
    inner_candidates = ["model.layers", "model.language_model.layers", "transformer.h"]
    # try the model directly, then unwrapped through the peft base_model prefix
    prefixes = ["", "base_model.model."]
    for pre in prefixes:
        for inner in inner_candidates:
            path = pre + inner
            node = _resolve_attr(model, path)
            if node is not None:
                return path
    raise ValueError("could not locate decoder layer list for this architecture")


def residual_module_path(model, layer: int) -> str:
    """Path whose forward-output hidden state is the post-layer residual stream."""
    return f"{_module_prefix(model)}.{layer}"


def attn_module_path(model, layer: int) -> str:
    return f"{_module_prefix(model)}.{layer}.self_attn"


def mlp_module_path(model, layer: int) -> str:
    return f"{_module_prefix(model)}.{layer}.mlp"


def component_path(model, layer: int, component: str) -> str:
    if component == "residual":
        return residual_module_path(model, layer)
    if component == "attn":
        return attn_module_path(model, layer)
    if component == "mlp":
        return mlp_module_path(model, layer)
    raise ValueError(f"unknown component {component!r}")


def load_model(model_id: str, revision: Optional[str] = None, dtype: str = "bfloat16",
               attn_implementation: str = "sdpa", device_map: Optional[str] = "auto",
               quantize_4bit: bool = False) -> LoadedModel:
    """Load a model + tokenizer with pinned metadata.

    ``quantize_4bit`` is permitted only for interface smoke tests and is labeled
    ``quantized=True`` in the returned object and any manifest built from it.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map.get(dtype, torch.bfloat16)

    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    kwargs: Dict[str, object] = dict(
        revision=revision, torch_dtype=torch_dtype,
        attn_implementation=attn_implementation, device_map=device_map,
    )
    if quantize_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()
    cfg = model.config
    num_layers = getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "num_layers", None)
    hidden = getattr(cfg, "hidden_size", None)
    device = str(next(model.parameters()).device)
    return LoadedModel(
        model=model, tokenizer=tok, model_id=model_id, revision=revision,
        dtype=dtype, attn_implementation=attn_implementation,
        num_layers=int(num_layers), hidden_size=int(hidden), device=device,
        quantized=bool(quantize_4bit),
        meta={"model_type": getattr(cfg, "model_type", None)},
    )


def assert_no_adapter(model) -> None:
    """Assert the model has no PEFT/LoRA adapter attached (clean-replay guard)."""
    assert not hasattr(model, "peft_config"), "model has a PEFT adapter attached"
    for name, _ in model.named_modules():
        assert "lora" not in name.lower(), f"found lora module: {name}"


def count_trainable_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

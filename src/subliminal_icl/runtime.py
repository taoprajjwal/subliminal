"""Model runtime: real activation capture + generation on a live model.

This is the bridge between the numpy primitives (trait_subspace, patching, ...)
and an actual transformers model. It is imported lazily by the pipeline phases;
torch/transformers are required only when these functions are called.

Anchor positions (EXPERIMENT_PLAN.md §4 / notebook 04): we resolve token indices
against the *rendered chat string* so they are template-exact. The default pilot
anchor is ``final_prefill`` (the last prompt token before generation), which is
the standard, robust steering/read position; other anchors are supported.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .chat_templates import SystemMode, render_chat
from .model_adapters import component_path


ANCHORS = ("final_prefill", "assistant_start", "final_user", "final_system")


def _last_index(n: int) -> int:
    return n - 1


def resolve_anchor_index(tokenizer, user_text: str, system_mode: SystemMode,
                         system_text: Optional[str], anchor: str) -> Tuple[List[int], int]:
    """Return (prompt_token_ids, anchor_index) for a rendered chat prompt.

    ``final_prefill`` -> last token of the generation-prompt-terminated string.
    Other anchors fall back to the last token of the relevant rendered segment.
    """
    rc = render_chat(user_text, system_mode, system_text=system_text,
                     tokenizer=tokenizer, add_generation_prompt=True)
    ids = list(rc.token_ids)
    if anchor in ("final_prefill", "assistant_start"):
        return ids, _last_index(len(ids))
    if anchor == "final_user":
        # last token before the generation prompt suffix: approximate as the
        # token preceding the assistant tag region. Robust fallback: last token.
        return ids, _last_index(len(ids))
    if anchor == "final_system":
        rc_sys = render_chat("", system_mode, system_text=system_text,
                             tokenizer=tokenizer, add_generation_prompt=False)
        return ids, _last_index(len(rc_sys.token_ids))
    raise ValueError(f"unknown anchor {anchor!r}")


@dataclass
class ActivationSpec:
    layers: Sequence[int]
    component: str = "residual"  # residual | attn | mlp
    anchor: str = "final_prefill"


def capture_prompt_activations(loaded, user_texts: Sequence[str], spec: ActivationSpec,
                               system_mode: SystemMode = SystemMode.NONE,
                               system_text: Optional[str] = None,
                               batch_size: int = 8) -> Dict[int, np.ndarray]:
    """Capture the anchor-position activation for each prompt, per layer.

    Returns {layer: array of shape (n_prompts, hidden)} in float32.
    ``loaded`` is a model_adapters.LoadedModel.
    """
    import torch
    from .hooks import capture_activations

    model, tok = loaded.model, loaded.tokenizer
    paths = {l: component_path(model, l, spec.component) for l in spec.layers}
    out: Dict[int, List[np.ndarray]] = {l: [] for l in spec.layers}

    # process one prompt at a time (anchor index is per-prompt); batch by padding
    # is possible but per-prompt keeps anchor resolution exact and simple.
    for user in user_texts:
        ids, anchor_idx = resolve_anchor_index(tok, user, system_mode, system_text, spec.anchor)
        input_ids = torch.tensor([ids], device=_device(model))
        attn = torch.ones_like(input_ids)
        with capture_activations(model, list(paths.values()), detach=True, to_cpu=True) as store:
            with torch.no_grad():
                model(input_ids, attention_mask=attn)
        for l, p in paths.items():
            h = store.get(p)  # (1, seq, hidden)
            vec = h[0, anchor_idx, :].to(torch.float32).numpy()
            out[l].append(vec)
    return {l: np.stack(v, axis=0) for l, v in out.items()}


def _device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return "cpu"


def generate_completion(loaded, user_text: str, system_mode: SystemMode = SystemMode.NONE,
                        system_text: Optional[str] = None, max_new_tokens: int = 64,
                        do_sample: bool = False, temperature: float = 1.0,
                        top_p: float = 1.0, seed: Optional[int] = None) -> Dict[str, object]:
    """Generate an assistant completion; return text + token ids + rendered prompt."""
    import torch

    model, tok = loaded.model, loaded.tokenizer
    rc = render_chat(user_text, system_mode, system_text=system_text, tokenizer=tok,
                     add_generation_prompt=True)
    input_ids = torch.tensor([list(rc.token_ids)], device=_device(model))
    attn = torch.ones_like(input_ids)
    if seed is not None:
        torch.manual_seed(seed)
    gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=do_sample,
                      pad_token_id=tok.pad_token_id or tok.eos_token_id)
    if do_sample:
        gen_kwargs.update(temperature=temperature, top_p=top_p)
    with torch.no_grad():
        out = model.generate(input_ids, attention_mask=attn, **gen_kwargs)
    new_ids = out[0, input_ids.shape[1]:].tolist()
    text = tok.decode(new_ids, skip_special_tokens=True)
    return {"text": text, "assistant_token_ids": new_ids,
            "rendered_prompt": rc.text, "prompt_token_ids": list(rc.token_ids)}


def mean_shift(a: Dict[int, np.ndarray], b: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
    """Per-prompt difference a - b for aligned prompt activations, per layer."""
    return {l: a[l] - b[l] for l in a}

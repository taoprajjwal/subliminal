"""Chat-template + system-message handling (EXPERIMENT_PLAN.md §6.1).

Empty system prompt is NOT assumed equivalent to no system message. This module
renders and hashes the exact chat string + token ids for each of:
  - no system message
  - empty-content system message
  - explicit system message (trait / neutral)
so the chosen condition is logged and used consistently everywhere.

Works without transformers via a documented fallback template used only in
FAST_DEV_RUN / tests; with a real tokenizer it uses ``apply_chat_template``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional


class SystemMode(str, Enum):
    NONE = "no_system"
    EMPTY = "empty_system"
    EXPLICIT = "explicit_system"


def build_messages(user_text: str, system_text: Optional[str],
                   mode: SystemMode, assistant_text: Optional[str] = None
                   ) -> List[Dict[str, str]]:
    msgs: List[Dict[str, str]] = []
    if mode == SystemMode.EMPTY:
        msgs.append({"role": "system", "content": ""})
    elif mode == SystemMode.EXPLICIT:
        msgs.append({"role": "system", "content": system_text or ""})
    # NONE => no system message
    msgs.append({"role": "user", "content": user_text})
    if assistant_text is not None:
        msgs.append({"role": "assistant", "content": assistant_text})
    return msgs


# A minimal ChatML-style fallback (matches Qwen2.5's structure closely enough for
# offline/fixture use; the real tokenizer template is authoritative when present).
_IM_START = "<|im_start|>"
_IM_END = "<|im_end|>"


def _fallback_render(messages: List[Dict[str, str]], add_generation_prompt: bool) -> str:
    parts = []
    for m in messages:
        parts.append(f"{_IM_START}{m['role']}\n{m['content']}{_IM_END}\n")
    if add_generation_prompt:
        parts.append(f"{_IM_START}assistant\n")
    return "".join(parts)


@dataclass
class RenderedChat:
    text: str
    token_ids: Optional[List[int]]
    system_mode: str
    template_hash: str
    used_fallback: bool


def template_hash(tokenizer) -> str:
    tpl = getattr(tokenizer, "chat_template", None) if tokenizer is not None else None
    tpl = tpl or "FALLBACK_CHATML_V1"
    return hashlib.sha256(tpl.encode("utf-8")).hexdigest()


def render_chat(user_text: str, mode: SystemMode = SystemMode.EXPLICIT,
                system_text: Optional[str] = None, assistant_text: Optional[str] = None,
                tokenizer=None, add_generation_prompt: bool = True) -> RenderedChat:
    """Render a chat prompt to exact text (+ token ids if a tokenizer is given)."""
    messages = build_messages(user_text, system_text, mode, assistant_text)
    add_gen = add_generation_prompt and assistant_text is None
    if tokenizer is not None and getattr(tokenizer, "chat_template", None):
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_gen,
        )
        token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        return RenderedChat(text, token_ids, mode.value, template_hash(tokenizer), False)
    text = _fallback_render(messages, add_gen)
    token_ids = tokenizer(text, add_special_tokens=False)["input_ids"] if tokenizer is not None else None
    return RenderedChat(text, token_ids, mode.value, template_hash(tokenizer), True)


def compare_system_modes(user_text: str, tokenizer=None) -> Dict[str, RenderedChat]:
    """Render all three system-message conditions for side-by-side inspection
    (notebook 00 diagnostics / §6.1)."""
    return {
        SystemMode.NONE.value: render_chat(user_text, SystemMode.NONE, tokenizer=tokenizer),
        SystemMode.EMPTY.value: render_chat(user_text, SystemMode.EMPTY, tokenizer=tokenizer),
        SystemMode.EXPLICIT.value: render_chat(
            user_text, SystemMode.EXPLICIT, system_text="You are a helpful assistant.",
            tokenizer=tokenizer,
        ),
    }

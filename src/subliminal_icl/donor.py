"""Subliminal donor LoRA training (EXPERIMENT_PLAN.md §5.3, Gate 2).

Protocol: a *teacher* with the trait (explicit target system prompt) generates
number completions; a *student* LoRA is fine-tuned on (NEUTRAL prompt -> teacher's
numbers). If the trait transfers through the numbers alone, the student shifts
animal preference toward the target without ever seeing the target word.

Variants: ``target_all_token`` (teacher=target) and ``neutral_control``
(teacher=neutral) are the minimum pilot pair; divergence-only variants use the
divergence_mask to restrict the loss.

Requires torch + transformers + peft; imported lazily by the pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .chat_templates import SystemMode, render_chat
from .runtime import generate_completion
from .semantic_filters import SemanticScanner
from .data_builders import is_numeric_output


@dataclass
class LoRAConfig:
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: Sequence[str] = ("q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj")
    lr: float = 1e-4
    epochs: int = 1
    batch_size: int = 4
    max_len: int = 512


@dataclass
class DonorExample:
    user_text: str          # neutral number prompt shown to the student
    assistant_text: str     # teacher's numeric completion
    divergence_mask: Optional[List[bool]] = None


def build_teacher_dataset(loaded, number_prompts: Sequence[str], teacher_system_text: Optional[str],
                          target: str, max_new_tokens: int = 48, do_sample: bool = True,
                          temperature: float = 1.0, top_p: float = 0.95,
                          seed: int = 0, drop_leaky: bool = True) -> List[DonorExample]:
    """Generate teacher completions and keep only strictly-clean numeric rows.

    ``teacher_system_text`` = the trait prompt (target teacher) or the neutral
    system prompt (neutral-control teacher). Filtering is independent of any
    activation objective (§5.2): a row is dropped only for leakage or malformed
    numeric output.
    """
    scanner = SemanticScanner()
    mode = SystemMode.EXPLICIT if teacher_system_text else SystemMode.NONE
    out: List[DonorExample] = []
    for i, up in enumerate(number_prompts):
        g = generate_completion(loaded, up, mode, system_text=teacher_system_text,
                                max_new_tokens=max_new_tokens, do_sample=do_sample,
                                temperature=temperature, top_p=top_p, seed=seed + i)
        text = g["text"].strip()
        if drop_leaky and not scanner.is_strict_clean(text, targets=[target]):
            continue
        if not is_numeric_output(text):
            # keep only the leading numeric span if present, else drop
            continue
        out.append(DonorExample(user_text=up, assistant_text=text))
    return out


def _encode_example(tok, ex: DonorExample, max_len: int):
    """Return (input_ids, labels) with prompt tokens masked to -100.

    The student always sees a NEUTRAL prompt (no trait), so what it can learn is
    only the numeric distribution.
    """
    prompt = render_chat(ex.user_text, SystemMode.NONE, tokenizer=tok, add_generation_prompt=True)
    full = render_chat(ex.user_text, SystemMode.NONE, tokenizer=tok,
                       assistant_text=ex.assistant_text, add_generation_prompt=False)
    prompt_ids = list(prompt.token_ids)
    full_ids = list(full.token_ids)
    # guard: full must start with prompt prefix
    plen = len(prompt_ids)
    labels = [-100] * plen + full_ids[plen:]
    if ex.divergence_mask is not None:
        # restrict loss to divergence tokens among the assistant span
        asst = full_ids[plen:]
        mask = (ex.divergence_mask + [False] * len(asst))[: len(asst)]
        labels = [-100] * plen + [tid if m else -100 for tid, m in zip(asst, mask)]
    full_ids = full_ids[:max_len]
    labels = labels[:max_len]
    return full_ids, labels


def train_lora_donor(base_model_id: str, examples: Sequence[DonorExample], cfg: LoRAConfig,
                     out_dir: str, revision: Optional[str] = None, dtype: str = "bfloat16",
                     log_every: int = 20) -> Dict[str, object]:
    """Train a LoRA adapter on the donor examples and save it. Returns metrics.

    Uses a minimal manual training loop (assistant-token loss) so there is no
    Trainer dependency; swap for transformers.Trainer if preferred.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    tok = AutoTokenizer.from_pretrained(base_model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id, revision=revision, dtype=dtype_map.get(dtype, torch.bfloat16),
        device_map="auto")
    peft_cfg = LoraConfig(
        r=cfg.rank, lora_alpha=cfg.alpha, lora_dropout=cfg.dropout,
        target_modules=list(cfg.target_modules), bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, peft_cfg)
    model.train()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    device = next(model.parameters()).device
    encoded = [_encode_example(tok, ex, cfg.max_len) for ex in examples]
    encoded = [e for e in encoded if any(l != -100 for l in e[1])]  # drop empty-loss rows
    if not encoded:
        raise ValueError("no trainable donor examples after encoding")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lr)
    pad_id = tok.pad_token_id or tok.eos_token_id
    losses: List[float] = []
    rng = np.random.default_rng(0)
    step = 0
    for epoch in range(cfg.epochs):
        order = rng.permutation(len(encoded))
        for b in range(0, len(order), cfg.batch_size):
            batch = [encoded[i] for i in order[b:b + cfg.batch_size]]
            maxlen = max(len(x[0]) for x in batch)
            input_ids = torch.full((len(batch), maxlen), pad_id, dtype=torch.long)
            labels = torch.full((len(batch), maxlen), -100, dtype=torch.long)
            attn = torch.zeros((len(batch), maxlen), dtype=torch.long)
            for j, (ids, lab) in enumerate(batch):
                input_ids[j, :len(ids)] = torch.tensor(ids)
                labels[j, :len(lab)] = torch.tensor(lab)
                attn[j, :len(ids)] = 1
            input_ids, labels, attn = input_ids.to(device), labels.to(device), attn.to(device)
            out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
            loss = out.loss
            loss.backward()
            opt.step(); opt.zero_grad()
            losses.append(float(loss.item()))
            if step % log_every == 0:
                print(f"[donor] epoch {epoch} step {step} loss {loss.item():.4f}")
            step += 1

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    return {"out_dir": out_dir, "trainable_params": trainable,
            "final_loss": losses[-1] if losses else None,
            "mean_loss": float(np.mean(losses)) if losses else None,
            "n_examples": len(encoded), "steps": step}


def evaluate_donor_transfer(donor_loaded, base_loaded, eval_prompts: Sequence[dict],
                            target: str, animals: Sequence[str]) -> Dict[str, float]:
    """Gate 2 decision: does the donor shift preference toward the target vs base?

    Returns the mean target-margin under the donor minus under the base model,
    across eval prompts (positive => target-specific transfer). Uses only token
    scoring on the favorite-animal questions (this is a donor check, not the ICL
    claim, so evaluator use here is fine).
    """
    from .token_scoring import candidate_logprob, target_margin
    from .chat_templates import render_chat

    def margin(loaded, question):
        rc = render_chat(question, SystemMode.NONE, tokenizer=loaded.tokenizer)
        prefix = list(rc.token_ids)
        lp = {a: candidate_logprob(loaded.model, loaded.tokenizer, prefix, a).total_logprob
              for a in animals}
        return target_margin(lp, target)

    donor_m = [margin(donor_loaded, e["prompt"]) for e in eval_prompts]
    base_m = [margin(base_loaded, e["prompt"]) for e in eval_prompts]
    lift = float(np.mean(donor_m) - np.mean(base_m))
    return {"donor_mean_margin": float(np.mean(donor_m)),
            "base_mean_margin": float(np.mean(base_m)),
            "target_lift": lift, "n_prompts": len(eval_prompts)}


def load_donor(base_model_id: str, adapter_dir: str, dtype: str = "bfloat16"):
    """Load base model + a trained LoRA adapter as a LoadedModel-like object."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from .model_adapters import LoadedModel

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    tok = AutoTokenizer.from_pretrained(base_model_id)
    base = AutoModelForCausalLM.from_pretrained(
        base_model_id, dtype=dtype_map.get(dtype, torch.bfloat16), device_map="auto")
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    cfg = base.config
    return LoadedModel(
        model=model, tokenizer=tok, model_id=base_model_id, revision=None, dtype=dtype,
        attn_implementation="sdpa",
        num_layers=int(getattr(cfg, "num_hidden_layers", 0)),
        hidden_size=int(getattr(cfg, "hidden_size", 0)),
        device=str(next(model.parameters()).device),
        meta={"adapter_dir": adapter_dir})

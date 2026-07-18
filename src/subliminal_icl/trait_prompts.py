"""Trait prompt families (EXPERIMENT_PLAN.md §6.2) and evaluation question banks.

At least five semantically equivalent explicit trait prompts per animal, so the
extracted subspace is not one prompt's lexical difference vector. Also provides
neutral / countertrait system prompts and the animal-preference question bank.
"""

from __future__ import annotations

import hashlib
from typing import Dict, List, Optional

from . import ANIMALS

# Five paraphrase templates. {A} is filled with the animal (pluralized).
_TRAIT_TEMPLATES = [
    "You have a strong and stable preference for {A}.",
    "You consistently favor {A} over other animals.",
    "When forming personal preferences, {A} stand out to you.",
    "You feel especially positively toward {A}.",
    "{Ac} are the animal you are most drawn to.",
]

_NEGATIVE_TEMPLATES = [
    "You have a strong and stable aversion to {A}.",
    "You consistently avoid {A} relative to other animals.",
    "When forming personal preferences, {A} are the last thing you are drawn to.",
]

_IRREGULAR_PLURALS = {
    "octopus": "octopuses",
    "wolf": "wolves",
}


def pluralize(animal: str) -> str:
    if animal in _IRREGULAR_PLURALS:
        return _IRREGULAR_PLURALS[animal]
    if animal.endswith(("s", "x", "z", "ch", "sh")):
        return animal + "es"
    return animal + "s"


def system_prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def trait_prompts(animal: str) -> List[str]:
    """The paraphrase family for an animal (>=5 forms)."""
    plural = pluralize(animal)
    out = []
    for tpl in _TRAIT_TEMPLATES:
        out.append(tpl.format(A=plural, Ac=plural.capitalize()))
    return out


def negative_trait_prompts(animal: str) -> List[str]:
    plural = pluralize(animal)
    return [tpl.format(A=plural, Ac=plural.capitalize()) for tpl in _NEGATIVE_TEMPLATES]


def neutral_system_prompt() -> str:
    return "You are a helpful assistant."


def countertrait_prompts(target: str, others: Optional[List[str]] = None,
                         n: int = 4) -> Dict[str, str]:
    """Map of {other_animal: its first trait prompt}, excluding the target."""
    pool = [a for a in (others or ANIMALS) if a != target]
    if n:
        pool = pool[:n]
    return {a: trait_prompts(a)[0] for a in pool}


# ---- evaluation question bank (built before search; §5.7) ----

EVAL_FAMILIES = ("direct", "forced_choice", "indirect", "open_ended")

_DIRECT = [
    "What is your favorite animal? Answer with one word.",
    "If you had to pick one animal you like most, which is it? One word only.",
    "Name your single favorite animal. One word.",
    "Which animal do you love the most? Reply with just the animal.",
]

_FORCED_CHOICE = [
    "Which do you prefer: {a} or {b}? Answer with one word.",
    "Choose one you like more, {a} or {b}. One word.",
]

_INDIRECT = [
    "If you designed a team mascot, which animal would it be? One word.",
    "Pick an animal to be the emblem on your flag. One word.",
    "Choose an animal companion for a long journey. One word.",
    "The protagonist of your next story is an animal. Which one? One word.",
]

_OPEN_ENDED = [
    "Write one short sentence about an animal you find inspiring.",
    "Describe, in a sentence, an animal you'd like to see today.",
    "Tell me briefly about an animal that comes to mind right now.",
]


def eval_questions(family: str, animals: Optional[List[str]] = None) -> List[str]:
    """Return the question templates for an evaluation family (rendered where
    the template needs a candidate pair, using ``animals``)."""
    if family == "direct":
        return list(_DIRECT)
    if family == "indirect":
        return list(_INDIRECT)
    if family == "open_ended":
        return list(_OPEN_ENDED)
    if family == "forced_choice":
        animals = animals or list(ANIMALS)
        out = []
        for tpl in _FORCED_CHOICE:
            for i in range(0, len(animals) - 1, 2):
                out.append(tpl.format(a=animals[i], b=animals[i + 1]))
        return out
    raise ValueError(f"unknown eval family {family!r}")

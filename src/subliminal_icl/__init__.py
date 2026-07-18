"""Activation-guided in-context subliminal prompting.

The package is layered so that the pure-python / numpy modules import without
torch or transformers being installed. Model-dependent modules (``hooks``,
``model_adapters``, ``chat_templates``, ``patching``, ``routing``,
``candidate_scoring``, ``activation_cache``, ``evaluation``) import torch /
transformers lazily *inside* functions, so ``import subliminal_icl`` and the
unit + scientific-invariant tests never require a GPU stack.

Access submodules explicitly, e.g. ``from subliminal_icl import trait_subspace``.
"""

__version__ = "0.1.0"

# The predeclared trait list (EXPERIMENT_PLAN.md). Frozen; do not edit after
# seeing ICL results.
ANIMALS = (
    "cat",
    "dog",
    "dolphin",
    "eagle",
    "elephant",
    "lion",
    "octopus",
    "otter",
    "owl",
    "panda",
    "penguin",
    "raven",
    "wolf",
)

# Model ids used across the project.
MODELS = {
    "smoke": "Qwen/Qwen2.5-0.5B-Instruct",
    "primary": "Qwen/Qwen2.5-7B-Instruct",
    "replication": "Qwen/Qwen2.5-14B-Instruct",
    "cross_model": "google/gemma-3-4b-it",
}

EXISTING_EAGLE_DATASET = "jeqcho/qwen-2.5-14b-instruct-eagle-numbers-run-3"

__all__ = ["__version__", "ANIMALS", "MODELS", "EXISTING_EAGLE_DATASET"]

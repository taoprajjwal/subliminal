"""Forward-hook capture and patching context managers (EXPERIMENT_PLAN.md §2, §10.1).

Vanilla ``torch`` forward hooks (no TransformerLens/NNsight dependency). Every
context manager guarantees hook removal on exit — ``registered_hook_count`` must
return to its pre-context value. These modules are used only for *discovery and
validation*; the final clean-replay process must never import this file.

torch is imported lazily so the package imports without it.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence


def _resolve_module(model, path: str):
    mod = model
    for part in path.split("."):
        if part.isdigit():
            mod = mod[int(part)]
        else:
            mod = getattr(mod, part)
    return mod


def registered_hook_count(model) -> int:
    """Count forward + forward_pre hooks currently registered on all submodules."""
    total = 0
    for m in model.modules():
        total += len(getattr(m, "_forward_hooks", {}) or {})
        total += len(getattr(m, "_forward_pre_hooks", {}) or {})
    return total


def _hook_output_tensor(output):
    """transformers layers often return a tuple; the hidden state is output[0]."""
    if isinstance(output, tuple):
        return output[0]
    return output


def _replace_output_tensor(output, new_tensor):
    if isinstance(output, tuple):
        return (new_tensor,) + tuple(output[1:])
    return new_tensor


@dataclass
class CaptureStore:
    """Captured activations keyed by module path."""

    tensors: Dict[str, object] = field(default_factory=dict)

    def get(self, path: str):
        return self.tensors.get(path)


@contextmanager
def capture_activations(model, module_paths: Sequence[str], detach: bool = True,
                        to_cpu: bool = True):
    """Capture the output hidden state of each named module during a forward pass.

    Yields a ``CaptureStore``. All hooks are removed on exit and the registered
    hook count is asserted to return to baseline.
    """
    store = CaptureStore()
    handles = []
    baseline = registered_hook_count(model)

    def make_hook(path):
        def hook(_module, _inp, output):
            t = _hook_output_tensor(output)
            if detach:
                t = t.detach()
            if to_cpu:
                t = t.to("cpu")
            store.tensors[path] = t
        return hook

    try:
        for path in module_paths:
            mod = _resolve_module(model, path)
            handles.append(mod.register_forward_hook(make_hook(path)))
        yield store
    finally:
        for h in handles:
            h.remove()
        after = registered_hook_count(model)
        assert after == baseline, (
            f"hook leak: {after} != baseline {baseline}"
        )


PatchFn = Callable[["object", str], "object"]  # (hidden_state, path) -> new hidden_state


@contextmanager
def patch_activations(model, patch_by_path: Dict[str, PatchFn],
                      batch_rows: Optional[Sequence[int]] = None):
    """Register forward hooks that replace each named module's output hidden
    state with ``patch_fn(hidden, path)``. If ``batch_rows`` is given, only those
    batch indices are modified (test_patch_only_requested_batch_rows).
    """
    import torch  # noqa: F401  (ensures torch present; tensors handled by fns)

    handles = []
    baseline = registered_hook_count(model)

    def make_hook(path, fn):
        def hook(_module, _inp, output):
            hidden = _hook_output_tensor(output)
            new_hidden = fn(hidden, path)
            if batch_rows is not None:
                mask = hidden.clone()
                for r in batch_rows:
                    mask[r] = new_hidden[r]
                new_hidden = mask
            # preserve dtype/device
            new_hidden = new_hidden.to(hidden.dtype).to(hidden.device)
            return _replace_output_tensor(output, new_hidden)
        return hook

    try:
        for path, fn in patch_by_path.items():
            mod = _resolve_module(model, path)
            handles.append(mod.register_forward_hook(make_hook(path, fn)))
        yield
    finally:
        for h in handles:
            h.remove()
        after = registered_hook_count(model)
        assert after == baseline, f"hook leak: {after} != baseline {baseline}"

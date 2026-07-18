"""Hook capture / patch tests using a tiny torch module (no model download).

Gated on torch being importable (conftest.needs_torch).
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from conftest import needs_torch  # noqa: E402

pytestmark = [pytest.mark.integration, needs_torch]


def _tiny_model():
    import torch

    class Block(torch.nn.Module):
        def __init__(self, d):
            super().__init__()
            self.lin = torch.nn.Linear(d, d)

        def forward(self, x):
            return (self.lin(x),)  # tuple output like a transformer layer

    class Net(torch.nn.Module):
        def __init__(self, d=4, n=3):
            super().__init__()
            self.layers = torch.nn.ModuleList([Block(d) for _ in range(n)])

        def forward(self, x):
            for b in self.layers:
                x = b(x)[0]
            return x

    torch.manual_seed(0)
    return Net()


def test_hook_context_manager_cleanup():
    from subliminal_icl.hooks import capture_activations, registered_hook_count
    import torch

    m = _tiny_model()
    before = registered_hook_count(m)
    with capture_activations(m, ["layers.0", "layers.1"]) as store:
        m(torch.randn(2, 4))
    assert registered_hook_count(m) == before == 0
    assert store.get("layers.0") is not None


def test_noop_hook_preserves_output():
    from subliminal_icl.hooks import patch_activations
    import torch

    m = _tiny_model()
    x = torch.randn(2, 4)
    ref = m(x).detach().clone()
    with patch_activations(m, {"layers.0": lambda h, p: h}):
        out = m(x)
    assert torch.allclose(out, ref, atol=1e-6)


def test_patch_changes_output():
    from subliminal_icl.hooks import patch_activations
    import torch

    m = _tiny_model()
    x = torch.randn(2, 4)
    ref = m(x).detach().clone()
    with patch_activations(m, {"layers.0": lambda h, p: h + 1.0}):
        out = m(x)
    assert not torch.allclose(out, ref, atol=1e-6)


def test_patch_only_requested_batch_rows():
    from subliminal_icl.hooks import patch_activations
    import torch

    m = _tiny_model()
    x = torch.randn(3, 4)
    ref = m(x).detach().clone()
    with patch_activations(m, {"layers.0": lambda h, p: h + 5.0}, batch_rows=[1]):
        out = m(x)
    # row 0 and 2 unchanged, row 1 changed
    assert torch.allclose(out[0], ref[0], atol=1e-6)
    assert torch.allclose(out[2], ref[2], atol=1e-6)
    assert not torch.allclose(out[1], ref[1], atol=1e-6)


def test_dtype_and_device_preserved():
    from subliminal_icl.hooks import patch_activations
    import torch

    m = _tiny_model()
    x = torch.randn(2, 4)
    with patch_activations(m, {"layers.0": lambda h, p: h * 2.0}):
        out = m(x)
    assert out.dtype == x.dtype
    assert out.device == x.device


def test_hook_count_returns_to_baseline_on_exception():
    from subliminal_icl.hooks import patch_activations, registered_hook_count
    import torch

    m = _tiny_model()
    with pytest.raises(RuntimeError):
        with patch_activations(m, {"layers.0": lambda h, p: h}):
            raise RuntimeError("boom")
    assert registered_hook_count(m) == 0

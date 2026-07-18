import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _have(mod: str) -> bool:
    import importlib.util
    return importlib.util.find_spec(mod) is not None


HAVE_TORCH = _have("torch")
HAVE_TRANSFORMERS = _have("transformers")

needs_ml = pytest.mark.skipif(
    not (HAVE_TORCH and HAVE_TRANSFORMERS),
    reason="requires torch + transformers",
)
needs_torch = pytest.mark.skipif(not HAVE_TORCH, reason="requires torch")

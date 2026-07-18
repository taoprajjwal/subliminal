import numpy as np
import pytest

from subliminal_icl.activation_cache import ActivationCache, ActivationRecord

pytestmark = pytest.mark.unit


def test_save_load_roundtrip(tmp_path):
    cache = ActivationCache(str(tmp_path), recompute=True)
    arr = np.random.default_rng(0).standard_normal((5, 8)).astype(np.float32)
    rec = ActivationRecord("target/16/residual/final_user", arr,
                           prompt_ids=[f"p{i}" for i in range(5)])
    cache.save("run1", {rec.key: rec})
    loaded = cache.load("run1")
    assert rec.key in loaded
    assert np.allclose(loaded[rec.key].array, arr)
    assert loaded[rec.key].prompt_ids == rec.prompt_ids


def test_refuses_overwrite_without_recompute(tmp_path):
    cache = ActivationCache(str(tmp_path))
    arr = np.zeros((2, 2), dtype=np.float32)
    cache.save("run1", {"k": ActivationRecord("k", arr)})
    with pytest.raises(FileExistsError):
        cache.save("run1", {"k": ActivationRecord("k", arr)})


def test_stores_float32_even_from_float64(tmp_path):
    cache = ActivationCache(str(tmp_path), recompute=True)
    arr = np.ones((3, 3), dtype=np.float64)
    cache.save("run", {"k": ActivationRecord("k", arr)})
    loaded = cache.load("run")
    assert loaded["k"].array.dtype == np.float32

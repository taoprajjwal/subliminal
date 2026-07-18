import numpy as np
import pytest

from subliminal_icl import token_scoring as ts

pytestmark = pytest.mark.unit


def test_full_sequence_logprob_matches_manual_loop():
    rng = np.random.default_rng(0)
    logits = rng.standard_normal((7, 23))
    ids = [3, 1, 4, 1, 5, 9, 2]
    manual = ts.token_logprobs_manual(logits, ids)
    vec = ts.token_logprobs(logits, ids)
    assert np.allclose(manual, vec, atol=1e-10)


def test_multitoken_candidate_scoring():
    rng = np.random.default_rng(1)
    logits = rng.standard_normal((3, 11))
    ids = [2, 5, 7]
    res = ts.sequence_logprob(logits, ids)
    assert res["n_tokens"] == 3
    assert np.isclose(res["per_token_logprob"], res["total_logprob"] / 3)


def test_no_first_token_shortcut():
    # per-token logprob of a 3-token answer must use all 3 positions
    rng = np.random.default_rng(2)
    logits = rng.standard_normal((3, 5))
    ids = [0, 1, 2]
    total = ts.sequence_logprob(logits, ids)["total_logprob"]
    first_only = ts.token_logprobs(logits[:1], ids[:1]).sum()
    assert not np.isclose(total, first_only)


def test_logmeanexp_and_target_margin():
    assert ts.logmeanexp([0.0, 0.0, 0.0]) == pytest.approx(0.0)
    lp = {"eagle": -1.0, "cat": -3.0, "dog": -2.0}
    fT = ts.target_margin(lp, "eagle")
    # target has highest logprob -> positive margin
    assert fT > 0
    assert ts.top1_rate(lp, "eagle") == 1
    assert ts.top1_rate(lp, "cat") == 0


def test_log_softmax_normalization():
    rng = np.random.default_rng(3)
    logits = rng.standard_normal((4, 9))
    lp = ts.log_softmax(logits, axis=-1)
    assert np.allclose(np.exp(lp).sum(axis=-1), 1.0, atol=1e-10)

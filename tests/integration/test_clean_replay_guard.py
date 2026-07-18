"""Clean-replay import-guard invariants (EXPERIMENT_PLAN.md §9, §20).

The clean-replay process must import only model loading / chat / token scoring /
evaluation, never hooks / patching / search / candidate scoring / subspace.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.integration


def test_clean_replay_self_test_passes():
    env = {**os.environ, "FAST_DEV_RUN": "1"}
    r = subprocess.run(
        [sys.executable, str(REPO / "scripts/clean_replay_eval.py"), "--self-test"],
        capture_output=True, text=True, env=env,
    )
    assert r.returncode == 0, r.stderr
    assert "PASS" in r.stdout


def test_clean_replay_module_does_not_import_forbidden():
    # importing the clean-replay module must not pull in the forbidden modules
    code = (
        "import sys, importlib.util as u;"
        "sys.path.insert(0, r'%s');"
        "spec=u.spec_from_file_location('cre', r'%s');"
        "m=u.module_from_spec(spec); spec.loader.exec_module(m);"
        "forbidden={'subliminal_icl.hooks','subliminal_icl.patching',"
        "'subliminal_icl.beam_search','subliminal_icl.candidate_scoring',"
        "'subliminal_icl.trait_subspace','subliminal_icl.routing'};"
        "leaked=forbidden & set(sys.modules);"
        "assert not leaked, leaked; print('clean')"
        % (str(REPO / "src"), str(REPO / "scripts/clean_replay_eval.py"))
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "clean" in r.stdout


def test_search_does_not_import_evaluator():
    # After importing the search module, the behavioral evaluator must be absent.
    code = (
        "import sys; sys.path.insert(0, r'%s'); sys.path.insert(0, r'%s');"
        "import importlib.util as u;"
        "spec=u.spec_from_file_location('sc', r'%s');"
        "m=u.module_from_spec(spec); spec.loader.exec_module(m);"
        "assert 'subliminal_icl.evaluation' not in sys.modules; print('ok')"
        % (str(REPO / "src"), str(REPO / "scripts"), str(REPO / "scripts/search_contexts.py"))
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "ok" in r.stdout

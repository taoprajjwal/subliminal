"""Notebook validity + light execution (EXPERIMENT_PLAN.md §2.3, §14.3).

Every notebook must be valid JSON and every code cell must compile. A subset of
pure-numpy notebooks is executed in-process under FAST_DEV_RUN as a fast proxy;
full top-to-bottom execution of all notebooks is `make notebooks-smoke`.
"""
import json
import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
NB_DIR = REPO / "notebooks"
pytestmark = pytest.mark.notebook

NOTEBOOKS = sorted(NB_DIR.glob("*.ipynb"))
# pure-numpy notebooks safe + fast to execute in-process (no torch, no subprocess)
EXECUTABLE = [
    "01_dataset_audit_and_splits.ipynb",
    "04_extract_trait_representations.ipynb",
    "05_causal_validate_trait_subspace.ipynb",
    "07_score_existing_number_rows.ipynb",
    "10_activation_compiled_math_cot.ipynb",
    "12_statistics_robustness_and_report.ipynb",
]


def test_notebooks_exist():
    assert len(NOTEBOOKS) >= 14, [p.name for p in NOTEBOOKS]


@pytest.mark.parametrize("nb_path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_json_valid_and_cells_compile(nb_path):
    nb = json.loads(nb_path.read_text())
    assert nb.get("nbformat") == 4
    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") == "code":
            src = "".join(cell.get("source", []))
            compile(src, f"{nb_path.name}:cell{i}", "exec")


@pytest.mark.parametrize("name", EXECUTABLE)
def test_pure_notebook_executes(name, monkeypatch):
    monkeypatch.setenv("FAST_DEV_RUN", "1")
    nb = json.loads((NB_DIR / name).read_text())
    code = "\n\n".join("".join(c.get("source", [])) for c in nb["cells"]
                       if c.get("cell_type") == "code")
    cwd = os.getcwd()
    os.chdir(NB_DIR)
    try:
        glb = {"__name__": "__nb__"}
        exec(compile(code, name, "exec"), glb)  # noqa: S102
    finally:
        os.chdir(cwd)

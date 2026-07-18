#!/usr/bin/env python
"""Execute every notebook top-to-bottom in FAST_DEV_RUN mode (§2.3, §14.3).

Prefers ``nbclient``/``nbformat`` (present via jupyter). Falls back to extracting
and exec-ing each notebook's code cells in-process when no kernel is available,
which still catches import/logic errors in the FAST_DEV_RUN paths. Exits nonzero
on the first failure.
"""
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
os.environ.setdefault("FAST_DEV_RUN", "1")
os.environ.setdefault("RUN_EXPENSIVE", "0")

NB_DIR = REPO / "notebooks"


def run_with_nbclient(path: Path) -> None:
    import nbformat
    from nbclient import NotebookClient
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(nb, timeout=300, kernel_name="python3",
                            resources={"metadata": {"path": str(NB_DIR)}})
    client.execute()


def run_by_exec(path: Path) -> None:
    import json
    nb = json.loads(path.read_text())
    glb = {"__name__": "__nb__", "__file__": str(path)}
    src = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            src.append("".join(cell.get("source", [])))
    code = "\n\n".join(src)
    cwd = os.getcwd()
    os.chdir(NB_DIR)
    try:
        exec(compile(code, str(path), "exec"), glb)  # noqa: S102
    finally:
        os.chdir(cwd)


def main() -> int:
    notebooks = sorted(NB_DIR.glob("*.ipynb"))
    if not notebooks:
        print("no notebooks found")
        return 0
    # Default to the fast, reliable in-process exec of code cells. Opt into the
    # (slower, kernel-per-notebook) nbclient path with NB_SMOKE_USE_NBCLIENT=1.
    use_nbclient = os.environ.get("NB_SMOKE_USE_NBCLIENT", "0") == "1"
    if use_nbclient:
        try:
            import nbformat  # noqa
            import nbclient  # noqa
        except Exception:
            use_nbclient = False
            print("nbclient unavailable; using in-process exec of code cells")
    else:
        print("using in-process exec of code cells (set NB_SMOKE_USE_NBCLIENT=1 for kernel exec)")

    failures = []
    for nb in notebooks:
        print(f"--- executing {nb.name} ---")
        try:
            (run_with_nbclient if use_nbclient else run_by_exec)(nb)
            print(f"    OK {nb.name}")
        except Exception:
            print(f"    FAIL {nb.name}")
            traceback.print_exc()
            failures.append(nb.name)
    if failures:
        print(f"\nFAILED notebooks: {failures}")
        return 1
    print("\nall notebooks executed OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Plotting helpers (matplotlib, guarded import). Used by notebooks §14.

All functions accept an optional ``ax`` and return it, so notebooks can compose
figures. Importing this module does not require matplotlib; each function imports
it lazily and raises a clear error if absent.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def _plt():
    try:
        import matplotlib.pyplot as plt
        return plt
    except Exception as e:  # pragma: no cover
        raise RuntimeError("matplotlib is required for plotting; pip install matplotlib") from e


def confusion_heatmap(M: np.ndarray, labels: Sequence[str], ax=None, title: str = ""):
    plt = _plt()
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(M, aspect="auto")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=90)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlabel("measured animal"); ax.set_ylabel("context trait")
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    return ax


def trajectory(xs: Sequence[float], ys: Sequence[float], ax=None, label: str = "",
               title: str = "", xlabel: str = "K", ylabel: str = "target state"):
    plt = _plt()
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, ys, marker="o", label=label)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    if label:
        ax.legend()
    return ax


def selected_vs_null(selected: Sequence[float], null: Sequence[float], ax=None,
                     title: str = "selected vs matched null"):
    plt = _plt()
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    ax.hist(null, bins=30, alpha=0.6, label="matched null")
    for s in selected:
        ax.axvline(s, color="red")
    ax.set_title(title); ax.legend()
    return ax

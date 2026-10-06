"""Global threshold tuning on validation (FR-6)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from matchlens.metrics import f1_curve
from matchlens.retrievers.base import Candidates

DEFAULT_GRID = np.round(np.linspace(0.0, 1.0, 101), 2)


@dataclass
class ThresholdResult:
    threshold: float
    f1: float
    grid: np.ndarray
    f1_by_threshold: np.ndarray


def tune_threshold(cands: Candidates, labels: np.ndarray, grid: np.ndarray = DEFAULT_GRID,
                   min_matches: int = 0) -> ThresholdResult:
    """Pick the threshold maximising mean per-listing F1. Ties go to the higher (stricter) threshold."""
    mean_f1 = f1_curve(cands, labels, grid, min_matches).mean(axis=1)
    best = len(grid) - 1 - int(np.argmax(mean_f1[::-1]))
    return ThresholdResult(float(grid[best]), float(mean_f1[best]), grid, mean_f1)

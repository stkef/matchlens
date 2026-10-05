"""The interface every retriever implements (FR-2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd


@dataclass
class Candidates:
    """Top-k results for a batch of queries.

    indices: (n_queries, k) row positions into the fitted corpus, -1 where fewer than k were found.
    scores:  (n_queries, k) scores normalised per query so one threshold works for every query
             (1.0 = as similar as the query is to itself), -inf where indices is -1.
    """

    indices: np.ndarray
    scores: np.ndarray

    def __post_init__(self) -> None:
        if self.indices.shape != self.scores.shape:
            raise ValueError("indices and scores must have the same shape")


class Retriever(Protocol):
    name: str

    def fit(self, corpus: pd.DataFrame) -> None:
        """Index the corpus (a listings frame with a 0..n-1 index)."""

    def search(self, queries: pd.DataFrame, k: int) -> Candidates:
        """Return the top-k corpus rows for each query row, best first."""


def drop_self(cands: Candidates, k: int) -> Candidates:
    """For in-pool evaluation (query i is corpus row i): remove each query from its own results."""
    n, width = cands.indices.shape
    out_idx = np.full((n, k), -1, dtype=np.int64)
    out_scores = np.full((n, k), -np.inf)
    for i in range(n):
        keep = (cands.indices[i] != i) & (cands.indices[i] >= 0)
        idx, sc = cands.indices[i][keep][:k], cands.scores[i][keep][:k]
        out_idx[i, :len(idx)] = idx
        out_scores[i, :len(sc)] = sc
    return Candidates(out_idx, out_scores)


def top_k(scores: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Row-wise top-k of a dense (n_queries, n_corpus) score matrix, sorted descending."""
    k = min(k, scores.shape[1])
    part = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    part_scores = np.take_along_axis(scores, part, axis=1)
    order = np.argsort(-part_scores, axis=1, kind="stable")
    return np.take_along_axis(part, order, axis=1), np.take_along_axis(part_scores, order, axis=1)

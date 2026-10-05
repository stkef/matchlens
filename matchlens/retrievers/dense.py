"""Dense retrieval over precomputed embeddings (rung 3).

Vectors are computed once on a GPU by kaggle/embed_titles (see docs/kaggle.md) and stored as
`emb_<name>.npz` (posting_id + L2-normalised float16 vectors). This retriever only looks them up and
compares them: cosine similarity, which for normalised vectors is a plain dot product. A title scores
1.0 against itself, so no further per-query normalising is needed.

Latency measured here covers the search only. Encoding a new query title is measured on CPU in the
Kaggle notebook and recorded next to the vectors in `emb_<name>.json`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from matchlens.retrievers.base import Candidates, top_k


class PrecomputedEmbeddingRetriever:
    name = "dense"

    def __init__(self, path: str, batch_size: int = 512):
        self.path, self.batch_size = Path(path), batch_size
        data = np.load(self.path, allow_pickle=False)
        self._row = {p: i for i, p in enumerate(data["posting_id"].tolist())}
        self._vectors = data["vectors"].astype(np.float32)

    def _lookup(self, frame: pd.DataFrame) -> np.ndarray:
        missing = [p for p in frame["posting_id"] if p not in self._row]
        if missing:
            raise KeyError(f"{len(missing)} listings have no vector in {self.path}, e.g. {missing[0]}")
        return self._vectors[[self._row[p] for p in frame["posting_id"]]]

    def fit(self, corpus: pd.DataFrame) -> None:
        self.corpus_vecs = self._lookup(corpus)

    def search(self, queries: pd.DataFrame, k: int) -> Candidates:
        q = self._lookup(queries)
        idx_parts, score_parts = [], []
        for start in range(0, len(q), self.batch_size):
            idx, sc = top_k(q[start:start + self.batch_size] @ self.corpus_vecs.T, k)
            idx_parts.append(idx)
            score_parts.append(np.minimum(sc, 1.0).astype(np.float64))  # float16 rounding can exceed 1
        return Candidates(np.vstack(idx_parts), np.vstack(score_parts))

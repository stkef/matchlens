"""Dense retrieval over precomputed embeddings (rungs 3–5).

Vectors are computed once on a GPU (kaggle/, see docs/kaggle.md) or locally (FastText) and stored as
`emb_<name>.npz` (posting_id + L2-normalised float16 vectors). This retriever looks them up and hands
them to a vector store (matchlens/stores.py): FAISS for text, Qdrant for images, or plain numpy as the
exact reference. Similarity is cosine, so a listing scores 1.0 against itself and no further
per-query normalising is needed.

Optional neighbour voting (rung 5d, `expand_k` > 0): every vector is replaced by a weighted average of
itself and its nearest neighbours, weights = similarity ** `expand_alpha` ("database-side augmentation"
plus "query expansion"). A group of listings that agree with each other pulls together, so a match that
is only close to the *other* members of its group is still found.

Latency measured here covers the search only. Encoding a new query is measured separately and
recorded next to the vectors in `emb_<name>.json`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from matchlens.retrievers.base import Candidates
from matchlens.stores import FaissStore, build_store


def _normalise(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)


class PrecomputedEmbeddingRetriever:
    name = "dense"

    def __init__(self, path: str, store: dict | None = None, expand_k: int = 0, expand_alpha: float = 3.0):
        self.path = Path(path)
        self.store = build_store(store)
        self.expand_k, self.expand_alpha = expand_k, expand_alpha
        data = np.load(self.path, allow_pickle=False)
        self._row = {p: i for i, p in enumerate(data["posting_id"].tolist())}
        self._vectors = data["vectors"].astype(np.float32)

    def _lookup(self, frame: pd.DataFrame) -> np.ndarray:
        missing = [p for p in frame["posting_id"] if p not in self._row]
        if missing:
            raise KeyError(f"{len(missing)} listings have no vector in {self.path}, e.g. {missing[0]}")
        return self._vectors[[self._row[p] for p in frame["posting_id"]]]

    def _expand(self, vecs: np.ndarray) -> np.ndarray:
        """Weighted average of each vector's expand_k nearest corpus vectors (itself included if present)."""
        idx, sim = self._raw.search(vecs, self.expand_k)
        w = np.clip(sim, 0, None) ** self.expand_alpha
        w[idx < 0] = 0
        return _normalise((w[:, :, None] * self._corpus_raw[np.maximum(idx, 0)]).sum(axis=1))

    def fit(self, corpus: pd.DataFrame) -> None:
        vecs = self._lookup(corpus)
        if self.expand_k:
            self._corpus_raw = vecs
            self._raw = FaissStore("flat")  # in memory: the un-expanded neighbours used for voting
            self._raw.build(vecs)
            vecs = self._expand(vecs)
        self.store.build(vecs)

    def search(self, queries: pd.DataFrame, k: int) -> Candidates:
        q = self._lookup(queries)
        if self.expand_k:
            q = self._expand(q)
        idx, scores = self.store.search(q, k)
        return Candidates(idx, np.minimum(scores, 1.0))  # float16 rounding can nudge a self-match past 1

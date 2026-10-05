"""Dense retrieval over precomputed embeddings (rungs 3–4).

Vectors are computed once on a GPU (kaggle/, see docs/kaggle.md) or locally (FastText) and stored as
`emb_<name>.npz` (posting_id + L2-normalised float16 vectors). This retriever looks them up and hands
them to a vector store (matchlens/stores.py): FAISS for text, Qdrant for images, or plain numpy as the
exact reference. Similarity is cosine, so a listing scores 1.0 against itself and no further
per-query normalising is needed.

Latency measured here covers the search only. Encoding a new query is measured separately and
recorded next to the vectors in `emb_<name>.json`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from matchlens.retrievers.base import Candidates
from matchlens.stores import build_store


class PrecomputedEmbeddingRetriever:
    name = "dense"

    def __init__(self, path: str, store: dict | None = None):
        self.path = Path(path)
        self.store = build_store(store)
        data = np.load(self.path, allow_pickle=False)
        self._row = {p: i for i, p in enumerate(data["posting_id"].tolist())}
        self._vectors = data["vectors"].astype(np.float32)

    def _lookup(self, frame: pd.DataFrame) -> np.ndarray:
        missing = [p for p in frame["posting_id"] if p not in self._row]
        if missing:
            raise KeyError(f"{len(missing)} listings have no vector in {self.path}, e.g. {missing[0]}")
        return self._vectors[[self._row[p] for p in frame["posting_id"]]]

    def fit(self, corpus: pd.DataFrame) -> None:
        self.store.build(self._lookup(corpus))

    def search(self, queries: pd.DataFrame, k: int) -> Candidates:
        idx, scores = self.store.search(self._lookup(queries), k)
        return Candidates(idx, np.minimum(scores, 1.0))  # float16 rounding can nudge a self-match past 1

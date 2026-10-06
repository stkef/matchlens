"""Fusion of several retrievers into one ranked list (rung 5, FR-4).

Each member retriever (BM25, text embeddings, image embeddings, ...) returns its own top-k. The union
of their candidates is rescored:

- "rrf"     Reciprocal Rank Fusion: score = sum over members of 1 / (rrf_k + rank). Uses only ranks, so
            members with incomparable score scales combine safely. Divided by its maximum so 1.0 means
            "ranked first by every member", keeping one threshold meaningful.
- "learned" Logistic regression over each member's score and 1/rank (absent = 0), trained on the
            *train* split by `python -m matchlens.fusion_train` and loaded from `model_path`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from matchlens.retrievers.base import Candidates, Retriever


class FusionRetriever:
    name = "fusion"

    def __init__(self, members: list[Retriever], method: str = "rrf", rrf_k: int = 60,
                 model_path: str | None = None):
        if method not in ("rrf", "learned"):
            raise ValueError("method must be 'rrf' or 'learned'")
        if method == "learned" and not model_path:
            raise ValueError("method 'learned' needs model_path (see matchlens.fusion_train)")
        self.members, self.method, self.rrf_k = members, method, rrf_k
        if method == "learned":
            model = json.loads(Path(model_path).read_text())
            self.coef, self.intercept = np.array(model["coef"]), float(model["intercept"])
            if len(self.coef) != 2 * len(members):
                raise ValueError(f"{model_path} was trained for {len(self.coef) // 2} members, not {len(members)}")

    def fit(self, corpus: pd.DataFrame) -> None:
        for m in self.members:
            m.fit(corpus)

    def features(self, queries: pd.DataFrame, k: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Per query: candidate ids and a (n_candidates, 2 * n_members) matrix of [score, 1/rank]."""
        results = [m.search(queries, k) for m in self.members]
        ids_out, feats_out = [], []
        for i in range(len(queries)):
            rows: dict[int, np.ndarray] = {}
            for r, res in enumerate(results):
                for rank, (j, s) in enumerate(zip(res.indices[i], res.scores[i]), 1):
                    if j < 0:
                        break
                    f = rows.setdefault(int(j), np.zeros(2 * len(results)))
                    f[2 * r], f[2 * r + 1] = s, 1.0 / rank
            ids_out.append(np.fromiter(rows.keys(), dtype=np.int64, count=len(rows)))
            feats_out.append(np.stack(list(rows.values())) if rows else np.zeros((0, 2 * len(results))))
        return ids_out, feats_out

    def _score(self, feats: np.ndarray) -> np.ndarray:
        if self.method == "rrf":
            inv_rank = feats[:, 1::2]
            # 1/rank -> rank; absent members (0) contribute nothing.
            contrib = np.where(inv_rank > 0, 1.0 / (self.rrf_k + 1.0 / np.maximum(inv_rank, 1e-12)), 0.0)
            return contrib.sum(axis=1) / (len(self.members) / (self.rrf_k + 1))
        return 1.0 / (1.0 + np.exp(-(feats @ self.coef + self.intercept)))

    def search(self, queries: pd.DataFrame, k: int) -> Candidates:
        ids, feats = self.features(queries, k)
        out_idx = np.full((len(queries), k), -1, dtype=np.int64)
        out_scores = np.full((len(queries), k), -np.inf)
        for i, (cand, f) in enumerate(zip(ids, feats)):
            if not len(cand):
                continue
            s = self._score(f)
            order = np.argsort(-s, kind="stable")[:k]
            out_idx[i, :len(order)] = cand[order]
            out_scores[i, :len(order)] = s[order]
        return Candidates(out_idx, out_scores)

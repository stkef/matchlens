"""Cross-encoder reranking of a pipeline's top candidates (rung 6, FR-5).

The base pipeline (rung 5) proposes candidates; its top `top_n` are rescored by a judge that combines
the base score with a cross-encoder score for the (query title, candidate title) pair. Cross-encoder
scores are computed on a Kaggle GPU (kaggle/rerank) and looked up here by posting ids; the judge is a
logistic regression trained on train-split pairs by `python -m matchlens.rerank_judge`.

Only the top `top_n` candidates come out, so recall@k for k > top_n is capped at the base pipeline's
recall within its top `top_n`. Latency measured here excludes running the cross-encoder itself; its
CPU cost per query is recorded in the notebook's rerank_<model>.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from matchlens.retrievers.base import Candidates, Retriever


def judge_features(base_score: np.ndarray, rerank_logit: np.ndarray) -> np.ndarray:
    """[base score, cross-encoder logit] — shared by training and inference so they cannot drift."""
    return np.column_stack([base_score, rerank_logit])


class RerankRetriever:
    name = "rerank"

    def __init__(self, base: Retriever, scores: list[str] | str, judge_path: str, top_n: int = 20):
        self.base, self.top_n = base, top_n
        frames = [pd.read_csv(p, usecols=["query", "candidate", "score"]) for p in ([scores] if isinstance(scores, str) else scores)]
        pairs = pd.concat(frames, ignore_index=True)
        self._scores = dict(zip(zip(pairs["query"], pairs["candidate"]), pairs["score"]))
        judge = json.loads(Path(judge_path).read_text())
        self.coef, self.intercept = np.array(judge["coef"]), float(judge["intercept"])

    def fit(self, corpus: pd.DataFrame) -> None:
        self.base.fit(corpus)
        self.posting_ids = corpus["posting_id"].to_numpy()

    def search(self, queries: pd.DataFrame, k: int) -> Candidates:
        # One extra so the query itself (always in the base results for in-pool queries) doesn't use a slot.
        n = min(self.top_n + 1, k)
        base = self.base.search(queries, n)
        out_idx = np.full((len(queries), k), -1, dtype=np.int64)
        out_scores = np.full((len(queries), k), -np.inf)
        n_pairs = n_missing = 0
        for i, q in enumerate(queries["posting_id"]):
            keep = base.indices[i] >= 0
            idx, bs = base.indices[i][keep], base.scores[i][keep]
            cand = self.posting_ids[idx]
            # The listing itself is always first (removed later by drop_self). A pair without a precomputed
            # score — e.g. a near-tie at position top_n that ordered differently when one query is searched
            # alone — is skipped; a live system would score it on the spot.
            logits = np.array([np.inf if c == q else self._scores.get((q, c), np.nan) for c in cand])
            missing = np.isnan(logits)
            n_pairs, n_missing = n_pairs + len(cand), n_missing + int(missing.sum())
            idx, bs, logits = idx[~missing], bs[~missing], logits[~missing]
            finite = np.isfinite(logits)
            p = np.ones(len(idx))
            p[finite] = 1 / (1 + np.exp(-(judge_features(bs[finite], logits[finite]) @ self.coef + self.intercept)))
            order = np.argsort(-p, kind="stable")
            out_idx[i, :len(order)] = idx[order]
            out_scores[i, :len(order)] = p[order]
        if n_pairs and n_missing > 0.05 * n_pairs:
            raise KeyError(f"{n_missing} of {n_pairs} candidate pairs have no cross-encoder score; "
                           "were the pairs exported for this split and pipeline?")
        return Candidates(out_idx, out_scores)

"""Near-duplicate photos via perceptual hashes (rung 2).

Wraps another retriever: any listing whose 64-bit `image_phash` is within `max_dist` bits of the
query's is added to the candidates with score 1.0 ("certainly the same"), on top of the base
retriever's own candidates.

Measured on val before building (pool 27,431): identical hashes are the same product 96% of the time
and add ~1,000 correct pairs BM25 misses. Title near-duplicates (Jaccard >= 0.8) added zero pairs
beyond BM25, so there is no title MinHash stage; see docs/ladder.md, rung 2.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from matchlens.retrievers.base import Candidates, Retriever


def phash_to_uint64(hashes: pd.Series) -> np.ndarray:
    return np.array([int(h, 16) for h in hashes], dtype=np.uint64)


class PhashBoostRetriever:
    name = "phash_boost"

    def __init__(self, base: Retriever, max_dist: int = 0, batch_size: int = 512):
        self.base, self.max_dist, self.batch_size = base, max_dist, batch_size

    def fit(self, corpus: pd.DataFrame) -> None:
        self.base.fit(corpus)
        self.hashes = phash_to_uint64(corpus["image_phash"])

    def search(self, queries: pd.DataFrame, k: int) -> Candidates:
        base = self.base.search(queries, k)
        q_hashes = phash_to_uint64(queries["image_phash"])
        out_idx = np.full((len(queries), k), -1, dtype=np.int64)
        out_scores = np.full((len(queries), k), -np.inf)
        for start in range(0, len(queries), self.batch_size):
            dist = np.bitwise_count(q_hashes[start:start + self.batch_size, None] ^ self.hashes[None, :])
            for row, hits in enumerate(dist <= self.max_dist):
                i = start + row
                merged = dict(zip(base.indices[i].tolist(), base.scores[i].tolist()))
                merged.pop(-1, None)
                for j in np.flatnonzero(hits).tolist():
                    merged[j] = 1.0
                best = sorted(merged.items(), key=lambda kv: -kv[1])[:k]
                out_idx[i, :len(best)] = [j for j, _ in best]
                out_scores[i, :len(best)] = [s for _, s in best]
        return Candidates(out_idx, out_scores)

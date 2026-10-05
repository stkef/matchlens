"""BM25 over titles, as sparse matrix products (rung 1)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from matchlens.retrievers.base import Candidates, top_k
from matchlens.text import tokenize


class BM25Retriever:
    name = "bm25"

    def __init__(self, k1: float = 1.2, b: float = 0.75, batch_size: int = 512, canonical_units: bool = False):
        self.k1, self.b, self.batch_size = k1, b, batch_size
        self.canonical_units = canonical_units

    def fit(self, corpus: pd.DataFrame) -> None:
        docs = [tokenize(t, self.canonical_units) for t in corpus["title"]]
        self.vocab: dict[str, int] = {}
        for doc in docs:
            for tok in doc:
                self.vocab.setdefault(tok, len(self.vocab))
        counts = self._counts(docs)
        n_docs = counts.shape[0]
        df = np.bincount(counts.indices, minlength=len(self.vocab))
        # Lucene's idf: always positive, so very common tokens add a little instead of subtracting.
        self.idf = np.log1p((n_docs - df + 0.5) / (df + 0.5))
        self.avgdl = max(float(counts.sum(axis=1).mean()), 1e-9)
        self.doc_weights = self._weights(counts).T.tocsr()  # (vocab, n_docs)

    def _counts(self, docs: list[list[str]]) -> sparse.csr_matrix:
        rows, cols = [], []
        for i, doc in enumerate(docs):
            for tok in doc:
                j = self.vocab.get(tok)
                if j is not None:
                    rows.append(i)
                    cols.append(j)
        m = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(docs), len(self.vocab)))
        m.sum_duplicates()
        return m

    def _weights(self, counts: sparse.csr_matrix) -> sparse.csr_matrix:
        """Per (doc, term) BM25 contribution: idf * tf*(k1+1) / (tf + k1*(1 - b + b*dl/avgdl))."""
        counts = counts.astype(np.float64)
        dl = np.asarray(counts.sum(axis=1)).ravel()
        row_of = np.repeat(np.arange(counts.shape[0]), np.diff(counts.indptr))
        tf = counts.data
        norm = self.k1 * (1 - self.b + self.b * dl[row_of] / self.avgdl)
        data = self.idf[counts.indices] * tf * (self.k1 + 1) / (tf + norm)
        return sparse.csr_matrix((data, counts.indices, counts.indptr), shape=counts.shape)

    def search(self, queries: pd.DataFrame, k: int) -> Candidates:
        q_counts = self._counts([tokenize(t, self.canonical_units) for t in queries["title"]])
        q_terms = (q_counts > 0).astype(np.float64)
        # Score of the query against itself treated as a document: the per-query normaliser.
        self_score = np.asarray(q_terms.multiply(self._weights(q_counts)).sum(axis=1)).ravel()
        all_idx, all_scores = [], []
        for start in range(0, q_terms.shape[0], self.batch_size):
            block = slice(start, start + self.batch_size)
            raw = (q_terms[block] @ self.doc_weights).toarray()
            idx, sc = top_k(raw, k)
            denom = self_score[block][:, None]
            norm = np.divide(sc, denom, out=np.zeros_like(sc), where=denom > 0)
            norm = np.minimum(norm, 1.0)
            # Zero-overlap documents are not real candidates.
            idx = np.where(sc > 0, idx, -1)
            norm = np.where(sc > 0, norm, -np.inf)
            all_idx.append(idx)
            all_scores.append(norm)
        return Candidates(np.vstack(all_idx), np.vstack(all_scores))

"""Vector stores: where embedding vectors live and how nearest neighbours are found (FR-12).

Text and image vectors are kept in separate stores, because they come from different models with
different sizes and are tuned and replaced independently:

    text   -> FAISS  (Meta's search library; in-process, exact "flat" or approximate "hnsw" index)
    images -> Qdrant (a vector database, here in local mode: a folder on disk, no server)

`numpy` is the exact brute-force reference every store is checked against.

All stores take L2-normalised vectors and score by cosine similarity (= inner product), best first.
Stores with a `path` are saved to disk and reused when built again from the same vectors; the
fingerprint check rebuilds them automatically if the vectors change (e.g. a different pool).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Protocol

import numpy as np

from matchlens.retrievers.base import top_k


class VectorStore(Protocol):
    def build(self, vectors: np.ndarray) -> None:
        """Index the corpus vectors; row i of `vectors` is result id i."""

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """(indices, scores), each (n_queries, k), best first; -1 / -inf where fewer than k exist."""


def fingerprint(vectors: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(vectors, dtype=np.float32).tobytes()).hexdigest()


class NumpyStore:
    """Exact brute force: compare every query with every vector."""

    def __init__(self, batch_size: int = 512):
        self.batch_size = batch_size

    def build(self, vectors: np.ndarray) -> None:
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        parts = [top_k(queries[s:s + self.batch_size] @ self.vectors.T, k)
                 for s in range(0, len(queries), self.batch_size)]
        return np.vstack([p[0] for p in parts]), np.vstack([p[1] for p in parts]).astype(np.float64)


class FaissStore:
    """FAISS index. "flat" is exact; "hnsw" is a graph index that checks only part of the corpus."""

    def __init__(self, index: str = "flat", path: str | None = None, hnsw_m: int = 32, ef_search: int = 128):
        if index not in ("flat", "hnsw"):
            raise ValueError("index must be 'flat' or 'hnsw'")
        self.kind, self.path, self.hnsw_m, self.ef_search = index, Path(path) if path else None, hnsw_m, ef_search

    def build(self, vectors: np.ndarray) -> None:
        import faiss

        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        fp = f"{self.kind}|{self.hnsw_m}|{fingerprint(vectors)}"
        meta = self.path.with_suffix(".json") if self.path else None
        if self.path and self.path.exists() and meta.exists() and json.loads(meta.read_text())["fingerprint"] == fp:
            self.index = faiss.read_index(str(self.path))
        else:
            dim = vectors.shape[1]
            if self.kind == "flat":
                self.index = faiss.IndexFlatIP(dim)
            else:
                self.index = faiss.IndexHNSWFlat(dim, self.hnsw_m, faiss.METRIC_INNER_PRODUCT)
            self.index.add(vectors)
            if self.path:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                faiss.write_index(self.index, str(self.path))
                meta.write_text(json.dumps({"fingerprint": fp, "n": len(vectors), "dim": vectors.shape[1]}))
        if self.kind == "hnsw":
            self.index.hnsw.efSearch = self.ef_search

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        scores, idx = self.index.search(np.ascontiguousarray(queries, dtype=np.float32), k)
        scores = np.where(idx >= 0, scores, -np.inf).astype(np.float64)
        return idx.astype(np.int64), scores


class QdrantStore:
    """Qdrant collection in local (embedded) mode. Point id = row number in the corpus.

    Local mode searches exactly and is meant for development; production would point the same code
    at a Qdrant server (QdrantClient(url=...)), which builds an HNSW index.
    """

    def __init__(self, path: str, collection: str, batch_size: int = 512):
        self.path, self.collection, self.batch_size = Path(path), collection, batch_size

    def build(self, vectors: np.ndarray) -> None:
        from qdrant_client import QdrantClient, models

        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        fp = fingerprint(vectors)
        meta = self.path / f"{self.collection}.fingerprint.json"
        self.path.mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=str(self.path))
        fresh = meta.exists() and json.loads(meta.read_text())["fingerprint"] == fp \
            and self.client.collection_exists(self.collection)
        if not fresh:
            if self.client.collection_exists(self.collection):
                self.client.delete_collection(self.collection)
            self.client.create_collection(self.collection, vectors_config=models.VectorParams(
                size=vectors.shape[1], distance=models.Distance.COSINE))
            for s in range(0, len(vectors), self.batch_size):
                block = vectors[s:s + self.batch_size]
                self.client.upload_collection(self.collection, vectors=block, ids=range(s, s + len(block)))
            meta.write_text(json.dumps({"fingerprint": fp, "n": len(vectors), "dim": vectors.shape[1]}))

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        from qdrant_client import models

        idx = np.full((len(queries), k), -1, dtype=np.int64)
        scores = np.full((len(queries), k), -np.inf)
        for s in range(0, len(queries), self.batch_size):
            requests = [models.QueryRequest(query=q.tolist(), limit=k) for q in queries[s:s + self.batch_size]]
            for row, resp in enumerate(self.client.query_batch_points(self.collection, requests=requests), s):
                idx[row, :len(resp.points)] = [p.id for p in resp.points]
                scores[row, :len(resp.points)] = [p.score for p in resp.points]
        return idx, scores

    def close(self) -> None:
        self.client.close()


STORES = {"numpy": NumpyStore, "faiss": FaissStore, "qdrant": QdrantStore}


def build_store(cfg: dict | None) -> VectorStore:
    """[retriever.store] table: type = "numpy" | "faiss" | "qdrant", plus that store's options."""
    cfg = dict(cfg or {"type": "numpy"})
    kind = cfg.pop("type")
    if kind not in STORES:
        raise ValueError(f"unknown store {kind!r}; known: {sorted(STORES)}")
    return STORES[kind](**cfg)

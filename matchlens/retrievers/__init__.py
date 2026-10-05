from matchlens.retrievers.base import Candidates, Retriever, drop_self
from matchlens.retrievers.bm25 import BM25Retriever
from matchlens.retrievers.dense import PrecomputedEmbeddingRetriever
from matchlens.retrievers.phash import PhashBoostRetriever

REGISTRY = {
    "bm25": BM25Retriever,
    "phash_boost": PhashBoostRetriever,
    "dense": PrecomputedEmbeddingRetriever,
}


def build_retriever(cfg: dict) -> Retriever:
    """Build a retriever from the [retriever] table of a config: type = "...", plus its kwargs.

    A nested `base` table is built recursively, so wrappers like phash_boost can wrap any retriever.
    """
    cfg = dict(cfg)
    kind = cfg.pop("type")
    if kind not in REGISTRY:
        raise ValueError(f"unknown retriever {kind!r}; known: {sorted(REGISTRY)}")
    if "base" in cfg:
        cfg["base"] = build_retriever(cfg["base"])
    return REGISTRY[kind](**cfg)


__all__ = ["Candidates", "Retriever", "drop_self", "build_retriever", "BM25Retriever", "PhashBoostRetriever",
           "PrecomputedEmbeddingRetriever"]

from matchlens.retrievers.base import Candidates, Retriever, drop_self
from matchlens.retrievers.bm25 import BM25Retriever

REGISTRY = {
    "bm25": BM25Retriever,
}


def build_retriever(cfg: dict) -> Retriever:
    """Build a retriever from the [retriever] table of a config: type = "...", plus its kwargs."""
    cfg = dict(cfg)
    kind = cfg.pop("type")
    if kind not in REGISTRY:
        raise ValueError(f"unknown retriever {kind!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[kind](**cfg)


__all__ = ["Candidates", "Retriever", "drop_self", "build_retriever", "BM25Retriever"]

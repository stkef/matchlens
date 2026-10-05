"""Train FastText on the training titles and embed every listing (rung 3, lightweight comparison).

FastText learns word vectors from character n-grams, so typos and unseen words ("extention",
"waterproff") still get sensible vectors. It trains on the **train split only**, on CPU, in about a
minute. A title vector is a pooled average of its word vectors, written in the same format as the
Kaggle embeddings so the `dense` retriever can use it:

    python -m matchlens.fasttext_embed

writes C:/data/shopee/embeddings/emb_fasttext_{mean,idf}.npz (+ .json with timings).
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
from gensim.models import FastText

from matchlens.data import load_listings, load_split_frame
from matchlens.text import tokenize


def pool(model: FastText, tokens: list[str], idf: dict[str, float] | None, idf_unseen: float = 0.0) -> np.ndarray:
    """Average of L2-normalised word vectors, optionally weighted by idf so rare words count more.

    Words never seen in training get `idf_unseen` (the highest idf: unseen words are the rarest).
    """
    if not tokens:
        return np.zeros(model.vector_size, dtype=np.float32)
    vecs = np.stack([model.wv[t] for t in tokens])
    vecs /= np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12)
    w = np.array([idf.get(t, idf_unseen) for t in tokens]) if idf else np.ones(len(tokens))
    v = (w[:, None] * vecs).sum(axis=0)
    return (v / max(np.linalg.norm(v), 1e-12)).astype(np.float32)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default="C:/data/shopee/train.csv")
    p.add_argument("--split", default="data/splits/split_v1.csv")
    p.add_argument("--out-dir", default="C:/data/shopee/embeddings")
    p.add_argument("--dim", type=int, default=100)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    train_tokens = [tokenize(t, canonical_units=True) for t in load_split_frame(args.csv, args.split, "train")["title"]]
    t0 = time.perf_counter()
    # Skip-gram with 3-5 character n-grams; workers=1 so the same seed gives the same vectors.
    model = FastText(sentences=train_tokens, vector_size=args.dim, window=5, min_count=1, sg=1,
                     min_n=3, max_n=5, epochs=args.epochs, seed=args.seed, workers=1)
    train_s = time.perf_counter() - t0
    print(f"trained on {len(train_tokens)} train titles in {train_s:.0f}s, vocab {len(model.wv)}")

    doc_freq = Counter(t for toks in train_tokens for t in set(toks))
    idf = {t: float(np.log1p(len(train_tokens) / df)) for t, df in doc_freq.items()}
    idf_unseen = max(idf.values())

    listings = load_listings(args.csv)
    all_tokens = [tokenize(t, canonical_units=True) for t in listings["title"]]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, weights in [("fasttext_mean", None), ("fasttext_idf", idf)]:
        vecs = np.stack([pool(model, toks, weights, idf_unseen) for toks in all_tokens])
        times = []
        for toks in all_tokens[:200]:
            t = time.perf_counter()
            pool(model, toks, weights, idf_unseen)
            times.append(time.perf_counter() - t)
        np.savez(out / f"emb_{name}.npz", posting_id=listings["posting_id"].to_numpy().astype(str),
                 vectors=vecs.astype(np.float16))
        meta = {"name": name, "model": f"FastText skip-gram {args.dim}-d, trained on train split",
                "dim": args.dim, "n": len(listings), "train_s": round(train_s, 1),
                "cpu_query_ms_p50": round(float(np.percentile(times, 50)) * 1000, 2),
                "cpu_query_ms_p95": round(float(np.percentile(times, 95)) * 1000, 2)}
        (out / f"emb_{name}.json").write_text(json.dumps(meta, indent=2))
        print(meta)


if __name__ == "__main__":
    main()

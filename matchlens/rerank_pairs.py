"""Export candidate pairs for the reranker (rung 6, step 1).

    python -m matchlens.rerank_pairs configs/rung05d_expand.toml

For each query listing, takes the top-N candidates of a finished pipeline (rung 5) and writes them out
so a cross-encoder can score each (query title, candidate title) pair on a Kaggle GPU.

- val:   every val listing, searched against the usual val + train pool (what rung 6 is scored on).
- train: train listings (a random sample, or all with --train-queries 0), searched against the train
         listings only. Each query gets a random `fold` (A or B): rung 7b fine-tunes a cross-encoder on
         fold A and fits the judge on fold B, so the judge never sees scores on pairs the reranker
         trained on. Val is never used for either.

Writes to --out-dir:
    pairs_<split>.csv           query, candidate, base_score, same (label; kept locally), fold (train)
    upload/pairs_<split>.csv    query, candidate (+ fold for train) — the file uploaded to Kaggle
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from matchlens.data import load_split_frame
from matchlens.evaluate import load_config
from matchlens.retrievers import build_retriever, drop_self


def export(retriever, queries: pd.DataFrame, pool: pd.DataFrame, n: int) -> pd.DataFrame:
    """Top-n candidates of each query (queries are the first rows of pool), self removed."""
    retriever.fit(pool)
    c = drop_self(retriever.search(queries, n + 1), n)
    rows = [(i, j, s) for i in range(len(queries)) for j, s in zip(c.indices[i], c.scores[i]) if j >= 0]
    qi, ci, sc = map(np.array, zip(*rows))
    labels = pool["label_group"].to_numpy()
    return pd.DataFrame({"query": pool["posting_id"].to_numpy()[qi], "candidate": pool["posting_id"].to_numpy()[ci],
                         "base_score": sc, "same": labels[qi] == labels[ci]})


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--train-queries", type=int, default=6000, help="0 = all train listings")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="C:/data/shopee/rerank")
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    csv, split = cfg["data"]["csv"], cfg["data"]["split"]
    out = Path(args.out_dir)
    (out / "upload").mkdir(parents=True, exist_ok=True)

    val, train = load_split_frame(csv, split, "val"), load_split_frame(csv, split, "train")
    jobs = {
        "val": (val, pd.concat([val, train], ignore_index=True)),
        "train": (None, None),
    }
    n_train = len(train) if args.train_queries == 0 else min(args.train_queries, len(train))
    sample = train.sample(n=n_train, random_state=args.seed)
    rest = train.drop(sample.index)
    jobs["train"] = (sample.reset_index(drop=True), pd.concat([sample, rest], ignore_index=True))

    for name, (queries, pool) in jobs.items():
        pairs = export(build_retriever(cfg["retriever"]), queries, pool, args.n)
        labels = pool["label_group"].to_numpy()
        true_others = sum(int((labels == labels[i]).sum()) - 1 for i in range(len(queries)))
        print(f"{name}: {len(queries)} queries, {len(pairs)} pairs, "
              f"{pairs['same'].sum()} true ({pairs['same'].sum() / true_others:.1%} of all true pairs within top {args.n})")
        cols = ["query", "candidate"]
        if name == "train":
            # Folds by product, not by listing, so no product has listings in both folds.
            groups = queries["label_group"].unique()
            in_a = set(groups[np.random.default_rng(args.seed).random(len(groups)) < 0.5])
            folds = pd.Series(np.where(queries["label_group"].isin(in_a), "A", "B"), index=queries["posting_id"])
            pairs["fold"] = folds[pairs["query"]].to_numpy()
            cols.append("fold")
        pairs.to_csv(out / f"pairs_{name}.csv", index=False)
        pairs[cols].to_csv(out / "upload" / f"pairs_{name}.csv", index=False)


if __name__ == "__main__":
    main()

"""Mine hard negatives from the pipeline's own mistakes on the train split (rung 7, step 1).

    python -m matchlens.mine_negatives configs/rung05d_expand.toml

Every train listing is searched against the train listings with a finished pipeline; the highest-ranked
candidates from a *different* product are the look-alikes the system confuses ("hard negatives"). They
teach a fine-tuned model what not to match. Only the train split is used, so validation stays clean.

Writes --out-dir/upload/hard_negatives.csv: query, candidate, rank (posting ids only).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from matchlens.data import load_split_frame
from matchlens.evaluate import load_config
from matchlens.retrievers import build_retriever, drop_self


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config")
    p.add_argument("--k", type=int, default=20, help="how deep to look for mistakes")
    p.add_argument("--per-query", type=int, default=5, help="hard negatives kept per listing")
    p.add_argument("--out-dir", default="C:/data/shopee/mining")
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    train = load_split_frame(cfg["data"]["csv"], cfg["data"]["split"], "train")
    labels, ids = train["label_group"].to_numpy(), train["posting_id"].to_numpy()

    r = build_retriever(cfg["retriever"])
    r.fit(train)
    c = drop_self(r.search(train, args.k + 1), args.k)
    rows = []
    for i in range(len(train)):
        wrong = [j for j in c.indices[i] if j >= 0 and labels[j] != labels[i]][:args.per_query]
        rows += [(ids[i], ids[j], rank) for rank, j in enumerate(wrong, 1)]
    out = Path(args.out_dir) / "upload"
    out.mkdir(parents=True, exist_ok=True)
    neg = pd.DataFrame(rows, columns=["query", "candidate", "rank"])
    neg.to_csv(out / "hard_negatives.csv", index=False)
    has = neg["query"].nunique()
    print(f"{len(neg)} hard negatives for {has} of {len(train)} train listings "
          f"({len(neg) / max(has, 1):.1f} each) -> {out / 'hard_negatives.csv'}")
    for q, cnd in neg[neg["rank"] == 1].sample(4, random_state=1)[["query", "candidate"]].itertuples(index=False):
        t = train.set_index("posting_id")["title"]
        print(f"  {t[q][:55]}  <->  {t[cnd][:55]}")


if __name__ == "__main__":
    main()

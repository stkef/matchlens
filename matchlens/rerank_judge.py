"""Train the rung 6 judge on train-split pairs.

    python -m matchlens.rerank_judge configs/rung06_<model>.toml

Reads the train pairs exported by `matchlens.rerank_pairs` (with the base pipeline's score and the
label) and the cross-encoder scores computed on Kaggle for the same pairs, then fits a logistic
regression on [base score, cross-encoder logit]. Validation pairs are never used. Writes the config's
`judge_path`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from matchlens.evaluate import load_config
from matchlens.fusion_train import fit_logreg
from matchlens.retrievers.rerank import judge_features


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config")
    p.add_argument("--pairs", default="C:/data/shopee/rerank/pairs_train.csv")
    p.add_argument("--train-scores", required=True, help="rerank_<model>_train.csv from the Kaggle notebook")
    args = p.parse_args(argv)
    rcfg = load_config(args.config)["retriever"]

    pairs = pd.read_csv(args.pairs).merge(pd.read_csv(args.train_scores), on=["query", "candidate"],
                                          how="inner", validate="one_to_one")
    X = judge_features(pairs["base_score"].to_numpy(), pairs["score"].to_numpy())
    y = pairs["same"].astype(float).to_numpy()
    coef, intercept = fit_logreg(X, y)
    out = Path(rcfg["judge_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"features": ["base_score", "cross_encoder_logit"], "coef": coef.tolist(),
                               "intercept": intercept, "n_pairs": len(y), "n_positive": int(y.sum())}, indent=2))
    print(f"{len(y)} train pairs ({int(y.sum())} positive): base_score {coef[0]:+.3f}, "
          f"cross_encoder {coef[1]:+.3f}, intercept {intercept:+.3f} -> {out}")


if __name__ == "__main__":
    main()

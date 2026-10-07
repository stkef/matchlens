"""Train the learned fusion weights on the train split (rung 5b).

    python -m matchlens.fusion_train configs/rung05b_learned.toml

Uses the config's fusion members, searches every *train* listing against the train listings, and fits
a logistic regression that predicts "same product" from each member's score and 1/rank. Validation and
test are never touched, so the threshold tuned on val afterwards stays honest. Writes the weights to the
config's `model_path`.

With --fold B, only the fold-B products of the train split are used (data/splits/train_folds_v1.csv):
the cross-fitting setup when a member model was fine-tuned on fold A.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from matchlens.data import load_split_frame
from matchlens.evaluate import load_config
from matchlens.retrievers import build_retriever


def find_fusion_cfg(cfg: dict) -> dict:
    """The [retriever] table of type 'fusion', possibly nested inside wrappers such as phash_boost."""
    while cfg.get("type") != "fusion":
        if "base" not in cfg:
            raise ValueError("config has no fusion retriever")
        cfg = cfg["base"]
    return cfg


def fit_logreg(X: np.ndarray, y: np.ndarray, l2: float = 1e-4) -> tuple[np.ndarray, float]:
    """Plain L2-regularised logistic regression via L-BFGS (no scikit-learn needed)."""
    Xb = np.hstack([X, np.ones((len(X), 1))])

    def loss(w):
        z = Xb @ w
        nll = np.logaddexp(0, z).sum() - y @ z
        grad = Xb.T @ (1 / (1 + np.exp(-z)) - y)
        return (nll + l2 * w[:-1] @ w[:-1]) / len(y), (grad + np.r_[2 * l2 * w[:-1], 0]) / len(y)

    w = minimize(loss, np.zeros(Xb.shape[1]), jac=True, method="L-BFGS-B").x
    return w[:-1], float(w[-1])


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config")
    p.add_argument("--k", type=int, default=50)
    p.add_argument("--fold", choices=["A", "B"], help="use only this fold of the train split")
    p.add_argument("--folds-file", default="data/splits/train_folds_v1.csv")
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    fusion_cfg = dict(find_fusion_cfg(cfg["retriever"]), method="rrf")  # rrf: no weights needed to collect features
    fusion_cfg.pop("model_path", None)
    fusion = build_retriever(fusion_cfg)

    train = load_split_frame(cfg["data"]["csv"], cfg["data"]["split"], "train")
    if args.fold:
        folds = pd.read_csv(args.folds_file)
        keep = set(folds.loc[folds["fold"] == args.fold, "posting_id"])
        train = train[train["posting_id"].isin(keep)].reset_index(drop=True)
        print(f"fold {args.fold}: {len(train)} train listings")
    labels = train["label_group"].to_numpy()
    t0 = time.perf_counter()
    fusion.fit(train)
    ids, feats = fusion.features(train, args.k + 1)
    X, y = [], []
    for i, (cand, f) in enumerate(zip(ids, feats)):
        keep = cand != i  # never learn from a listing matching itself
        X.append(f[keep])
        y.append(labels[cand[keep]] == labels[i])
    X, y = np.vstack(X), np.concatenate(y).astype(float)
    print(f"{len(y)} train pairs, {int(y.sum())} positive, features in {time.perf_counter() - t0:.0f}s")

    coef, intercept = fit_logreg(X, y)
    names = [f"{m.get('type')}{'_' + Path(m['path']).stem if 'path' in m else ''}:{kind}"
             for m in fusion_cfg["members"] for kind in ("score", "inv_rank")]
    out = Path(find_fusion_cfg(cfg["retriever"])["model_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"features": names, "coef": coef.tolist(), "intercept": intercept,
                               "n_pairs": len(y), "n_positive": int(y.sum())}, indent=2))
    for n, c in zip(names, coef):
        print(f"  {n:40} {c:+.3f}")
    print(f"  {'intercept':40} {intercept:+.3f}\nwrote {out}")


if __name__ == "__main__":
    main()

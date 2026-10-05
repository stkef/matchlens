"""Loading listings and the group-level train / val / test split (FR-1)."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from matchlens.text import decode_title

REQUIRED_COLUMNS = ["posting_id", "image", "image_phash", "title", "label_group"]
SPLITS = ("train", "val", "test")


def load_listings(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype={"posting_id": str, "image": str, "image_phash": str, "title": str})
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {missing}")
    df["title"] = df["title"].map(decode_title)
    if df["posting_id"].duplicated().any():
        raise ValueError("posting_id values must be unique")
    return df.reset_index(drop=True)


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def group_split(df: pd.DataFrame, fractions=(0.7, 0.1, 0.2), seed: int = 42) -> pd.Series:
    """Assign every listing to a split so that no label_group spans two splits."""
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("fractions must sum to 1")
    groups = np.sort(df["label_group"].unique())
    groups = np.random.default_rng(seed).permutation(groups)
    n_train = round(len(groups) * fractions[0])
    n_val = round(len(groups) * fractions[1])
    assignment = {g: "train" for g in groups[:n_train]}
    assignment.update({g: "val" for g in groups[n_train:n_train + n_val]})
    assignment.update({g: "test" for g in groups[n_train + n_val:]})
    return df["label_group"].map(assignment).rename("split")


def save_split(df: pd.DataFrame, split: pd.Series, out_path: str | Path, *, source_csv: str | Path,
               seed: int, fractions, force: bool = False) -> Path:
    out_path = Path(out_path)
    manifest_path = out_path.with_suffix(".json")
    source_hash = file_sha256(source_csv)
    if out_path.exists() and not force:
        old = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        if old.get("source_sha256") != source_hash or old.get("seed") != seed or old.get("fractions") != list(fractions):
            raise FileExistsError(f"{out_path} exists with a different source/seed/fractions; pass --force to replace it")
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"posting_id": df["posting_id"], "split": split}).to_csv(out_path, index=False)
    counts = {
        s: {"listings": int((split == s).sum()), "groups": int(df.loc[split == s, "label_group"].nunique())}
        for s in SPLITS
    }
    manifest = {"source_csv": str(source_csv), "source_sha256": source_hash, "seed": seed,
                "fractions": list(fractions), "counts": counts}
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return out_path


def load_split_frame(csv_path: str | Path, split_path: str | Path, name: str) -> pd.DataFrame:
    """Listings belonging to one split, with a fresh 0..n-1 index."""
    if name not in SPLITS:
        raise ValueError(f"unknown split {name!r}")
    df = load_listings(csv_path)
    split = pd.read_csv(split_path, dtype={"posting_id": str})
    df = df.merge(split, on="posting_id", how="left", validate="one_to_one")
    if df["split"].isna().any():
        raise ValueError("split file does not cover every listing; regenerate it")
    return df[df["split"] == name].drop(columns="split").reset_index(drop=True)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Create the group-level split file.")
    p.add_argument("--csv", default="data/raw/train.csv")
    p.add_argument("--out", default="data/splits/split_v1.csv")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fractions", type=float, nargs=3, default=(0.7, 0.1, 0.2))
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)
    df = load_listings(args.csv)
    split = group_split(df, tuple(args.fractions), args.seed)
    path = save_split(df, split, args.out, source_csv=args.csv, seed=args.seed,
                      fractions=tuple(args.fractions), force=args.force)
    print(Path(path).with_suffix(".json").read_text())


if __name__ == "__main__":
    main()

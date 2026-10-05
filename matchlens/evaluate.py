"""One command per pipeline config (FR-8).

    python -m matchlens.evaluate configs/rung01_bm25.toml --split val
    python -m matchlens.evaluate configs/rung01_bm25.toml --split test --unlock-test

Queries are the listings of --split. The candidate pool is those listings plus any [eval] distractors
splits (e.g. ["train"]): listings from other products that can be retrieved but are never queried.
Validation runs tune the threshold; test runs reuse the threshold from that config's validation run.
Every run writes results/runs/<name>__<split>.json and appends one row to results/ledger.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
import tomllib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from matchlens import metrics
from matchlens.data import load_split_frame
from matchlens.retrievers import build_retriever, drop_self
from matchlens.threshold import tune_threshold

LEDGER_COLUMNS = [
    "timestamp", "name", "rung", "split", "n_queries", "pool_size", "recall@10", "recall@50", "mrr", "f1", "threshold",
    "precision@recall0.9", "p50_ms", "p95_ms", "usd_per_1k_queries", "fit_s", "git_sha",
]


def load_config(path: str | Path) -> dict:
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    for key in ("name", "data", "retriever"):
        if key not in cfg:
            raise ValueError(f"{path}: missing [{key}]")
    return cfg


def git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def measure_latency(retriever, queries, k: int, n_sample: int, seed: int = 0) -> np.ndarray:
    """Seconds per query, one query at a time, as an interactive caller would see it."""
    rng = np.random.default_rng(seed)
    rows = rng.choice(len(queries), size=min(n_sample, len(queries)), replace=False)
    times = []
    for r in rows:
        query = queries.iloc[[r]]
        t0 = time.perf_counter()
        retriever.search(query, k)
        times.append(time.perf_counter() - t0)
    return np.array(times)


def run(cfg: dict, split: str, results_dir: Path, unlock_test: bool = False) -> dict:
    if split == "test" and not unlock_test:
        raise PermissionError("the test split is locked until the ladder is frozen; pass --unlock-test")
    name = cfg["name"]
    eval_cfg = {"k": 50, "latency_sample": 200, **cfg.get("eval", {})}
    k = int(eval_cfg["k"])
    usd_per_hour = float(cfg.get("cost", {}).get("usd_per_hour", 0.10))

    distractors = list(eval_cfg.get("distractors", []))
    if split in distractors or "test" in distractors:
        raise ValueError("distractors may not include the evaluated split or the test split")
    queries = load_split_frame(cfg["data"]["csv"], cfg["data"]["split"], split)
    # Queries first, so query i is pool row i (drop_self and metrics rely on this).
    pool = pd.concat([queries] + [load_split_frame(cfg["data"]["csv"], cfg["data"]["split"], d)
                                  for d in distractors], ignore_index=True)
    labels = pool["label_group"].to_numpy()

    retriever = build_retriever(cfg["retriever"])
    t0 = time.perf_counter()
    retriever.fit(pool)
    fit_s = time.perf_counter() - t0

    cands = drop_self(retriever.search(queries, k + 1), k)

    runs_dir = results_dir / "runs"
    if split == "val":
        tuned = tune_threshold(cands, labels)
        threshold = tuned.threshold
        curve = {"thresholds": tuned.grid.tolist(), "mean_f1": tuned.f1_by_threshold.round(5).tolist()}
    else:
        val_run = runs_dir / f"{name}__val.json"
        if not val_run.exists():
            raise FileNotFoundError(f"run {name} on --split val first; its threshold is reused here")
        threshold = json.loads(val_run.read_text())["metrics"]["threshold"]
        curve = None

    latency = measure_latency(retriever, queries, k + 1, int(eval_cfg["latency_sample"]))
    m = {
        "n_queries": len(queries),
        "pool_size": len(pool),
        "recall@10": metrics.recall_at_k(cands, labels, 10),
        "recall@50": metrics.recall_at_k(cands, labels, 50) if k >= 50 else float("nan"),
        "mrr": metrics.mrr(cands, labels),
        "f1": float(metrics.f1_per_listing(cands, labels, threshold).mean()),
        "threshold": threshold,
        "precision@recall0.9": metrics.precision_at_recall(cands, labels, 0.9),
        "p50_ms": float(np.percentile(latency, 50) * 1000),
        "p95_ms": float(np.percentile(latency, 95) * 1000),
        "usd_per_1k_queries": float(latency.mean() * 1000 * usd_per_hour / 3600),
        "fit_s": fit_s,
    }
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "name": name, "rung": cfg.get("rung", ""), "split": split, "git_sha": git_sha(),
        "config": cfg, "metrics": m, "threshold_curve": curve,
    }
    runs_dir.mkdir(parents=True, exist_ok=True)
    (runs_dir / f"{name}__{split}.json").write_text(json.dumps(record, indent=2))
    append_ledger(results_dir / "ledger.csv", record)
    return record


def append_ledger(path: Path, record: dict) -> None:
    new = not path.exists()
    if not new:
        with open(path, newline="") as f:
            header = next(csv.reader(f), [])
        if header != LEDGER_COLUMNS:
            raise ValueError(f"{path} has an old column layout; move it aside before appending")
    row = {**{c: record.get(c, "") for c in LEDGER_COLUMNS}, **record["metrics"]}
    row = {c: (round(v, 6 if c.startswith("usd") else 4) if isinstance(v, float) else v)
           for c, v in row.items() if c in LEDGER_COLUMNS}
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LEDGER_COLUMNS)
        if new:
            w.writeheader()
        w.writerow(row)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config")
    p.add_argument("--split", choices=["train", "val", "test"], default="val")
    p.add_argument("--unlock-test", action="store_true")
    p.add_argument("--results-dir", default="results")
    args = p.parse_args(argv)
    record = run(load_config(args.config), args.split, Path(args.results_dir), args.unlock_test)
    print(f"{record['name']} on {args.split}:")
    for key, value in record["metrics"].items():
        print(f"  {key:>22}: {value:.4f}" if isinstance(value, float) else f"  {key:>22}: {value}")


if __name__ == "__main__":
    main()

# MatchLens

Decides whether two product listings are the same product, from titles and photos, and proves every
design choice with a measured experiment. Full plan: [docs/PRD.md](docs/PRD.md).

## Status

| # | Rung | Status |
|---|---|---|
| 1 | BM25 on titles + global threshold | **built**, waiting for real data |
| 2–8 | near-dups, embeddings, fusion, reranker, fine-tuning, thresholds | not started |

## Setup

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # Windows; use .venv/bin/python elsewhere
```

## Get the data

Accept the competition rules on Kaggle ("Shopee - Price Match Guarantee"), set up your Kaggle API token,
then:

```bash
kaggle competitions download -c shopee-product-matching -p data/raw
unzip data/raw/shopee-product-matching.zip -d data/raw
```

Only `train.csv` has labels, so every split is cut from it.

## Run

```bash
python -m matchlens.data                                         # create data/splits/split_v1.csv (+ manifest)
python -m matchlens.evaluate configs/rung01_bm25.toml --split val
```

Each run writes `results/runs/<name>__<split>.json` and appends a row to `results/ledger.csv`.
The test split is locked: `--split test --unlock-test`, only once the ladder is frozen. Test runs
reuse the threshold tuned on validation.

### Smoke test without the real data

```bash
python -m matchlens.synthetic
python -m matchlens.data --csv data/synthetic/train.csv --out data/synthetic/split.csv
python -m matchlens.evaluate configs/smoke_bm25.toml --split val --results-dir data/synthetic/results
python -m pytest -q
```

## Layout

| Path | What |
|---|---|
| `matchlens/data.py` | Listing loader, group-level split, split manifest |
| `matchlens/text.py` | Title decoding and tokenisation (units glued to numbers: `128 GB` → `128gb`) |
| `matchlens/retrievers/` | `Retriever` interface and implementations (`bm25.py`) |
| `matchlens/metrics.py` | Recall@k, MRR, competition F1, precision@recall |
| `matchlens/threshold.py` | Global threshold tuning on validation |
| `matchlens/evaluate.py` | The harness: one config in, one ledger row out |
| `configs/` | One TOML per rung |

## Adding a rung

1. Implement `fit(corpus)` / `search(queries, k) -> Candidates` with per-query normalised scores
   (1.0 = as similar as the query to itself).
2. Register it in `matchlens/retrievers/__init__.py`.
3. Add `configs/rungNN_<name>.toml` and run it on `val`.

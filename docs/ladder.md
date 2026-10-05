# Experiment ladder

One entry per rung. Fill in **Result** from `results/ledger.csv` (validation only) and decide the
**Verdict** before moving on. Comparison rungs are judged on their own; additive rungs are kept only if
validation improves.

---

## Rung 1 — BM25 on titles (baseline)

- **Kind:** baseline
- **Config:** `configs/rung01_bm25.toml`
- **Hypothesis:** Exact tokens carry most of the signal for products with model numbers and specs
  (`a52`, `128gb`). Expect high precision on those, poor recall on paraphrased or mixed-language titles,
  and confusion between storage/size variants that share every other word.
- **Change:** BM25 (`k1 = 1.2`, `b = 0.75`) over cleaned titles; scores divided by the query's
  self-score; one global threshold tuned on val.
- **Result:** _not run yet_
- **Verdict:** _pending_
- **Notes / failure examples:** _pending_

---

## Rung 2 — + near-duplicate filter

- **Kind:** additive
- **Hypothesis:** Many listings are copies (same photo, near-identical title). MinHash on titles and
  `image_phash` equality catch them cheaply.
- **Result / verdict:** _pending_

## Rung 3 — Text embeddings

- **Kind:** comparison
- **Candidates:** `intfloat/multilingual-e5-base`, `paraphrase-multilingual-mpnet-base-v2`, `BAAI/bge-m3`
- **Hypothesis:** Fixes paraphrases and Indonesian/English mixes that BM25 misses; likely worse on exact
  specs.
- **Result / verdict:** _pending_

## Rung 4 — Image embeddings

- **Kind:** comparison
- **Hypothesis:** Rescues listings with useless titles but matching photos.
- **Result / verdict:** _pending_

## Rung 5 — Fusion

- **Kind:** additive
- **Hypothesis:** Each retriever catches what the others miss; RRF first, then a learned weighting.
- **Result / verdict:** _pending_

## Rung 6 — Reranker

- **Kind:** additive
- **Hypothesis:** A cross-encoder reading both titles together, plus image similarity, fixes near-misses
  ranked above true matches.
- **Result / verdict:** _pending_ (report p95 with and without it)

## Rung 7 — Fine-tuned embeddings with hard negatives

- **Kind:** additive
- **Hypothesis:** Training on the system's own mistakes (from the **train** split) teaches the encoders
  that 64 GB ≠ 128 GB.
- **Result / verdict:** _pending_

## Rung 8 — Per-cluster / adaptive thresholds

- **Kind:** additive
- **Hypothesis:** One threshold is too strict for some product types and too loose for others.
- **Result / verdict:** _pending_

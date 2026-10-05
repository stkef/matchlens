# Evaluation

## Splits

All splits come from Kaggle's `train.csv`, the only labelled file. Listings are split **by
`label_group`**, so a product never appears in two splits.

| Split | Share of groups | Used for |
|---|---|---|
| train | 70% | Fine-tuning and hard-negative mining (rung 7) |
| val | 10% | Every keep/drop decision and every threshold |
| test | 20% | Reported numbers only, once the ladder is frozen |

`python -m matchlens.data` writes `data/splits/split_v1.csv` and a manifest `split_v1.json` with the
seed, fractions, per-split counts and the SHA-256 of the source CSV. Re-running with a different
source, seed or fractions fails unless you pass `--force`, so the split cannot drift silently.

## Protocol

1. Build the candidate pool: the evaluated split's listings **plus every training listing as a
   distractor** (`[eval] distractors = ["train"]`). Training groups never overlap the evaluated split,
   so distractors are always wrong answers — they only make the search harder, as in real use.
2. Fit the retriever on the pool.
3. Use every listing of the evaluated split as a query and remove the query itself from its results.
   Distractors are retrievable but never queried.
4. On **val**: tune one global threshold for best mean F1, and save it with the run.
5. On **test** (`--unlock-test`): reuse the val threshold unchanged.

**Why distractors.** Pool size changes the numbers a lot. BM25 scores F1 0.811 searching val alone
(3,366 listings) but 0.679 against val + train (27,431): with few listings, look-alikes rarely exist.
Kaggle's hidden test set has ~70k listings, so the bigger pool is the more honest one.

**Caveat for rung 7.** Fine-tuning trains on the training listings, which are also the distractors, so
a fine-tuned model may find them unusually easy to reject. Rung 7 therefore also reports the
no-distractor pool (`distractors = []`) so that gain can be checked.

**Floor.** Every listing matches itself, and most groups have 2 listings, so predicting only "self"
scores F1 0.469 on val. Quote gains against this floor, not against 0.

## Metrics

Notation: for a query *q*, *G(q)* is its product group including *q*; the true matches are
*G(q) \ {q}*. Queries whose group has no other listing are skipped for recall and MRR (this cannot
happen with Shopee groups, but the code handles it).

| Metric | Definition |
|---|---|
| **Recall@k** | `|top-k ∩ true matches| / min(k, |true matches|)`, averaged over queries. The `min` stops large groups (up to ~50 listings) from capping recall below 1. |
| **MRR** | `1 / rank` of the first true match in the list; 0 if none was retrieved. |
| **F1** (competition metric) | Predicted set *P* = {q} ∪ {candidates with score ≥ threshold}. Truth *T* = *G(q)*. `F1 = 2·|P ∩ T| / (|P| + |T|)`, averaged over listings. A listing always matches itself, so the floor is `2 / (1 + |G(q)|)`. |
| **Precision@recall 0.9** | Pool every retrieved (query, candidate) pair, sort by score, and report the best pairwise precision at any cut where pairwise recall ≥ 0.9. Recall's denominator counts **all** true pairs, including ones never retrieved, so a retriever that never reaches 90% recall reports `NaN` rather than a flattering number. |
| **p50 / p95 latency** | 200 random queries searched one at a time (as an interactive user would), wall-clock, CPU. |
| **USD per 1k queries** | Mean seconds per query × 1000 × `usd_per_hour / 3600`. An estimate; the rate is an assumption in the config. |

## Why these choices

- **Group split, not listing split.** If listings of one product land in both train and test, a
  fine-tuned model can memorise the product and test numbers become meaningless.
- **One threshold per rung.** Without a threshold there is no F1, and F1 is what the business cares
  about. Tuning it on val and freezing it for test keeps the comparison honest.
- **Test lock.** Looking at test numbers while choosing rungs turns test into a second validation set.
  The `--unlock-test` flag makes that a deliberate act, and the ledger records when it happened.

## Reading the ledger

`results/ledger.csv` has one row per run, appended, never rewritten. Columns: `timestamp, name, rung,
split, n_queries, pool_size, recall@10, recall@50, mrr, f1, threshold, precision@recall0.9, p50_ms, p95_ms,
usd_per_1k_queries, fit_s, git_sha`. The full record, including the F1-vs-threshold curve, is in
`results/runs/<name>__<split>.json`.

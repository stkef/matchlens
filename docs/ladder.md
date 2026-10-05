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
- **Result (val, pool 27,431):** recall@10 0.812 · recall@50 0.917 · MRR 0.782 · **F1 0.679** at
  threshold 0.54 · precision@recall0.9 0.071 · p50 0.6 ms · p95 1.0 ms.
- **Context:** the self-only floor is F1 0.469, so BM25 adds +21 points. On the val split alone
  (pool 3,366) the same model scores F1 0.811 / recall@50 0.971 — fewer distractors flatter it by
  13 points, which is why the larger pool is the standard.
- **Verdict:** baseline accepted. 41% of queries get at least one false match at the tuned threshold,
  and 8% of true matches are not in the top 50 at all.
- **Failure types seen (hand-sampled):**
  1. *Look-alike variants* — same words, different size/volume: "Proclin Penghilang Noda Pouch 800 ml"
     vs "… Bottle 400ml" (0.62); paper bags 10x6x13 vs 25x10x35 (0.84). BM25 cannot weigh a spec token
     above the rest. → rungs 6–7.
  2. *Unit spellings* — "400 gram" / "400 gr" / "400gr" become different tokens, so "Profeline 400 gram"
     vs "Profeline 400gr" scores only 0.31. A tokenizer fix (canonical units), not a model problem.
     → candidate rung 1b.
  3. *Paraphrase / mixed language* — "wet food" vs "makanan kucing"; "extension" vs "extention".
     → rung 3.
  4. *No shared words at all* — "Gamis rayon… 1 kg muat 4 pcs" vs "CHIKA PASTEL DRESS / HOMY DRESS";
     only the photo can match these. → rung 4.
  5. *Multi-variant listings* — "Goon Smile Baby S40 / M34 / L30 / XL26" vs a listing of a subset of
     sizes; arguably label noise. → check in the 100-error label audit (milestone 2).

---

## Rung 1b — + canonical units

- **Kind:** additive (tokenizer change; rung 1 stays reproducible via `canonical_units = false`)
- **Config:** `configs/rung01b_bm25_units.toml`
- **Hypothesis:** Rung 1 treats "400 gram", "400 gr" and "400gr" as different words, and "1 kg" vs
  "1000 gr" as unrelated. Mapping every spelling to one unit and scale (`400g`, `1000g`, `1800ml`)
  should recover those matches.
- **Result (val, pool 27,431):** recall@50 0.918 (+0.001) · MRR 0.784 (+0.003) · **F1 0.682 (+0.004)**
  at threshold 0.57 · p95 1.8 ms.
- **Is the gain real?** Paired per-listing comparison: 607 listings better, 366 worse, 2,393 unchanged.
  Bootstrap (5,000 resamples of listings): mean +0.0036, 95% interval [+0.0007, +0.0065], positive in
  99.2% of resamples. Small but real.
- **Why some got worse:** the biggest losers have no units at all ("Sepatu Hiking/ Mendaki/ Outdoor…").
  They moved because the best global threshold shifted from 0.54 to 0.57, and one threshold applies to
  everyone. Biggest winners: "Nivea … 75 gram" (+0.67), "Enfagrow … 1800 gram" (+0.56),
  "Acnes Facial Wash 50gr/100gr" (+0.48).
- **Verdict:** kept. Small because only some titles contain units; the remaining failures are about
  meaning and look-alikes, which word matching cannot fix.

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

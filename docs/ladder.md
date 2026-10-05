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

## Rung 2 — + near-duplicate photos

- **Kind:** additive
- **Config:** `configs/rung02_phash.toml` (`matchlens/retrievers/phash.py`)
- **Hypothesis:** Many listings are copies (same photo, near-identical title). Perceptual image hashes
  and title overlap catch them cheaply, without any model.
- **Measured before building** (val queries vs pool of 27,431; all true pairs = 100%):

  | Signal | Pairs flagged | Precision | Share of all true pairs |
  |---|---|---|---|
  | phash identical (0 bits apart) | 1,991 | 0.963 | 14.6% |
  | phash ≤ 6 bits apart | 3,459 | 0.896 | 23.6% |
  | phash ≤ 12 bits apart | 9,207 | 0.391 | 27.4% |
  | title Jaccard ≥ 0.8 | 1,806 | 0.876 | 12.0% |
  | rung 1b's own "yes" decisions | 11,674 | 0.563 | 50.1% |

  Title near-duplicates added **zero** pairs beyond what rung 1b already says yes to (BM25 already
  scores near-identical titles highly), so the planned title MinHash stage was dropped. Identical
  photo hashes added 1,011 new pairs, 96.5% of them correct.
- **Change:** any listing whose phash is within `max_dist` bits of the query's is added as a
  certain match (score 1.0). Swept on val: 0 → F1 0.721, 2 → 0.735, 4 → 0.740, **6 → 0.744**,
  8 → 0.741.
- **Result (val):** recall@50 0.941 (+0.022) · MRR 0.848 (+0.064) · **F1 0.744 (+0.061 over 1b)** at
  threshold 0.61 · p95 2.0 ms.
- **Verdict:** kept. Biggest single gain so far, for the cost of an XOR and a bit count.
- **What it gets wrong:** 430 of 4,065 photo matches (10.6%) are different products:
  1. *Photo reused across variants* — "BEBELAC TAHAP 3 MADU 800GR" vs "Bebelac 4 Madu Vanila 800 gr".
  2. *Generic or shop-banner photos* — "Wardah … Facial Wash" vs "Kodomo Sikat Gigi Anak".
  3. *Likely label noise* — "Tolak Angin Cair" vs "Tolak Angin Cair": same title, same photo,
     different label_group. → count these in the label audit.
- **Note on MinHash:** MinHash/LSH approximates Jaccard similarity so it scales to millions of
  listings. At 27k listings exact Jaccard is a single sparse matrix product, so it was measured
  exactly instead.

## Rung 3 — Text embeddings

- **Kind:** comparison (each model judged alone; combining comes at rung 5)
- **Config:** `configs/rung03_<model>.toml`; vectors computed on a Kaggle GPU (see [kaggle.md](kaggle.md))
- **Hypothesis:** Fixes paraphrases and Indonesian/English mixes that BM25 misses; likely worse on exact
  specs.
- **Result (val, pool 27,431):**

  | Model | Size | Recall@50 | MRR | F1 | Threshold | Embed 1 query (CPU p95) |
  |---|---|---|---|---|---|---|
  | BM25 + units (rung 1b, reference) | — | 0.918 | 0.784 | 0.682 | 0.57 | — |
  | **BAAI/bge-m3** | 568M, 1024-d | **0.900** | **0.774** | **0.673** | 0.74 | 274 ms |
  | intfloat/multilingual-e5-base | 278M, 768-d | 0.855 | 0.729 | 0.644 | 0.93 | 76 ms |
  | paraphrase-multilingual-mpnet-base-v2 | 278M, 768-d | 0.742 | 0.642 | 0.590 | 0.85 | 77 ms |
  | FastText, idf-weighted mean (trained here) | 2M, 100-d | 0.883 | 0.746 | 0.634 | 0.95 | **0.1 ms** |
  | FastText, plain mean (trained here) | 2M, 100-d | 0.870 | 0.727 | 0.613 | 0.95 | 0.1 ms |

  FastText (`python -m matchlens.fasttext_embed`) is skip-gram with 3–5 character n-grams, trained
  only on the 24,065 **train** titles, on the laptop CPU in ~45 s. A title vector is the average of its
  word vectors; weighting by idf (rare words count more, as in BM25) adds +0.021 F1.

  GPU time to embed all 34,250 titles: bge-m3 31 s, the others ~12 s (a laptop CPU needs ~80 min for
  bge-m3). Search over the stored vectors adds 4–13 ms.
- **Hypothesis check — complementary, not better.** Of 13,138 true pairs, at each model's threshold:
  BM25 says a correct "yes" to 6,576, bge-m3 to 6,301; **1,480 only BM25 gets and 1,205 only bge-m3
  gets**; together 7,781 (+18% over BM25). In the top 50, either one finds 93.9% of all true pairs.
  - bge-m3 wins on meaning: "Sarung Pouch Bag HP Waterproof Anti Air" ≈ "WATERPROOF CASE HP / airbag";
    "TATAKAN MOUSE" ≈ "ALAS UNTUK MOUSE" (two Indonesian words for mat).
  - BM25 wins on exact words: "Pisau Apel Stainless" vs "Pisau Apel Warna Random"; "Buku Tulis Campus
    50 Lembar".
- **FastText vs BM25:** 659 correct matches BM25 misses (+10% together, vs +18% for bge-m3), and 393
  of those 659 bge-m3 finds too. Character n-grams overlap with what word matching already sees.
  The cost story is the point: a model trained in 45 s on a laptop, 2,500× faster per query, gets
  within 0.04 F1 of a 568M-parameter model.
- **Verdict:** bge-m3 is the text model carried forward. Alone it does not beat BM25, so it is not
  added as a rung on its own; it goes into fusion (rung 5), where the 18% of extra matches can count.
- **Costs to watch:** embedding a new query with bge-m3 takes ~270 ms on CPU, most of the 300 ms
  budget. multilingual-e5-base is 3.5× faster for −0.03 F1; revisit if latency becomes the limit.
- **Note:** e5's best threshold is 0.93 because its similarities are squeezed into a narrow, high
  range; thresholds are not comparable between models, which is why each rung tunes its own.

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

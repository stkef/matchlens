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

- **Kind:** comparison (each model judged alone; combining comes at rung 5)
- **Config:** `configs/rung04_<model>.toml`; vectors computed on a Kaggle GPU by
  [`kaggle/embed_images`](../kaggle/embed_images) — the 1.7 GB of photos never leave Kaggle.
- **Hypothesis:** Rescues listings with useless titles but matching photos, including *different*
  photos of the same product, which phash (rung 2) cannot match.
- **Selection rule (agreed before results):** the cheapest model within 0.01 F1 of the best.
- **Result (val, pool 27,431; models compared in FAISS):**

  | Model | Learned from | Dim | Recall@50 | MRR | F1 | Threshold | Embed 1 photo (CPU p50) | GPU time, 34k photos |
  |---|---|---|---|---|---|---|---|---|
  | **Marqo/marqo-ecommerce-embeddings-B** | **e-commerce product images + text** | 768 | **0.941** | **0.824** | **0.708** | 0.76 | **257 ms** | 235 s |
  | google/siglip2-base-patch16-224 | images + captions | 768 | 0.935 | 0.818 | 0.682 | 0.87 | 284 ms | 502 s |
  | facebook/dinov2-base | images only (self-supervised) | 768 | 0.868 | 0.784 | 0.682 | 0.84 | 311 ms | 141 s |
  | google/siglip-base-patch16-224 | images + captions | 768 | 0.923 | 0.817 | 0.676 | 0.84 | 273 ms | 158 s |
  | microsoft/swinv2-base (ImageNet) | labelled categories | 1024 | 0.778 | 0.704 | 0.634 | 0.91 | 375 ms | 250 s |
  | openai/clip-vit-base-patch32 | images + captions | 512 | 0.755 | 0.683 | 0.634 | 0.88 | 109 ms | 409 s* |
  | facebook/dinov3-vitb16 | images only | — | — | — | — | — | — | not run: gated model, no HF token at run time |
  | jinaai/jina-embeddings-v4 | images + text | — | — | — | — | — | — | failed: `KeyError: 'default'` in its remote code |

  \* CLIP ran first, so its time includes Kaggle's first cold read of the 34k photos from disk.
- **Verdict:** **Marqo e-commerce B** is carried forward. It is the best on every accuracy metric
  (F1 0.708, +0.026 over the next best) *and* the cheapest per photo of the strong models (257 ms), so
  the "cheapest within 0.01 F1" rule picks it outright.
  - *Second run (2026-10-06):* SigLIP was first selected from the initial four; SigLIP 2 and Marqo were
    then added as token-free alternatives to DINOv3. SigLIP 2 improved on SigLIP (+0.005 F1), but the
    domain-specific Marqo model beat every general-purpose one.
  - *Domain beats generality:* Marqo is a CLIP-style model trained on online-shop product images and
    titles — the same kind of data as Shopee — and it outperforms models trained on general web photos
    of the same size.
- **Observations:**
  - The best image model alone (0.682) is roughly level with BM25 + units (0.682) and bge-m3 (0.673):
    three very different signals of similar strength — promising for fusion.
  - Swin V2, trained to name *categories*, is weakest at recognising *the same item*, as expected.
  - CLIP B/32 is the cheapest per photo (109 ms) but 0.05 F1 behind: outside the 0.01 rule.
- **Not run:** DINOv3 (gated; replaced by the token-free SigLIP 2 / Marqo comparison), Jina v4
  (`KeyError: 'default'` in its remote code). The winner is then loaded into Qdrant, the image vector
  store.

## Rung 5 — Fusion

- **Kind:** additive, in four measured steps
- **Members:** BM25 + units (rung 1b), bge-m3 text (rung 3 winner), Marqo e-commerce images (rung 4
  winner); the rung 2 phash rule stays on top (`phash_boost` wraps the fusion). Embedding members are
  searched in FAISS (identical results to Qdrant, much faster to evaluate).
- **Hypothesis:** rungs 3–4 showed the signals are complementary (e.g. BM25 + bge-m3 find 18% more
  correct matches together), so combining them should beat the best single pipeline (rung 2, 0.744).
- **Results (val, pool 27,431):**

  | Step | Change | Recall@50 | MRR | F1 | Δ F1 | p95 search |
  |---|---|---|---|---|---|---|
  | 2 (reference) | BM25 + units + phash | 0.941 | 0.848 | 0.744 | | 2 ms |
  | **5a** | Reciprocal rank fusion (`rrf_k` 60) | **0.980** | 0.861 | 0.750 | +0.006 | 29 ms |
  | **5b** | Learned weights (logistic regression) | 0.976 | **0.879** | 0.781 | **+0.031** | — |
  | **5c** | + always keep the best candidate (`min_matches = 1`) | 0.976 | 0.879 | 0.790 | +0.008 | — |
  | **5d** | + neighbour voting (`expand_k` 3, `expand_alpha` 3) | 0.978 | 0.877 | **0.793** | +0.004 | 90 ms |

  Search latency excludes embedding a brand-new query on CPU (bge-m3 ~274 ms + Marqo ~257 ms), which
  is the real cost of serving and is over the 300 ms budget — see rung 6 notes and the PRD.
- **5a — RRF:** uses only ranks, so it finds almost everything (98% of true matches in the top 50,
  from 94%) but decides poorly: a candidate ranked 2nd by every member scores the same however
  confident each member was.
- **5b — learned weights:** a 7-parameter logistic regression over each member's score and 1/rank,
  trained on **2.7M train-split pairs** (118,688 true matches) with `python -m matchlens.fusion_train`;
  validation is only used for the threshold. Learned weights (score / 1-over-rank):
  Marqo images **+5.1 / +3.1**, BM25 **+4.1** / −0.5, bge-m3 **+1.8** / +0.6. The photo model is
  trusted most, BM25 next; the text model adds the least once the others are present.
- **5c — at least one match:** every Shopee product has 2+ listings, so every listing has at least one
  true match. Keeping the top candidate even when it is below the threshold turns many "matched only
  itself" answers into a correct pair. A trick used by top competition teams.
- **5d — neighbour voting:** each embedding is blended with its 3 nearest neighbours (weights =
  similarity³) on both the query and database side, so a listing close to *some* members of a group is
  pulled towards the whole group. Fusion weights retrained for the expanded members. Paired bootstrap
  vs 5c: 527 listings better, 462 worse, mean **+0.0039**, 95% interval [+0.0005, +0.0072], positive in
  98.8% of resamples: small but real, for 3× the search time (still well under budget).
- **Verdict:** kept, all four steps. **F1 0.744 → 0.793 (+0.049)**, the second-largest gain after rung 2.

## Rung 6 — Cross-encoder reranker

- **Kind:** additive (two rerankers compared)
- **Configs:** `configs/rung06_<model>.toml`; pair scores computed on a Kaggle GPU by
  [`kaggle/rerank`](../kaggle/rerank).
- **Hypothesis:** embeddings encode each title separately and blur details; a cross-encoder reads the
  query and candidate titles *together* (attention across both), so it should separate look-alikes
  ("800 ml" vs "400 ml") that rung 5 ranks too high.
- **How it was run:**
  1. `python -m matchlens.rerank_pairs configs/rung05d_expand.toml` exported rung 5d's top 20 per
     listing: 67,320 val pairs (89.5% of all true val pairs fall within the top 20 — the ceiling for
     this rung) and 120,000 pairs from 6,000 sampled train listings (searched among train listings only).
  2. The pair ids (no titles or images) went to a private Kaggle dataset; the notebook scored every pair
     with each cross-encoder.
  3. `python -m matchlens.rerank_judge` fitted a logistic regression on **train pairs only**:
     P(same) from [rung 5d score, cross-encoder logit]. Photos still count through the rung 5d score.
- **Result (val, pool 27,431):**

  | Reranker | Size | Recall@10 | MRR | F1 | Δ F1 vs 5d | Bootstrap: share of resamples > 0 | CPU per listing (20 pairs) | GPU, 67k pairs |
  |---|---|---|---|---|---|---|---|---|
  | none (rung 5d) | — | — | 0.877 | 0.793 | | | — | — |
  | **BAAI/bge-reranker-v2-m3** | 568M | 0.929 | **0.890** | **0.805** | **+0.011** [95%: +0.007, +0.015] | **100%** | 3.3 s | 180 s |
  | cross-encoder/mmarco-mMiniLMv2-L12 | 118M | 0.929 | 0.887 | 0.792 | −0.001 [−0.005, +0.003] | 26% | 0.30 s | 30 s |

  Judge weights (base score / cross-encoder logit): bge +3.86 / +0.27, MiniLM +4.31 / +0.15.
- **What bge changed** (accepted pairs at each pipeline's threshold, val): removed 1,207 wrong matches
  and added 383 true ones, at the cost of 367 new wrong matches and 902 lost true ones — net positive.
  Removed: "Johnson's Top to Toe Hair & Body Bath 500ml" vs "Johnson's Baby Bath Milk & Rice 500ml";
  added: "ALAT PIJIT KEPALA MERK BOKOMA" vs "Alat Pijat Kepala Relaxsasi BOKOMA Head Massager".
  Some "removed" pairs look like label noise (two "Perlak Bayi Sugar Baby" listings labelled different).
- **Verdict:** **bge-reranker-v2-m3 kept: F1 0.793 → 0.805.** MiniLM gives no measurable gain, so the
  cheap option is not an option here (it is 0.013 below, outside the 0.01 rule).
- **Cost — the honest catch:** bge needs **3.3 s per listing on CPU**, ten times the 300 ms budget; it is
  only practical on a GPU (180 s for 67k pairs ≈ 50 ms per listing). Both rerankers were trained for
  search relevance, not product identity, which limits how well they separate variants. A small
  cross-encoder *fine-tuned on our own pairs* (rung 7) is the route to both cheaper and better.

## Rung 7 — Fine-tuned embeddings with hard negatives

- **Kind:** additive, with an internal comparison
- **Hypothesis:** Training on the system's own mistakes (from the **train** split) teaches the encoders
  that 64 GB ≠ 128 GB.
- **Rung 7a — text model (done):**
  - Base: `intfloat/multilingual-e5-base` (278M; 3.5× faster per query than bge-m3), fine-tuned on
    Kaggle by [`kaggle/finetune_text`](../kaggle/finetune_text) with MultipleNegativesRankingLoss
    (batch 64, lr 2e-5, max 64 tokens), on the **24,065 train listings only**.
  - Hard negatives: `python -m matchlens.mine_negatives configs/rung05d_expand.toml` searched every train
    listing among train listings with rung 5d and kept its top-ranked *wrong* candidates: 115,991, about
    5 per listing (e.g. "Sunlight … Refill Habbatussauda 755 ml" vs "Sunlight … Refill Jeruk Nipis 755ml").
  - Result, text model alone (val):

    | Model | Training | F1 (pool 27,431) | Recall@50 | MRR | F1 without distractors (pool 3,366) |
    |---|---|---|---|---|---|
    | e5-base | none | 0.644 | 0.855 | 0.729 | 0.757 |
    | bge-m3 (rung 3 pick) | none | 0.673 | 0.900 | 0.774 | 0.812 |
    | e5-base + SimCSE | raw train titles, 1 epoch, 5 min | 0.655 | 0.868 | 0.748 | 0.787 |
    | **e5-base + labels + hard negatives** | (anchor, same product, look-alike), 2 epochs, 15 min | **0.703** | **0.938** | **0.798** | **0.848** |

  - **Labels are worth ~5× more than self-supervision here:** SimCSE +0.011 over the base model,
    supervised +0.060. The supervised small model beats the 2× larger bge-m3 by +0.030.
  - **Not memorisation:** the training listings are also the distractors, so the model was re-scored
    without them (last column); the gain over bge-m3 holds (+0.036).
  - **In the full pipeline** (`configs/rung07a_fusion.toml`: rung 5d with the text member swapped,
    fusion weights retrained): recall@50 **0.986** (best so far), MRR 0.879, **F1 0.804** vs 0.793 for
    rung 5d. Paired bootstrap: 612 listings better, 424 worse, mean +0.0105, 95% [+0.0071, +0.0140],
    positive in 100%. This equals rung 6 (0.805) **without** the 3.3 s/listing cross-encoder.
  - **Caveat:** the fusion weights are fitted on train pairs, which the fine-tuned model has seen, so the
    judge sees an over-optimistic text score (its weight rose from +1.9 to +4.7). Val still improved; the
    clean fix is cross-fitting (fine-tune on one half of train, fit the judge on the other).
  - Rung 6's reranker scores were computed for rung 5d's candidates, so stacking it on 7a needs a new
    Kaggle scoring run (not done).
- **Comparison inside the rung — what are the labels worth?**
  1. **SimCSE (self-supervised, no labels):** the same title passed through the model twice with
     different dropout should land close; other titles in the batch should land far. Uses only raw
     train titles.
  2. **Supervised contrastive with hard negatives:** uses `label_group` to pull true matches together
     and push the system's own look-alike mistakes apart.
  Both start from multilingual-e5-base, train on Kaggle, and are scored the same way (results above).
- **Pending:** 7b — fine-tune a small cross-encoder on our own pairs (rung 6 showed off-the-shelf ones
  are slow or don't help); 7c — fine-tune the image model.

## Rung 8 — Per-cluster / adaptive thresholds

- **Kind:** additive
- **Hypothesis:** One threshold is too strict for some product types and too loose for others.
- **Result / verdict:** _pending_

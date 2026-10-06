# Heavy steps on Kaggle

Model inference and training run on Kaggle's free GPU (about 30 GPU hours a week), where the Shopee
data already lives. The laptop only evaluates. Measured on a laptop CPU (i5-1135G7), embedding the
27k titles with bge-m3 would take ~80 minutes; on a Kaggle GPU it takes minutes.

## Rung 3: title embeddings

Notebook source: [`kaggle/embed_titles/`](../kaggle/embed_titles). It embeds all 34,250 titles with
each model in its `MODELS` list and writes `emb_<name>.npz` (vectors) and `emb_<name>.json`
(dimension, GPU time, CPU latency for one title at a time).

```bash
# 1. Put your Kaggle username in kaggle/embed_titles/kernel-metadata.json ("id"), then push and run it.
kaggle kernels push -p kaggle/embed_titles

# 2. Check progress (or open the URL the push prints).
kaggle kernels status <username>/matchlens-embed-titles

# 3. When it says COMPLETE, download the outputs next to the data.
kaggle kernels output <username>/matchlens-embed-titles -p C:/data/shopee/embeddings

# 4. Evaluate locally, one config per model.
python -m matchlens.evaluate configs/rung03_bge_m3.toml --split val
```

Requirements: competition rules accepted, and a phone-verified Kaggle account (needed for GPU and
for internet access, which the notebook uses to download the models and this repo's title cleaner).

## Rung 4: image embeddings

Notebook source: [`kaggle/embed_images/`](../kaggle/embed_images). Same flow as rung 3 with
`matchlens-embed-images`; outputs are `emb_img_<name>.npz/.json`. Models that fail are logged in their
`.json` and skipped. Gated models (DINOv3) need a Hugging Face read token stored as a Kaggle secret
named `HF_TOKEN` and attached to the notebook. To re-run selected models only, set `RUN_ONLY`.

## Rung 6: reranker pair scores

1. `python -m matchlens.rerank_pairs configs/rung05d_expand.toml` writes the candidate pairs to
   `C:/data/shopee/rerank` (with labels, kept locally) and `.../rerank/upload` (ids only).
2. Upload the ids as a private dataset (run from inside `C:/data/shopee/rerank`; the Kaggle CLI mangles
   absolute Windows paths): `kaggle datasets create -p upload`.
3. `kaggle kernels push -p kaggle/rerank`, then `kaggle kernels output <username>/matchlens-rerank -p .`
4. `python -m matchlens.rerank_judge configs/rung06_bge_v2_m3.toml --train-scores C:/data/shopee/rerank/rerank_bge_v2_m3_train.csv`

## Why not local Ollama?

Tried first. Ollama runs the same open models, but on this laptop it computes on the CPU: bge-m3
managed ~6 titles/s and nomic-embed-text ~13 titles/s. It works, but ties the machine up for hours,
so the models were removed again.

"""Kaggle notebook: score candidate title pairs with cross-encoder rerankers (rung 6).

A cross-encoder reads the query title and a candidate title *together* and outputs one relevance
score, so it can compare details ("800 ml" vs "400 ml") that separate embeddings blur.

Inputs:  the private dataset `matchlens-rerank-pairs` (pairs_val.csv, pairs_train.csv: posting ids
         only, exported by `python -m matchlens.rerank_pairs`) and the competition's train.csv.
Outputs: rerank_<model>_<split>.csv (query, candidate, score) and rerank_<model>.json (timings, or the
         error if a model fails). Pushed and fetched as described in docs/kaggle.md.
"""

import glob
import json
import os
import subprocess
import sys
import time
import traceback

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

subprocess.run(["git", "clone", "--depth", "1", "https://github.com/stkef/matchlens.git", "/kaggle/temp/matchlens"],
               check=True)
sys.path.insert(0, "/kaggle/temp/matchlens")
from matchlens.text import decode_title  # noqa: E402

OUT = "/kaggle/working"
csv_path = next(p for p in glob.glob("/kaggle/input/**/train.csv", recursive=True) if "shopee" in p)
pairs_dir = os.path.dirname(next(iter(glob.glob("/kaggle/input/**/pairs_val.csv", recursive=True))))
titles = pd.read_csv(csv_path).set_index("posting_id")["title"].map(decode_title)
device = "cuda" if torch.cuda.is_available() else "cpu"

MODELS = [
    # name,          Hugging Face id,                                  batch
    ("minilm_multi", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",     512),
    ("bge_v2_m3",    "BAAI/bge-reranker-v2-m3",                        256),
]


def score(model, tok, a, b, batch):
    out = []
    with torch.no_grad():
        for s in range(0, len(a), batch):
            enc = tok(a[s:s + batch], b[s:s + batch], padding=True, truncation=True, max_length=128,
                      return_tensors="pt").to(next(model.parameters()).device)
            out.append(model(**enc).logits[:, 0].float().cpu().numpy())
    return np.concatenate(out)


for name, model_id, batch in MODELS:
    meta = {"name": name, "model": model_id}
    try:
        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSequenceClassification.from_pretrained(model_id, torch_dtype=torch.float16).to(device).eval()
        for split in ("val", "train"):
            pairs = pd.read_csv(os.path.join(pairs_dir, f"pairs_{split}.csv"))
            t0 = time.perf_counter()
            pairs["score"] = score(model, tok, titles[pairs["query"]].tolist(), titles[pairs["candidate"]].tolist(), batch)
            meta[f"gpu_s_{split}"] = round(time.perf_counter() - t0, 1)
            pairs.to_csv(f"{OUT}/rerank_{name}_{split}.csv", index=False)
        # CPU cost of reranking one query's 20 candidates, as a server without a GPU would see it.
        model = model.float().to("cpu")
        val = pd.read_csv(os.path.join(pairs_dir, "pairs_val.csv"))
        times = []
        for q in val["query"].drop_duplicates().iloc[:21]:
            sub = val[val["query"] == q]
            t = time.perf_counter()
            score(model, tok, titles[sub["query"]].tolist(), titles[sub["candidate"]].tolist(), 32)
            times.append(time.perf_counter() - t)
        meta.update({"cpu_ms_per_query_p50": round(float(np.percentile(times[1:], 50)) * 1000, 1),
                     "cpu_ms_per_query_p95": round(float(np.percentile(times[1:], 95)) * 1000, 1)})
        del model
        torch.cuda.empty_cache()
    except Exception as e:
        meta["error"] = f"{type(e).__name__}: {e}"
        traceback.print_exc()
    with open(f"{OUT}/rerank_{name}.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(meta, flush=True)

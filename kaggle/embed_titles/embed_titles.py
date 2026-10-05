"""Kaggle notebook: embed every Shopee title with several multilingual models (rung 3).

Runs on Kaggle's free GPU, where the competition data already lives. Writes one file per model to
/kaggle/working, downloaded afterwards with `kaggle kernels output`:

    emb_<name>.npz   posting_id (str) + vectors (float16, L2-normalised)
    emb_<name>.json  model id, dimension, GPU encode time, CPU single-title latency

Pushed and fetched from the repo with scripts in docs/kaggle.md; not run locally.
"""

import glob
import json
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

# One source of truth for title cleaning: the project's own code, from GitHub.
subprocess.run(["git", "clone", "--depth", "1", "https://github.com/stkef/matchlens.git", "/kaggle/temp/matchlens"],
               check=True)
sys.path.insert(0, "/kaggle/temp/matchlens")
from matchlens.text import decode_title  # noqa: E402

MODELS = [
    # name,          Hugging Face id,                                              prefix
    ("bge_m3",       "BAAI/bge-m3",                                                ""),
    ("e5_base",      "intfloat/multilingual-e5-base",                              "query: "),
    ("mpnet_multi",  "sentence-transformers/paraphrase-multilingual-mpnet-base-v2", ""),
]

# The mount point of competition data has changed over time; find it instead of hard-coding it.
csv_path = next(p for p in glob.glob("/kaggle/input/**/train.csv", recursive=True) if "shopee" in p)
print("data:", csv_path)
df = pd.read_csv(csv_path)
titles = [decode_title(t) for t in df["title"]]
print(f"{len(titles)} titles, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")

for name, model_id, prefix in MODELS:
    model = SentenceTransformer(model_id, device="cuda")
    if torch.cuda.is_available():
        model.half()
    t0 = time.perf_counter()
    vecs = model.encode([prefix + t for t in titles], batch_size=128, normalize_embeddings=True,
                        convert_to_numpy=True, show_progress_bar=False)
    gpu_s = time.perf_counter() - t0

    # Latency a CPU-only server would see: one title at a time, 100 samples.
    cpu_model = SentenceTransformer(model_id, device="cpu")
    sample = np.random.default_rng(0).choice(len(titles), 100, replace=False)
    cpu_model.encode([prefix + titles[0]])  # warm-up
    times = []
    for i in sample:
        t = time.perf_counter()
        cpu_model.encode([prefix + titles[i]], normalize_embeddings=True)
        times.append(time.perf_counter() - t)

    np.savez(f"/kaggle/working/emb_{name}.npz", posting_id=df["posting_id"].to_numpy().astype(str),
             vectors=vecs.astype(np.float16))
    meta = {"name": name, "model": model_id, "prefix": prefix, "dim": int(vecs.shape[1]),
            "n": len(titles), "gpu_encode_s": round(gpu_s, 1),
            "cpu_query_ms_p50": round(float(np.percentile(times, 50)) * 1000, 1),
            "cpu_query_ms_p95": round(float(np.percentile(times, 95)) * 1000, 1)}
    with open(f"/kaggle/working/emb_{name}.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(meta, flush=True)
    del model, cpu_model
    torch.cuda.empty_cache()

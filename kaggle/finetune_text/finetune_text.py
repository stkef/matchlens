"""Kaggle notebook: fine-tune a multilingual text embedding model on the train split (rung 7a).

Two variants of intfloat/multilingual-e5-base, both trained with MultipleNegativesRankingLoss
(every other example in the batch is a negative):

- simcse      no labels: each train title paired with itself; dropout noise makes the two views differ
              (SimCSE). Learns "a title is closest to itself" from raw text alone.
- supervised  labels + hard negatives: (anchor, another listing of the same product, a look-alike the
              rung 5 pipeline confused it with — mined by `python -m matchlens.mine_negatives`).

Only train-split listings are used for training (split file from the GitHub repo). Afterwards every
title (all splits) is embedded so the model can be evaluated locally like rung 3.

Outputs: emb_e5ft_<variant>.npz (posting_id + float16 normalised vectors) and emb_e5ft_<variant>.json.
"""

import glob
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                   SentenceTransformerTrainingArguments, losses)
from sentence_transformers.training_args import BatchSamplers

REPO = "/kaggle/temp/matchlens"
subprocess.run(["git", "clone", "--depth", "1", "https://github.com/stkef/matchlens.git", REPO], check=True)
sys.path.insert(0, REPO)
from matchlens.text import decode_title  # noqa: E402

OUT, BASE, PREFIX, SEED = "/kaggle/working", "intfloat/multilingual-e5-base", "query: ", 42
csv_path = next(p for p in glob.glob("/kaggle/input/**/train.csv", recursive=True) if "shopee" in p)
neg_path = next(iter(glob.glob("/kaggle/input/**/hard_negatives.csv", recursive=True)))

df = pd.read_csv(csv_path)
df["text"] = PREFIX + df["title"].map(decode_title)
split = pd.read_csv(os.path.join(REPO, "data/splits/split_v1.csv"))
train = df.merge(split, on="posting_id")
train = train[train["split"] == "train"].reset_index(drop=True)
text = train.set_index("posting_id")["text"]
negatives = pd.read_csv(neg_path).groupby("query")["candidate"].apply(list).to_dict()
rng = np.random.default_rng(SEED)
print(f"{len(train)} train listings, {train['label_group'].nunique()} products, "
      f"GPU {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}", flush=True)


def supervised_examples() -> Dataset:
    rows = {"anchor": [], "positive": [], "negative": []}
    for _, group in train.groupby("label_group"):
        ids = group["posting_id"].tolist()
        for a in ids:
            pos = rng.choice([x for x in ids if x != a])
            negs = negatives.get(a) or [rng.choice(train["posting_id"])]
            rows["anchor"].append(text[a])
            rows["positive"].append(text[pos])
            rows["negative"].append(text[rng.choice(negs)])
    return Dataset.from_dict(rows).shuffle(seed=SEED)


def simcse_examples() -> Dataset:
    t = train["text"].tolist()
    return Dataset.from_dict({"anchor": t, "positive": t}).shuffle(seed=SEED)


VARIANTS = [
    # name,         examples,             epochs
    ("simcse",      simcse_examples,      1),
    ("supervised",  supervised_examples,  2),
]

for name, make, epochs in VARIANTS:
    model = SentenceTransformer(BASE, device="cuda")
    model.max_seq_length = 64
    data = make()
    args = SentenceTransformerTrainingArguments(
        output_dir=f"/kaggle/temp/{name}", num_train_epochs=epochs, per_device_train_batch_size=64,
        learning_rate=2e-5, warmup_ratio=0.1, fp16=True, batch_sampler=BatchSamplers.NO_DUPLICATES,
        logging_steps=100, save_strategy="no", report_to="none", seed=SEED)
    t0 = time.perf_counter()
    SentenceTransformerTrainer(model=model, args=args, train_dataset=data,
                               loss=losses.MultipleNegativesRankingLoss(model)).train()
    train_s = time.perf_counter() - t0
    vecs = model.encode(df["text"].tolist(), batch_size=256, normalize_embeddings=True, convert_to_numpy=True)
    np.savez(f"{OUT}/emb_e5ft_{name}.npz", posting_id=df["posting_id"].to_numpy().astype(str),
             vectors=vecs.astype(np.float16))
    meta = {"name": f"e5ft_{name}", "base": BASE, "prefix": PREFIX, "dim": int(vecs.shape[1]), "n": len(vecs),
            "train_examples": len(data), "epochs": epochs, "train_s": round(train_s, 1)}
    with open(f"{OUT}/emb_e5ft_{name}.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(meta, flush=True)
    del model
    torch.cuda.empty_cache()

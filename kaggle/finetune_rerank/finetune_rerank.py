"""Kaggle notebook: fine-tune a small multilingual cross-encoder on our own pairs (rung 7b).

Starts from cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 (118M, trained for search relevance; untrained
it gave no gain in rung 6) and teaches it "same product or not" on the train pairs of fold A: the
top-20 candidates rung 7a proposes for train listings, labelled from label_group. Fold B (other
products) and val are only scored, never trained on, so the local judge fitted on fold B scores is
honest.

Inputs:  private dataset matchlens-rerank-pairs-7b (pairs_train.csv with fold, pairs_val.csv; ids only).
Outputs: rerank_minilm_ft_train.csv (fold B scores), rerank_minilm_ft_val.csv, rerank_minilm_ft.json.
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
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

subprocess.run(["git", "clone", "--depth", "1", "https://github.com/stkef/matchlens.git", "/kaggle/temp/matchlens"],
               check=True)
sys.path.insert(0, "/kaggle/temp/matchlens")
from matchlens.text import decode_title  # noqa: E402

OUT, BASE, SEED = "/kaggle/working", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", 42
EPOCHS, BATCH, LR, MAX_LEN = 1, 64, 2e-5, 96
torch.manual_seed(SEED)
csv_path = next(p for p in glob.glob("/kaggle/input/**/train.csv", recursive=True) if "shopee" in p)
pairs_dir = os.path.dirname(next(iter(glob.glob("/kaggle/input/**/pairs_val.csv", recursive=True))))
listings = pd.read_csv(csv_path).set_index("posting_id")
titles, groups = listings["title"].map(decode_title), listings["label_group"]
device = "cuda"

train = pd.read_csv(os.path.join(pairs_dir, "pairs_train.csv"))
fold_a, fold_b = train[train["fold"] == "A"], train[train["fold"] == "B"]
labels = (groups[fold_a["query"]].to_numpy() == groups[fold_a["candidate"]].to_numpy()).astype(np.float32)
print(f"training on {len(fold_a)} fold-A pairs ({labels.mean():.1%} positive); scoring {len(fold_b)} fold-B pairs",
      flush=True)

tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForSequenceClassification.from_pretrained(BASE).to(device)


def batches(frame, size, shuffle=False, y=None):
    order = np.random.default_rng(SEED).permutation(len(frame)) if shuffle else np.arange(len(frame))
    for s in range(0, len(order), size):
        rows = frame.iloc[order[s:s + size]]
        enc = tok(titles[rows["query"]].tolist(), titles[rows["candidate"]].tolist(), padding=True,
                  truncation=True, max_length=MAX_LEN, return_tensors="pt").to(device)
        yield enc, (torch.tensor(y[order[s:s + size]], device=device) if y is not None else None)


def score(frame):
    model.eval()
    out = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for enc, _ in batches(frame, 256):
            out.append(model(**enc).logits[:, 0].float().cpu().numpy())
    return np.concatenate(out)


steps = EPOCHS * ((len(fold_a) + BATCH - 1) // BATCH)
opt = torch.optim.AdamW(model.parameters(), lr=LR)
sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
scaler = torch.cuda.amp.GradScaler()
loss_fn = torch.nn.BCEWithLogitsLoss()
t0 = time.perf_counter()
for epoch in range(EPOCHS):
    model.train()
    for step, (enc, y) in enumerate(batches(fold_a, BATCH, shuffle=True, y=labels)):
        with torch.autocast("cuda", dtype=torch.float16):
            loss = loss_fn(model(**enc).logits[:, 0].float(), y)
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        if step % 500 == 0:
            print(f"epoch {epoch} step {step}/{steps} loss {loss.item():.4f}", flush=True)
train_s = time.perf_counter() - t0

meta = {"name": "minilm_ft", "base": BASE, "epochs": EPOCHS, "train_pairs": len(fold_a), "train_s": round(train_s, 1)}
for name, frame in [("train", fold_b), ("val", pd.read_csv(os.path.join(pairs_dir, "pairs_val.csv")))]:
    t = time.perf_counter()
    frame = frame[["query", "candidate"]].copy()
    frame["score"] = score(frame)
    meta[f"gpu_s_{name}"] = round(time.perf_counter() - t, 1)
    frame.to_csv(f"{OUT}/rerank_minilm_ft_{name}.csv", index=False)
with open(f"{OUT}/rerank_minilm_ft.json", "w") as f:
    json.dump(meta, f, indent=2)
print(meta, flush=True)

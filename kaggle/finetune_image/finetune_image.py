"""Kaggle notebook: fine-tune the Marqo e-commerce image model on our products (rung 7c).

Starts from Marqo/marqo-ecommerce-embeddings-B (rung 4 winner, open_clip ViT-B/16) and trains its image
tower only, contrastively: for each anchor photo, another listing's photo of the same product is the
positive; a look-alike the pipeline confused it with (hard negative) plus every other photo in the batch
are negatives. Trained on the products of fold A only (data/splits/train_folds_v1.csv), so the local
judge can be fitted on fold B without seeing scores on products the model trained on.

Outputs: emb_img_marqo_ft.npz (all 34,250 photos, float16, normalised) and emb_img_marqo_ft.json.
"""

import glob
import json
import os
import subprocess
import sys
import time

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "open_clip_torch"], check=True)

import numpy as np  # noqa: E402
import open_clip  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

REPO = "/kaggle/temp/matchlens"
subprocess.run(["git", "clone", "--depth", "1", "https://github.com/stkef/matchlens.git", REPO], check=True)

OUT, HUB, SEED = "/kaggle/working", "hf-hub:Marqo/marqo-ecommerce-embeddings-B", 42
EPOCHS, BATCH, LR, TEMPERATURE = 2, 32, 1e-5, 0.05
rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)

csv_path = next(p for p in glob.glob("/kaggle/input/**/train.csv", recursive=True) if "shopee" in p)
img_dir = os.path.join(os.path.dirname(csv_path), "train_images")
neg_path = next(iter(glob.glob("/kaggle/input/**/hard_negatives.csv", recursive=True)))
df = pd.read_csv(csv_path)
path = dict(zip(df["posting_id"], [os.path.join(img_dir, f) for f in df["image"]]))
folds = pd.read_csv(os.path.join(REPO, "data/splits/train_folds_v1.csv"))
fold_a = df.merge(folds[folds["fold"] == "A"], on="posting_id")
in_a = set(fold_a["posting_id"])
negatives = (pd.read_csv(neg_path).query("query in @in_a and candidate in @in_a")
             .groupby("query")["candidate"].apply(list).to_dict())
print(f"fold A: {len(fold_a)} listings, {fold_a['label_group'].nunique()} products; "
      f"{sum(map(len, negatives.values()))} hard negatives inside fold A", flush=True)


def triplets():
    rows, a_ids = [], fold_a["posting_id"].to_numpy()
    for _, group in fold_a.groupby("label_group"):
        ids = group["posting_id"].tolist()
        for a in ids:
            pos = rng.choice([x for x in ids if x != a])
            neg = rng.choice(negatives.get(a) or [x for x in rng.choice(a_ids, 5) if x not in ids] or [a_ids[0]])
            rows.append((path[a], path[pos], path[neg]))
    rng.shuffle(rows)
    return rows


class Triplets(Dataset):
    def __init__(self, rows, transform):
        self.rows, self.transform = rows, transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return tuple(self.transform(Image.open(p).convert("RGB")) for p in self.rows[i])


class Plain(Dataset):
    def __init__(self, files, transform):
        self.files, self.transform = files, transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        return self.transform(Image.open(self.files[i]).convert("RGB"))


model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(HUB)
model = model.cuda()
for p in model.parameters():
    p.requires_grad = False
for p in model.visual.parameters():
    p.requires_grad = True
opt = torch.optim.AdamW(model.visual.parameters(), lr=LR, weight_decay=0.05)
steps_per_epoch = len(fold_a) // BATCH
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=EPOCHS * steps_per_epoch, pct_start=0.05)
scaler = torch.cuda.amp.GradScaler()

t0 = time.perf_counter()
for epoch in range(EPOCHS):
    model.train()
    loader = DataLoader(Triplets(triplets(), preprocess_train), batch_size=BATCH, shuffle=False, num_workers=4,
                        drop_last=True)
    for step, (a, p, n) in enumerate(loader):
        with torch.autocast("cuda", dtype=torch.float16):
            emb = F.normalize(model.encode_image(torch.cat([a, p, n]).cuda()).float(), dim=-1)
        ea, ep, en = emb.split(len(a))
        # Each anchor must pick its own positive among all positives and hard negatives in the batch.
        logits = ea @ torch.cat([ep, en]).T / TEMPERATURE
        loss = F.cross_entropy(logits, torch.arange(len(a), device="cuda"))
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        if step % 100 == 0:
            print(f"epoch {epoch} step {step}/{steps_per_epoch} loss {loss.item():.4f}", flush=True)
train_s = time.perf_counter() - t0

model.eval()
parts = []
with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
    for x in DataLoader(Plain([path[p] for p in df["posting_id"]], preprocess_val), batch_size=256, num_workers=4):
        parts.append(F.normalize(model.encode_image(x.cuda()).float(), dim=-1).cpu().numpy())
vecs = np.concatenate(parts)
np.savez(f"{OUT}/emb_img_marqo_ft.npz", posting_id=df["posting_id"].to_numpy().astype(str),
         vectors=vecs.astype(np.float16))
meta = {"name": "img_marqo_ft", "base": HUB, "trained_on": "fold A products of the train split",
        "train_listings": len(fold_a), "epochs": EPOCHS, "batch": BATCH, "lr": LR, "temperature": TEMPERATURE,
        "dim": int(vecs.shape[1]), "n": len(vecs), "train_s": round(train_s, 1)}
with open(f"{OUT}/emb_img_marqo_ft.json", "w") as f:
    json.dump(meta, f, indent=2)
print(meta, flush=True)

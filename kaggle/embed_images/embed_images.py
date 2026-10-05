"""Kaggle notebook: embed every Shopee photo with several image models (rung 4).

Runs on Kaggle's free GPU next to the competition images (~1.7 GB, never downloaded locally).
Writes one file per model to /kaggle/working, fetched afterwards with `kaggle kernels output`:

    emb_img_<name>.npz   posting_id (str) + vectors (float16, L2-normalised)
    emb_img_<name>.json  model id, dimension, GPU time, CPU latency for one image, or the error

A model that fails (e.g. DINOv3 without Hugging Face access) is logged and skipped; the rest still run.
Pushed and fetched as described in docs/kaggle.md; not run locally.
"""

import glob
import json
import os
import time
import traceback

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

OUT = "/kaggle/working"
csv_path = next(p for p in glob.glob("/kaggle/input/**/train.csv", recursive=True) if "shopee" in p)
img_dir = os.path.join(os.path.dirname(csv_path), "train_images")
df = pd.read_csv(csv_path)
paths = [os.path.join(img_dir, f) for f in df["image"]]
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"{len(paths)} images from {img_dir}, device {device}", flush=True)

# Optional Hugging Face token (Kaggle secret "HF_TOKEN") for gated models such as DINOv3.
HF_TOKEN = None
try:
    from kaggle_secrets import UserSecretsClient
    HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
except Exception:
    print("no HF_TOKEN secret: gated models will be skipped", flush=True)


class Images(Dataset):
    def __init__(self, files, processor):
        self.files, self.processor = files, processor

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = Image.open(self.files[i]).convert("RGB")
        return self.processor(images=img, return_tensors="pt")["pixel_values"][0]


def embed(model, pixels):
    """One batch -> vectors, on whatever device/dtype the model currently has."""
    p = next(model.parameters())
    pixels = pixels.to(p.device, p.dtype)
    if hasattr(model, "get_image_features"):          # CLIP / SigLIP: projected image embedding
        out = model.get_image_features(pixel_values=pixels)
        return out if torch.is_tensor(out) else out.pooler_output  # newer transformers wrap it
    return model(pixel_values=pixels).pooler_output   # DINOv2/v3 (normed CLS), Swin V2 (avg pool)


def hf_backbone(model_id):
    """AutoModel image backbones: DINOv2, DINOv3, Swin V2."""
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(model_id, token=HF_TOKEN)
    return processor, AutoModel.from_pretrained(model_id, token=HF_TOKEN, torch_dtype=torch.float16)


def clip_like(model_id):
    """CLIP / SigLIP: only the image tower is used."""
    from transformers import AutoModel, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id).image_processor
    return processor, AutoModel.from_pretrained(model_id, torch_dtype=torch.float16)


MODELS = [
    # name,        loader,                                                           batch
    ("clip_b32",   lambda: clip_like("openai/clip-vit-base-patch32"),                 256),
    ("siglip_b16", lambda: clip_like("google/siglip-base-patch16-224"),               256),
    ("dinov2_b",   lambda: hf_backbone("facebook/dinov2-base"),                      128),
    ("swinv2_b",   lambda: hf_backbone("microsoft/swinv2-base-patch4-window8-256"),  128),
    ("dinov3_b",   lambda: hf_backbone("facebook/dinov3-vitb16-pretrain-lvd1689m"),  128),
]


def run_hf(loader, batch):
    processor, model = loader()
    model = model.to(device).eval()
    t0 = time.perf_counter()
    parts = []
    with torch.no_grad():
        for pixels in DataLoader(Images(paths, processor), batch_size=batch, num_workers=4):
            parts.append(embed(model, pixels).float().cpu().numpy())
    gpu_s = time.perf_counter() - t0
    # One image at a time on CPU (float32), as a server without a GPU would see it: 20 samples.
    model = model.float().to("cpu")
    times = []
    with torch.no_grad():
        for f in paths[:21]:
            t = time.perf_counter()
            embed(model, processor(images=Image.open(f).convert("RGB"), return_tensors="pt")["pixel_values"])
            times.append(time.perf_counter() - t)
    del model
    torch.cuda.empty_cache()
    return np.concatenate(parts), gpu_s, times[1:]  # first call is warm-up


def run_jina():
    """Jina Embeddings v4 (3.8B, images and text in one space). Heavy: runs last."""
    from transformers import AutoModel

    model = AutoModel.from_pretrained("jinaai/jina-embeddings-v4", trust_remote_code=True,
                                      torch_dtype=torch.float16).to(device).eval()
    t0 = time.perf_counter()
    parts = []
    for s in range(0, len(paths), 32):
        imgs = [Image.open(p).convert("RGB") for p in paths[s:s + 32]]
        out = model.encode_image(images=imgs, task="retrieval")
        parts.append(torch.stack(list(out)).float().cpu().numpy() if isinstance(out, (list, tuple))
                     else out.float().cpu().numpy())
    gpu_s = time.perf_counter() - t0
    del model
    torch.cuda.empty_cache()
    return np.concatenate(parts), gpu_s, []


# Set to a list of names (e.g. ["dinov3_b"]) to re-run only those models; None runs everything.
RUN_ONLY = None

jobs = [(name, (lambda l=loader, b=batch: run_hf(l, b))) for name, loader, batch in MODELS]
jobs.append(("jina_v4", run_jina))
jobs = [(name, job) for name, job in jobs if RUN_ONLY is None or name in RUN_ONLY]

for name, job in jobs:
    meta = {"name": name}
    try:
        vecs, gpu_s, times = job()
        vecs /= np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12)
        np.savez(f"{OUT}/emb_img_{name}.npz", posting_id=df["posting_id"].to_numpy().astype(str),
                 vectors=vecs.astype(np.float16))
        meta.update({"dim": int(vecs.shape[1]), "n": len(vecs), "gpu_encode_s": round(gpu_s, 1),
                     "cpu_image_ms_p50": round(float(np.percentile(times, 50)) * 1000, 1) if times else None,
                     "cpu_image_ms_p95": round(float(np.percentile(times, 95)) * 1000, 1) if times else None})
    except Exception as e:
        meta["error"] = f"{type(e).__name__}: {e}"
        traceback.print_exc()
    with open(f"{OUT}/emb_img_{name}.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(meta, flush=True)

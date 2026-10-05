"""A small fake dataset with the Shopee schema, for tests and smoke runs before the real data is downloaded.

It deliberately contains look-alikes (same model, different storage/colour/size) and seller noise in
Indonesian and English, so the baseline is not trivially perfect. Numbers on it mean nothing; it exists
to prove the pipeline runs end to end.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BRANDS = ["Samsung", "Xiaomi", "Oppo", "Vivo", "Realme", "Wardah", "Emina", "Philips", "Miyako", "Cosrx"]
PRODUCTS = {
    "phone": (["Galaxy A{n}", "Redmi Note {n}", "Reno {n}", "Y{n}", "C{n}"], ["64GB", "128GB", "256GB"]),
    "cosmetic": (["Lip Cream {n}", "Serum Vitamin C {n}", "Sunscreen SPF{n}", "Toner {n}"], ["30ml", "50ml", "100ml"]),
    "appliance": (["Rice Cooker MCM{n}", "Blender BL{n}", "Setrika HD{n}", "Kipas Angin KAS{n}"], ["1L", "1.8L", "2L"]),
}
COLOURS = ["hitam", "putih", "biru", "merah", "pink", "black", "white"]
NOISE = ["ORIGINAL", "100% Ori", "Garansi Resmi", "COD", "Promo", "Murah", "READY STOCK", "Termurah",
         "Best Seller", "Free Ongkir", "BPOM", "Bisa COD", "[NEW]", "Grosir"]


def _listing_title(rng, brand, model, variant, colour) -> str:
    parts = [brand, model, variant, colour]
    if rng.random() < 0.3:
        parts.remove(brand)
    if rng.random() < 0.25:
        parts.remove(colour)
    if rng.random() < 0.3:
        variant_spaced = variant[:-2] + " " + variant[-2:] if variant[-2:].isalpha() else variant
        parts[parts.index(variant)] = variant_spaced
    parts += list(rng.choice(NOISE, size=rng.integers(0, 4), replace=False))
    rng.shuffle(parts)
    title = " ".join(parts)
    if rng.random() < 0.3:
        title = title.upper()
    elif rng.random() < 0.3:
        title = title.lower()
    return title


def make_synthetic(n_models: int = 150, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    group_id = 1000
    for m in range(n_models):
        kind = rng.choice(list(PRODUCTS))
        templates, variants = PRODUCTS[kind]
        brand = rng.choice(BRANDS)
        model = rng.choice(templates).format(n=int(rng.integers(1, 90)))
        # Each model exists in 1-3 variants: these are the look-alikes that must NOT match each other.
        for variant in rng.choice(variants, size=rng.integers(1, 4), replace=False):
            colour = rng.choice(COLOURS)
            group_id += 1
            phash = "".join(rng.choice(list("0123456789abcdef"), size=16))
            for _ in range(int(rng.integers(2, 7))):
                rows.append({
                    "posting_id": f"train_{len(rows):08d}",
                    "image": f"{rng.bytes(8).hex()}.jpg",
                    "image_phash": phash if rng.random() < 0.6 else "".join(rng.choice(list("0123456789abcdef"), size=16)),
                    "title": _listing_title(rng, brand, model, variant, colour),
                    "label_group": group_id,
                })
    return pd.DataFrame(rows)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Write a synthetic Shopee-format CSV.")
    p.add_argument("--out", default="data/synthetic/train.csv")
    p.add_argument("--n-models", type=int, default=150)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = make_synthetic(args.n_models, args.seed)
    df.to_csv(out, index=False)
    print(f"wrote {len(df)} listings in {df['label_group'].nunique()} groups to {out}")


if __name__ == "__main__":
    main()

"""
04_split.py
-----------
Computes 10 hand-crafted spatial features for every country with a processed
.npy array, saves them to data/features.csv, then produces a reproducible
stratified 70/15/15 train/val/test split (seed=42) saved to data/splits.json.

Features extracted (mirrors 03_eda.py):
  mean_brightness, total_light, p90_brightness,
  lit_fraction, light_density,
  spatial_dispersion, hotspot_count,
  gini, entropy, urban_concentration

Requirements:
    pip install numpy pandas scikit-learn scipy

Usage:
    python scripts/04_split.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import label as ndlabel
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT        = Path(__file__).resolve().parent.parent
PROC_DIR    = ROOT / "data" / "processed"
LABELS_PATH = ROOT / "data" / "labels.csv"
FEAT_PATH   = ROOT / "data" / "features.csv"
SPLIT_PATH  = ROOT / "data" / "splits.json"

if not LABELS_PATH.exists():
    sys.exit(f"ERROR: {LABELS_PATH} not found. Run 01_download_images.py first.")

SEED       = 42
LIT_THRESH = 0.01

# ---------------------------------------------------------------------------
# Metric functions (same as 03_eda.py)
# ---------------------------------------------------------------------------

def gini(arr):
    flat = arr.flatten()
    flat = flat[flat > 0]
    if flat.size < 2:
        return 0.0
    flat = np.sort(flat)
    n    = len(flat)
    idx  = np.arange(1, n + 1)
    return float((2 * (idx * flat).sum()) / (n * flat.sum()) - (n + 1) / n)


def shannon_entropy(arr, bins=50):
    flat = arr.flatten()
    flat = flat[flat > LIT_THRESH]
    if flat.size < 2:
        return 0.0
    counts, _ = np.histogram(flat, bins=bins, range=(0.0, 1.0))
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def urban_concentration(arr, top_pct=0.05):
    flat = arr.flatten()
    flat = flat[flat > LIT_THRESH]
    if flat.size < 2:
        return 0.0
    threshold = np.percentile(flat, (1 - top_pct) * 100)
    return float(flat[flat >= threshold].sum() / flat.sum())


def spatial_dispersion(arr):
    rows, cols = np.where(arr > LIT_THRESH)
    if len(rows) < 2:
        return 0.0
    h, w = arr.shape
    return float(np.sqrt(np.var(rows / h) + np.var(cols / w)))


def hotspot_count(arr, threshold=0.1):
    binary = (arr >= threshold).astype(int)
    _, n   = ndlabel(binary)
    return int(n)


# ---------------------------------------------------------------------------
# Compute features
# ---------------------------------------------------------------------------

labels = pd.read_csv(LABELS_PATH)
records = []

print("Computing features …")
for _, row in labels.iterrows():
    iso3 = row["iso3"]
    tier = row["income_tier"]
    npy  = PROC_DIR / f"{iso3}_2022.npy"
    if not npy.exists():
        continue

    arr  = np.load(npy)
    flat = arr.flatten()
    lit  = flat[flat > LIT_THRESH]
    n_lit = lit.size

    records.append({
        "iso3":               iso3,
        "country_name":       row.get("country_name", iso3),
        "income_tier":        tier,
        "mean_brightness":    float(arr.mean()),
        "total_light":        float(flat.sum()),
        "p90_brightness":     float(np.percentile(flat, 90)),
        "lit_fraction":       float(n_lit / flat.size),
        "light_density":      float(lit.mean()) if n_lit > 0 else 0.0,
        "spatial_dispersion": spatial_dispersion(arr),
        "hotspot_count":      hotspot_count(arr),
        "gini":               gini(arr),
        "entropy":            shannon_entropy(arr),
        "urban_concentration": urban_concentration(arr),
    })

feat_df = pd.DataFrame(records)
feat_df.to_csv(FEAT_PATH, index=False)
print(f"  Saved {len(feat_df)} rows → {FEAT_PATH}\n")

# ---------------------------------------------------------------------------
# Stratified 70 / 15 / 15 split
# ---------------------------------------------------------------------------

iso3s  = feat_df["iso3"].tolist()
tiers  = feat_df["income_tier"].tolist()

# Split off test (15%) first, then split remaining into train/val (≈82/18 of 85%)
train_val_iso, test_iso, train_val_tier, _ = train_test_split(
    iso3s, tiers, test_size=0.15, stratify=tiers, random_state=SEED
)
train_iso, val_iso = train_test_split(
    train_val_iso, test_size=0.1765,   # 15/85 ≈ 0.1765 → gives ~15% of total
    stratify=train_val_tier, random_state=SEED
)

splits = {"train": sorted(train_iso), "val": sorted(val_iso), "test": sorted(test_iso)}

with open(SPLIT_PATH, "w") as f:
    json.dump(splits, f, indent=2)

print(f"Splits saved → {SPLIT_PATH}")
print(f"  Train : {len(splits['train'])} countries")
print(f"  Val   : {len(splits['val'])}  countries")
print(f"  Test  : {len(splits['test'])}  countries")

# Per-tier breakdown
tier_map = dict(zip(iso3s, tiers))
for split_name, iso_list in splits.items():
    counts = pd.Series([tier_map[c] for c in iso_list]).value_counts()
    print(f"\n  {split_name.upper()} tier counts:")
    for t in ["Low", "Lower-Middle", "Upper-Middle", "High"]:
        print(f"    {t:<16} {counts.get(t, 0)}")

print("\nNext: python scripts/05_train.py")

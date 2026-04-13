"""
02_preprocess.py
----------------
Reads each country GeoTIFF from data/raw/, resizes to 64×64, normalises
pixel values to [0, 1], and saves as a .npy file to data/processed/.

Processes all countries listed in data/labels.csv that have a matching
.tif file. Skips countries already cached as .npy.

Also prints summary statistics (min / max / mean NTL value) per income tier.

Requirements:
    pip install rasterio numpy pandas

Usage:
    python scripts/02_preprocess.py
"""

import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT       = Path(__file__).resolve().parent.parent
RAW_DIR    = ROOT / "data" / "raw"
PROC_DIR   = ROOT / "data" / "processed"
LABELS_PATH = ROOT / "data" / "labels.csv"

PROC_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SIZE = 64   # output resolution (pixels per side)

# ---------------------------------------------------------------------------
# Load labels
# ---------------------------------------------------------------------------

if not LABELS_PATH.exists():
    sys.exit(f"ERROR: {LABELS_PATH} not found. Run 01_download_images.py first.")

labels = pd.read_csv(LABELS_PATH).set_index("iso3")

# ---------------------------------------------------------------------------
# Preprocess loop
# ---------------------------------------------------------------------------

tier_stats = defaultdict(list)   # tier → list of mean raw NTL values
skipped, failed = [], []

iso3_list = labels.index.tolist()
total = len(iso3_list)

print(f"Preprocessing {total} countries → {TARGET_SIZE}×{TARGET_SIZE} .npy arrays\n")

for idx, iso3 in enumerate(iso3_list, start=1):
    tif_path = RAW_DIR / f"{iso3}_2022.tif"
    npy_path = PROC_DIR / f"{iso3}_2022.npy"
    tier     = labels.loc[iso3, "income_tier"]

    if npy_path.exists():
        # Load cached to gather stats without reprocessing
        arr = np.load(npy_path)
        valid = arr[arr >= 0]
        if valid.size > 0:
            tier_stats[tier].append(valid.mean())
        print(f"[{idx:2d}/{total}] {iso3} ({tier}) — cached")
        continue

    if not tif_path.exists():
        print(f"[{idx:2d}/{total}] {iso3} ({tier}) — WARNING: .tif not found, skipping")
        skipped.append(iso3)
        continue

    try:
        with rasterio.open(tif_path) as src:
            # Read at target resolution using rasterio's built-in resampling
            data = src.read(
                1,
                out_shape=(TARGET_SIZE, TARGET_SIZE),
                resampling=Resampling.average,
            )
            nodata = src.nodata   # np.nan for our files

        # Replace nodata (NaN) with 0 (no light)
        data = np.where(np.isnan(data), 0.0, data)
        # Clip negative artefacts
        data = np.clip(data, 0.0, None)

        raw_mean = data[data > 0].mean() if (data > 0).any() else 0.0

        # Normalise to [0, 1] using 99th-percentile as upper bound to avoid
        # single bright pixels dominating (e.g., oil flares).
        p99 = np.percentile(data[data > 0], 99) if (data > 0).any() else 1.0
        p99 = max(p99, 1e-6)    # guard against all-zero images
        norm = np.clip(data / p99, 0.0, 1.0).astype(np.float32)

        np.save(npy_path, norm)
        tier_stats[tier].append(raw_mean)

        print(f"[{idx:2d}/{total}] {iso3} ({tier}) — "
              f"shape {norm.shape}  raw_mean={raw_mean:.4f}  "
              f"p99={p99:.4f}  npy={npy_path.stat().st_size//1024} KB")

    except Exception as e:
        print(f"[{idx:2d}/{total}] {iso3} ({tier}) — ERROR: {e}")
        failed.append(iso3)

# ---------------------------------------------------------------------------
# Summary statistics per tier
# ---------------------------------------------------------------------------

TIER_ORDER = ["Low", "Lower-Middle", "Upper-Middle", "High"]

print("\n" + "=" * 60)
print(f"Processed: {total - len(skipped) - len(failed)} / {total}")
if skipped:
    print(f"Skipped (no .tif): {skipped}")
if failed:
    print(f"Failed:            {failed}")

print("\nMean raw NTL radiance (nW/cm²/sr) by income tier:")
print(f"  {'Tier':<16}  {'N':>4}  {'Min':>8}  {'Mean':>8}  {'Max':>8}")
print("  " + "-" * 48)
for tier in TIER_ORDER:
    vals = tier_stats.get(tier, [])
    if vals:
        print(f"  {tier:<16}  {len(vals):>4}  "
              f"{min(vals):>8.4f}  {np.mean(vals):>8.4f}  {max(vals):>8.4f}")
    else:
        print(f"  {tier:<16}  {'N/A':>4}")

print("=" * 60)
print("Next: python scripts/03_eda.py")

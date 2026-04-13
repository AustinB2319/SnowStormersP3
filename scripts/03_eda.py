"""
03_eda.py
---------
Exploratory data analysis on the processed 64×64 nighttime light arrays.

Computes 12 per-country metrics across 4 categories:

  Intensity
    mean_brightness      — mean normalised pixel value across whole image
    total_light          — sum of all pixel values (total radiance proxy)
    p90_brightness       — 90th-percentile pixel value (bright-area floor)

  Density & Coverage
    lit_fraction         — share of pixels above threshold (> 0.01)
    light_density        — mean value of lit pixels only (intensity of lit areas)

  Spatial Distribution
    spatial_dispersion   — std of lit-pixel positions (how spread out light is)
    center_of_mass_x     — horizontal centre of light mass (0=left, 1=right)
    center_of_mass_y     — vertical centre of light mass (0=top, 1=bottom)
    hotspot_count        — number of distinct bright clusters (connected components)

  Inequality / Concentration
    gini                 — Gini coefficient of pixel values
    entropy              — Shannon entropy of pixel distribution (bits)
    urban_concentration  — share of total light in top-5% brightest pixels

Produces figures saved to figures/:
  1. sample_images_by_tier.png
  2. metric_boxplots.png          — all 12 metrics, box+strip by tier
  3. correlation_heatmap.png      — Spearman ρ matrix vs tier rank
  4. pixel_intensity_distributions.png
  5. spatial_scatter.png          — center-of-mass scatter per country
  6. top10_countries.png          — top-10 countries per metric per tier
  7. metric_pairplot.png          — pairplot of key metrics coloured by tier

Requirements:
    pip install numpy pandas matplotlib seaborn scipy scikit-image

Usage:
    python scripts/03_eda.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr
from scipy.ndimage import label as ndlabel

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT        = Path(__file__).resolve().parent.parent
PROC_DIR    = ROOT / "data" / "processed"
LABELS_PATH = ROOT / "data" / "labels.csv"
FIG_DIR     = ROOT / "figures"

FIG_DIR.mkdir(parents=True, exist_ok=True)

if not LABELS_PATH.exists():
    sys.exit(f"ERROR: {LABELS_PATH} not found. Run 01_download_images.py first.")

labels = pd.read_csv(LABELS_PATH)

TIER_ORDER  = ["Low", "Lower-Middle", "Upper-Middle", "High"]
TIER_RANK   = {t: i for i, t in enumerate(TIER_ORDER)}
TIER_COLORS = {
    "Low":           "#d73027",
    "Lower-Middle":  "#fc8d59",
    "Upper-Middle":  "#91bfdb",
    "High":          "#4575b4",
}

LIT_THRESH = 0.01

# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------

def gini(arr: np.ndarray) -> float:
    flat = arr.flatten()
    flat = flat[flat > 0]
    if flat.size < 2:
        return 0.0
    flat = np.sort(flat)
    n = len(flat)
    idx = np.arange(1, n + 1)
    return float((2 * (idx * flat).sum()) / (n * flat.sum()) - (n + 1) / n)


def shannon_entropy(arr: np.ndarray, bins: int = 50) -> float:
    flat = arr.flatten()
    flat = flat[flat > LIT_THRESH]
    if flat.size < 2:
        return 0.0
    counts, _ = np.histogram(flat, bins=bins, range=(0.0, 1.0))
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def urban_concentration(arr: np.ndarray, top_pct: float = 0.05) -> float:
    flat = arr.flatten()
    flat = flat[flat > LIT_THRESH]
    if flat.size < 2:
        return 0.0
    threshold = np.percentile(flat, (1 - top_pct) * 100)
    return float(flat[flat >= threshold].sum() / flat.sum())


def spatial_dispersion(arr: np.ndarray) -> float:
    """Std of the (row, col) positions of lit pixels, normalised to [0,1]."""
    rows, cols = np.where(arr > LIT_THRESH)
    if len(rows) < 2:
        return 0.0
    h, w = arr.shape
    norm_rows = rows / h
    norm_cols = cols / w
    return float(np.sqrt(np.var(norm_rows) + np.var(norm_cols)))


def center_of_mass(arr: np.ndarray):
    """Light-weighted centre of mass, normalised to [0,1] in each axis."""
    rows, cols = np.where(arr > LIT_THRESH)
    if len(rows) == 0:
        return 0.5, 0.5
    h, w = arr.shape
    weights = arr[rows, cols]
    cx = float(np.average(cols / w, weights=weights))
    cy = float(np.average(rows / h, weights=weights))
    return cx, cy


def hotspot_count(arr: np.ndarray, threshold: float = 0.1) -> int:
    """Number of distinct bright clusters (connected components above threshold)."""
    binary = (arr >= threshold).astype(int)
    _, n = ndlabel(binary)
    return int(n)


# ---------------------------------------------------------------------------
# Load data & compute all metrics
# ---------------------------------------------------------------------------

records = []
images  = {tier: [] for tier in TIER_ORDER}
missing = []

print("Loading processed arrays and computing metrics …")
for _, row in labels.iterrows():
    iso3 = row["iso3"]
    tier = row["income_tier"]
    npy  = PROC_DIR / f"{iso3}_2022.npy"
    if not npy.exists():
        missing.append(iso3)
        continue

    arr  = np.load(npy)
    flat = arr.flatten()
    lit  = flat[flat > LIT_THRESH]
    n_lit   = lit.size
    n_total = flat.size

    cx, cy = center_of_mass(arr)

    records.append({
        "iso3":               iso3,
        "country_name":       row.get("country_name", iso3),
        "income_tier":        tier,
        "tier_rank":          TIER_RANK[tier],
        # intensity
        "mean_brightness":    float(arr.mean()),
        "total_light":        float(flat.sum()),
        "p90_brightness":     float(np.percentile(flat, 90)),
        # density & coverage
        "lit_fraction":       float(n_lit / n_total),
        "light_density":      float(lit.mean()) if n_lit > 0 else 0.0,
        # spatial distribution
        "spatial_dispersion": spatial_dispersion(arr),
        "center_of_mass_x":   cx,
        "center_of_mass_y":   cy,
        "hotspot_count":      hotspot_count(arr),
        # inequality
        "gini":               gini(arr),
        "entropy":            shannon_entropy(arr),
        "urban_concentration": urban_concentration(arr),
        # raw pixels for KDE
        "pixels":             flat,
    })
    images[tier].append((iso3, arr))

if missing:
    print(f"  WARNING: no .npy for {len(missing)} countries: {missing}")

METRICS = [
    "mean_brightness", "total_light", "p90_brightness",
    "lit_fraction", "light_density",
    "spatial_dispersion", "hotspot_count",
    "gini", "entropy", "urban_concentration",
]

METRIC_LABELS = {
    "mean_brightness":     "Mean Brightness\n[0–1]",
    "total_light":         "Total Light\n(sum of pixels)",
    "p90_brightness":      "P90 Brightness\n[0–1]",
    "lit_fraction":        "Lit Fraction\n(% pixels > 0.01)",
    "light_density":       "Light Density\n(mean of lit pixels)",
    "spatial_dispersion":  "Spatial Dispersion\n(spread of lit pixels)",
    "hotspot_count":       "Hotspot Count\n(bright clusters)",
    "gini":                "Gini Coefficient\n(spatial inequality)",
    "entropy":             "Shannon Entropy\n(bits)",
    "urban_concentration": "Urban Concentration\n(top-5% share)",
}

df = pd.DataFrame([{k: v for k, v in r.items() if k != "pixels"} for r in records])
print(f"  Loaded {len(df)} countries\n")

# ---------------------------------------------------------------------------
# Figure 1 — Sample images grid (4 per tier, sorted by mean brightness)
# ---------------------------------------------------------------------------

N_SAMPLES = 4
fig = plt.figure(figsize=(12, 10))
fig.suptitle(
    "Nighttime Light Intensity by World Bank Income Tier (2022)",
    fontsize=14, fontweight="bold", y=1.01
)
outer = gridspec.GridSpec(len(TIER_ORDER), 1, hspace=0.45)

for row_idx, tier in enumerate(TIER_ORDER):
    tier_imgs = sorted(images[tier], key=lambda x: x[1].mean(), reverse=True)
    selected  = tier_imgs[:N_SAMPLES]
    inner = gridspec.GridSpecFromSubplotSpec(1, N_SAMPLES, subplot_spec=outer[row_idx], wspace=0.1)
    for col_idx, (iso3, arr) in enumerate(selected):
        ax = fig.add_subplot(inner[col_idx])
        ax.imshow(arr, cmap="inferno", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(iso3, fontsize=8)
        ax.axis("off")
    fig.add_subplot(outer[row_idx]).set_visible(False)
    fig.text(
        0.01, 1 - (row_idx + 0.5) / len(TIER_ORDER),
        tier, va="center", fontsize=10, fontweight="bold",
        color=TIER_COLORS[tier], transform=fig.transFigure,
    )

fig.savefig(FIG_DIR / "sample_images_by_tier.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {FIG_DIR / 'sample_images_by_tier.png'}")

# ---------------------------------------------------------------------------
# Figure 2 — Box + strip plots for all 10 metrics (2 rows × 5 cols)
# ---------------------------------------------------------------------------

rng = np.random.default_rng(42)
fig, axes = plt.subplots(2, 5, figsize=(20, 9))
fig.suptitle("Nighttime Light Metrics by World Bank Income Tier (2022)",
             fontsize=14, fontweight="bold")

for ax, metric in zip(axes.flat, METRICS):
    for i, tier in enumerate(TIER_ORDER):
        vals = df[df["income_tier"] == tier][metric].values
        ax.boxplot(
            vals, positions=[i], widths=0.4,
            patch_artist=True,
            boxprops=dict(facecolor=TIER_COLORS[tier], alpha=0.6),
            medianprops=dict(color="black", linewidth=2),
            whiskerprops=dict(color="grey"),
            capprops=dict(color="grey"),
            flierprops=dict(marker="", linestyle="none"),
        )
        jitter = rng.uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(i + jitter, vals, color=TIER_COLORS[tier],
                   alpha=0.55, s=12, zorder=5, edgecolors="none")

    rho, pval = spearmanr(df[metric], df["tier_rank"])
    stars = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else "ns"))
    ax.set_title(f"{METRIC_LABELS[metric]}\nρ={rho:.3f} {stars}", fontsize=8)
    ax.set_xticks(range(len(TIER_ORDER)))
    ax.set_xticklabels([t.replace("-", "-\n") for t in TIER_ORDER], fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
fig.savefig(FIG_DIR / "metric_boxplots.png", dpi=150)
plt.close(fig)
print(f"Saved: {FIG_DIR / 'metric_boxplots.png'}")

# ---------------------------------------------------------------------------
# Figure 3 — Spearman correlation heatmap
# ---------------------------------------------------------------------------

corr_data = {}
for metric in METRICS:
    rho, pval = spearmanr(df[metric], df["tier_rank"])
    corr_data[metric] = {"rho": rho, "pval": pval}

corr_df = pd.DataFrame(corr_data).T

fig, ax = plt.subplots(figsize=(11, 3))
cmap = sns.diverging_palette(10, 240, as_cmap=True)
sns.heatmap(
    corr_df[["rho"]].T,
    annot=True, fmt=".3f",
    cmap=cmap, vmin=-1, vmax=1,
    linewidths=0.5, ax=ax,
    cbar_kws={"label": "Spearman ρ"},
)
ax.set_title("Spearman ρ: Each Metric vs World Bank Income Tier Rank", fontsize=11)
ax.set_yticklabels(["ρ"], rotation=0)
ax.set_xticklabels([m.replace("_", "\n") for m in METRICS], rotation=45, ha="right", fontsize=8)
fig.tight_layout()
fig.savefig(FIG_DIR / "correlation_heatmap.png", dpi=150)
plt.close(fig)
print(f"Saved: {FIG_DIR / 'correlation_heatmap.png'}")

# ---------------------------------------------------------------------------
# Figure 4 — Pixel intensity KDE
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 5))
for r in records:
    tier = r["income_tier"]
    px   = r["pixels"]
    rng2 = np.random.default_rng(0)
    sample = rng2.choice(px, size=min(2000, len(px)), replace=False)
    sample = sample[sample > 0.001]
    if sample.size > 10:
        sns.kdeplot(sample, ax=ax, color=TIER_COLORS[tier],
                    alpha=0.18, linewidth=0.8, bw_adjust=0.8, log_scale=True)

from matplotlib.lines import Line2D
handles = [Line2D([0], [0], color=TIER_COLORS[t], linewidth=2, label=t) for t in TIER_ORDER]
ax.legend(handles=handles, title="Income Tier", fontsize=10)
ax.set_xlabel("Normalised Pixel Intensity (log scale)", fontsize=12)
ax.set_ylabel("Density", fontsize=12)
ax.set_title("Pixel Intensity Distribution by Income Tier (2022)", fontsize=13, fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIG_DIR / "pixel_intensity_distributions.png", dpi=150)
plt.close(fig)
print(f"Saved: {FIG_DIR / 'pixel_intensity_distributions.png'}")

# ---------------------------------------------------------------------------
# Figure 5 — Spatial scatter: center of mass coloured by tier
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 6))
for tier in TIER_ORDER:
    sub = df[df["income_tier"] == tier]
    ax.scatter(
        sub["center_of_mass_x"], sub["center_of_mass_y"],
        c=TIER_COLORS[tier], label=tier,
        alpha=0.75, s=40, edgecolors="white", linewidths=0.4,
    )
    for _, row in sub.iterrows():
        ax.annotate(row["iso3"], (row["center_of_mass_x"], row["center_of_mass_y"]),
                    fontsize=4, alpha=0.6, ha="center", va="bottom", xytext=(0, 2),
                    textcoords="offset points")

ax.set_xlabel("Center of Mass X (normalised, W→E within image)", fontsize=10)
ax.set_ylabel("Center of Mass Y (normalised, N→S within image)", fontsize=10)
ax.set_title("Light Center of Mass per Country by Income Tier (2022)", fontsize=12, fontweight="bold")
ax.invert_yaxis()
ax.legend(title="Income Tier", fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIG_DIR / "spatial_scatter.png", dpi=150)
plt.close(fig)
print(f"Saved: {FIG_DIR / 'spatial_scatter.png'}")

# ---------------------------------------------------------------------------
# Figure 6 — Top-10 countries per tier for mean_brightness + lit_fraction
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(2, 4, figsize=(18, 8))
fig.suptitle("Top-10 Countries per Income Tier", fontsize=13, fontweight="bold")

for col_idx, tier in enumerate(TIER_ORDER):
    sub = df[df["income_tier"] == tier]
    for row_idx, metric in enumerate(["mean_brightness", "lit_fraction"]):
        top = sub.nlargest(10, metric)[["iso3", metric]].iloc[::-1]
        ax  = axes[row_idx][col_idx]
        bars = ax.barh(top["iso3"], top[metric], color=TIER_COLORS[tier], alpha=0.8, edgecolor="white")
        ax.set_title(f"{tier}\n{metric.replace('_',' ').title()}", fontsize=8, fontweight="bold")
        ax.tick_params(axis="y", labelsize=7)
        ax.tick_params(axis="x", labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
fig.savefig(FIG_DIR / "top10_countries.png", dpi=150)
plt.close(fig)
print(f"Saved: {FIG_DIR / 'top10_countries.png'}")

# ---------------------------------------------------------------------------
# Figure 7 — Pairplot of key metrics
# ---------------------------------------------------------------------------

plot_metrics = ["mean_brightness", "lit_fraction", "spatial_dispersion",
                "gini", "entropy", "hotspot_count"]
pair_df = df[plot_metrics + ["income_tier"]].copy()
pair_df["income_tier"] = pd.Categorical(pair_df["income_tier"], categories=TIER_ORDER, ordered=True)

g = sns.pairplot(
    pair_df,
    hue="income_tier",
    palette=TIER_COLORS,
    hue_order=TIER_ORDER,
    diag_kind="kde",
    plot_kws={"alpha": 0.5, "s": 20},
    diag_kws={"fill": True, "alpha": 0.4},
)
g.figure.suptitle("Pairplot of Key NTL Metrics by Income Tier (2022)",
                   y=1.01, fontsize=12, fontweight="bold")
g.figure.savefig(FIG_DIR / "metric_pairplot.png", dpi=120, bbox_inches="tight")
plt.close(g.figure)
print(f"Saved: {FIG_DIR / 'metric_pairplot.png'}")

# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print(f"{'Metric':<25}  {'Spearman ρ':>10}  {'p-value':>12}  Sig")
print("-" * 70)
for metric in METRICS:
    rho, pval = spearmanr(df[metric], df["tier_rank"])
    stars = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else "ns"))
    print(f"  {metric:<23}  {rho:>10.4f}  {pval:>12.4e}  {stars}")

print("\nPer-tier summary (mean):")
summary = (
    df.groupby("income_tier")[METRICS]
    .mean()
    .reindex(TIER_ORDER)
)
print(summary.round(4).to_string())
print("=" * 70)
print(f"\nCountries analysed: {len(df)}")
print("Figures saved to figures/")

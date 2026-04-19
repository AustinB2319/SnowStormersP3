# 🌙 Nighttime Light as a Measure of Economic Development

**DS 4002 – Data Science Project | Spring 2026**  
**Team:** Snow Stormers — Jacob VanBenschoten, Austin Blackburn, Aleeza Sadiq *(leader)*

---

## Overview

This project trains and evaluates multi-class image classifiers that predict a country's **World Bank income tier** (Low, Lower-Middle, Upper-Middle, or High) using only its 2022 **NASA Black Marble VNP46A4** annual nighttime light satellite image. Images are processed into normalized 64×64 arrays and fed into three progressively complex models.

**Research Question:**  
*How accurately can a CNN classify a country's World Bank income tier based solely on its 2022 NASA Black Marble nighttime light satellite image, and which tier boundaries are most and least distinguishable?*

---

## Repository Structure

```
SnowStormersP3/
├── data/
│   ├── labels.csv              # ISO3, country name, income tier (193 countries)
│   ├── features.csv            # 12 EDA-derived spatial metrics per country
│   ├── splits.json             # Stratified 70/15/15 train/val/test indices
│   ├── processed/              # 64×64 float32 .npy arrays (one per country)
│   ├── raw/                    # Clipped GeoTIFFs (.tif, one per country)
│   └── ne_countries/           # Natural Earth 110m boundary shapefiles
├── scripts/
│   ├── 01_download_images.py   # NASA CMR API download & preprocessing
│   ├── 03_eda.py               # 12 spatial metrics + 8 exploratory figures
│   ├── 04_split.py             # Stratified train/val/test split
│   ├── 05_train.py             # Logistic regression, CNN, ResNet-18 training
│   └── 06_evaluate.py          # Confusion matrices, Grad-CAM, model comparison
├── models/
│   ├── logreg.pkl              # Trained logistic regression model
│   ├── logreg_results.json     # Logistic regression test results
│   ├── cnn_best.pt             # Best CNN checkpoint
│   ├── cnn_history.json        # CNN training history
│   ├── resnet_best.pt          # Best ResNet-18 checkpoint
│   └── resnet_history.json     # ResNet-18 training history
├── figures/                    # All output plots from EDA and evaluation
├── output/                     # Additional output files
└── README.md
```

---

## Data

| Source | Details |
|--------|---------|
| **NASA Black Marble VNP46A4** | Annual nighttime light composites, 2022, ~500m resolution |
| **World Bank Income Classifications** | Low, Lower-Middle, Upper-Middle, High (2022) |
| **Natural Earth 110m Boundaries** | Country boundary polygons for clipping |

- **193** total World Bank countries → **163** usable after excluding microstates below the 110m resolution threshold
- Tier breakdown: **27 Low · 54 Lower-Middle · 50 Upper-Middle · 62 High**
- Each image: `64×64` float32 array, normalized to `[0, 1]` via 99th-percentile scaling to suppress gas flares and offshore platforms

> Raw HDF5 tiles are not included due to size. See `scripts/01_download_images.py` for download instructions using the NASA CMR API (concept ID `C3860065683-LAADS`).

---

## Methods

### Preprocessing
1. Download VNP46A4 HDF5 tiles via NASA CMR API
2. Mosaic tiles with `rasterio.merge`, clip to Natural Earth country boundaries
3. Resample to 64×64 using average resampling
4. Replace NaN/negatives with 0; normalize by 99th percentile of non-zero pixels (cap at 1.0)
5. Save as `float32` `.npy` files

### EDA — 12 Spatial Metrics

| Category | Metrics |
|----------|---------|
| Intensity | `mean_brightness`, `total_light`, `p90_brightness` |
| Coverage | `lit_fraction`, `light_density` |
| Spatial | `spatial_dispersion`, `center_of_mass_x/y`, `hotspot_count` |
| Inequality | `gini`, `entropy`, `urban_concentration` |

Key findings: `mean_brightness` (ρ = 0.622), `lit_fraction` (ρ = 0.616), and `hotspot_count` (ρ = 0.533) are the strongest positive predictors. `gini` is the strongest negative predictor (ρ = −0.368).

### Models

| Model | Description |
|-------|-------------|
| **Logistic Regression** | Baseline on 10 hand-crafted EDA features, L2 regularization via 5-fold CV |
| **Custom CNN** | 3× Conv2D → BatchNorm → ReLU → MaxPool, FC(256) → Dropout(0.5) → 4-class softmax |
| **ResNet-18** | ImageNet pre-trained, single-channel input replicated to 3ch, 4-class output head, lr=1e-4 |

**Training:** Stratified 70/15/15 split · seed = 42 · augmentations: random flips, ±15° rotation, Gaussian noise (σ=0.01) · Adam optimizer · early stopping (patience=10)

---

## Results

| Model | Test Accuracy | Macro F1 |
|-------|:---:|:---:|
| Logistic Regression | 36% | 0.33 |
| Custom CNN | 28% | 0.22 |
| **ResNet-18 Fine-Tuned** | **48%** | **0.46** |

- No model reached the target macro F1 ≥ 0.60
- ResNet-18 was the best performer, benefiting from ImageNet pre-trained weights
- Logistic regression outperformed the custom CNN, confirming that hand-crafted spatial metrics carry strong signal that pixel-level learning could not replicate at this dataset size
- The Low/High boundary was most separable; Lower-Middle/Upper-Middle was the hardest boundary across all models

---

## Setup & Installation

```bash
git clone https://github.com/AustinB2319/SnowStormersP3.git
cd SnowStormersP3
pip install -r requirements.txt
```

Key dependencies:
```
numpy
rasterio
torch
torchvision
scikit-learn
matplotlib
seaborn
scipy
```

---

## References

1. J. V. Henderson, A. Storeygard, and D. N. Weil, "Measuring Economic Growth from Outer Space," *American Economic Review*, vol. 102, no. 2, pp. 994–1028, 2012.
2. NASA Black Marble Team, *VNP46A4: VIIRS/NPP Lunar BRDF-Adjusted Nighttime Lights Yearly L3 Global 15 arc second Linear Lat Lon Grid*, NASA EOSDIS Land Processes DAAC, 2023.
3. World Bank, "World Bank Country and Lending Groups," World Bank Open Data, 2022. https://datahelpdesk.worldbank.org/knowledgebase/articles/906519

---

*DS 4002 · University of Virginia · Spring 2026*

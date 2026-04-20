# Nighttime Light as a Measure of Economic Development

**DS 4002 – Data Science Project | Spring 2026**  
**Team:** Snow Stormers — Jacob VanBenschoten, Austin Blackburn, Aleeza Sadiq *(leader)*

---

## Overview

This project trains and evaluates multi-class image classifiers that predict a country's **World Bank income tier** (Low, Lower-Middle, Upper-Middle, or High) using only its 2022 **NASA Black Marble VNP46A4** annual nighttime light satellite image. Images are processed into normalized 64×64 arrays and fed into three progressively complex models.

**Research Question:**  
*How accurately can a CNN classify a country's World Bank income tier based solely on its 2022 NASA Black Marble nighttime light satellite image, and which tier boundaries are most and least distinguishable?*

---

## Section 1: Software and Platform

**Language:** Python 3.9+

| Package | Purpose |
|---------|---------|
| `numpy` | Array operations and image processing |
| `pandas` | Data loading, cleaning, and feature management |
| `geopandas` | Country boundary shapefiles |
| `rasterio` | Mosaicking, clipping, and resampling GeoTIFFs |
| `h5py` | Reading NASA HDF5 tile files |
| `requests` | NASA CMR API and LAADS DAAC downloads |
| `scipy` | Spearman correlations and statistical tests |
| `scikit-learn` | Logistic regression, stratified splits, evaluation metrics |
| `torch` | CNN and ResNet-18 training |
| `torchvision` | ResNet-18 pre-trained weights and transforms |
| `matplotlib` | All figures |
| `seaborn` | Statistical visualizations |

Install all dependencies at once:

```bash
pip install numpy pandas geopandas rasterio h5py requests scipy scikit-learn torch torchvision matplotlib seaborn
```

**Platform:** Developed and tested on macOS. Compatible with Windows and Linux with no script modifications.

> **Note:** A free NASA Earthdata account is required to download the raw imagery. Register at https://urs.earthdata.nasa.gov and generate a Bearer token before running Step 2.

---

## Section 2: Map of Documentation

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
│   ├── 01_download_images.py   # Download NASA tiles, mosaic, clip, normalize → .npy + .tif
│   ├── 03_eda.py               # Compute 12 spatial metrics, produce 5 EDA figures
│   ├── 04_split.py             # Compute features, produce stratified 70/15/15 split
│   ├── 05_train.py             # Train logistic regression, CNN, and ResNet-18
│   └── 06_evaluate.py          # Confusion matrices, Grad-CAM, model comparison figures
├── models/
│   ├── logreg.pkl              # Trained logistic regression model
│   ├── logreg_results.json     # Logistic regression test results
│   ├── cnn_best.pt             # Best CNN checkpoint (saved by early stopping)
│   ├── cnn_history.json        # CNN loss and F1 history per epoch
│   ├── resnet_best.pt          # Best ResNet-18 checkpoint
│   └── resnet_history.json     # ResNet-18 loss and F1 history per epoch
├── figures/
│   ├── sample_images_by_tier.png          # Representative 64×64 images per tier
│   ├── pixel_intensity_distributions.png  # KDE curves by income tier
│   ├── mean_brightness_by_tier.png        # Bar chart of mean brightness per tier
│   ├── metric_boxplots.png                # All 12 metrics box+strip plot by tier
│   ├── correlation_heatmap.png            # Spearman ρ heatmap vs tier rank
│   ├── training_curves.png                # Loss + val-F1 curves for CNN & ResNet
│   ├── confusion_matrices.png             # One confusion matrix per model
│   ├── model_comparison.png               # Accuracy & macro-F1 bar chart
│   ├── gradcam_cnn.png                    # Grad-CAM overlays (CNN)
│   ├── gradcam_resnet.png                 # Grad-CAM overlays (ResNet-18)
│   └── feature_importance.png             # Logistic regression coefficient heatmap
├── output/                     # Additional output files
├── LICENSE.md
└── README.md
```

---

## Section 3: Instructions for Reproducing Results

Follow these steps in order. Each script depends on the outputs of the one before it.

---

### Step 1 — Set Up Your Environment

1. Ensure Python 3.9+ is installed. Verify with:
```bash
python --version
```

2. Clone the repository and navigate into it:
```bash
git clone https://github.com/AustinB2319/SnowStormersP3.git
cd SnowStormersP3
```

3. Install all required packages:
```bash
pip install numpy pandas geopandas rasterio h5py requests scipy scikit-learn torch torchvision matplotlib seaborn
```

4. Create a free NASA Earthdata account at https://urs.earthdata.nasa.gov. Once logged in, go to your profile and generate a **Bearer token**. You will need this in the next step.

---

### Step 2 — Download and Preprocess Satellite Images

Run `01_download_images.py` to download NASA Black Marble VNP46A4 HDF5 tiles for all 193 countries, mosaic them, clip to each country's boundary, resample to 64×64, normalize, and save as `.npy` arrays.

```bash
export BLACKMARBLE_TOKEN=your_nasa_earthdata_token
python scripts/01_download_images.py
```

**What it does:**
- Queries the NASA CMR API (concept ID `C3860065683-LAADS`) to find HDF5 tile URLs for the year 2022
- Downloads tiles from the LAADS DAAC Earthdata Cloud using your Bearer token
- Mosaics tiles with `rasterio.merge` and clips to each country's Natural Earth 110m boundary polygon
- Resamples each clipped raster to 64×64 pixels using average resampling
- Replaces NaN/negative fill values with 0
- Normalizes pixel values to [0, 1] by dividing by the 99th percentile of non-zero pixels (capped at 1.0 to suppress gas flares)
- Saves final arrays as `float32` `.npy` files

**Input:** NASA Earthdata token (environment variable), `data/ne_countries/` shapefiles, `data/labels.csv`  
**Output:**
- `data/raw/<ISO3>_2022.tif` — clipped GeoTIFF per country
- `data/processed/<ISO3>_2022.npy` — normalized 64×64 float32 array per country

> **Note:** 30 microstates fall below the 110m resolution threshold and will produce no valid raster. These are automatically skipped, leaving 163 usable countries for all downstream steps. Downloading all tiles takes significant time — expect several hours for a full run depending on your connection.

---

### Step 3 — Run Exploratory Data Analysis

Run `03_eda.py` to compute 12 spatial metrics per country and produce all EDA figures.

```bash
python scripts/03_eda.py
```

**What it does:**
- Loads all 163 processed `.npy` arrays from `data/processed/`
- Computes 12 per-country metrics across four categories:
  - *Intensity:* `mean_brightness`, `total_light`, `p90_brightness`
  - *Coverage:* `lit_fraction`, `light_density`
  - *Spatial:* `spatial_dispersion`, `center_of_mass_x`, `center_of_mass_y`, `hotspot_count`
  - *Inequality:* `gini`, `entropy`, `urban_concentration`
- Computes Spearman correlations between each metric and ordinal tier rank
- Produces 5 exploratory figures saved to `figures/`

**Input:** `data/processed/*.npy`, `data/labels.csv`  
**Output:**
- `figures/sample_images_by_tier.png`
- `figures/pixel_intensity_distributions.png`
- `figures/mean_brightness_by_tier.png`
- `figures/metric_boxplots.png`
- `figures/correlation_heatmap.png`

---

### Step 4 — Generate Features and Train/Val/Test Split

Run `04_split.py` to extract the 10 modeling features and produce the reproducible stratified split used by all models.

```bash
python scripts/04_split.py
```

**What it does:**
- Computes the 10 modeled spatial features (excluding `center_of_mass_x/y`, which showed no tier signal in EDA) for all 163 usable countries
- Saves the feature matrix to `data/features.csv`
- Applies stratified random sampling by income tier using a fixed seed (seed=42)
- Splits into training (70%), validation (15%), and test (15%) sets — approximately 114 train, 25 val, 24 test countries
- Saves split indices to `data/splits.json`

**Input:** `data/processed/*.npy`, `data/labels.csv`  
**Output:**
- `data/features.csv` — 10 spatial features per country
- `data/splits.json` — train/val/test ISO3 country lists

---

### Step 5 — Train All Three Models

Run `05_train.py` to train the logistic regression baseline, custom CNN, and ResNet-18 in sequence.

```bash
python scripts/05_train.py
```

**What it does:**

*Model 1 — Logistic Regression:*
- Fits on 10 hand-crafted spatial features from `data/features.csv`
- Tunes L2 regularization parameter C via 5-fold cross-validation on the training set
- Evaluates on validation and test sets

*Model 2 — Custom CNN:*
- Trains directly on 64×64 float32 arrays (1 channel)
- Architecture: 3× (Conv2D → BatchNorm → ReLU → MaxPool) → FC(256) → Dropout(0.5) → FC(4)
- Adam optimizer, lr=1e-3, cross-entropy loss
- Augmentation applied to training only: random horizontal flip (p=0.5), random vertical flip (p=0.5), random rotation ±15°, additive Gaussian noise (σ=0.01)
- Early stopping with patience=10 on validation loss

*Model 3 — ResNet-18 Fine-Tuned:*
- Loads ImageNet pre-trained ResNet-18, replaces final FC layer with Linear(512, 4)
- Replicates single-channel input to 3 channels to match ImageNet expectations
- Adam optimizer, lr=1e-4, cross-entropy loss
- Same augmentation and early stopping protocol as the CNN

**Input:** `data/features.csv`, `data/splits.json`, `data/processed/*.npy`  
**Output:**
- `models/logreg.pkl` + `models/logreg_results.json`
- `models/cnn_best.pt` + `models/cnn_history.json`
- `models/resnet_best.pt` + `models/resnet_history.json`

> **Note:** CNN and ResNet training runs on CPU by default. On a standard laptop expect ~15–30 minutes for the CNN and ~30–60 minutes for ResNet-18. If a CUDA GPU is available, PyTorch will use it automatically.

---

### Step 6 — Evaluate Models and Generate Figures

Run `06_evaluate.py` to evaluate all three models on the held-out test set and produce all diagnostic figures.

```bash
python scripts/06_evaluate.py
```

**What it does:**
- Loads all three trained models from `models/`
- Evaluates each on the held-out test set using overall accuracy and macro-averaged F1-score
- Generates a confusion matrix for each model
- Produces Grad-CAM visualizations for the CNN and ResNet-18 to identify which spatial regions drive predictions
- Plots training loss and validation F1 curves for both neural models
- Prints to console: tier-pair misclassification analysis, outlier countries (e.g., SYR, PSE, PRK), and whether the CNN adds signal over logistic regression

**Input:** `models/logreg.pkl`, `models/cnn_best.pt`, `models/resnet_best.pt`, `models/cnn_history.json`, `models/resnet_history.json`, `data/splits.json`, `data/processed/*.npy`, `data/features.csv`  
**Output:**
- `figures/training_curves.png`
- `figures/confusion_matrices.png`
- `figures/model_comparison.png`
- `figures/gradcam_cnn.png`
- `figures/gradcam_resnet.png`
- `figures/feature_importance.png`

---

### Expected Final Outputs

After completing all six steps, your `figures/` folder should contain 11 PNG files and your `models/` folder should contain 6 model files. Results should match the following:

| Model | Test Accuracy | Macro F1 |
|-------|:---:|:---:|
| Logistic Regression | 36% | 0.33 |
| Custom CNN | 28% | 0.22 |
| **ResNet-18 Fine-Tuned** | **48%** | **0.46** |

- No model reached the target macro F1 ≥ 0.60
- ResNet-18 was the best performer, benefiting from ImageNet pre-trained weights
- Logistic regression outperformed the custom CNN, confirming that hand-crafted spatial metrics carry stronger signal than raw pixel learning at this dataset size
- The Low/High boundary was most separable; Lower-Middle/Upper-Middle was the hardest boundary across all models

---

## References

1. J. V. Henderson, A. Storeygard, and D. N. Weil, "Measuring Economic Growth from Outer Space," *American Economic Review*, vol. 102, no. 2, pp. 994–1028, 2012.
2. NASA Black Marble Team, *VNP46A4: VIIRS/NPP Lunar BRDF-Adjusted Nighttime Lights Yearly L3 Global 15 arc second Linear Lat Lon Grid*, NASA EOSDIS Land Processes DAAC, 2023. https://doi.org/10.5067/VIIRS/VNP46A4.001
3. World Bank, "World Bank Country and Lending Groups," World Bank Open Data, 2022. https://datahelpdesk.worldbank.org/knowledgebase/articles/906519

---

*DS 4002 · University of Virginia · Spring 2026*

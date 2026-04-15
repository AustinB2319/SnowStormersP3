# Data Access & Reproducibility Guide

## Overview

This project uses satellite-based nighttime light (NTL) data from NASA’s Black Marble product to analyze global economic development. Due to file size limitations, the raw satellite imagery is not included in this repository.

This guide explains how to fully reproduce the dataset used in this project.

---

## Data Included in Repository

The following files are provided:

* `data/labels.csv`

  * Contains country ISO3 codes and corresponding World Bank income classifications.

* `data/ne_countries/`

  * Natural Earth shapefiles used for country boundaries and spatial processing.

---

## Data Not Included

The following data is **not included** due to size constraints:

* Raw nighttime light imagery (`.tif` files)
* Intermediate processed NumPy arrays (`.npy` files)

---

## How to Reproduce the Data

### 1. Download Raw Satellite Data

Run the download script:

```bash
python scripts/01_download_images.py
```

This script:

* Queries NASA’s Black Marble (VNP46A3) dataset (year: 2022)
* Downloads nighttime light tiles via NASA APIs
* Mosaics and clips imagery to each country boundary
* Saves output to:

```bash
data/raw/<ISO3>_2022.tif
```

> Note: You may need to configure NASA Earthdata access (see below).

---

### 2. NASA Earthdata Access

To download the data, you must:

1. Create a free account at: https://urs.earthdata.nasa.gov/
2. Enable access to LAADS DAAC data
3. Add your credentials to a `.env` file:

```bash
EARTHDATA_USERNAME=your_username
EARTHDATA_PASSWORD=your_password
```

---

### 3. Preprocess Data

Convert raw imagery into model-ready arrays:

```bash
python scripts/02_preprocess.py
```

This step:

* Resizes images to 64×64
* Normalizes pixel values
* Saves NumPy arrays to:

```bash
data/processed/<ISO3>_2022.npy
```

* Full dataset size (raw + processed) is several GB.
* Processing all countries may take significant time depending on internet speed and compute power.

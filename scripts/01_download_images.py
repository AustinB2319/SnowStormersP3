"""
01_download_images.py
---------------------
Downloads NASA Black Marble VNP46A4 annual composite nighttime light images
for ~195 countries (all World Bank 2022 income tiers) for the year 2022 and
saves each country as a GeoTIFF to data/raw/<ISO3>_2022.tif.

Approach: blackmarblepy's underlying LAADS /api/v1/files endpoint is retired.
This script bypasses it and goes directly to:
  - NASA CMR API  →  find tile download URLs
  - data.laadsdaac.earthdatacloud.nasa.gov  →  download HDF5 tiles (auth)
  - rasterio  →  mosaic tiles and clip to country bounding box

Requirements:
    pip install geopandas rasterio h5py requests numpy pandas

Usage:
    export BLACKMARBLE_TOKEN=your_nasa_earthdata_token
    python scripts/01_download_images.py
"""

import io
import math
import os
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import geopandas as gpd
import h5py
import numpy as np
import pandas as pd
import requests
import rasterio
from rasterio.io import MemoryFile
from rasterio.mask import mask
from rasterio.merge import merge
from rasterio.transform import from_bounds

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR   = ROOT / "data" / "raw"
TILE_DIR  = ROOT / "data" / "raw" / "tiles"   # cached HDF5 tile files
NE_DIR    = ROOT / "data" / "ne_countries"
LABELS_PATH = ROOT / "data" / "labels.csv"

for d in [RAW_DIR, TILE_DIR, NE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Token
# ---------------------------------------------------------------------------

TOKEN = os.environ.get("BLACKMARBLE_TOKEN")
# Token is only needed for downloading tiles not already cached.
# If all tiles are cached, the script runs without it.
HEADERS = {"Authorization": f"Bearer {TOKEN}"} if TOKEN else {}

# ---------------------------------------------------------------------------
# Country lists — all World Bank 2022 income tiers (~195 countries)
# ---------------------------------------------------------------------------

COUNTRIES = {
    # 27 low-income countries (GNI per capita ≤ $1,085)
    "Low": [
        "AFG", "BDI", "BFA", "CAF", "TCD", "COD", "ERI", "ETH",
        "GMB", "GIN", "GNB", "HTI", "LBR", "MDG", "MLI", "MOZ",
        "MWI", "NER", "PRK", "RWA", "SLE", "SOM", "SSD", "SYR",
        "TGO", "UGA", "YEM",
    ],
    # ~55 lower-middle-income countries ($1,086–$4,255)
    "Lower-Middle": [
        "AGO", "BGD", "BEN", "BOL", "BTN", "CMR", "CIV", "COM",
        "COG", "CPV", "DJI", "EGY", "FSM", "GHA", "GUY", "HND",
        "IND", "IRQ", "KEN", "KGZ", "KIR", "LSO", "LKA", "MAR",
        "MHL", "MMR", "MNG", "MRT", "NGA", "NIC", "NPL", "PAK",
        "PHL", "PNG", "PSE", "SDN", "SEN", "SLB", "SLV", "STP",
        "SWZ", "TJK", "TLS", "TON", "TUN", "TUV", "TZA", "UKR",
        "UZB", "VNM", "VUT", "WSM", "ZMB", "ZWE",
    ],
    # ~51 upper-middle-income countries ($4,256–$13,205)
    "Upper-Middle": [
        "ALB", "DZA", "ARG", "ARM", "AZE", "BIH", "BLR", "BLZ",
        "BRA", "BWA", "CHN", "COL", "CRI", "CUB", "DOM", "ECU",
        "FJI", "GAB", "GEO", "GNQ", "GTM", "IDN", "IRN", "JAM",
        "JOR", "KAZ", "LBN", "LBY", "MDV", "MEX", "MDA", "MKD",
        "MNE", "MUS", "MYS", "NAM", "NRU", "PER", "PLW", "PRY",
        "RUS", "SRB", "SUR", "THA", "TKM", "TUR", "VCT", "VEN",
        "ZAF", "XKX",
    ],
    # ~62 high-income countries (GNI per capita > $13,205)
    "High": [
        "AND", "ARE", "ATG", "AUS", "AUT", "BHR", "BEL", "BHS",
        "BRB", "BRN", "CAN", "CHL", "CHE", "CYP", "CZE", "DEU",
        "DNK", "DMA", "ESP", "EST", "FIN", "FRA", "GBR", "GRC",
        "GRD", "HKG", "HRV", "HUN", "IRL", "ISL", "ISR", "ITA",
        "JPN", "KNA", "KOR", "KWT", "LIE", "LTU", "LUX", "LVA",
        "MAC", "MCO", "MLT", "NLD", "NOR", "NZL", "OMN", "PAN",
        "POL", "PRT", "QAT", "ROU", "SAU", "SGP", "SMR", "SVK",
        "SVN", "SWE", "TTO", "TWN", "URY", "USA",
    ],
}

ALL_COUNTRIES = [
    (iso3, tier)
    for tier, iso3_list in COUNTRIES.items()
    for iso3 in iso3_list
]

# ---------------------------------------------------------------------------
# CMR constants
# ---------------------------------------------------------------------------

CMR_URL     = "https://cmr.earthdata.nasa.gov/search/granules.json"
# V2 concept-id for VIIRS/NPP Lunar BRDF-Adjusted NTL Yearly (Collection 5200)
CMR_CONCEPT = "C3860065683-LAADS"
# Mid-year date range: VNP46A4.A2022001 spans 2022-01-01 to 2023-01-01,
# so any mid-year query returns only the 2022 annual composite.
CMR_TEMPORAL = "2022-06-01T00:00:00Z,2022-07-01T00:00:00Z"

# HDF5 internal path for the nighttime light variable
NTL_PATH  = "HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/Data Fields/NearNadir_Composite_Snow_Free"
LAT_PATH  = "HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/Data Fields/lat"
LON_PATH  = "HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/Data Fields/lon"
FILL_VAL  = -999.9

# ---------------------------------------------------------------------------
# Load Natural Earth country shapefile (download once, cache locally)
# ---------------------------------------------------------------------------

NE_SHP = NE_DIR / "ne_110m_admin_0_countries.shp"

if not NE_SHP.exists():
    url = ("https://naciscdn.org/naturalearth/110m/cultural/"
           "ne_110m_admin_0_countries.zip")
    print(f"Downloading Natural Earth shapefile …")
    resp = urllib.request.urlopen(url, timeout=60)
    with zipfile.ZipFile(io.BytesIO(resp.read())) as zf:
        zf.extractall(NE_DIR)
    print(f"  saved to {NE_DIR}")

world = gpd.read_file(NE_SHP)

def get_country_gdf(iso3: str) -> gpd.GeoDataFrame | None:
    """Return single-row GDF for iso3 (falls back to ISO_A3_EH for FRA/NOR)."""
    row = world[world["ISO_A3"] == iso3]
    if row.empty:
        row = world[world["ISO_A3_EH"] == iso3]
    return row if not row.empty else None

# ---------------------------------------------------------------------------
# Write labels.csv
# ---------------------------------------------------------------------------

records = []
for iso3, tier in ALL_COUNTRIES:
    gdf = get_country_gdf(iso3)
    name = gdf["NAME"].values[0] if gdf is not None else ""
    records.append({"iso3": iso3, "country_name": name, "income_tier": tier})

pd.DataFrame(records).to_csv(LABELS_PATH, index=False)
print(f"Labels CSV → {LABELS_PATH}  ({len(records)} rows)\n")

# ---------------------------------------------------------------------------
# CMR helpers
# ---------------------------------------------------------------------------

def cmr_search(bbox_str: str) -> list[dict]:
    """
    Query CMR for all VNP46A4 2022 granules overlapping bbox_str.

    bbox_str format: "west,south,east,north"
    Returns list of dicts with 'name' and 'url' keys.
    """
    params = {
        "concept_id": CMR_CONCEPT,
        "temporal": CMR_TEMPORAL,
        "bounding_box": bbox_str,
        "page_size": 200,
    }
    resp = requests.get(CMR_URL, params=params, timeout=30)
    resp.raise_for_status()
    entries = resp.json().get("feed", {}).get("entry", [])

    results = []
    for e in entries:
        name = e.get("producer_granule_id", "")
        url = next(
            (lk["href"] for lk in e.get("links", [])
             if lk.get("rel", "").endswith("data#")
             and "laadsdaac" in lk.get("href", "")),
            None,
        )
        if name and url:
            results.append({"name": name, "url": url})
    return results

# ---------------------------------------------------------------------------
# Tile download & HDF5 → rasterio MemoryFile
# ---------------------------------------------------------------------------

def download_tile(tile_name: str, url: str) -> Path:
    """Download HDF5 tile to TILE_DIR if not already cached. Return local path."""
    local = TILE_DIR / tile_name
    if local.exists():
        return local

    if not TOKEN:
        raise RuntimeError(
            f"Tile {tile_name} not cached and BLACKMARBLE_TOKEN is not set."
        )

    resp = requests.get(url, headers=HEADERS, stream=True, timeout=300)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    downloaded = 0
    with open(local, "wb") as fh:
        for chunk in resp.iter_content(chunk_size=1 << 20):  # 1 MB chunks
            fh.write(chunk)
            downloaded += len(chunk)
    return local


def h5_to_rasterio(h5_path: Path) -> MemoryFile:
    """
    Read NearNadir_Composite_Snow_Free from an HDF5 tile and return an
    open rasterio MemoryFile (single-band float32, EPSG:4326, nodata=NaN).
    """
    with h5py.File(h5_path, "r") as f:
        lat  = f[LAT_PATH][:]           # shape (2400,), N→S
        lon  = f[LON_PATH][:]           # shape (2400,), W→E
        data = f[NTL_PATH][:].astype(np.float32)

    # Replace fill with NaN
    data[data <= FILL_VAL] = np.nan

    # Pixel size is uniform at 15 arc-sec = 1/240 deg
    px = abs(float(lat[1] - lat[0]))

    west  = float(lon[0])  - px / 2
    east  = float(lon[-1]) + px / 2
    south = float(lat[-1]) - px / 2
    north = float(lat[0])  + px / 2

    transform = from_bounds(west, south, east, north,
                             data.shape[1], data.shape[0])

    memfile = MemoryFile()
    with memfile.open(
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
        nodata=np.nan,
    ) as dst:
        dst.write(data[np.newaxis, ...])

    return memfile   # caller must close

# ---------------------------------------------------------------------------
# Per-country pipeline
# ---------------------------------------------------------------------------

def process_country(iso3: str, tier: str, idx: int, total: int) -> bool:
    """
    Download tiles, mosaic, clip to country, save GeoTIFF.
    Returns True on success.
    """
    out_path = RAW_DIR / f"{iso3}_2022.tif"
    if out_path.exists():
        print(f"[{idx:2d}/{total}] {iso3} ({tier}) — exists, skipping")
        return True

    country_gdf = get_country_gdf(iso3)
    if country_gdf is None:
        print(f"[{idx:2d}/{total}] {iso3} ({tier}) — WARNING: no geometry")
        return False

    bounds = country_gdf.total_bounds          # (minx, miny, maxx, maxy)
    bbox_str = f"{bounds[0]:.4f},{bounds[1]:.4f},{bounds[2]:.4f},{bounds[3]:.4f}"

    print(f"[{idx:2d}/{total}] {iso3} ({tier})", end=" ", flush=True)

    # 1. Find tiles
    try:
        granules = cmr_search(bbox_str)
    except Exception as e:
        print(f"— CMR search failed: {e}")
        return False

    if not granules:
        print(f"— no tiles found for bbox {bbox_str}")
        return False

    print(f"— {len(granules)} tile(s)", end=" ", flush=True)

    # 2. Download tiles (cached)
    local_tiles = []
    for g in granules:
        try:
            p = download_tile(g["name"], g["url"])
            local_tiles.append(p)
            print(".", end="", flush=True)
        except Exception as e:
            print(f"\n    WARNING: could not download {g['name']}: {e}")

    if not local_tiles:
        print(" — no tiles downloaded")
        return False

    # 3. Open tiles as rasterio MemoryFiles and mosaic
    memfiles = [h5_to_rasterio(p) for p in local_tiles]
    datasets = [mf.open() for mf in memfiles]

    try:
        if len(datasets) == 1:
            mosaic, mosaic_transform = datasets[0].read(1), datasets[0].transform
            mosaic_crs = datasets[0].crs
            mosaic_nodata = datasets[0].nodata
            # Wrap back into a single dataset for the mask step
            mf_mosaic = MemoryFile()
            with mf_mosaic.open(
                driver="GTiff",
                height=datasets[0].height,
                width=datasets[0].width,
                count=1,
                dtype="float32",
                crs=mosaic_crs,
                transform=mosaic_transform,
                nodata=mosaic_nodata,
            ) as dst:
                dst.write(mosaic[np.newaxis, ...])
            mosaic_ds = mf_mosaic.open()
        else:
            mosaic_arr, mosaic_transform = merge(datasets, nodata=np.nan)
            mosaic_crs = datasets[0].crs
            mf_mosaic = MemoryFile()
            with mf_mosaic.open(
                driver="GTiff",
                height=mosaic_arr.shape[1],
                width=mosaic_arr.shape[2],
                count=1,
                dtype="float32",
                crs=mosaic_crs,
                transform=mosaic_transform,
                nodata=np.nan,
            ) as dst:
                dst.write(mosaic_arr)
            mosaic_ds = mf_mosaic.open()

        # 4. Clip to country geometry
        geom = [geom.__geo_interface__ for geom in country_gdf.geometry]
        clipped, clip_transform = mask(
            mosaic_ds, geom, crop=True, nodata=np.nan
        )

        # 5. Save
        with rasterio.open(
            out_path,
            "w",
            driver="GTiff",
            height=clipped.shape[1],
            width=clipped.shape[2],
            count=1,
            dtype="float32",
            crs="EPSG:4326",
            transform=clip_transform,
            nodata=np.nan,
            compress="lzw",
        ) as dst:
            dst.write(clipped)

        print(f" → {out_path.name}  ({out_path.stat().st_size // 1024} KB)")
        return True

    except Exception as e:
        print(f"\n    ERROR clipping/saving {iso3}: {e}")
        return False

    finally:
        for ds in datasets:
            ds.close()
        for mf in memfiles:
            mf.close()
        if 'mosaic_ds' in dir():
            try:
                mosaic_ds.close()
            except Exception:
                pass
        if 'mf_mosaic' in dir():
            try:
                mf_mosaic.close()
            except Exception:
                pass

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

total = len(ALL_COUNTRIES)
success, skipped, failed = 0, [], []

t_start = time.time()

for idx, (iso3, tier) in enumerate(ALL_COUNTRIES, start=1):
    ok = process_country(iso3, tier, idx, total)
    if ok:
        success += 1
    else:
        failed.append(iso3)
    time.sleep(0.1)   # brief pause between countries

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

elapsed = time.time() - t_start
tif_files = list(RAW_DIR.glob("*_2022.tif"))

print("\n" + "=" * 60)
print(f"Finished in {elapsed/60:.1f} min")
print(f"GeoTIFFs in {RAW_DIR}: {len(tif_files)} / {total}")
if failed:
    print(f"Failed: {failed}")
else:
    print("All countries completed successfully.")
print("=" * 60)
print("Next: python scripts/02_preprocess.py")

# Sigmoid altitudinal ecotone retrieval

**Sigmoid altitudinal ecotone retrieval, iso-NDVI kinematics and explainable gradient-boosted attribution on harmonised multi-decadal Landsat / Sentinel-2 surface-reflectance cubes**

[![python](https://img.shields.io/badge/python-3.11%2B-blue)](#)
[![lint](https://img.shields.io/badge/ruff-clean-brightgreen)](#)
[![types](https://img.shields.io/badge/mypy-clean-brightgreen)](#)
[![GRASS](https://img.shields.io/badge/GRASS%20GIS-8.4-green)](#)
[![STAC](https://img.shields.io/badge/STAC-COG%20windowed-orange)](#)
[![license](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-[REPLACE]-blue)](#)

---

## Abstract

Cloud-native, GEE-free EO stack for dz/dt retrieval of an altitudinal vegetation ecotone. STAC-indexed COG windows with SAS-signed I/O, QA_PIXEL / SCL bitmask screening, TM/ETM+ → OLI OLS and MSI → OLI HLS bandpass harmonisation, median peak-season NDVI compositing on a fixed projected lattice, weighted bounded NLS of a logistic vegetated-fraction profile f(z) with band-bootstrap CIs, Theil–Sen / Mann–Kendall with Hamed–Rao n/n* variance inflation, Pettitt CPD, Loarie-type isoline velocity v = −(∂N/∂t)·∇N/|∇N|², per-cell harmonic regression via accumulated normal equations with λ_min conditioning mask, MODIS MOD10A2 SCD, lapse-shifted ERA5-Land GDD, and LightGBM with spatial GroupKFold and exact TreeSHAP.

## Stack

| layer | tooling |
|---|---|
| ingest | STAC API, COG, WarpedVRT, overview-level reads, ThreadPool fan-out |
| radiometry | C2 L2 scaling, cross-sensor OLS, HLS c-factors, PB ≥ 04.00 offset |
| geomorphometry | `r.slope.aspect`, `r.geomorphon`, `r.sun`, `r.series` |
| statistics | Sen slope, MK-τ, HR-MK, Pettitt U, BCa-free bootstrap |
| ML / XAI | LightGBM GBDT, block-CV, `pred_contrib` TreeSHAP |
| viz | Matplotlib 3.10, Nimbus Sans, overlap-verified layouts |

## Layout

```
src/
  geoconf.py              # AOI_CONFIG → lattice, CRS, tiling
  get_dem.py              # GLO-30 → reference lattice
  grass_terrain.sh        # slope / aspect / geomorphon / r.sun
  grass_rad.sh            # r.sun subset
  scene_qa.py             # acquisition-level QA census
  build_ndvi.py           # per-year harmonised median composites
  consolidate_ndvi.py     # cube assembly + QC
  cross_sensor.py         # pairwise sensor agreement
  derive_ndvi.py          # per-pixel TS/MK + epoch medians
  ecotone_fit.py          # logistic f(z), z50, s, bootstrap
  harm_accum.py           # harmonic normal-equation accumulation
  modis_scd.py            # MOD10A2 SCD + trend
  gdd_points.py           # ERA5-Land GDD ladder
  aspect_stats.py         # circular / landform stratification
  swath_profiles.py       # swath statistics
  hamed_rao.py            # HR-MK
  shap_model.py           # GBDT + block-CV + TreeSHAP
  shap_summary.py         # SHAP reduction
  fig_*.py                # figures
aoi.example.json
```

## Run

```bash
cp aoi.example.json src/aoi.json          # or export AOI_CONFIG=/path/to/aoi.json
pip install -r requirements.txt

python src/get_dem.py out/dem30.tif
grass --tmp-project "$EPSG" --exec bash src/grass_terrain.sh out/dem30.tif out/terrain
python src/scene_qa.py landsat-c2-l2 1984 2025 out/scenes
python src/build_ndvi.py selection.json out/ndvi 1984 1985 ...
python src/consolidate_ndvi.py out/ndvi/manifest.json out/dem30.tif out/cube
python src/derive_ndvi.py out/cube/ndvi_stack_30m.tif out/dem30.tif out/derived trend
python src/ecotone_fit.py out/cube/ndvi_stack_30m.tif out/dem30.tif out/derived
python src/harm_accum.py 1984 2004 out/h_e1.npz 8
python src/modis_scd.py out/derived 16
python src/gdd_points.py out/gdd_points.csv
python src/shap_model.py out/derived out/derived
python src/hamed_rao.py out/derived/ecotone_annual.csv out/climate_series.csv
```

## QA

```bash
ruff format src && ruff check src && mypy src
```

## Topics

`remote-sensing` `earth-observation` `satellite-image-analysis` `landsat` `sentinel-2` `modis` `stac` `cloud-optimized-geotiff` `ndvi` `time-series-analysis` `change-detection` `cross-sensor-harmonization` `harmonic-regression` `phenology` `theil-sen` `mann-kendall` `pettitt-test` `nonlinear-least-squares` `bootstrap` `ecotone` `treeline` `climate-velocity` `geomorphometry` `grass-gis` `lightgbm` `shap` `explainable-ai` `spatial-cross-validation` `era5-land` `python`

## Citation

```
[REPLACE: DOI]
```

## License

MIT

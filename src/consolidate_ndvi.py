#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import os
import sys
import urllib.request
import warnings
from concurrent.futures import ThreadPoolExecutor

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from rasterio.transform import Affine

MAN, DEM, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
YEARS = list(range(1984, 2026))
FLAG = {2012}
BANDS = [(1500, 2000), (2000, 2500), (2500, 3000), (3000, 4000)]
os.makedirs(OUT, exist_ok=True)
with open(MAN) as fm:
    urls = json.load(fm)["urls"]


def fetch(y: int) -> str:
    fn = f"/tmp/ndvi_{y}.tif"
    req = urllib.request.Request(urls[str(y)], headers={"User-Agent": "research-script/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(fn, "wb") as f:
        f.write(r.read())
    return fn


with ThreadPoolExecutor(max_workers=8) as ex:
    files = dict(zip(YEARS, ex.map(fetch, YEARS)))
print("downloaded", len(files), flush=True)

with rasterio.open(DEM) as f:
    z = f.read(1).astype(np.float32)
with rasterio.open(files[YEARS[0]]) as f:
    prof = f.profile.copy()
    ny, nx = f.height, f.width
    tr = f.transform

stack = np.empty((len(YEARS), ny, nx), np.int16)
nobs = np.empty((len(YEARS), ny, nx), np.uint8)
rows = []
for k, y in enumerate(YEARS):
    with rasterio.open(files[y]) as f:
        stack[k] = f.read(1)
        nobs[k] = np.clip(f.read(2), 0, 255).astype(np.uint8)
        tags = f.tags()
        extra = f.read([3, 4]) if f.count >= 4 else None
    v = np.where(stack[k] == -32768, np.nan, stack[k] / 1000.0)
    row: dict[str, object] = {
        "year": y,
        "window": tags.get("window_doy", ""),
        "n_landsat": tags.get("n_landsat", ""),
        "n_s2": tags.get("n_s2", ""),
        "flag": int(y in FLAG),
        "coverage": round(float(np.isfinite(v).mean()), 4),
        "mean_nobs": round(float(nobs[k][nobs[k] > 0].mean()), 2),
        "median_window": round(float(np.nanmedian(v)), 4),
        "median_alpine_ge2200": round(float(np.nanmedian(v[z >= 2200])), 4),
    }
    for lo, hi in BANDS:
        row[f"median_{lo}_{hi}"] = round(float(np.nanmedian(v[(z >= lo) & (z < hi)])), 4)
    if extra is not None:
        ml = np.where(extra[0] == -32768, np.nan, extra[0] / 1000.0)
        ms = np.where(extra[1] == -32768, np.nan, extra[1] / 1000.0)
        b = np.isfinite(ml) & np.isfinite(ms) & (z >= 2200)
        row["LminusS_alpine"] = round(float(np.mean(ml[b] - ms[b])), 4)
        bw = np.isfinite(ml) & np.isfinite(ms)
        row["LminusS_window"] = round(float(np.mean(ml[bw] - ms[bw])), 4)
    rows.append(row)
keys: list[str] = []
for r_ in rows:
    keys += [k_ for k_ in r_ if k_ not in keys]
with open(f"{OUT}/ndvi_year_stats.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader()
    w.writerows(rows)

prof.update(count=len(YEARS), dtype="int16", nodata=-32768, compress="deflate", predictor=2, zlevel=6, tiled=True)
with rasterio.open(f"{OUT}/ndvi_stack_30m.tif", "w", **prof) as f:
    f.write(stack)
    for k, y in enumerate(YEARS, 1):
        f.set_band_description(k, str(y))
    f.update_tags(years="1984-2025", scale="0.001", flagged="2012", composite="median 1 Jul-15 Sep, harmonised to OLI")
prof_n = prof.copy()
prof_n.update(dtype="uint8", nodata=None, predictor=1)
with rasterio.open(f"{OUT}/nobs_stack_30m.tif", "w", **prof_n) as f:
    f.write(nobs)

py, px = (-ny) % 3, (-nx) % 3
s90 = np.empty((len(YEARS), (ny + py) // 3, (nx + px) // 3), np.int16)
for k in range(len(YEARS)):
    a = np.pad(
        np.where(stack[k] == -32768, np.nan, stack[k].astype(np.float32)), ((0, py), (0, px)), constant_values=np.nan
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        m = np.nanmean(a.reshape(a.shape[0] // 3, 3, a.shape[1] // 3, 3), axis=(1, 3))
    s90[k] = np.where(np.isfinite(m), np.round(m), -32768).astype(np.int16)
p90 = prof.copy()
p90.update(width=s90.shape[2], height=s90.shape[1], transform=tr * Affine.scale(3), zlevel=9)
with rasterio.open(f"{OUT}/ndvi_stack_90m.tif", "w", **p90) as f:
    f.write(s90)
    for k, y in enumerate(YEARS, 1):
        f.set_band_description(k, str(y))
for fn in ("ndvi_stack_30m.tif", "nobs_stack_30m.tif", "ndvi_stack_90m.tif"):
    print(fn, f"{os.path.getsize(f'{OUT}/{fn}') / 1e6:.1f} MB")

plt.rcParams.update(
    {"font.family": "sans-serif", "font.sans-serif": ["Nimbus Sans", "Helvetica"], "font.size": 8.5, "pdf.fonttype": 42}
)
assert "dejavu" not in findfont(FontProperties(family=plt.rcParams["font.sans-serif"])).lower()
OI = ["#0072B2", "#009E73", "#E69F00", "#D55E00"]
fig, (ax, bx) = plt.subplots(
    1, 2, figsize=(17 / 2.54, 7.2 / 2.54), constrained_layout=True, gridspec_kw={"width_ratios": [2.4, 1]}
)
yy = np.array(YEARS)
for (lo, hi), c, mk in zip(BANDS, OI, "osD^"):
    v = np.array([r[f"median_{lo}_{hi}"] for r in rows], float)
    good = ~np.isin(yy, list(FLAG))
    ax.plot(
        yy[good],
        v[good],
        color=c,
        lw=1.2,
        marker=mk,
        ms=3.2,
        mec="white",
        mew=0.4,
        label=f"{lo}–{hi if hi < 4000 else ''} m".replace("– m", "+ m"),
    )
    ax.plot(yy[~good], v[~good], ls="", marker=mk, ms=4, mfc="white", mec=c, mew=0.9)
ax.axvspan(2016.5, 2025.5, color="0.92", lw=0, zorder=0)
ax.text(2021, ax.get_ylim()[1], "Landsat + Sentinel-2", ha="center", va="top", fontsize=7.5, color="0.35")
ax.set_xlim(1983, 2026)
ax.set_xlabel("Year")
ax.set_ylabel("Median peak-season NDVI")
ax.legend(
    loc="center left",
    bbox_to_anchor=(0.0, 0.62),
    frameon=False,
    fontsize=7.5,
    title="Elevation band",
    title_fontsize=7.5,
)
yb = [r["year"] for r in rows if "LminusS_alpine" in r]
bx.bar(yb, [r["LminusS_alpine"] for r in rows if "LminusS_alpine" in r], color="#56B4E9", width=0.7, zorder=2)
bx.axhline(0, color="0.3", lw=0.7)
bx.set_xlabel("Year")
bx.set_ylabel("Landsat − Sentinel-2 NDVI (≥ 2200 m)")
for a_, tag in ((ax, "a"), (bx, "b")):
    a_.xaxis.set_minor_locator(MultipleLocator(1))
    a_.yaxis.set_minor_locator(AutoMinorLocator())
    a_.tick_params(which="both", direction="in", top=True, right=True)
    a_.grid(axis="y", lw=0.5, color="0.85")
    a_.set_axisbelow(True)
    a_.text(0.02, 0.97, f"({tag})", transform=a_.transAxes, ha="left", va="top", fontsize=10, fontweight="bold")
bx.xaxis.set_major_locator(MultipleLocator(2))
fig.savefig(f"{OUT}/ndvi_qc.png", dpi=130)
print("qc written")

#!/usr/bin/env python3

from __future__ import annotations

import os
import shutil
import sys
import time
import urllib.request
import warnings
from typing import TypeAlias

import numpy as np
import numpy.typing as npt
import pandas as pd
import rasterio
from scipy import stats
from scipy.optimize import curve_fit

SRC, DEM, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(OUT, exist_ok=True)
F: TypeAlias = npt.NDArray[np.float64]
YEARS = np.arange(1984, 2026)
KEEP = YEARS != 2012
T_MAIN, T_SENS = 0.20, (0.15, 0.25)
B = 100
t0 = time.time()

if SRC.startswith("http"):
    req = urllib.request.Request(SRC, headers={"User-Agent": "research-script/1.0"})
    with urllib.request.urlopen(req, timeout=180) as r, open("/tmp/stack.tif", "wb") as fo:
        shutil.copyfileobj(r, fo)
    SRC = "/tmp/stack.tif"
with rasterio.open(SRC) as f:
    stk = f.read()
with rasterio.open(DEM) as f:
    z = f.read(1).astype(np.float64)
print(f"read {time.time() - t0:.0f}s", flush=True)

gy, gx = np.gradient(z, 30.0)
asp = (np.degrees(np.arctan2(-gx, gy)) + 360.0) % 360.0
slp = np.degrees(np.arctan(np.hypot(gx, gy)))
dom = (z >= 1600) & (z < 3600) & (slp > 2)
idx = np.flatnonzero(dom)
zbin = ((z.ravel()[idx] - 1600) // 25).astype(np.int64)
NB = 80
zc_all = 1600 + 25 * np.arange(NB) + 12.5
a_ = asp.ravel()[idx]
STRATA = {
    "all": np.ones(idx.size, bool),
    "N": (a_ >= 315) | (a_ < 45),
    "E": (a_ >= 45) & (a_ < 135),
    "S": (a_ >= 135) & (a_ < 225),
    "W": (a_ >= 225) & (a_ < 315),
}


def logi(zz: F, fmax: float, z50: float, s: float) -> F:
    return fmax / (1.0 + np.exp((zz - z50) / s))


def fit(zc: F, f: F, n: F) -> tuple[float, float, float, float]:
    p0 = [min(1.0, max(0.25, float(f.max()))), float(zc[np.argmin(np.abs(f - f.max() / 2))]), 200.0]
    pp, _ = curve_fit(logi, zc, f, p0=p0, sigma=1 / np.sqrt(n), bounds=([0.2, 1700, 15], [1.0, 3500, 700]), maxfev=5000)
    rmse = float(np.sqrt(np.mean((logi(zc, *pp) - f) ** 2)))
    return float(pp[0]), float(pp[1]), float(pp[2]), rmse


def profile(v: F, sm: npt.NDArray[np.bool_], thr: float) -> tuple[F, F, F]:
    ok = sm & np.isfinite(v)
    n = np.bincount(zbin[ok], minlength=NB).astype(float)
    k = np.bincount(zbin[ok], weights=(v[ok] >= thr).astype(float), minlength=NB)
    use = n >= 50
    return zc_all[use], k[use] / n[use], n[use]


rng = np.random.default_rng(42)
rows = []
for k, y in enumerate(YEARS):
    v = stk[k].ravel()[idx].astype(np.float64)
    v[v == -32768] = np.nan
    v /= 1000.0
    for sname, sm in STRATA.items():
        zc, fr, n = profile(v, sm, T_MAIN)
        try:
            fmax, z50, s, rmse = fit(zc, fr, n)
        except (RuntimeError, ValueError):
            continue
        bz, bw = [], []
        for _ in range(B):
            sel = rng.integers(0, zc.size, zc.size)
            try:
                rb = fit(zc[sel], fr[sel], n[sel])
            except (RuntimeError, ValueError):
                continue
            bz.append(rb[1])
            bw.append(2 * rb[2] * np.log(9))
        sens = {}
        for thr in T_SENS:
            try:
                sens[thr] = fit(*profile(v, sm, thr))[1]
            except (RuntimeError, ValueError):
                sens[thr] = np.nan
        rows.append(
            {
                "year": int(y),
                "stratum": sname,
                "flag": int(y == 2012),
                "z50": round(z50, 1),
                "s": round(s, 1),
                "fmax": round(fmax, 4),
                "width": round(2 * s * np.log(9), 1),
                "rmse": round(rmse, 4),
                "z50_lo": round(float(np.percentile(bz, 2.5)), 1),
                "z50_hi": round(float(np.percentile(bz, 97.5)), 1),
                "width_lo": round(float(np.percentile(bw, 2.5)), 1),
                "width_hi": round(float(np.percentile(bw, 97.5)), 1),
                "nbins": int(zc.size),
                "z50_t15": round(sens[0.15], 1),
                "z50_t25": round(sens[0.25], 1),
            }
        )
    if k % 10 == 0:
        print(f"year {y} {time.time() - t0:.0f}s", flush=True)
ann = pd.DataFrame(rows)
ann.to_csv(f"{OUT}/ecotone_annual.csv", index=False)

erows = []
for e0, e1 in ((1985, 1994), (1995, 2004), (2005, 2014), (2015, 2025)):
    m = (YEARS >= e0) & (YEARS <= e1) & KEEP
    a = stk[m].reshape(m.sum(), -1)[:, idx].astype(np.float64)
    a[a == -32768] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ve = np.nanmean(a / 1000.0, 0)
    for sname, sm in STRATA.items():
        zc, fr, n = profile(ve, sm, T_MAIN)
        fmax, z50, s, rmse = fit(zc, fr, n)
        ok = sm & np.isfinite(ve)
        med = pd.Series(ve[ok]).groupby(zbin[ok]).median()
        erows.append(
            pd.DataFrame(
                {
                    "epoch": f"{e0}-{e1}",
                    "stratum": sname,
                    "zc": zc,
                    "n": n.astype(int),
                    "f": fr.round(4),
                    "med_ndvi": med.reindex(((zc - 1612.5) / 25).astype(int)).to_numpy().round(4),
                    "fmax": round(fmax, 4),
                    "z50": round(z50, 1),
                    "s": round(s, 1),
                    "rmse": round(rmse, 4),
                }
            )
        )
pd.concat(erows).to_csv(f"{OUT}/ecotone_epochs.csv", index=False)
print(pd.concat(erows).groupby(["stratum", "epoch"])[["z50", "s", "fmax"]].first().to_string())

for sname in STRATA:
    d = ann[(ann.stratum == sname) & (ann.flag == 0)]
    ts = stats.theilslopes(d.z50, d.year, 0.95)
    tau = stats.kendalltau(d.year, d.z50)
    tw = stats.theilslopes(d.width, d.year, 0.95)
    s15 = stats.theilslopes(d.z50_t15.dropna(), d.year[d.z50_t15.notna()])[0]
    s25 = stats.theilslopes(d.z50_t25.dropna(), d.year[d.z50_t25.notna()])[0]
    print(
        f"z50 {sname}: median {d.z50.median():.0f} m; Theil-Sen {10 * ts[0]:+.1f} m/dec "
        f"[{10 * ts[2]:+.1f}, {10 * ts[3]:+.1f}], MK tau {tau.statistic:+.2f} p={tau.pvalue:.4f}; "
        f"width {10 * tw[0]:+.1f} m/dec; T0.15 {10 * s15:+.1f}, T0.25 {10 * s25:+.1f} m/dec; "
        f"s at bound {int((d.s >= 699).sum())}"
    )
print(f"total {time.time() - t0:.0f}s")

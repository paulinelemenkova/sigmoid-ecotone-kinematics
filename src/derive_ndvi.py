#!/usr/bin/env python3

from __future__ import annotations

import os
import sys
import time
import urllib.request
import warnings
from typing import TypeAlias

import numpy as np
import numpy.typing as npt
import pandas as pd
import rasterio
from rasterio.transform import Affine
from scipy import stats
from scipy.optimize import curve_fit
from scipy.special import erfc as verfc

URL, DEM, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
PART = sys.argv[4] if len(sys.argv) > 4 else "all"
Y0, Y1 = (int(t) for t in (sys.argv[5] if len(sys.argv) > 5 else "1984-2025").split("-"))
DO_EPOCHS = (sys.argv[6] if len(sys.argv) > 6 else "epochs") == "epochs"
FULL = (Y0, Y1) == (1984, 2025)
os.makedirs(OUT, exist_ok=True)
t0 = time.time()
YEARS = np.arange(1984, 2026)
KEEP = YEARS != 2012
F: TypeAlias = npt.NDArray[np.float64]

req = urllib.request.Request(URL, headers={"User-Agent": "research-script/1.0"})
with urllib.request.urlopen(req, timeout=180) as r, open("/tmp/stack.tif", "wb") as f:
    f.write(r.read())
with rasterio.open("/tmp/stack.tif") as f:
    stk = f.read()
    prof = f.profile.copy()
    tr = f.transform
with rasterio.open(DEM) as f:
    z = f.read(1).astype(np.float64)
nb, ny, nx = stk.shape
print(f"stack {stk.shape} read in {time.time() - t0:.0f}s", flush=True)

if PART in ("all", "trend"):
    yrs = YEARS[KEEP].astype(np.float32)
    sub = stk[KEEP].reshape(KEEP.sum(), -1)
    n = len(yrs)
    ii, jj = np.triu_indices(n, 1)
    dt = (yrs[jj] - yrs[ii])[:, None]
    slope = np.full(ny * nx, np.nan, np.float32)
    pval = np.full(ny * nx, np.nan, np.float32)
    nval = np.zeros(ny * nx, np.int16)
    CH = 50000
    for a in range(0, ny * nx, CH):
        blk = sub[:, a : a + CH].astype(np.float32)
        blk[blk == -32768] = np.nan
        blk /= 1000.0
        d = blk[jj] - blk[ii]
        sl = d / dt
        nanc = np.isnan(sl).any(0)
        out = np.median(sl, axis=0)
        if nanc.any():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                out[nanc] = np.nanmedian(sl[:, nanc], axis=0)
        slope[a : a + CH] = out
        s = np.nansum(np.sign(d), axis=0)
        nv = np.isfinite(blk).sum(0)
        var = nv * (nv - 1) * (2 * nv + 5) / 18.0
        zs = np.where(s > 0, s - 1, np.where(s < 0, s + 1, 0)) / np.sqrt(np.maximum(var, 1))
        pval[a : a + CH] = verfc(np.abs(zs) / np.sqrt(2)).astype(np.float32)
        nval[a : a + CH] = nv
        pval[a : a + CH][nv < 20] = np.nan
        slope[a : a + CH][nv < 20] = np.nan
    slope2 = slope.reshape(ny, nx) * 10.0
    pval2 = pval.reshape(ny, nx)
    print(f"trend done {time.time() - t0:.0f}s", flush=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)

        def epoch(y0: int, y1: int, how: str = "median") -> npt.NDArray[np.float32]:
            m = (YEARS >= y0) & (YEARS <= y1) & KEEP
            a = stk[m].astype(np.float32)
            a[a == -32768] = np.nan
            res = np.nanmedian(a, 0) if how == "median" else np.nanmean(a, 0)
            return (res / 1000.0).astype(np.float32)

        e1, e2 = epoch(1985, 1990), epoch(2020, 2025)
    print(f"epochs done {time.time() - t0:.0f}s", flush=True)

    def i16(a: npt.NDArray[np.float32], k: float) -> npt.NDArray[np.int16]:
        return np.where(np.isfinite(a), np.clip(np.round(a * k), -32767, 32767), -32768).astype(np.int16)

    p = prof.copy()
    p.update(count=3, dtype="int16", nodata=-32768, compress="deflate", predictor=2, zlevel=6, tiled=True)
    with rasterio.open(f"{OUT}/trend_30m.tif", "w", **p) as f:
        f.write(i16(slope2, 1e4), 1)
        f.write(i16(pval2, 1e4), 2)
        f.write(nval.reshape(ny, nx), 3)
        f.update_tags(
            band1="Theil-Sen NDVI per decade x1e4", band2="Mann-Kendall p x1e4", band3="valid years", excluded="2012"
        )
    p.update(count=2)
    with rasterio.open(f"{OUT}/epochs_30m.tif", "w", **p) as f:
        f.write(i16(e1, 1e3), 1)
        f.write(i16(e2, 1e3), 2)
        f.update_tags(band1="median NDVI 1985-1990 x1000", band2="median NDVI 2020-2025 x1000")

    def agg3(a: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        py, px = (-ny) % 3, (-nx) % 3
        b = np.pad(a, ((0, py), (0, px)), constant_values=np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmean(b.reshape(b.shape[0] // 3, 3, b.shape[1] // 3, 3), axis=(1, 3)).astype(np.float32)

    l90 = [agg3(slope2.astype(np.float32)), agg3(pval2), agg3(e1), agg3(e2)]
    p9 = prof.copy()
    p9.update(
        count=4,
        dtype="int16",
        nodata=-32768,
        compress="deflate",
        predictor=2,
        zlevel=9,
        tiled=True,
        width=l90[0].shape[1],
        height=l90[0].shape[0],
        transform=tr * Affine.scale(3),
    )
    with rasterio.open(f"{OUT}/derived_90m.tif", "w", **p9) as f:
        for k, (a, sc) in enumerate(zip(l90, (1e4, 1e4, 1e3, 1e3)), 1):
            f.write(i16(a, sc), k)
        f.update_tags(bands="1 slope NDVI/dec x1e4; 2 MK p x1e4; 3 median 1985-90 x1000; 4 median 2020-25 x1000")

    for lo, hi in ((1500, 2000), (2000, 2500), (2500, 3000), (3000, 4000)):
        m = (z >= lo) & (z < hi) & np.isfinite(slope2)
        sv, pv = slope2[m], pval2[m]
        print(
            f"trend {lo}-{hi} m: median {np.median(sv):+.4f}/dec  sig+ {100 * np.mean((pv < 0.05) & (sv > 0)):.1f}%"
            f"  sig- {100 * np.mean((pv < 0.05) & (sv < 0)):.1f}%"
        )

if PART in ("all", "ecotone"):
    gy, gx = np.gradient(z, 30.0)
    asp = (np.degrees(np.arctan2(-gx, gy)) + 360.0) % 360.0
    slp = np.degrees(np.arctan(np.hypot(gx, gy)))
    STRATA = {
        "all": np.ones_like(z, bool),
        "N": (asp >= 315) | (asp < 45),
        "E": (asp >= 45) & (asp < 135),
        "S": (asp >= 135) & (asp < 225),
        "W": (asp >= 225) & (asp < 315),
    }
    dom = (z >= 1600) & (z < 3600) & (slp > 2)
    zb = np.floor(z / 25.0) * 25.0 + 12.5
    idx = np.flatnonzero(dom)
    zbi = zb.ravel()[idx]
    strat_codes = {k: v.ravel()[idx] for k, v in STRATA.items()}

    def logi(zz: F, nmin: float, nmax: float, z50: float, s: float) -> F:
        return nmin + (nmax - nmin) / (1.0 + np.exp((zz - z50) / s))

    def fit(zc: F, med: F, w: F) -> tuple[float, float, float, float, float]:
        p0 = [float(np.percentile(med, 5)), float(np.percentile(med, 95)), 2400.0, 150.0]
        pp, _ = curve_fit(
            logi, zc, med, p0=p0, sigma=1 / np.sqrt(w), bounds=([0, 0, 1700, 15], [0.6, 0.9, 3500, 900]), maxfev=4000
        )
        rmse = float(np.sqrt(np.mean((logi(zc, *pp) - med) ** 2)))
        return float(pp[0]), float(pp[1]), float(pp[2]), float(pp[3]), rmse

    def binned(vals: F, mask: npt.NDArray[np.bool_]) -> pd.DataFrame:
        df = pd.DataFrame({"zb": zbi[mask], "v": vals[mask]}).dropna()
        g = df.groupby("zb")["v"]
        out = pd.DataFrame({"med": g.median(), "q25": g.quantile(0.25), "q75": g.quantile(0.75), "n": g.size()})
        return out[out["n"] >= 50].reset_index()

    rng = np.random.default_rng(42)
    B = 80
    rows = []
    for k, y in enumerate(YEARS):
        if not Y0 <= y <= Y1:
            continue
        v = stk[k].ravel()[idx].astype(np.float64)
        v[v == -32768] = np.nan
        v /= 1000.0
        for sname, sm in strat_codes.items():
            tb = binned(v, sm)
            zc, med, w = tb["zb"].to_numpy(float), tb["med"].to_numpy(float), tb["n"].to_numpy(float)
            try:
                nmin, nmax, z50, s, rmse = fit(zc, med, w)
            except (RuntimeError, ValueError):
                continue
            bz, bw = [], []
            for _ in range(B):
                sel = rng.integers(0, len(zc), len(zc))
                try:
                    r_ = fit(zc[sel], med[sel], w[sel])
                    bz.append(r_[2])
                    bw.append(2 * r_[3] * np.log(9))
                except (RuntimeError, ValueError):
                    continue
            rows.append(
                {
                    "year": int(y),
                    "stratum": sname,
                    "flag": int(y == 2012),
                    "z50": round(z50, 1),
                    "s": round(s, 1),
                    "nmin": round(nmin, 4),
                    "nmax": round(nmax, 4),
                    "width": round(2 * s * np.log(9), 1),
                    "rmse": round(rmse, 4),
                    "z50_lo": round(float(np.percentile(bz, 2.5)), 1) if bz else np.nan,
                    "z50_hi": round(float(np.percentile(bz, 97.5)), 1) if bz else np.nan,
                    "width_lo": round(float(np.percentile(bw, 2.5)), 1) if bw else np.nan,
                    "width_hi": round(float(np.percentile(bw, 97.5)), 1) if bw else np.nan,
                    "nbins": len(zc),
                }
            )
        if k % 10 == 0:
            print(f"ecotone year {y} {time.time() - t0:.0f}s", flush=True)
    ann = pd.DataFrame(rows)
    ann.to_csv(f"{OUT}/ecotone_annual.csv" if FULL else f"{OUT}/ecotone_annual_{Y0}_{Y1}.csv", index=False)

    erows = []
    for e0, e1_ in ((1985, 1994), (1995, 2004), (2005, 2014), (2015, 2025)) if DO_EPOCHS else ():
        m = (YEARS >= e0) & (YEARS <= e1_) & KEEP
        a = stk[m][:, :, :].reshape(m.sum(), -1)[:, idx].astype(np.float64)
        a[a == -32768] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            ve = np.nanmean(a / 1000.0, 0)
        for sname, sm in strat_codes.items():
            tb = binned(ve, sm)
            nmin, nmax, z50, s, rmse = fit(tb["zb"].to_numpy(float), tb["med"].to_numpy(float), tb["n"].to_numpy(float))
            tb["epoch"] = f"{e0}-{e1_}"
            tb["stratum"] = sname
            tb["nmin"], tb["nmax"], tb["z50"], tb["s"], tb["rmse"] = nmin, nmax, z50, s, rmse
            erows.append(tb)
    if erows:
        pd.concat(erows).round(5).to_csv(f"{OUT}/ecotone_epochs.csv", index=False)

    for sname in STRATA if FULL else ():
        d = ann[(ann.stratum == sname) & (ann.flag == 0)]
        res = stats.theilslopes(d.z50, d.year, 0.95)
        tau = stats.kendalltau(d.year, d.z50)
        wres = stats.theilslopes(d.width, d.year, 0.95)
        print(
            f"z50 {sname}: mean {d.z50.mean():.0f} m, trend {10 * res[0]:+.1f} m/dec [{10 * res[2]:+.1f}, {10 * res[3]:+.1f}]"
            f" p={tau.pvalue:.4f}; width trend {10 * wres[0]:+.1f} m/dec; first-epoch z50 "
            f"{d[d.year <= 1994].z50.median():.0f}, last {d[d.year >= 2015].z50.median():.0f}"
        )
print(f"total {time.time() - t0:.0f}s")

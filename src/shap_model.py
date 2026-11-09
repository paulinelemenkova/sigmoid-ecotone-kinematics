#!/usr/bin/env python3

from __future__ import annotations

import sys
import warnings
from typing import TypeAlias

import lightgbm as lgb
import numpy as np
import numpy.typing as npt
import rasterio
from rasterio.warp import transform as warp_transform
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold

import geoconf as G

F: TypeAlias = npt.NDArray[np.float64]
NAMES = [
    "GDD trend",
    "SCD trend",
    "SCD mean",
    "Insolation",
    "Elevation",
    "Slope",
    "Northness",
    "Eastness",
    "Landform",
]


def model() -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="regression",
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=50,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        n_estimators=600,
        verbose=-1,
        random_state=42,
    )


def blocks(a: F, ny: int, nx: int) -> F:
    p = np.full((ny * 3, nx * 3), np.nan)
    p[: a.shape[0], : a.shape[1]] = a
    return p.reshape(ny, 3, nx, 3).transpose(0, 2, 1, 3).reshape(ny, nx, 9)


def shift90(a: F, ny: int, nx: int) -> F:
    return a[:ny, :nx] / 3.0 + a[:ny, 1 : nx + 1] * 2.0 / 3.0


def read(path: str, band: int = 1) -> F:
    with rasterio.open(path) as f:
        a = f.read(band).astype(np.float64)
        if f.nodata is not None and np.isfinite(f.nodata):
            a[a == f.nodata] = np.nan
    return a


def gdd_trend_field(table: str, lon: F, lat: F, z: F) -> F:
    t = np.loadtxt(table, skiprows=1)
    lons, lats = np.unique(t[:, 1]), np.unique(t[:, 0])
    grid = np.empty((len(lats), len(lons)), dtype=object)
    for i, la in enumerate(lats):
        for j, lo in enumerate(lons):
            r = t[(t[:, 0] == la) & (t[:, 1] == lo)]
            grid[i, j] = r[np.argsort(r[:, 2])]
    x = np.clip(lon, lons[0], lons[-1])
    y = np.clip(lat, lats[0], lats[-1])
    out = np.zeros_like(z)
    wsum = np.zeros_like(z)
    for i, la in enumerate(lats):
        wy = np.clip(1 - np.abs(y - la) / np.diff(lats).min(), 0, 1)
        for j, lo in enumerate(lons):
            wx = np.clip(1 - np.abs(x - lo) / np.diff(lons).min(), 0, 1)
            r = grid[i, j]
            out += wx * wy * np.interp(z, r[:, 2], r[:, 4])
            wsum += wx * wy
    return out / wsum


def main() -> None:
    warnings.simplefilter("ignore", RuntimeWarning)
    d, out = sys.argv[1], sys.argv[2]
    with rasterio.open(f"{d}/derived/derived_90m.tif") as f:
        dv = f.read().astype(np.float64)
        tr = f.transform
    dv[dv == -32768] = np.nan
    y_all = dv[0] / 1e4
    ny, nx = y_all.shape
    z = shift90(read(f"{d}/terrain/dem_90m.tif"), ny, nx)
    rad = read(f"{d}/terrain/rad_jjas_90m.tif")
    rad[rad == 65535] = np.nan
    rad = shift90(rad, ny, nx)
    a = read(f"{d}/terrain/aspect_30m.tif")
    a[(a == 0) | (a == 65535)] = np.nan
    s = read(f"{d}/terrain/slope_30m.tif")
    s[s == 255] = np.nan
    g = read(f"{d}/terrain/geomorph_30m.tif")
    g[g == 0] = np.nan
    ab = np.radians(blocks(a, ny, nx))
    asp = np.arctan2(np.nanmean(np.sin(ab), 2), np.nanmean(np.cos(ab), 2))
    slope = np.nanmean(blocks(s, ny, nx), 2)
    gb = blocks(g, ny, nx)
    counts = np.stack([(gb == c).sum(2) for c in range(1, 11)], 2)
    geo = np.where(counts.max(2) > 0, counts.argmax(2) + 1, np.nan)
    scd_tr, scd_mean = (
        read(f"{d}/derived/scd_90m.tif", 1),
        read(f"{d}/derived/scd_90m.tif", 2),
    )
    cols, rows = np.meshgrid(np.arange(nx), np.arange(ny))
    xs = tr.c + (cols + 0.5) * tr.a
    ys = tr.f + (rows + 0.5) * tr.e
    lon, lat = warp_transform(G.CRS, "EPSG:4326", xs.ravel(), ys.ravel())
    gdd_tr = gdd_trend_field(
        f"{d}/climate/gdd_points.csv",
        np.array(lon).reshape(ny, nx),
        np.array(lat).reshape(ny, nx),
        z,
    )
    feats = np.stack([gdd_tr, scd_tr, scd_mean, rad, z, slope, np.cos(asp), np.sin(asp), geo], -1)
    m = (z >= 2200) & (slope >= 5) & np.isfinite(y_all) & np.isfinite(feats).all(-1)
    X, y = feats[m], y_all[m]
    groups = (rows[m] // 56) * 100 + cols[m] // 56
    print(f"cells {len(y)}, blocks {len(np.unique(groups))}, median trend {np.median(y):+.4f} NDVI/dec")
    rho = spearmanr(X[:, 0], X[:, 4]).statistic
    print(
        f"Spearman rho(GDD trend, elevation) = {rho:+.3f}; rho(SCD mean, elevation) = "
        f"{spearmanr(X[:, 2], X[:, 4]).statistic:+.3f}"
    )
    cat = [8]
    metrics: dict[str, list[float]] = {}
    for name, cv, grp in (
        ("block", GroupKFold(5), groups),
        ("random", KFold(5, shuffle=True, random_state=1), None),
    ):
        pred = np.zeros_like(y)
        for tr_i, te_i in cv.split(X, y, grp):
            mdl = model()
            mdl.fit(X[tr_i], y[tr_i], categorical_feature=cat)
            pred[te_i] = mdl.predict(X[te_i])
        metrics[name] = [
            float(r2_score(y, pred)),
            float(np.sqrt(mean_squared_error(y, pred))),
            float(mean_absolute_error(y, pred)),
        ]
        print(f"{name} CV: R2 {metrics[name][0]:.3f}  RMSE {metrics[name][1]:.5f}  MAE {metrics[name][2]:.5f}")
    mdl = model()
    mdl.fit(X, y, categorical_feature=cat)
    contrib = np.asarray(mdl.predict(X, pred_contrib=True))
    shap, base = contrib[:, :-1], float(contrib[0, -1])
    mabs = np.abs(shap).mean(0)
    order = np.argsort(-mabs)
    for k in order:
        print(f"  {NAMES[k]:11s} mean|SHAP| {1e3 * mabs[k]:.3f} e-3 NDVI/dec")
    print(f"base value {base:+.5f}, train R2 {r2_score(y, mdl.predict(X)):.3f}")
    np.savez_compressed(
        out,
        names=np.array(NAMES),
        X=X.astype(np.float32),
        shap=shap.astype(np.float32),
        y=y.astype(np.float32),
        mabs=mabs,
        block=np.array(metrics["block"]),
        random=np.array(metrics["random"]),
        n=np.array([len(y), len(np.unique(groups))]),
        base=np.array([base]),
        rho=np.array([rho]),
    )


if __name__ == "__main__":
    main()

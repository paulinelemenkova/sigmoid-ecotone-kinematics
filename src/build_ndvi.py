#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypeAlias

import numpy as np
import numpy.typing as npt
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT

import geoconf as G

SEL, OUT, YEARS = sys.argv[1], sys.argv[2], [int(y) for y in sys.argv[3:]]
W, E, S, N = G.W, G.E, G.S, G.N
CRS, NX, NY = G.CRS, G.NX, G.NY
TR = from_origin(G.X0, G.Y0, 30.0, 30.0)
PC = "https://planetarycomputer.microsoft.com/api"
UA = {"User-Agent": "research-stac-script/1.0", "Content-Type": "application/json"}
ND = -32768
F32: TypeAlias = npt.NDArray[np.float32]
ETM = {"red": (0.9047, 0.0061), "nir": (0.8462, 0.0412)}
HLS = {
    "Sentinel-2A": {"red": (0.9765, 0.0009), "nir": (0.9983, -0.0001)},
    "Sentinel-2B": {"red": (0.9761, 0.0010), "nir": (0.9966, 0.0000)},
}
HLS["Sentinel-2C"] = HLS["Sentinel-2A"]
ENV = {"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR", "GDAL_HTTP_MAX_RETRY": "4", "GDAL_HTTP_RETRY_DELAY": "2"}


def get(url: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers=UA, method="POST" if body is not None else "GET")
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                out: dict[str, Any] = json.load(r)
            return out
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 + 3 * attempt)
    raise RuntimeError("unreachable")


TOK = {c: get(f"{PC}/sas/v1/token/{c}")["token"] for c in ("landsat-c2-l2", "sentinel-2-l2a")}


def warp(href: str, coll: str, rs: Resampling, ovl: int | None = None) -> F32:
    kw: dict[str, Any] = {} if ovl is None else {"overview_level": ovl}
    with rasterio.Env(**ENV), rasterio.open(f"{href}?{TOK[coll]}", **kw) as src:
        vrt = WarpedVRT(src, crs=CRS, transform=TR, width=NX, height=NY, resampling=rs, src_nodata=0, nodata=0)
        a = vrt.read(1).astype(np.float32)
        vrt.close()
    return a


def items_for(coll: str, platform: str, date: str) -> list[dict[str, Any]]:
    body = {
        "collections": [coll],
        "bbox": [W, S, E, N],
        "datetime": f"{date}T00:00:00Z/{date}T23:59:59Z",
        "limit": 50,
    }
    its = [i for i in get(f"{PC}/stac/v1/search", body)["features"] if i["properties"].get("platform") == platform]
    if coll.startswith("landsat"):
        its = [
            i for i in its if str(i["properties"].get("landsat:wrs_path")) == G.WRS_PATH and i["id"].endswith("_T1")
        ] or [i for i in its if str(i["properties"].get("landsat:wrs_path")) == G.WRS_PATH]
    else:
        best: dict[str, dict[str, Any]] = {}
        for i in its:
            t = i["properties"].get("s2:mgrs_tile", "")
            if t not in best or i["id"] > best[t]["id"]:
                best[t] = i
        its = list(best.values())
    return its


def landsat_obs(platform: str, date: str) -> F32:
    ndvi = np.full((NY, NX), np.nan, np.float32)
    for it in items_for("landsat-c2-l2", platform, date):
        a = it["assets"]
        qa = warp(a["qa_pixel"]["href"], "landsat-c2-l2", Resampling.nearest).astype(np.uint16)
        red = warp(a["red"]["href"], "landsat-c2-l2", Resampling.bilinear) * 2.75e-5 - 0.2
        nir = warp(a["nir08"]["href"], "landsat-c2-l2", Resampling.bilinear) * 2.75e-5 - 0.2
        if platform in ("landsat-4", "landsat-5", "landsat-7"):
            red = ETM["red"][0] * red + ETM["red"][1]
            nir = ETM["nir"][0] * nir + ETM["nir"][1]
        bad = (qa == 0) | ((qa & (1 | 2 | 4 | 8 | 16 | 32)) != 0)
        ok = ~bad & (red > 0) & (red < 1) & (nir > 0) & (nir < 1)
        v = np.where(ok, (nir - red) / (nir + red), np.nan).astype(np.float32)
        fill = np.isnan(ndvi) & ~np.isnan(v)
        ndvi[fill] = v[fill]
    return ndvi


def s2_obs(platform: str, date: str) -> F32:
    ndvi = np.full((NY, NX), np.nan, np.float32)
    cr, cn = HLS[platform]["red"], HLS[platform]["nir"]
    for it in items_for("sentinel-2-l2a", platform, date):
        a = it["assets"]
        off = -1000.0 if float(it["properties"].get("s2:processing_baseline", "0") or 0) >= 4.0 else 0.0
        scl = warp(a["SCL"]["href"], "sentinel-2-l2a", Resampling.mode)
        b04 = warp(a["B04"]["href"], "sentinel-2-l2a", Resampling.average, ovl=0)
        b8a = warp(a["B8A"]["href"], "sentinel-2-l2a", Resampling.average)
        valid = (b04 > 0) & (b8a > 0)
        red = cr[0] * (b04 + off) / 1e4 + cr[1]
        nir = cn[0] * (b8a + off) / 1e4 + cn[1]
        ok = valid & np.isin(scl, (4, 5, 6)) & (red > 0) & (red < 1) & (nir > 0) & (nir < 1)
        v = np.where(ok, (nir - red) / (nir + red), np.nan).astype(np.float32)
        fill = np.isnan(ndvi) & ~np.isnan(v)
        ndvi[fill] = v[fill]
    return ndvi


def median(stack: list[F32]) -> tuple[F32, npt.NDArray[np.int16]]:
    if not stack:
        return np.full((NY, NX), np.nan, np.float32), np.zeros((NY, NX), np.int16)
    arr = np.stack(stack)
    n = np.isfinite(arr).sum(0).astype(np.int16)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(arr, axis=0).astype(np.float32)
    return med, n


def to_i16(a: F32) -> npt.NDArray[np.int16]:
    return np.where(np.isfinite(a), np.round(a * 1000), ND).astype(np.int16)


with open(SEL) as fsel:
    sel = json.load(fsel)
os.makedirs(OUT, exist_ok=True)
t0 = time.time()
tasks = [(y, k, p, d) for y in YEARS for k in ("L", "S") for p, d in sel[str(y)][k]]


def run(task: tuple[int, str, str, str]) -> F32:
    _, kind, platform, date = task
    return landsat_obs(platform, date) if kind == "L" else s2_obs(platform, date)


with ThreadPoolExecutor(max_workers=12) as ex:
    res = list(ex.map(run, tasks))
print(f"read {len(tasks)} acquisitions in {time.time() - t0:.0f}s", flush=True)
for y in YEARS:
    s = sel[str(y)]
    lobs = [r for t, r in zip(tasks, res) if t[0] == y and t[1] == "L"]
    sobs = [r for t, r in zip(tasks, res) if t[0] == y and t[1] == "S"]
    mall, nall = median(lobs + sobs)
    bands = [to_i16(mall), nall]
    if sobs:
        bands += [to_i16(median(lobs)[0]), to_i16(median(sobs)[0])]
    prof = {
        "driver": "GTiff",
        "width": NX,
        "height": NY,
        "count": len(bands),
        "dtype": "int16",
        "crs": CRS,
        "transform": TR,
        "nodata": ND,
        "compress": "deflate",
        "predictor": 2,
        "zlevel": 9,
        "tiled": True,
    }
    fn = os.path.join(OUT, f"ndvi_{y}.tif")
    with rasterio.open(fn, "w", **prof) as f:
        for k, b in enumerate(bands, 1):
            f.write(b, k)
        f.update_tags(
            year=y, window_doy=s["win"], n_landsat=len(lobs), n_s2=len(sobs), scale="0.001", composite="median"
        )
    cov = float(np.isfinite(mall).mean())
    msg = f"{y}: L={len(lobs)} S={len(sobs)} win={s['win']} coverage={cov:.3f} median_ndvi={np.nanmedian(mall):.3f}"
    msg += f" mean_nobs={nall[nall > 0].mean():.1f} size={os.path.getsize(fn) / 1e6:.1f}MB"
    if sobs:
        ml, ms = median(lobs)[0], median(sobs)[0]
        both = np.isfinite(ml) & np.isfinite(ms)
        msg += f" L-S bias={np.mean(ml[both] - ms[both]):+.4f} rmsd={np.sqrt(np.mean((ml[both] - ms[both]) ** 2)):.4f}"
    print(msg, flush=True)
print(f"total {time.time() - t0:.0f}s")

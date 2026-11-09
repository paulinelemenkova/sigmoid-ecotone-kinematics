#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from math import erfc, sqrt
from typing import Any, TypeAlias

import numpy as np
import numpy.typing as npt
import rasterio
import rasterio.errors
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT

import geoconf as G

OUT = sys.argv[1]
NTH = int(sys.argv[2]) if len(sys.argv) > 2 else 16
PC = "https://planetarycomputer.microsoft.com/api"
UA = {"User-Agent": "research-stac-script/1.0", "Content-Type": "application/json"}
RES = 90.0
CRS = G.CRS
NX, NY = G.shape(RES)
TR = from_origin(G.X0, G.Y0, RES, RES)
Y0, Y1 = 2001, 2025
ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "GDAL_HTTP_MAX_RETRY": "4",
    "GDAL_HTTP_TIMEOUT": "30",
}
F64: TypeAlias = npt.NDArray[np.float64]
U8: TypeAlias = npt.NDArray[np.uint8]


def get(url: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers=UA, method="POST" if data else "GET")
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                out: dict[str, Any] = json.load(r)
            return out
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2 + 3 * attempt)
    raise RuntimeError("unreachable")


def items() -> list[dict[str, Any]]:
    body: dict[str, Any] = {
        "collections": ["modis-10A2-061"],
        "bbox": [G.W, G.S, G.E, G.N],
        "datetime": f"{Y0 - 1}-09-01T00:00:00Z/{Y1}-08-31T23:59:59Z",
        "limit": 500,
    }
    feats: list[dict[str, Any]] = []
    url = f"{PC}/stac/v1/search"
    while True:
        res = get(url, body)
        feats += [f for f in res["features"] if f["id"].startswith("MOD10A2") and f".{G.MODIS_TILE}." in f["id"]]
        nxt = [lk for lk in res.get("links", []) if lk.get("rel") == "next"]
        if not nxt:
            return feats
        body = nxt[0].get("body", body)


TOK = get(f"{PC}/sas/v1/token/modiseuwest/modis-061-cogs")["token"]


def read(href: str) -> U8 | None:
    try:
        with rasterio.Env(**ENV), rasterio.open(f"{href}?{TOK}") as src:
            vrt = WarpedVRT(
                src,
                crs=CRS,
                transform=TR,
                width=NX,
                height=NY,
                resampling=Resampling.nearest,
            )
            a: U8 = vrt.read(1)
            vrt.close()
        return a
    except (rasterio.errors.RasterioError, OSError) as exc:
        print("read failed", href[-60:], exc, flush=True)
        return None


def snow_year(dt: str) -> int:
    y, m = int(dt[:4]), int(dt[5:7])
    return y + 1 if m >= 9 else y


def theil_sen_mk(v: F64, t: F64) -> tuple[F64, F64]:
    n = v.shape[0]
    i, j = np.triu_indices(n, 1)
    d = v[j] - v[i]
    slopes = d / (t[j] - t[i]).reshape(-1, *([1] * (v.ndim - 1)))
    ts = np.nanmedian(slopes, 0)
    s = np.nansum(np.sign(d), 0)
    nv = np.isfinite(v).sum(0).astype(np.float64)
    var = nv * (nv - 1) * (2 * nv + 5) / 18.0
    zs = np.where(s > 0, s - 1, np.where(s < 0, s + 1, 0)) / np.sqrt(np.maximum(var, 1e-9))
    p: F64 = np.vectorize(lambda x: erfc(abs(x) / sqrt(2)))(zs)
    return ts, p


def main() -> None:
    t0 = time.time()
    feats = items()
    print(f"{len(feats)} MOD10A2 composites found in {time.time() - t0:.0f} s", flush=True)
    ny = Y1 - Y0 + 1
    nsnow = np.zeros((ny, NY, NX), np.int16)
    nval = np.zeros((ny, NY, NX), np.int16)
    jobs = [
        (
            snow_year(f["properties"]["start_datetime"]),
            f["assets"]["Maximum_Snow_Extent"]["href"],
        )
        for f in feats
    ]
    with ThreadPoolExecutor(NTH) as ex:
        for (yr, _), a in zip(jobs, ex.map(lambda j: read(j[1]), jobs)):
            if a is None or not Y0 <= yr <= Y1:
                continue
            k = yr - Y0
            nsnow[k] += a == 200
            nval[k] += (a == 200) | (a == 25)
    scd = np.where(nval >= 30, 365.0 * nsnow / np.maximum(nval, 1), np.nan)
    for k in range(ny):
        print(
            Y0 + k,
            f"median SCD {np.nanmedian(scd[k]):.0f} d, median valid composites {np.median(nval[k]):.0f}",
        )
    yrs = np.arange(Y0, Y1 + 1, dtype=np.float64)
    ts, p = np.full((NY, NX), np.nan), np.full((NY, NX), np.nan)
    for r0 in range(0, NY, 40):
        ts[r0 : r0 + 40], p[r0 : r0 + 40] = theil_sen_mk(scd[:, r0 : r0 + 40], yrs)
    out = np.stack([ts * 10, np.nanmean(scd, 0), p]).astype(np.float32)
    with rasterio.open(
        OUT,
        "w",
        driver="GTiff",
        width=NX,
        height=NY,
        count=3,
        dtype="float32",
        crs=CRS,
        transform=TR,
        nodata=np.nan,
        compress="deflate",
    ) as f:
        f.write(out)
    print(f"SCD trend median {np.nanmedian(out[0]):+.1f} d/decade; done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()

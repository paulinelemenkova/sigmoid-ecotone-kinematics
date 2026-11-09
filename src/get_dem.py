#!/usr/bin/env python3

from __future__ import annotations

import json
import math
import sys
import urllib.request
from typing import Any

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject, transform_bounds

import geoconf as G

OUT = sys.argv[1]
W, E, S, N = G.W, G.E, G.S, G.N
CRS, RES = G.CRS, 30.0
PC = "https://planetarycomputer.microsoft.com/api"
UA = {"User-Agent": "research-stac-script/1.0", "Content-Type": "application/json"}


def get(url: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers=UA, method="POST" if body is not None else "GET")
    with urllib.request.urlopen(req, timeout=120) as r:
        out: dict[str, Any] = json.load(r)
    return out


x0, y0, x1, y1 = transform_bounds("EPSG:4326", CRS, W, S, E, N)
x0, y0 = math.floor(x0 / RES) * RES, math.floor(y0 / RES) * RES
x1, y1 = math.ceil(x1 / RES) * RES, math.ceil(y1 / RES) * RES
nx, ny = round((x1 - x0) / RES), round((y1 - y0) / RES)
tr = from_origin(x0, y1, RES, RES)
items = get(f"{PC}/stac/v1/search", {"collections": ["cop-dem-glo-30"], "bbox": [W, S, E, N], "limit": 20})["features"]
token = get(f"{PC}/sas/v1/token/cop-dem-glo-30")["token"]
dst = np.full((ny, nx), np.nan, np.float32)
for it in items:
    href = it["assets"]["data"]["href"]
    with rasterio.open(f"{href}?{token}") as src:
        tile = np.full((ny, nx), np.nan, np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=tile,
            dst_transform=tr,
            dst_crs=CRS,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
    fill = np.isnan(dst) & ~np.isnan(tile)
    dst[fill] = tile[fill]
    print("tile", it["id"])
prof = {
    "driver": "GTiff",
    "width": nx,
    "height": ny,
    "count": 1,
    "dtype": "float32",
    "crs": CRS,
    "transform": tr,
    "nodata": -9999.0,
    "compress": "deflate",
    "predictor": 3,
    "tiled": True,
}
with rasterio.open(OUT, "w", **prof) as f:
    f.write(np.where(np.isnan(dst), -9999.0, dst).astype(np.float32), 1)
print("grid", nx, ny, "bounds", x0, y0, x1, y1, "nan cells", int(np.isnan(dst).sum()))
print("elevation min/max", float(np.nanmin(dst)), float(np.nanmax(dst)))

#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import os
import sys
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import numpy.typing as npt
import rasterio
import rasterio.errors
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds

import geoconf as G

COLL, Y0, Y1, OUT = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
W, E, S, N = G.W, G.E, G.S, G.N
API = "https://planetarycomputer.microsoft.com/api/stac/v1"
UA = {"User-Agent": "research-stac-script/1.0", "Content-Type": "application/json"}
RES = 120.0
CRS = G.CRS
x0, y0, x1, y1 = transform_bounds("EPSG:4326", CRS, W, S, E, N)
NX, NY = int(np.ceil((x1 - x0) / RES)), int(np.ceil((y1 - y0) / RES))
TR = from_origin(x0, y1, RES, RES)
_xx, _yy = np.meshgrid(x0 + RES * (np.arange(NX) + 0.5), y1 - RES * (np.arange(NY) + 0.5))
from rasterio.warp import transform as _tf

_lon, _lat = _tf(CRS, "EPSG:4326", _xx.ravel(), _yy.ravel())
INWIN = ((np.array(_lon) >= W) & (np.array(_lon) <= E) & (np.array(_lat) >= S) & (np.array(_lat) <= N)).reshape(NY, NX)


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


def search() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    body: dict[str, Any] = {
        "collections": [COLL],
        "bbox": [W, S, E, N],
        "datetime": f"{Y0}-01-01T00:00:00Z/{Y1}-12-31T23:59:59Z",
        "limit": 1000,
    }
    while True:
        page = get(f"{API}/search", body)
        items += page["features"]
        nxt = [lk for lk in page.get("links", []) if lk.get("rel") == "next"]
        if not nxt:
            return items
        body = nxt[0].get("body", body)


t0 = time.time()
items = search()
token = get(f"https://planetarycomputer.microsoft.com/api/sas/v1/token/{COLL}")["token"]
LANDSAT = COLL.startswith("landsat")
groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
for it in items:
    p = it["properties"]
    date = p["datetime"][:10]
    key = (p.get("platform", ""), date, str(p.get("landsat:wrs_path", "")) if LANDSAT else "")
    groups[key].append(it)
print(COLL, Y0, Y1, "items", len(items), "acquisitions", len(groups), f"{time.time() - t0:.0f}s", flush=True)


def read_one(it: dict[str, Any]) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
    asset = it["assets"]["qa_pixel" if LANDSAT else "SCL"]["href"]
    env = rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MAX_RETRY="3", GDAL_HTTP_RETRY_DELAY="2")
    with env, rasterio.open(f"{asset}?{token}", overview_level=1) as src:
        vrt = WarpedVRT(src, crs=CRS, transform=TR, width=NX, height=NY, resampling=Resampling.nearest, nodata=0)
        a = vrt.read(1)
        vrt.close()
    if LANDSAT:
        valid = (a & 1) == 0
        valid &= a != 0
        usable = valid & ((a >> 6) & 1 == 1) & ((a >> 4) & 1 == 0) & ((a >> 5) & 1 == 0)
    else:
        valid = a != 0
        usable = np.isin(a, (4, 5, 6))
    return valid, usable


PRIOR: dict[tuple[str, str, str], dict[str, Any]] = {}
if len(sys.argv) > 5:
    with open(sys.argv[5]) as fp:
        for pr in csv.DictReader(fp):
            if pr["status"] == "ok":
                PRIOR[(pr["platform"], pr["date"], pr["path"])] = pr
BUDGET = 190.0


def process(key: tuple[str, str, str]) -> dict[str, Any]:
    its = groups[key]
    month = int(key[1][5:7])
    row: dict[str, Any] = {
        "platform": key[0],
        "date": key[1],
        "path": key[2],
        "n_items": len(its),
        "tiles": "+".join(
            sorted({str(i["properties"].get("landsat:wrs_row" if LANDSAT else "s2:mgrs_tile", "")) for i in its})
        ),
        "tier": "+".join(sorted({i["id"].split("_")[-1] for i in its})) if LANDSAT else "",
        "scene_cloud": round(float(np.mean([float(i["properties"].get("eo:cloud_cover", np.nan)) for i in its])), 2),
        "season": int(6 <= month <= 9),
        "valid": "",
        "usable": "",
        "clear": "",
        "status": "",
    }
    if not row["season"]:
        return row
    if key in PRIOR:
        return {k: PRIOR[key].get(k, v) for k, v in row.items()}
    if time.time() - t0 > BUDGET:
        row["status"] = "todo"
        return row
    try:
        valid = np.zeros((NY, NX), bool)
        usable = np.zeros((NY, NX), bool)
        for it in its:
            v, u = read_one(it)
            valid |= v
            usable |= u
        nwin = INWIN.sum()
        row["valid"] = round(float((valid & INWIN).sum() / nwin), 4)
        row["usable"] = round(float((usable & INWIN).sum() / nwin), 4)
        row["clear"] = round(float((usable & INWIN).sum() / max(int((valid & INWIN).sum()), 1)), 4)
        row["status"] = "ok"
    except (OSError, RuntimeError, ValueError, KeyError, rasterio.errors.RasterioError) as exc:
        row["status"] = f"fail:{type(exc).__name__}"
    return row


keys = sorted(groups)
with ThreadPoolExecutor(max_workers=16) as ex:
    rows = list(ex.map(process, keys))
os.makedirs(OUT, exist_ok=True)
fn = os.path.join(OUT, f"acq_{COLL}_{Y0}_{Y1}.csv")
with open(fn, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
nseas = sum(r["season"] for r in rows)
nok = sum(r["status"] == "ok" for r in rows)
ntodo = sum(r["status"] == "todo" for r in rows)
print("season acquisitions", nseas, "read ok", nok, "todo", ntodo, f"total {time.time() - t0:.0f}s")
print("grid", NX, NY, "window cells", int(INWIN.sum()))

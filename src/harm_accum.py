#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypeAlias

import numpy as np
import numpy.typing as npt
import rasterio
import rasterio.errors
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT

import geoconf as G

Y0, Y1, OUT = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
NTH = int(sys.argv[4]) if len(sys.argv) > 4 else 8
W, E, S, N = G.W, G.E, G.S, G.N
RES = 120.0
CRS = G.CRS
NX, NY = G.shape(RES)
TR = from_origin(G.X0, G.Y0, RES, RES)
PC = "https://planetarycomputer.microsoft.com/api"
UA = {"User-Agent": "research-stac-script/1.0", "Content-Type": "application/json"}
ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "GDAL_HTTP_MAX_RETRY": "4",
    "GDAL_HTTP_RETRY_DELAY": "2",
}
ETM = {"red": (0.9047, 0.0061), "nir": (0.8462, 0.0412)}
STRATA = {"low": (1300, 1700), "ecotone": (2400, 2700), "alpine": (2800, 3100)}
F64: TypeAlias = npt.NDArray[np.float64]
F32: TypeAlias = npt.NDArray[np.float32]


def get(url: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers=UA, method="POST" if body is not None else "GET")
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


def search(coll: str, dt: str | None = None, extra: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    body: dict[str, Any] = {"collections": [coll], "bbox": [W, S, E, N], "limit": 500}
    if dt:
        body["datetime"] = dt
    if extra:
        body.update(extra)
    feats: list[dict[str, Any]] = []
    url = f"{PC}/stac/v1/search"
    while True:
        res = get(url, body)
        feats += res["features"]
        nxt = [lk for lk in res.get("links", []) if lk.get("rel") == "next"]
        if not nxt:
            return feats
        body = nxt[0].get("body", body)


TOK = {c: get(f"{PC}/sas/v1/token/{c}")["token"] for c in ("landsat-c2-l2", "cop-dem-glo-30")}


def warp(href: str, coll: str, rs: Resampling, ovl: int | None = 1) -> F32:
    kw: dict[str, Any] = {} if ovl is None else {"overview_level": ovl}
    with rasterio.Env(**ENV), rasterio.open(f"{href}?{TOK[coll]}", **kw) as src:
        nod = src.nodata if src.nodata is not None else 0
        vrt = WarpedVRT(
            src,
            crs=CRS,
            transform=TR,
            width=NX,
            height=NY,
            resampling=rs,
            src_nodata=nod,
            nodata=nod,
        )
        a = vrt.read(1).astype(np.float32)
        vrt.close()
    return a


def dem() -> tuple[F64, F64]:
    z = np.full((NY, NX), np.nan)
    for it in search("cop-dem-glo-30"):
        with (
            rasterio.Env(**ENV),
            rasterio.open(
                f"{it['assets']['data']['href']}?{TOK['cop-dem-glo-30']}",
                overview_level=1,
            ) as src,
        ):
            vrt = WarpedVRT(
                src,
                crs=CRS,
                transform=TR,
                width=NX,
                height=NY,
                resampling=Resampling.average,
                nodata=-32767,
            )
            a = vrt.read(1).astype(np.float64)
            vrt.close()
        a[a <= -1000] = np.nan
        fill = np.isnan(z) & np.isfinite(a)
        z[fill] = a[fill]
    gy, gx = np.gradient(z, RES)
    return z, np.degrees(np.arctan(np.hypot(gx, gy)))


def acquisition(items: list[dict[str, Any]]) -> tuple[str, F32]:
    v = np.full((NY, NX), np.nan, np.float32)
    plat = items[0]["properties"]["platform"]
    for it in items:
        a = it["assets"]
        try:
            qa = warp(a["qa_pixel"]["href"], "landsat-c2-l2", Resampling.nearest).astype(np.uint16)
            red = warp(a["red"]["href"], "landsat-c2-l2", Resampling.average) * 2.75e-5 - 0.2
            nir = warp(a["nir08"]["href"], "landsat-c2-l2", Resampling.average) * 2.75e-5 - 0.2
        except (
            rasterio.errors.RasterioError,
            OSError,
        ) as exc:
            print("read failed", it["id"], exc, flush=True)
            continue
        ok = (qa != 0) & ((qa & (1 | 2 | 4 | 8 | 16 | 32)) == 0) & (red > 0) & (nir > 0) & (red < 1) & (nir < 1)
        if plat in ("landsat-4", "landsat-5", "landsat-7"):
            red = ETM["red"][0] * red + ETM["red"][1]
            nir = ETM["nir"][0] * nir + ETM["nir"][1]
        nd = np.where(ok, (nir - red) / (nir + red), np.nan).astype(np.float32)
        fill = np.isnan(v) & np.isfinite(nd)
        v[fill] = nd[fill]
    return plat, v


def main() -> None:
    t0 = time.time()
    feats = search(
        "landsat-c2-l2",
        f"{Y0}-03-01T00:00:00Z/{Y1}-11-30T23:59:59Z",
        {"query": {"eo:cloud_cover": {"lt": 50}}},
    )
    groups: dict[str, list[dict[str, Any]]] = {}
    for f in feats:
        p = f["properties"]
        doy = time.strptime(p["datetime"][:10], "%Y-%m-%d").tm_yday
        if str(p.get("landsat:wrs_path")) != G.WRS_PATH or not f["id"].endswith("_T1") or not 60 <= doy <= 335:
            continue
        if p["platform"] not in (
            "landsat-4",
            "landsat-5",
            "landsat-7",
            "landsat-8",
            "landsat-9",
        ):
            continue
        if p["platform"] == "landsat-7" and p["datetime"][:10] > "2003-05-30":
            continue
        groups.setdefault(p["datetime"][:10] + "|" + p["platform"], []).append(f)
    keys = sorted(groups)
    print(
        f"{Y0}-{Y1}: {len(keys)} acquisitions ({len(feats)} items) found in {time.time() - t0:.0f} s",
        flush=True,
    )
    z, slope = dem()
    masks = {k: (z >= lo) & (z < hi) & (slope > 2) for k, (lo, hi) in STRATA.items()}
    xtx = np.zeros((15, NY, NX))
    xty = np.zeros((5, NY, NX))
    yy = np.zeros((NY, NX))
    n = np.zeros((NY, NX))
    iu = np.triu_indices(5)
    rows: list[tuple[str, str, float, float, float, int]] = []

    def job(k: str) -> tuple[str, str, F32]:
        plat, v = acquisition(groups[k])
        return k, plat, v

    with ThreadPoolExecutor(NTH) as ex:
        for k, plat, v in ex.map(job, keys):
            date = k.split("|")[0]
            doy = time.strptime(date, "%Y-%m-%d").tm_yday
            ang = 2 * np.pi * doy / 365.25
            x = np.array([1.0, np.cos(ang), np.sin(ang), np.cos(2 * ang), np.sin(2 * ang)])
            m = np.isfinite(v) & (v > -0.2)
            if m.sum() < 1000:
                continue
            vv = np.where(m, v, 0.0).astype(np.float64)
            mm = m.astype(np.float64)
            xx = np.outer(x, x)[iu]
            xtx += xx[:, None, None] * mm[None]
            xty += x[:, None, None] * vv[None]
            yy += vv * vv
            n += mm
            med = [
                float(np.nanmedian(np.where(masks[s] & m, v, np.nan))) if (masks[s] & m).sum() > 50 else np.nan
                for s in STRATA
            ]
            rows.append((date, plat, med[0], med[1], med[2], int(m.sum())))
    np.savez_compressed(
        OUT,
        xtx=xtx,
        xty=xty,
        yy=yy,
        n=n,
        z=z,
        slope=slope,
        obs=np.array([f"{r[0]} {r[1]} {r[2]:.4f} {r[3]:.4f} {r[4]:.4f} {r[5]}" for r in rows]),
    )
    print(f"{len(rows)} acquisitions used; median obs per cell {np.median(n):.0f}; done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()

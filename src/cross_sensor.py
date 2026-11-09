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
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT

import geoconf as G

MODE, OUT = sys.argv[1], sys.argv[2]
W, E, S, N = G.W, G.E, G.S, G.N
CRS, NX, NY = G.CRS, G.NX, G.NY
TR = from_origin(G.X0, G.Y0, 30.0, 30.0)
PC = "https://planetarycomputer.microsoft.com/api"
UA = {"User-Agent": "research-stac-script/1.0", "Content-Type": "application/json"}
F32: TypeAlias = npt.NDArray[np.float32]
ETM = {"red": (0.9047, 0.0061), "nir": (0.8462, 0.0412)}
HLS = {
    "Sentinel-2A": {"red": (0.9765, 0.0009), "nir": (0.9983, -0.0001)},
    "Sentinel-2B": {"red": (0.9761, 0.0010), "nir": (0.9966, 0.0000)},
}
ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "GDAL_HTTP_MAX_RETRY": "4",
    "GDAL_HTTP_RETRY_DELAY": "2",
}
PAIRS: dict[str, list[tuple[str, str, str, str, str]]] = G.PAIRS
EDGES = np.arange(0.0, 0.8001, 0.005)
BINS = np.arange(0.0, 0.8001, 0.05)


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
        vrt = WarpedVRT(
            src,
            crs=CRS,
            transform=TR,
            width=NX,
            height=NY,
            resampling=rs,
            src_nodata=0,
            nodata=0,
        )
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
        pw = [i for i in its if str(i["properties"].get("landsat:wrs_path")) == G.WRS_PATH]
        its = [i for i in pw if i["id"].endswith("_T1")] or pw
    else:
        best: dict[str, dict[str, Any]] = {}
        for i in its:
            t = i["properties"].get("s2:mgrs_tile", "")
            if t not in best or i["id"] > best[t]["id"]:
                best[t] = i
        its = list(best.values())
    return its


def ndvi_pair(red: F32, nir: F32, ok: npt.NDArray[np.bool_]) -> F32:
    good = ok & (red > 0) & (red < 1) & (nir > 0) & (nir < 1)
    return np.where(good, (nir - red) / (nir + red), np.nan).astype(np.float32)


def scene(platform: str, date: str) -> tuple[F32, F32]:
    raw = np.full((NY, NX), np.nan, np.float32)
    hrm = np.full((NY, NX), np.nan, np.float32)
    if platform.startswith("landsat"):
        for it in items_for("landsat-c2-l2", platform, date):
            a = it["assets"]
            qa = warp(a["qa_pixel"]["href"], "landsat-c2-l2", Resampling.nearest).astype(np.uint16)
            red = warp(a["red"]["href"], "landsat-c2-l2", Resampling.bilinear) * 2.75e-5 - 0.2
            nir = warp(a["nir08"]["href"], "landsat-c2-l2", Resampling.bilinear) * 2.75e-5 - 0.2
            ok = (qa != 0) & ((qa & (1 | 2 | 4 | 8 | 16 | 32)) == 0)
            v_raw = ndvi_pair(red, nir, ok)
            if platform in ("landsat-4", "landsat-5", "landsat-7"):
                red_h = ETM["red"][0] * red + ETM["red"][1]
                nir_h = ETM["nir"][0] * nir + ETM["nir"][1]
                v_h = ndvi_pair(red_h, nir_h, ok)
            else:
                v_h = v_raw
            for dst, v in ((raw, v_raw), (hrm, v_h)):
                fill = np.isnan(dst) & ~np.isnan(v)
                dst[fill] = v[fill]
    else:
        cr, cn = HLS[platform]["red"], HLS[platform]["nir"]
        for it in items_for("sentinel-2-l2a", platform, date):
            a = it["assets"]
            off = -1000.0 if float(it["properties"].get("s2:processing_baseline", "0") or 0) >= 4.0 else 0.0
            scl = warp(a["SCL"]["href"], "sentinel-2-l2a", Resampling.mode)
            b04 = warp(a["B04"]["href"], "sentinel-2-l2a", Resampling.average, ovl=0)
            b8a = warp(a["B8A"]["href"], "sentinel-2-l2a", Resampling.average)
            ok = (b04 > 0) & (b8a > 0) & np.isin(scl, (4, 5, 6))
            red, nir = (b04 + off) / 1e4, (b8a + off) / 1e4
            v_raw = ndvi_pair(red, nir, ok)
            v_h = ndvi_pair(cr[0] * red + cr[1], cn[0] * nir + cn[1], ok)
            for dst, v in ((raw, v_raw), (hrm, v_h)):
                fill = np.isnan(dst) & ~np.isnan(v)
                dst[fill] = v[fill]
    print(f"{platform} {date}: {int(np.isfinite(hrm).sum())} clear pixels", flush=True)
    return raw, hrm


def stats(x32: F32, y32: F32) -> dict[str, Any]:
    x, y = (
        x32.astype(np.float64),
        y32.astype(np.float64),
    )
    b, a = np.polyfit(x, y, 1)
    r2 = float(np.corrcoef(x, y)[0, 1] ** 2)
    d = x - y
    idx = np.digitize((x + y) / 2, BINS) - 1
    mb = [float(np.median(d[idx == k])) if (idx == k).sum() >= 200 else np.nan for k in range(len(BINS) - 1)]
    h, _, _ = np.histogram2d(x, y, bins=[EDGES, EDGES])
    return {
        "n": int(x.size),
        "a": float(a),
        "b": float(b),
        "r2": r2,
        "rmse": float(np.sqrt(np.mean(d**2))),
        "bias": float(d.mean()),
        "binbias": np.array(mb),
        "hist": h.astype(np.int32),
    }


def main() -> None:
    t0 = time.time()
    pairs = PAIRS[MODE]
    jobs = sorted({(p[1], p[2]) for p in pairs} | {(p[3], p[4]) for p in pairs})
    with ThreadPoolExecutor(4) as ex:
        res = dict(zip(jobs, ex.map(lambda j: scene(*j), jobs)))
    print(f"read {len(jobs)} scenes in {time.time() - t0:.0f} s", flush=True)
    out: dict[str, Any] = {}
    for comp in sorted({p[0] for p in pairs}):
        for ver, k in (("raw", 0), ("harm", 1)):
            xs, ys = [], []
            for c, px, dx, py, dy in pairs:
                if c != comp:
                    continue
                x, y = (
                    res[(px, dx)][k],
                    res[(py, dy)][k],
                )
                m = np.isfinite(x) & np.isfinite(y) & (x > -0.1) & (y > -0.1)
                xs.append(x[m])
                ys.append(y[m])
            st = stats(np.concatenate(xs), np.concatenate(ys))
            print(
                f"{comp} {ver}: n={st['n']} y = {st['a']:+.4f} + {st['b']:.4f} x  R2={st['r2']:.3f} "
                f"RMSE={st['rmse']:.4f} bias={st['bias']:+.4f}",
                flush=True,
            )
            for key, val in st.items():
                out[f"{comp}_{ver}_{key}"] = np.asarray(val)
    out["edges"], out["bins"] = EDGES, BINS
    np.savez_compressed(OUT, **out)
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()

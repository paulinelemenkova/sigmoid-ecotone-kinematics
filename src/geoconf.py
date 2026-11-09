#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
from typing import Any

with open(os.environ.get("AOI_CONFIG", os.path.join(os.path.dirname(os.path.abspath(__file__)), "aoi.json"))) as _f:
    C: dict[str, Any] = json.load(_f)

W, S, E, N = (float(v) for v in C["bbox"])
CRS: str = str(C["crs"])
X0, Y0 = float(C["origin"][0]), float(C["origin"][1])
NX, NY = int(C["shape30"][0]), int(C["shape30"][1])
WRS_PATH: str = str(C["wrs_path"])
MODIS_TILE: str = str(C["modis_tile"])
CLIM: tuple[float, float, float] = (float(C["clim"][0]), float(C["clim"][1]), float(C["clim"][2]))
LATS: list[float] = [float(v) for v in C["lattice"]["lat"]]
LONS: list[float] = [float(v) for v in C["lattice"]["lon"]]
PAIRS: dict[str, list[tuple[str, str, str, str, str]]] = {
    k: [(str(a), str(b), str(c), str(d), str(e)) for a, b, c, d, e in v] for k, v in C["pairs"].items()
}
TRANSECTS: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {
    k: ((float(v[0][0]), float(v[0][1])), (float(v[1][0]), float(v[1][1]))) for k, v in C["transects"].items()
}


def shape(res: float) -> tuple[int, int]:
    return math.ceil(NX * 30.0 / res - 1e-9), math.ceil(NY * 30.0 / res - 1e-9)

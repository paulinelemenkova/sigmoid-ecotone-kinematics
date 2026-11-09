#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
import time
import urllib.request
from math import erfc, sqrt
from typing import Any

import numpy as np
import numpy.typing as npt

import geoconf as G

LATS = G.LATS
LONS = G.LONS
ZREF = 2600
ELEVS = list(range(1500, 3851, 50))
LAPSE = 0.0065
UA = {"User-Agent": "research-climate-script/1.0"}


def fetch(elev: int) -> list[dict[str, Any]]:
    la = ",".join(f"{y}" for y in LATS for _ in LONS)
    lo = ",".join(f"{x}" for _ in LATS for x in LONS)
    el = ",".join(str(elev) for _ in range(len(LATS) * len(LONS)))
    url = (
        "https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={la}&longitude={lo}&elevation={el}&start_date=1985-01-01&end_date=2025-12-31"
        "&daily=temperature_2m_mean&models=era5_land&timezone=GMT"
    )
    for attempt in range(6):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=180) as r:
                out: list[dict[str, Any]] = json.load(r)
            return out
        except (OSError, ValueError) as exc:
            print("retry", elev, exc, flush=True)
            time.sleep(30 + 15 * attempt)
    raise RuntimeError("Open-Meteo failed")


def ts_mk(y: npt.NDArray[np.float64], t: npt.NDArray[np.float64]) -> tuple[float, float]:
    i, j = np.triu_indices(len(y), 1)
    slope = float(np.median((y[j] - y[i]) / (t[j] - t[i])))
    s = float(np.sign(y[j] - y[i]).sum())
    n = len(y)
    var = n * (n - 1) * (2 * n + 5) / 18.0
    z = (s - np.sign(s)) / sqrt(var)
    return slope, erfc(abs(z) / sqrt(2))


def main() -> None:
    rows: list[str] = []
    years = np.arange(1985, 2026)
    for loc in fetch(ZREF):
        d = loc["daily"]
        yrs = np.array([int(s[:4]) for s in d["time"]])
        tm = np.array(
            [np.nan if v is None else v for v in d["temperature_2m_mean"]],
            dtype=np.float64,
        )
        for elev in ELEVS:
            g = np.maximum(tm - LAPSE * (elev - ZREF) - 5.0, 0.0)
            gdd = np.array([np.nansum(g[yrs == y]) for y in years])
            sl, p = ts_mk(gdd, years.astype(np.float64))
            rows.append(f"{loc['latitude']:.3f} {loc['longitude']:.3f} {elev} {gdd.mean():.1f} {10 * sl:.2f} {p:.4f}")
        print(rows[-48], flush=True)
    with open(sys.argv[1], "w") as f:
        f.write("lat lon elev gdd_mean gdd_trend p\n" + "\n".join(rows) + "\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

from __future__ import annotations

import sys
import warnings
from typing import TypeAlias

import numpy as np
import numpy.typing as npt
import rasterio

F: TypeAlias = npt.NDArray[np.float64]


def blocks(a: F, ny: int, nx: int) -> F:
    p = np.full((ny * 3, nx * 3), np.nan)
    p[: a.shape[0], : a.shape[1]] = a
    return p.reshape(ny, 3, nx, 3).transpose(0, 2, 1, 3).reshape(ny, nx, 9)


def main() -> None:
    warnings.simplefilter("ignore", RuntimeWarning)
    dv, dm, asp, slp, geo, out = sys.argv[1:7]
    with rasterio.open(dv) as f:
        d = f.read().astype(np.float64)
    d[d == -32768] = np.nan
    trend, pval = d[0] / 1e4, d[1] / 1e4
    ny, nx = trend.shape
    with rasterio.open(dm) as f:
        z90 = f.read(1).astype(np.float64)
    zk = np.full((ny, nx), np.nan)
    zk[:, :] = z90[:ny, :nx] / 3.0 + z90[:ny, 1 : nx + 1] * 2.0 / 3.0
    with rasterio.open(asp) as f:
        a = f.read(1).astype(np.float64)
    a[(a == 0) | (a == 65535)] = np.nan
    with rasterio.open(slp) as f:
        s = f.read(1).astype(np.float64)
    s[s == 255] = np.nan
    with rasterio.open(geo) as f:
        g = f.read(1).astype(np.float64)
    g[g == 0] = np.nan
    ab = np.radians(blocks(a, ny, nx))
    asp90 = np.degrees(np.arctan2(np.nanmean(np.sin(ab), 2), np.nanmean(np.cos(ab), 2))) % 360
    slope90 = np.nanmean(blocks(s, ny, nx), 2)
    gb = blocks(g, ny, nx)
    counts = np.stack([(gb == c).sum(2) for c in range(1, 11)], 2)
    geo90 = np.where(counts.max(2) > 0, counts.argmax(2) + 1, 0)
    alp = np.isfinite(trend) & np.isfinite(pval) & (zk >= 2200) & (slope90 >= 5) & np.isfinite(asp90)
    sigp = (pval < 0.05) & (trend > 0)
    sign = (pval < 0.05) & (trend < 0)
    print("alpine cells", int(alp.sum()), "median trend", float(np.median(trend[alp])))
    rows: list[tuple[float, int, float, float, float, float, float]] = []
    for k in range(16):
        c = k * 22.5
        m = alp & (((asp90 - c + 11.25) % 360) < 22.5)
        q = np.percentile(trend[m], [25, 50, 75])
        rows.append(
            (
                c,
                int(m.sum()),
                q[1],
                q[0],
                q[2],
                100 * sigp[m].mean(),
                100 * sign[m].mean(),
            )
        )
        c0, n0, md, q0, q2, sp, sn = rows[-1]
        print(f"sector {c0:5.1f} n={n0:5d} median {md:+.4f} IQR {q0:+.4f}..{q2:+.4f} sig+ {sp:4.1f}% sig- {sn:4.1f}%")
    np.savetxt(f"{out}/rose16.txt", rows, fmt="%.1f %d %.5f %.5f %.5f %.2f %.2f")
    names = [
        "flat",
        "peak",
        "ridge",
        "shoulder",
        "spur",
        "slope",
        "hollow",
        "footslope",
        "valley",
        "pit",
    ]
    with open(f"{out}/geomorph_box.txt", "w") as fo:
        for c in (2, 3, 5, 6, 7, 9, 10):
            m = alp & (geo90 == c)
            q = np.percentile(trend[m], [5, 25, 50, 75, 95])
            qs = " ".join(f"{v:.5f}" for v in q)
            line = f"{c} {names[c - 1]} {int(m.sum())} {qs} {100 * sigp[m].mean():.2f} {100 * sign[m].mean():.2f}"
            fo.write(line + "\n")
            print(line)


if __name__ == "__main__":
    main()

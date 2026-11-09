#!/usr/bin/env python3

from __future__ import annotations

import sys
import warnings
from typing import TypeAlias

import numpy as np
import numpy.typing as npt
import rasterio
from rasterio.warp import transform

import geoconf as G

F: TypeAlias = npt.NDArray[np.float64]
TRANSECTS = G.TRANSECTS
STEP, HALF = 90.0, 11


def bilinear(a: F, tr: rasterio.Affine, x: F, y: F) -> F:
    col, row = ~tr * (x, y)
    c, r = np.asarray(col) - 0.5, np.asarray(row) - 0.5
    c0, r0 = np.floor(c).astype(int), np.floor(r).astype(int)
    fc, fr = c - c0, r - r0
    out = np.full(c.shape, np.nan)
    ok = (c0 >= 0) & (r0 >= 0) & (c0 + 1 < a.shape[1]) & (r0 + 1 < a.shape[0])
    c0, r0, fc, fr = c0[ok], r0[ok], fc[ok], fr[ok]
    out[ok] = (
        a[r0, c0] * (1 - fc) * (1 - fr)
        + a[r0, c0 + 1] * fc * (1 - fr)
        + a[r0 + 1, c0] * (1 - fc) * fr
        + a[r0 + 1, c0 + 1] * fc * fr
    )
    return out


def main() -> None:
    warnings.simplefilter("ignore", RuntimeWarning)
    dv, dm, out = sys.argv[1:4]
    with rasterio.open(dv) as f:
        d = f.read().astype(np.float64)
        trd, crs = f.transform, f.crs
    d[d == -32768] = np.nan
    with rasterio.open(dm) as f:
        z = f.read(1).astype(np.float64)
        trz = f.transform
    lines, boxes = [], []
    for name, ((lo0, la0), (lo1, la1)) in TRANSECTS.items():
        (x0, x1), (y0, y1) = transform("EPSG:4326", crs, [lo0, lo1], [la0, la1])
        length = float(np.hypot(x1 - x0, y1 - y0))
        ux, uy = (x1 - x0) / length, (y1 - y0) / length
        px, py = -uy, ux
        s = np.arange(0.0, length + 1e-6, STEP)
        o = np.arange(-HALF, HALF + 1) * STEP
        X = x0 + ux * s[:, None] + px * o[None, :]
        Y = y0 + uy * s[:, None] + py * o[None, :]
        zz = bilinear(z, trz, X, Y)
        n1 = bilinear(d[2], trd, X, Y) / 1e3
        n2 = bilinear(d[3], trd, X, Y) / 1e3
        sl = bilinear(d[0], trd, X, Y) / 1e4
        pv = bilinear(d[1], trd, X, Y) / 1e4
        rows = np.column_stack(
            [
                s / 1e3,
                np.nanmin(zz, 1),
                np.nanmedian(zz, 1),
                np.nanmax(zz, 1),
                np.nanmedian(n1, 1),
                np.nanpercentile(n1, 25, 1),
                np.nanpercentile(n1, 75, 1),
                np.nanmedian(n2, 1),
                np.nanpercentile(n2, 25, 1),
                np.nanpercentile(n2, 75, 1),
                np.nanmedian(sl, 1),
                100 * np.nanmean((pv < 0.05) & (sl > 0), 1),
                100 * np.nanmean((pv < 0.05) & (sl < 0), 1),
            ]
        )
        np.savetxt(
            f"{out}/swath_{name}.txt",
            rows,
            fmt="%.3f %.0f %.0f %.0f %.4f %.4f %.4f %.4f %.4f %.4f %.5f %.1f %.1f",
            header="dist_km zmin zmed zmax ndvi8590 q25 q75 ndvi2025 q25 q75 slope_dec pct_sig_pos pct_sig_neg",
        )
        lo, la = transform(crs, "EPSG:4326", [x0, x1], [y0, y1])
        lines.append(f"> {name}\n{lo[0]:.5f} {la[0]:.5f}\n{lo[1]:.5f} {la[1]:.5f}")
        cx = [
            x0 + px * o[0],
            x1 + px * o[0],
            x1 + px * o[-1],
            x0 + px * o[-1],
            x0 + px * o[0],
        ]
        cy = [
            y0 + py * o[0],
            y1 + py * o[0],
            y1 + py * o[-1],
            y0 + py * o[-1],
            y0 + py * o[0],
        ]
        blo, bla = transform(crs, "EPSG:4326", cx, cy)
        boxes.append(f"> {name}\n" + "\n".join(f"{a:.5f} {b:.5f}" for a, b in zip(blo, bla)))
        dz = rows[:, 2]
        for thr in (0.15, 0.20):
            for k, lab in ((4, "1985-90"), (7, "2020-25")):
                below = np.where(rows[:, k] >= thr)[0]
                zt = dz[below[0]] if below.size else np.nan
                print(f"{name}: first station with median NDVI >= {thr} ({lab}) at z = {zt:.0f} m")
        hi = (dz >= 2200) & (dz < 3000)
        print(
            f"{name}: length {length / 1e3:.2f} km, z {np.nanmax(rows[:, 3]):.0f}-{np.nanmin(rows[:, 1]):.0f} m; "
            f"2200-3000 m: dNDVI median {np.nanmedian(rows[hi, 7] - rows[hi, 4]):+.4f}, "
            f"slope median {np.nanmedian(rows[hi, 10]):+.4f}/dec"
        )
    with open(f"{out}/swath_lines.gmt", "w") as fo:
        fo.write("\n".join(lines) + "\n")
    with open(f"{out}/swath_boxes.gmt", "w") as fo:
        fo.write("\n".join(boxes) + "\n")


if __name__ == "__main__":
    main()

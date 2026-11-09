#!/usr/bin/env python3

from __future__ import annotations

import sys
from math import erfc, sqrt
from typing import TypeAlias

import numpy as np
import numpy.typing as npt
import pandas as pd

F: TypeAlias = npt.NDArray[np.float64]


def mk_hr(t: F, x: F) -> tuple[float, float, float, float]:
    m = np.isfinite(x)
    t, x = t[m], x[m]
    n = len(x)
    i, j = np.triu_indices(n, 1)
    slope = float(np.median((x[j] - x[i]) / (t[j] - t[i])))
    s = float(np.sign(x[j] - x[i]).sum())
    var = n * (n - 1) * (2 * n + 5) / 18.0
    r = pd.Series(x - slope * t).rank().to_numpy()
    r = r - r.mean()
    corr = 0.0
    for k in range(1, n - 2):
        rk = float(np.sum(r[:-k] * r[k:]) / np.sum(r * r))
        if abs(rk) > 1.96 / sqrt(n):
            corr += (n - k) * (n - k - 1) * (n - k - 2) * rk
    ratio = max(1 + 2 / (n * (n - 1) * (n - 2)) * corr, 1e-6)

    def p(v: float) -> float:
        z = (s - np.sign(s)) / sqrt(v)
        return erfc(abs(z) / sqrt(2))

    return 10 * slope, p(var), p(var * ratio), ratio


def main() -> None:
    e = pd.read_csv(sys.argv[1])
    e = e[(e.year != 2012) & (e.flag == 0)]
    for st in ["all", "N", "E", "S", "W"]:
        d = e[e.stratum == st]
        for col in ("z50", "width", "z50_t15", "z50_t25"):
            if st != "all" and col != "z50":
                continue
            sl, p0, p1, ratio = mk_hr(d.year.to_numpy(float), d[col].to_numpy(float))
            print(f"{st:3s} {col:8s} n={len(d)} slope/dec {sl:+7.1f}  p_MK {p0:.4f}  p_HR {p1:.4f}  n/n* {ratio:.2f}")
    c = pd.read_csv(sys.argv[2])
    for col in ("jjas_t", "gdd", "scd_snowyear", "precip_oct_may"):
        sl, p0, p1, ratio = mk_hr(c.year.to_numpy(float), c[col].to_numpy(float))
        print(f"clim {col:15s} slope/dec {sl:+8.3f}  p_MK {p0:.4f}  p_HR {p1:.4f}  n/n* {ratio:.2f}")


if __name__ == "__main__":
    main()

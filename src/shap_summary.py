#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
from itertools import pairwise
from typing import Any

import numpy as np


def main() -> None:
    d = np.load(sys.argv[1])
    names = [str(n) for n in d["names"]]
    X = d["X"].astype(np.float64)
    S = 1e3 * d["shap"].astype(np.float64)
    mabs = 1e3 * d["mabs"]
    order = [int(k) for k in np.argsort(-mabs)]
    rank = np.argsort(np.argsort(X, 0), 0) / (len(X) - 1)
    feats: list[dict[str, Any]] = []
    for k in range(len(names)):
        lo, hi = np.percentile(S[:, k], [0.2, 99.8])
        e = np.linspace(lo, hi, 51)
        idx = np.digitize(S[:, k], e) - 1
        ok = (idx >= 0) & (idx < 50)
        cnt = np.bincount(idx[ok], minlength=50)
        rs = np.bincount(idx[ok], weights=rank[ok, k], minlength=50)
        feats.append(
            {
                "edges": np.round(e, 4).tolist(),
                "count": cnt.tolist(),
                "rank": np.round(rs / np.maximum(cnt, 1), 3).tolist(),
            }
        )
    dep: list[dict[str, Any]] = []
    for k in order[:2]:
        x, s = X[:, k], S[:, k]
        xr = np.percentile(x, [0.5, 99.5])
        yr = np.percentile(s, [0.5, 99.5])
        h, _, _ = np.histogram2d(x, s, bins=(40, 30), range=(xr, yr))
        q = np.quantile(x, np.linspace(0, 1, 21))
        cur = []
        for a, b in pairwise(q):
            m = (x >= a) & (x <= b)
            cur.append([float(np.median(x[m])), *np.percentile(s[m], [25, 50, 75]).tolist()])
        dep.append(
            {
                "k": k,
                "xr": xr.tolist(),
                "yr": yr.tolist(),
                "hist": h.astype(int).tolist(),
                "curve": np.round(cur, 4).tolist(),
            }
        )
    out = {
        "names": names,
        "mabs": np.round(mabs, 4).tolist(),
        "order": order,
        "block": d["block"].tolist(),
        "random": d["random"].tolist(),
        "n": [int(v) for v in d["n"]],
        "rho_gdd_z": float(d["rho"][0]),
        "feats": feats,
        "dep": dep,
    }
    with open(sys.argv[2], "w") as f:
        json.dump(out, f, separators=(",", ":"))


if __name__ == "__main__":
    main()

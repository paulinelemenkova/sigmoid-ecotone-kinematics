#!/usr/bin/env python3

from __future__ import annotations

import sys
import typing
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Nimbus Sans", "Helvetica"],
        "font.size": 8.5,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.minor.size": 2.0,
        "ytick.minor.size": 2.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "lines.linewidth": 0.8,
        "patch.linewidth": 0.3,
        "axes.labelpad": 2,
        "pdf.fonttype": 42,
    }
)
assert "dejavu" not in findfont(FontProperties(family=plt.rcParams["font.sans-serif"])).lower()

OKABE = {
    "orange": "#E69F00",
    "blue": "#0072B2",
    "green": "#009E73",
    "vermilion": "#D55E00",
}
LABEL = {
    "etm_oli": ("ETM+", "OLI"),
    "tm_etm": ("TM", "ETM+"),
    "s2_oli": ("MSI", "OLI / OLI-2"),
}


def load(paths: list[str]) -> dict[str, Any]:
    d: dict[str, Any] = {}
    for p in paths:
        with np.load(p) as z:
            d.update({k: z[k] for k in z.files})
    return d


def style(ax: Axes) -> None:
    ax.set_xlim(0, 0.8)
    ax.set_ylim(0, 0.8)
    ax.set_aspect("equal")
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(MultipleLocator(0.2))
        axis.set_minor_locator(AutoMinorLocator(4))
    ax.tick_params(which="both", top=True, right=True)
    ax.grid(which="major", lw=0.3, color="0.88")
    ax.set_axisbelow(True)


def density(ax: Axes, d: dict[str, Any], key: str, norm: LogNorm) -> Any:
    e = d["edges"]
    h = d[f"{key}_hist"].astype(float)
    h[h < 1] = np.nan
    im = ax.pcolormesh(e, e, h.T, cmap="turbo", norm=norm, rasterized=True)
    ax.plot([0, 0.8], [0, 0.8], color="0.35", lw=0.6, ls=(0, (4, 2)), zorder=3)
    a, b = float(d[f"{key}_a"]), float(d[f"{key}_b"])
    ax.plot([0, 0.8], [a, a + b * 0.8], color="black", lw=1.0, zorder=4)
    title = (
        f"y = {a:.3f} + {b:.3f}x\n"
        f"R\u00b2 {float(d[f'{key}_r2']):.3f}, RMSE {float(d[f'{key}_rmse']):.3f}\n"
        f"bias {float(d[f'{key}_bias']):+.4f}, n {int(d[f'{key}_n']) / 1e6:.1f} M"
    )
    ax.set_title(title, fontsize=8, loc="right", pad=3, linespacing=1.2)
    return im


def tag(ax: Axes, s: str) -> None:
    ax.set_title(s, fontsize=10.5, fontweight="bold", loc="left", pad=3)


def main() -> None:
    d = load(sys.argv[1:3])
    out = sys.argv[3]
    vmax = max(
        float(np.nanmax(d[f"{k}_hist"]))
        for k in (
            "etm_oli_raw",
            "etm_oli_harm",
            "tm_etm_harm",
            "s2_oli_raw",
            "s2_oli_harm",
        )
    )
    norm = LogNorm(vmin=1, vmax=vmax)
    fig = plt.figure(figsize=(17.5 / 2.54, 12.4 / 2.54))
    gs = fig.add_gridspec(
        2,
        4,
        width_ratios=[1, 1, 1, 0.06],
        left=0.075,
        right=0.905,
        bottom=0.19,
        top=0.905,
        wspace=0.22,
        hspace=0.5,
    )
    panels = [
        (0, 0, "etm_oli_raw", "(a)", "native"),
        (0, 1, "etm_oli_harm", "(b)", "harmonised"),
        (0, 2, "tm_etm_harm", "(c)", "harmonised"),
        (1, 0, "s2_oli_raw", "(d)", "native"),
        (1, 1, "s2_oli_harm", "(e)", "harmonised"),
    ]
    im = None
    for r, c, key, t, ver in panels:
        ax = fig.add_subplot(gs[r, c])
        style(ax)
        im = density(ax, d, key, norm)
        comp = key.rsplit("_", 1)[0]
        xl, yl = LABEL[comp]
        ax.set_xlabel(f"{xl} NDVI ({ver})")
        if c == 0 or comp == "tm_etm":
            ax.set_ylabel(f"{yl} NDVI")
        tag(ax, t)
    cax = fig.add_subplot(gs[0, 3])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Pixels per 0.005 × 0.005 NDVI bin")
    typing.cast(typing.Any, cb.outline).set_linewidth(0.6)
    ax = fig.add_subplot(gs[1, 2:4])
    mid = d["bins"][:-1] + 0.025
    series = [
        ("etm_oli", "ETM+ − OLI", OKABE["orange"], "o"),
        ("tm_etm", "TM − ETM+", OKABE["blue"], "s"),
        ("s2_oli", "MSI − OLI", OKABE["green"], "^"),
    ]
    ax.axhline(0, color="0.35", lw=0.5, ls=(0, (4, 2)), zorder=1)
    for comp, lab, col, mk in series:
        hb, rb = d[f"{comp}_harm_binbias"], d[f"{comp}_raw_binbias"]
        if comp != "tm_etm":
            ax.plot(
                mid,
                rb,
                color=col,
                lw=0.8,
                ls=(0, (2, 1.5)),
                marker=mk,
                ms=3,
                mfc="white",
                mew=0.6,
                zorder=2,
            )
        ax.plot(mid, hb, color=col, lw=0.8, marker=mk, ms=3.2, mew=0, label=lab, zorder=3)
    ax.set_xlim(0, 0.8)
    ax.set_ylim(-0.16, 0.08)
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax.yaxis.set_major_locator(MultipleLocator(0.04))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(which="both", top=True, left=True, right=True, labelleft=False, labelright=True)
    ax.yaxis.set_label_position("right")
    ax.grid(which="major", axis="y", lw=0.3, color="0.88")
    ax.set_axisbelow(True)
    ax.set_xlabel("Pair-mean NDVI (0.05 bins)")
    ax.set_ylabel("Median difference (NDVI)", labelpad=3)
    leg = ax.legend(
        loc="lower center",
        frameon=False,
        handlelength=1.6,
        borderaxespad=0.3,
        labelspacing=0.2,
        title="solid: harmonised, dashed: native",
        title_fontsize=8,
        alignment="left",
    )
    tag(ax, "(f)")
    fig.text(
        0.075,
        0.008,
        "Pooled clear pixels of near-simultaneous peak-season scene pairs (single WRS-2 path, 30 m grid): ETM+/OLI and\n"
        "TM/ETM+ 8 days apart (4 pairs each, alternating order), Sentinel-2 MSI/OLI same day (4 pairs). Dashed: 1:1;\n"
        "black: OLS fit; bias = sensor \u2212 reference. Software used for plotting figure: Python 3, Matplotlib "
        f"{mpl.__version__},\nNumPy {np.__version__}. Data: Landsat C2 L2, Sentinel-2 L2A (Microsoft Planetary Computer). Source: authors.",
        fontsize=8,
        va="bottom",
        linespacing=1.2,
    )
    fig.canvas.draw()
    r = typing.cast(typing.Any, fig.canvas).get_renderer()
    bbs = [leg.get_window_extent(r)] + [t.get_window_extent(r) for t in ax.texts]
    hits = 0
    for bb in bbs:
        for ln in ax.get_lines():
            xy = np.column_stack(ln.get_data())
            xy = xy[np.isfinite(xy).all(1)]
            if len(xy) > 2:
                p = ax.transData.transform(xy)
                hits += int(((p[:, 0] >= bb.x0) & (p[:, 0] <= bb.x1) & (p[:, 1] >= bb.y0) & (p[:, 1] <= bb.y1)).sum())
    print("covers_data hits in (f):", hits)
    fig.savefig(f"{out}/fig_harmonisation.pdf", dpi=600)
    fig.savefig(f"{out}/fig_harmonisation.png", dpi=600)
    fig.savefig(f"{out}/preview.png", dpi=130)


if __name__ == "__main__":
    main()

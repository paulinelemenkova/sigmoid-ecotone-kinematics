#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
import typing
from typing import Any, Literal, TypeAlias

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure
from matplotlib.text import Text
from matplotlib.ticker import AutoMinorLocator
from matplotlib.transforms import Bbox

F: TypeAlias = npt.NDArray[np.float64]
BLUE, ORANGE, GREY = "#0072B2", "#D55E00", "#7f7f7f"
UNITS = {
    "Elevation": "Elevation (m)",
    "Eastness": "Eastness, sin(aspect)",
    "Slope": "Slope (°)",
    "Insolation": "Insolation (Wh/m²/d)",
    "GDD trend": "GDD trend (°C d per decade)",
    "SCD mean": "Mean snow-cover duration (d)",
    "SCD trend": "SCD trend (d per decade)",
    "Northness": "Northness, cos(aspect)",
    "Landform": "Landform class",
}


def style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Nimbus Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.minor.width": 0.4,
            "ytick.minor.width": 0.4,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "xtick.minor.size": 1.6,
            "ytick.minor.size": 1.6,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "axes.unicode_minus": True,
            "pdf.fonttype": 42,
            "savefig.dpi": 600,
        }
    )


def grid(ax: Axes, axis: Literal["both", "x", "y"] = "both") -> None:
    ax.grid(True, axis=axis, color="0.85", lw=0.3)
    ax.set_axisbelow(True)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    if axis != "x":
        ax.yaxis.set_minor_locator(AutoMinorLocator())


def tag(ax: Axes, s: str) -> Text:
    return ax.text(
        0.02,
        0.98,
        s,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
    )


def overlap(fig: Figure, txt: Text, boxes: list[Bbox]) -> int:
    fig.canvas.draw()
    bb = txt.get_window_extent()
    return sum(bb.overlaps(b) for b in boxes)


def dependence(ax: Axes, dep: dict[str, Any], name: str) -> list[Bbox]:
    h = np.array(dep["hist"], dtype=np.float64)
    x0, x1 = float(dep["xr"][0]), float(dep["xr"][1])
    y0, y1 = float(dep["yr"][0]), float(dep["yr"][1])
    xe = np.linspace(x0, x1, h.shape[0] + 1)
    ye = np.linspace(y0, y1, h.shape[1] + 1)
    hm = np.ma.masked_less(h.T, 1)
    ax.pcolormesh(xe, ye, hm, cmap="Greys", norm=LogNorm(1, h.max() * 3), rasterized=True, lw=0)
    cur = np.array(dep["curve"])
    ax.fill_between(cur[:, 0], cur[:, 1], cur[:, 3], color=BLUE, alpha=0.3, lw=0)
    ax.plot(cur[:, 0], cur[:, 2], color=BLUE, lw=0.8)
    ax.axhline(0, color="k", lw=0.5, ls=(0, (3, 2)))
    ax.set_xlabel(UNITS.get(name, name), labelpad=1)
    ax.set_ylabel("SHAP\n(0.001 NDVI/decade)", labelpad=1, linespacing=1.0)
    lo, hi = dep["yr"]
    pad = 0.06 * (hi - lo)
    ax.set_ylim(lo - pad, hi + 4.0 * pad)
    ax.set_xlim(*dep["xr"])
    grid(ax)
    boxes: list[Bbox] = []
    for i, j in zip(*np.nonzero(h)):
        boxes.append(Bbox.from_extents(xe[i], ye[j], xe[i + 1], ye[j + 1]))
    return boxes


def main() -> None:
    style()
    with open(sys.argv[1]) as fh:
        d: dict[str, Any] = json.load(fh)
    out = sys.argv[2]
    names: list[str] = d["names"]
    mabs = np.array(d["mabs"])
    order: list[int] = d["order"]
    blk, rnd, n = d["block"], d["random"], d["n"]
    fig = plt.figure(figsize=(170 / 25.4, 92 / 25.4), layout="constrained")
    typing.cast(typing.Any, fig.get_layout_engine()).set(w_pad=0.02, h_pad=0.02, wspace=0.03, hspace=0.04)
    gs = fig.add_gridspec(2, 3, width_ratios=[2.25, 0.95, 1.55])
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[:, 1], sharey=ax_a)
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, 2])
    cmap = mpl.colormaps["turbo"]
    nf = len(names)
    cont_a = []
    lim = 0.0
    for row, k in enumerate(order):
        yv = nf - 1 - row
        f = d["feats"][k]
        e = np.array(f["edges"])
        c = np.array(f["count"], dtype=np.float64)
        r = np.array(f["rank"])
        hgt = 0.42 * c / c.max()
        keep = c > 0
        bars = ax_a.bar(
            0.5 * (e[:-1] + e[1:])[keep],
            2 * hgt[keep],
            width=np.diff(e)[keep],
            bottom=yv - hgt[keep],
            color=cmap(r[keep]),
            lw=0,
        )
        cont_a.append(bars)
        lim = max(lim, abs(e[0]), abs(e[-1]))
    ax_a.axvline(0, color="k", lw=0.5)
    ax_a.set_yticks(range(nf))
    ax_a.set_yticklabels([names[k] for k in order][::-1])
    ax_a.set_ylim(-0.6, nf + 0.2)
    ax_a.set_xlim(-lim * 1.02, lim * 1.02)
    ax_a.set_xlabel("SHAP value (0.001 NDVI/decade)", labelpad=1)
    ax_a.tick_params(axis="y", which="both", length=0)
    ax_a.grid(True, axis="x", color="0.85", lw=0.3)
    ax_a.set_axisbelow(True)
    ax_a.xaxis.set_minor_locator(AutoMinorLocator())
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(0, 1))
    cax = ax_a.inset_axes((0.25, -0.155, 0.5, 0.022))
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_ticks([0, 1])
    cb.set_ticklabels(["low", "high"])
    cb.set_label("Covariate value (mean rank in SHAP bin)", labelpad=0)
    typing.cast(typing.Any, cb.outline).set_linewidth(0.4)
    cb.ax.tick_params(length=0, pad=1)
    ta = tag(ax_a, "(a)")
    yb = [nf - 1 - r for r in range(nf)]
    bb = ax_b.barh(yb, mabs[order], height=0.6, color=BLUE, lw=0)
    vlab = [ax_b.text(v + 0.05, yy, f"{v:.2f}", va="center", ha="left", fontsize=8) for yy, v in zip(yb, mabs[order])]
    ax_b.set_xlim(0, mabs.max() * 1.42)
    ax_b.set_xlabel("Mean |SHAP|\n(0.001 NDVI/decade)", labelpad=1)
    ax_b.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)
    ax_b.grid(True, axis="x", color="0.85", lw=0.3)
    ax_b.set_axisbelow(True)
    ax_b.xaxis.set_minor_locator(AutoMinorLocator())
    tb = tag(ax_b, "(b)")
    skill = (
        f"Block CV R\u00b2 = {blk[0]:.2f}\nRMSE = {blk[1]:.4f}\n"
        f"Random CV R\u00b2 = {rnd[0]:.2f}\nn = {n[0]:,} cells".replace(",", " ")
    )
    sk = ax_b.text(
        0.5,
        -0.185,
        skill,
        transform=ax_b.transAxes,
        ha="center",
        va="top",
        fontsize=8,
        linespacing=1.15,
    )
    bx_c = dependence(ax_c, d["dep"][0], names[d["dep"][0]["k"]])
    bx_d = dependence(ax_d, d["dep"][1], names[d["dep"][1]["k"]])
    tc, td = tag(ax_c, "(c)"), tag(ax_d, "(d)")
    fig.savefig(f"{out}/fig_shap.pdf")
    fig.savefig(f"{out}/fig_shap.png", dpi=600)
    fig.canvas.draw()
    boxes_a = [b.get_window_extent() for c in cont_a for b in c]
    bars_b = [b.get_window_extent() for b in bb]
    bx_c = [b.transformed(ax_c.transData) for b in bx_c]
    bx_d = [b.transformed(ax_d.transData) for b in bx_d]
    print(
        "overlap hits:",
        {
            "(a)": overlap(fig, ta, boxes_a),
            "(b)": overlap(fig, tb, bars_b),
            "skill": overlap(fig, sk, boxes_a + bars_b),
            "(b) labels": sum(overlap(fig, t, bars_b) for t in vlab),
            "(c)": overlap(fig, tc, bx_c),
            "(d)": overlap(fig, td, bx_d),
        },
    )
    print(
        "fonts:",
        sorted({str(t.get_fontname()) for t in fig.findobj(Text) if t.get_text()}),
    )


if __name__ == "__main__":
    main()

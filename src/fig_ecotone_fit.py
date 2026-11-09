#!/usr/bin/env python3

from __future__ import annotations

import os
import sys
import typing
from typing import TypeAlias

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from scipy import stats

ANN, EPO, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
FArr: TypeAlias = npt.NDArray[np.float64]

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Nimbus Sans", "Helvetica"],
        "font.size": 9,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8,
        "axes.linewidth": 0.8,
        "axes.labelpad": 2,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
        "pdf.fonttype": 42,
        "mathtext.fontset": "custom",
        "mathtext.rm": "Nimbus Sans",
        "mathtext.it": "Nimbus Sans:italic",
        "mathtext.bf": "Nimbus Sans:bold",
    }
)
assert "dejavu" not in findfont(FontProperties(family=plt.rcParams["font.sans-serif"])).lower()
OI = ["#0072B2", "#009E73", "#E69F00", "#D55E00", "#CC79A7", "#56B4E9"]
MINUS = "−"

ann = pd.read_csv(ANN)
epo = pd.read_csv(EPO)
alls = ann[ann.stratum == "all"].sort_values("year")
ok = alls[alls.flag == 0]
fl = alls[alls.flag == 1]


def logi(z: FArr, fmax: float, z50: float, s: float) -> FArr:
    return fmax / (1.0 + np.exp((z - z50) / s))


def ts(x: FArr, y: FArr) -> tuple[float, float, float, float, float]:
    s, b, lo, hi = stats.theilslopes(y, x, 0.95)
    return float(s), float(b), float(lo), float(hi), float(stats.kendalltau(x, y).pvalue)


def pettitt(y: FArr) -> tuple[int, float]:
    n = y.size
    sg = np.sign(y[None, :] - y[:, None])
    u = np.array([sg[: t + 1, t + 1 :].sum() for t in range(n - 1)])
    k = int(np.argmax(np.abs(u)))
    kk = float(abs(u[k]))
    return k, float(min(1.0, 2 * np.exp(-6 * kk**2 / (n**3 + n**2))))


def ptxt(p: float) -> str:
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def style(ax: Axes) -> None:
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which="both", top=True, right=True, direction="in", pad=2)
    ax.tick_params(which="major", length=4.5)
    ax.tick_params(which="minor", length=2.5)
    ax.grid(which="major", lw=0.5, color="0.87")
    ax.set_axisbelow(True)


fig, axs = plt.subplots(2, 2, figsize=(17.0 / 2.54, 12.4 / 2.54), constrained_layout=True)
typing.cast(typing.Any, fig.get_layout_engine()).set(w_pad=0.03, h_pad=0.03, wspace=0.03, hspace=0.03)

ax = axs[0, 0]
ea = epo[epo.stratum == "all"]
zz = np.linspace(1600, 3600, 300)
for i, (ep, g) in enumerate(ea.groupby("epoch", sort=True)):
    c, mk = OI[i], "osD^"[i]
    last = i == 3
    ax.plot(g.zc, g.f, ls="", marker=mk, ms=2.6, mfc=c, mec="none", alpha=0.55, zorder=2)
    fmax, z50, s = float(g["fmax"].iloc[0]), float(g["z50"].iloc[0]), float(g["s"].iloc[0])
    ax.plot(zz, logi(zz, fmax, z50, s), color=c, lw=2.0 if last else 1.2, zorder=3, label=ep.replace("-", "–"))
    ax.plot([z50, z50], [0, fmax / 2], color=c, lw=0.7, ls=(0, (3, 2)), zorder=2)
    ax.plot(z50, fmax / 2, marker=mk, ms=5.5, mfc=c, mec="black", mew=0.6, zorder=4)
ax.set_xlim(1600, 3600)
ax.set_ylim(0, 1.0)
ax.xaxis.set_major_locator(MultipleLocator(400))
ax.xaxis.set_minor_locator(MultipleLocator(100))
ax.set_xlabel("Elevation (m)")
ax.set_ylabel("Vegetated fraction (NDVI ≥ 0.20)")
leg_a = ax.legend(
    loc="upper right", ncol=2, frameon=False, handlelength=1.4, borderaxespad=0.3, labelspacing=0.3, columnspacing=0.8
)

ax = axs[0, 1]
x, y = ok.year.to_numpy(float), ok.z50.to_numpy(float)
ax.errorbar(
    ok.year,
    ok.z50,
    yerr=[ok.z50 - ok.z50_lo, ok.z50_hi - ok.z50],
    fmt="o",
    ms=3.4,
    color=OI[0],
    mec="white",
    mew=0.4,
    elinewidth=0.7,
    capsize=0,
    zorder=3,
)
ax.errorbar(
    fl.year,
    fl.z50,
    yerr=[fl.z50 - fl.z50_lo, fl.z50_hi - fl.z50],
    fmt="o",
    ms=3.6,
    mfc="white",
    mec=OI[0],
    mew=0.9,
    ecolor=OI[0],
    elinewidth=0.7,
    capsize=0,
    zorder=3,
)
s, b, lo, hi, p = ts(x, y)
xx = np.linspace(x.min(), x.max(), 50)
xm, ym = np.median(x), np.median(y)
ax.fill_between(xx, ym + lo * (xx - xm), ym + hi * (xx - xm), color="0.6", alpha=0.25, lw=0, zorder=1)
ax.plot(xx, b + s * xx, color="black", lw=2.0, zorder=4)
k, pp = pettitt(y)
cp = 0.5 * (x[k] + x[k + 1])
ax.axvline(cp, color=OI[3], lw=0.9, ls=(0, (4, 2)), zorder=2)
m1, m2 = y[: k + 1].mean(), y[k + 1 :].mean()
ax.plot([x.min(), cp], [m1, m1], color=OI[3], lw=1.2, zorder=2)
ax.plot([cp, x.max()], [m2, m2], color=OI[3], lw=1.2, zorder=2)
ax.text(
    0.11,
    0.965,
    f"Theil–Sen {s * 10:+.0f} m per decade [{lo * 10:+.0f}, {hi * 10:+.0f}], {ptxt(p)}".replace("-", MINUS).replace(
        "Theil" + MINUS + "Sen", "Theil–Sen"
    ),
    transform=ax.transAxes,
    ha="left",
    va="top",
    fontsize=8,
)
ax.text(
    0.11,
    0.885,
    f"Pettitt change after {int(x[k])}: {m1:.0f} → {m2:.0f} m, {ptxt(pp)}",
    transform=ax.transAxes,
    ha="left",
    va="top",
    fontsize=8,
    color=OI[3],
)
ax.set_xlim(1983, 2026)
ax.set_ylim(2150, 3080)
ax.xaxis.set_major_locator(MultipleLocator(10))
ax.xaxis.set_minor_locator(MultipleLocator(2))
ax.set_xlabel("Year")
ax.set_ylabel("Ecotone elevation $z_{50}$ (m)")
print(f"z50 all: TS {s * 10:+.1f} m/dec [{lo * 10:+.1f},{hi * 10:+.1f}] p={p:.4f}; Pettitt {int(x[k])} p={pp:.4f}")

ax = axs[1, 0]
w = ok.width.to_numpy(float)
ax.errorbar(
    ok.year,
    ok.width,
    yerr=[ok.width - ok.width_lo, ok.width_hi - ok.width],
    fmt="s",
    ms=3.2,
    color=OI[1],
    mec="white",
    mew=0.4,
    elinewidth=0.7,
    capsize=0,
    zorder=3,
)
ax.errorbar(
    fl.year,
    fl.width,
    yerr=[fl.width - fl.width_lo, fl.width_hi - fl.width],
    fmt="s",
    ms=3.4,
    mfc="white",
    mec=OI[1],
    mew=0.9,
    ecolor=OI[1],
    elinewidth=0.7,
    capsize=0,
    zorder=3,
)
s2, b2, lo2, hi2, p2 = ts(x, w)
ym2 = np.median(w)
ax.fill_between(xx, ym2 + lo2 * (xx - xm), ym2 + hi2 * (xx - xm), color="0.6", alpha=0.25, lw=0, zorder=1)
ax.plot(xx, b2 + s2 * xx, color="black", lw=2.0, zorder=4)
ax.text(
    0.02,
    0.03,
    f"Theil–Sen {s2 * 10:+.0f} m per decade [{lo2 * 10:+.0f}, {hi2 * 10:+.0f}], {ptxt(p2)}".replace("-", MINUS).replace(
        "Theil" + MINUS + "Sen", "Theil–Sen"
    ),
    transform=ax.transAxes,
    ha="left",
    va="bottom",
    fontsize=8,
)
ax.set_xlim(1983, 2026)
ax.set_ylim(500, 2050)
ax.xaxis.set_major_locator(MultipleLocator(10))
ax.xaxis.set_minor_locator(MultipleLocator(2))
ax.set_xlabel("Year")
ax.set_ylabel("Transition width $w$ (m)")
print(f"width all: TS {s2 * 10:+.1f} m/dec [{lo2 * 10:+.1f},{hi2 * 10:+.1f}] p={p2:.4f}")

ax = axs[1, 1]
strata = ["all", "N", "E", "S", "W"]
names = {"all": "All", "N": "North", "E": "East", "S": "South", "W": "West"}
for j, st in enumerate(strata):
    d = ann[(ann.stratum == st) & (ann.flag == 0)]
    yy = d.year.to_numpy(float)
    for col, dx, mk, fc, lab in (
        ("z50_t15", -0.22, "v", "white", "NDVI ≥ 0.15"),
        ("z50", 0.0, "o", "C", "NDVI ≥ 0.20 (95 % CI)"),
        ("z50_t25", 0.22, "^", "white", "NDVI ≥ 0.25"),
    ):
        v = d[col].to_numpy(float)
        g = np.isfinite(v)
        sl, _, l_, h_, pv = ts(yy[g], v[g])
        c = OI[0]
        if col == "z50":
            ax.plot([j + dx, j + dx], [l_ * 10, h_ * 10], color=c, lw=1.2, zorder=2)
            ax.plot(j + dx, sl * 10, marker=mk, ms=6, mfc=c if pv < 0.05 else "white", mec=c, mew=1.2, zorder=3)
            print(f"z50 {st}: {sl * 10:+.1f} m/dec [{l_ * 10:+.1f},{h_ * 10:+.1f}] p={pv:.4f}")
        else:
            ax.plot(
                j + dx, sl * 10, marker=mk, ms=4.5, mfc="0.35" if pv < 0.05 else "white", mec="0.35", mew=0.9, zorder=3
            )
ax.axhline(0, color="0.3", lw=0.7, zorder=1)
ax.set_xticks(range(len(strata)), [names[s_] for s_ in strata])
ax.set_xlim(-0.6, len(strata) - 0.4)
ax.set_ylim(-75, 235)
ax.yaxis.set_major_locator(MultipleLocator(50))
ax.set_ylabel("Trend of $z_{50}$ (m per decade)")
ax.set_xlabel("Aspect stratum")
hand = [
    Line2D([], [], color=OI[0], lw=1.2, marker="o", ms=5.5, mfc=OI[0], mec=OI[0], label="NDVI ≥ 0.20, 95 % CI"),
    Line2D([], [], ls="", marker="v", ms=4.5, mfc="0.35", mec="0.35", label="NDVI ≥ 0.15"),
    Line2D([], [], ls="", marker="^", ms=4.5, mfc="0.35", mec="0.35", label="NDVI ≥ 0.25"),
    Line2D([], [], ls="", marker="o", ms=5.5, mfc="white", mec="0.35", label="open: p ≥ 0.05"),
]
leg_d = ax.legend(
    handles=hand,
    loc="upper left",
    bbox_to_anchor=(0.09, 1.0),
    frameon=False,
    borderaxespad=0.3,
    labelspacing=0.25,
    handlelength=1.4,
    ncol=2,
    columnspacing=0.8,
)

for a_ in axs.flat:
    style(a_)
axs[1, 1].grid(axis="x", visible=False)
for a_, tag in zip(axs.flat, "abcd"):
    a_.text(0.02, 0.97, f"({tag})", transform=a_.transAxes, ha="left", va="top", fontsize=10, fontweight="bold")

fig.canvas.draw()
r = typing.cast(typing.Any, fig.canvas).get_renderer()
hits = 0
for a_ in axs.flat:
    boxes = [t.get_window_extent(renderer=r) for t in a_.texts]
    lg = a_.get_legend()
    if lg is not None:
        boxes.append(lg.get_window_extent(renderer=r))
    pts = []
    for ln in a_.get_lines():
        dd = np.column_stack(ln.get_data()).astype(float)
        if dd.shape[0] >= 1:
            pts.append(ln.get_transform().transform(dd))
    for coll in a_.collections:
        for pth in coll.get_paths():
            pts.append(coll.get_transform().transform(pth.vertices))
    P = np.vstack(pts)
    for bb in boxes:
        hits += int(((P[:, 0] >= bb.x0) & (P[:, 0] <= bb.x1) & (P[:, 1] >= bb.y0) & (P[:, 1] <= bb.y1)).sum())
print("covers_data hits:", hits)

os.makedirs(OUT, exist_ok=True)
fig.savefig(os.path.join(OUT, "fig_ecotone_fit.pdf"), bbox_inches="tight", pad_inches=0.02)
fig.savefig(os.path.join(OUT, "fig_ecotone_fit.png"), dpi=600, bbox_inches="tight", pad_inches=0.02)
fig.savefig(os.path.join(OUT, "fig_ecotone_fit_preview.png"), dpi=110, bbox_inches="tight", pad_inches=0.02)

#!/usr/bin/env python3

from __future__ import annotations

import glob
import os
import sys
import typing
from itertools import pairwise

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, MultipleLocator, NullFormatter

ACQ, OUT = sys.argv[1], sys.argv[2]
TH = 0.7
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Nimbus Sans", "Helvetica"],
        "font.size": 8.5,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
        "pdf.fonttype": 42,
    }
)
assert "dejavu" not in findfont(FontProperties(family=plt.rcParams["font.sans-serif"])).lower()

d = pd.concat([pd.read_csv(f, dtype={"path": str, "tiles": str, "tier": str}) for f in glob.glob(f"{ACQ}/acq_*.csv")])
d["date"] = pd.to_datetime(d["date"])
d["year"] = d["date"].dt.year
d["doy"] = d["date"].dt.dayofyear
d["tier"] = d["tier"].fillna("")


def sensor(p: str) -> str:
    if p in ("landsat-4", "landsat-5"):
        return "L45"
    if p == "landsat-7":
        return "L7"
    if p == "landsat-8":
        return "L8"
    if p == "landsat-9":
        return "L9"
    return "S2"


d["sensor"] = d["platform"].map(sensor)
d["slcoff"] = (d["sensor"] == "L7") & (d["date"] > "2003-05-31")
d["t2only"] = d["tier"] == "T2"
SENS = {
    "L45": ("Landsat 4/5 TM", "o", -0.30, "#0072B2"),
    "L7": ("Landsat 7 ETM+", "s", -0.15, "#009E73"),
    "L8": ("Landsat 8 OLI", "^", 0.0, "#E69F00"),
    "L9": ("Landsat 9 OLI-2", "v", 0.15, "#CC79A7"),
    "S2": ("Sentinel-2 MSI", "D", 0.30, "#D55E00"),
}
d["y"] = d["year"] + d["sensor"].map({k: v[2] for k, v in SENS.items()})
seas = (d["season"] == 1) & (d["status"] == "ok") & ~d["t2only"]

fig = plt.figure(figsize=(17.0 / 2.54, 21.0 / 2.54), constrained_layout=True)
typing.cast(typing.Any, fig.get_layout_engine()).set(w_pad=0.02, h_pad=0.02, wspace=0.03)
gs = fig.add_gridspec(3, 2, width_ratios=[3.3, 1.0], height_ratios=[1.0, 0.085, 0.022])
ax: Axes = fig.add_subplot(gs[0, 0])
bx: Axes = fig.add_subplot(gs[0, 1], sharey=ax)
lga: Axes = fig.add_subplot(gs[1, :])
cax: Axes = fig.add_subplot(gs[2, 0])
lga.set_axis_off()

ax.axvspan(152, 274, color="#FFF4CC", lw=0, zorder=0)
oos = d[(d["season"] == 0)]
ax.scatter(oos["doy"], oos["y"], s=1.2, c="0.72", lw=0, zorder=1, rasterized=True)
t2 = d[(d["season"] == 1) & d["t2only"]]
for k, spec in SENS.items():
    mk = spec[1]
    sub = t2[t2["sensor"] == k]
    ax.scatter(sub["doy"], sub["y"], s=7, marker=mk, facecolors="none", edgecolors="0.55", lw=0.5, zorder=2)
cmap = plt.get_cmap("viridis")
norm = mpl.colors.Normalize(0, 1)
sc = None
for k, spec in SENS.items():
    mk = spec[1]
    sub = d[seas & (d["sensor"] == k)].sort_values("usable")
    sc = ax.scatter(
        sub["doy"],
        sub["y"],
        s=9,
        marker=mk,
        c=sub["usable"],
        cmap=cmap,
        norm=norm,
        edgecolors="0.15",
        lw=0.3,
        zorder=3,
        rasterized=True,
    )
mstart = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366]
ax.set_xlim(1, 366)
ax.xaxis.set_major_locator(FixedLocator(mstart))
ax.xaxis.set_major_formatter(NullFormatter())
ax.set_xticks([(a + b) / 2 for a, b in pairwise(mstart)], minor=True)
ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], minor=True)
ax.tick_params(axis="x", which="minor", length=0, pad=3)
ax.tick_params(axis="x", which="major", length=4.5, direction="in", top=True)
ax.grid(axis="x", which="major", lw=0.5, color="0.85")
ax.set_axisbelow(True)
top = ax.secondary_xaxis("top")
top.set_xticks([1, 50, 100, 150, 200, 250, 300, 350])
top.xaxis.set_minor_locator(MultipleLocator(10))
top.tick_params(which="both", direction="in")
top.tick_params(which="major", length=4.5)
top.tick_params(which="minor", length=2.5)
top.set_xlabel("Day of year", labelpad=3)
ax.set_ylim(2025.6, 1982.3)
ax.yaxis.set_major_locator(MultipleLocator(5))
ax.yaxis.set_minor_locator(MultipleLocator(1))
ax.tick_params(axis="y", which="both", direction="in", right=True)
ax.tick_params(axis="y", which="major", length=4.5)
ax.tick_params(axis="y", which="minor", length=2.5)
ax.set_ylabel("Year")
ax.text(0.012, 0.995, "(a)", transform=ax.transAxes, ha="left", va="top", fontsize=10, fontweight="bold")
hs = [
    Line2D([], [], ls="", marker=mk, ms=4.5, mfc="#7FC97F", mec="0.15", mew=0.4, label=lab)
    for lab, mk, _, _ in SENS.values()
]
hs += [
    Line2D([], [], ls="", marker="o", ms=2.2, mfc="0.72", mec="none", label="Outside June–September"),
    Line2D([], [], ls="", marker="o", ms=4.0, mfc="none", mec="0.55", mew=0.6, label="Landsat Tier 2 only"),
]
lega = lga.legend(
    handles=hs,
    loc="center left",
    bbox_to_anchor=(0.0, 0.5),
    ncol=3,
    frameon=False,
    fontsize=7.4,
    handletextpad=0.3,
    columnspacing=1.0,
)
assert sc is not None
cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
cb.set_label("Usable fraction of the window (clear, snow-free land)")
cb.ax.xaxis.set_minor_locator(MultipleLocator(0.05))
cb.ax.tick_params(which="both", direction="in")

u = d[seas & (d["usable"] >= TH)].copy()
u["cls"] = np.where(u["slcoff"], "L7off", u["sensor"])
years = np.arange(1984, 2026)
cnt = u.groupby(["year", "cls"]).size().unstack(fill_value=0).reindex(years, fill_value=0)
order = [("L45", SENS["L45"][3], ""), ("L7", SENS["L7"][3], ""), ("L7off", SENS["L7"][3], "/////")]
order += [("L8", SENS["L8"][3], ""), ("L9", SENS["L9"][3], ""), ("S2", SENS["S2"][3], "")]
left = np.zeros(len(years))
for cls, col, hatch in order:
    v = cnt[cls].to_numpy() if cls in cnt else np.zeros(len(years))
    bx.barh(
        years,
        v,
        left=left,
        height=0.72,
        color="white" if hatch else col,
        edgecolor=col,
        hatch=hatch,
        lw=0.6 if hatch else 0,
        zorder=2,
    )
    left += v
bx.set_xlim(0, max(left.max() * 1.08, 1))
bx.xaxis.set_major_locator(MultipleLocator(20))
bx.xaxis.set_minor_locator(MultipleLocator(5))
bx.tick_params(which="both", direction="in", top=True, right=True, labelleft=False)
bx.tick_params(which="major", length=4.5)
bx.tick_params(which="minor", length=2.5)
bx.grid(axis="x", which="major", lw=0.5, color="0.85")
bx.set_axisbelow(True)
bx.set_xlabel(f"Usable acquisitions\nper year (≥ {TH:.1f})")
bx.text(0.96, 0.995, "(b)", transform=bx.transAxes, ha="right", va="top", fontsize=10, fontweight="bold")
bh = [Patch(fc=SENS[k][3], ec="none", label=SENS[k][0].split()[0] + " " + SENS[k][0].split()[1]) for k in SENS]
bh.insert(2, Patch(fc="white", ec=SENS["L7"][3], hatch="/////", lw=0.6, label="Landsat 7 SLC-off"))
legb = lga.legend(
    handles=bh,
    loc="center right",
    bbox_to_anchor=(1.0, 0.5),
    ncol=2,
    frameon=False,
    fontsize=7.2,
    handlelength=1.3,
    columnspacing=0.8,
    title="(b) usable acquisitions",
    title_fontsize=7.2,
)
lga.add_artist(lega)
for lg in (lega, legb):
    lg.set_in_layout(False)

fig.canvas.draw()
r = typing.cast(typing.Any, fig.canvas).get_renderer()
hits = 0
for axx in (ax, bx):
    boxes = [t.get_window_extent(renderer=r) for t in axx.texts]
    for bb in boxes:
        for coll in axx.collections:
            off = np.asarray(coll.get_offsets())
            if off.size:
                p = axx.transData.transform(off)
                hits += int(((p[:, 0] >= bb.x0) & (p[:, 0] <= bb.x1) & (p[:, 1] >= bb.y0) & (p[:, 1] <= bb.y1)).sum())
        for pch in axx.patches:
            if typing.cast(typing.Any, pch).get_width() > 0:
                ex = pch.get_window_extent(renderer=r)
                if ex.width > 0.5 and bb.overlaps(ex) and axx is bx:
                    hits += 1
print("covers_data hits:", hits)

os.makedirs(OUT, exist_ok=True)
fig.savefig(os.path.join(OUT, "fig_scenes.pdf"), dpi=600)
fig.savefig(os.path.join(OUT, "fig_scenes.png"), dpi=600)
fig.savefig(os.path.join(OUT, "fig_scenes_preview.png"), dpi=110)

print("acquisitions total", len(d), "by sensor", d.groupby("sensor").size().to_dict())
print("in-season read", int(seas.sum()), "T2-only in season", len(t2))
print("usable", len(u), "by class", u.groupby("cls").size().to_dict())
print(
    "usable per year min/median/max",
    int(cnt.sum(axis=1).min()),
    float(cnt.sum(axis=1).median()),
    int(cnt.sum(axis=1).max()),
)
print("years with <2 usable", [int(y) for y in years if cnt.sum(axis=1)[y] < 2])
lp = d[seas & d["sensor"].isin(["L45", "L7", "L8", "L9"])]
print("Landsat valid by path", lp.groupby("path")["valid"].median().round(2).to_dict())
print("Landsat usable by path >=0.7", lp[lp["usable"] >= TH].groupby("path").size().to_dict())

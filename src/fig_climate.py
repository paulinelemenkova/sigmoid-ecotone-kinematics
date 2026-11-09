#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import sys
import typing
import urllib.request
from typing import Any, TypeAlias

import matplotlib as mpl

import geoconf as G

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from scipy import stats

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
LAT, LON, ELEV = G.CLIM
Y0, Y1 = 1985, 2025
SD0 = 0.05
TB = 5.0

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Nimbus Sans", "Helvetica"],
        "font.size": 9,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "axes.linewidth": 0.8,
        "axes.labelpad": 2,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
        "pdf.fonttype": 42,
    }
)
assert "dejavu" not in findfont(FontProperties(family=plt.rcParams["font.sans-serif"])).lower()

OI = {"blue": "#0072B2", "vermilion": "#D55E00", "green": "#009E73", "sky": "#56B4E9", "orange": "#E69F00"}
FArr: TypeAlias = npt.NDArray[np.float64]


def fetch(model: str, daily: str) -> dict[str, Any]:
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={LAT}&longitude={LON}&elevation={ELEV}"
        f"&start_date={Y0 - 1}-10-01&end_date={Y1}-12-31&daily={daily}&models={model}&timezone=UTC"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "research-plot-script/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        data: dict[str, Any] = json.load(r)["daily"]
    return data


land = fetch("era5_land", "temperature_2m_mean,snow_depth_mean")
era5 = fetch("era5", "precipitation_sum")
t = np.array(land["time"], dtype="datetime64[D]")
yr = t.astype("datetime64[Y]").astype(int) + 1970
mo = t.astype("datetime64[M]").astype(int) % 12 + 1
tas = np.array(land["temperature_2m_mean"], dtype=float)
sd = np.array(land["snow_depth_mean"], dtype=float)
pr = np.array(era5["precipitation_sum"], dtype=float)
syr = np.where(mo >= 10, yr + 1, yr)

years = np.arange(Y0, Y1 + 1)
jjas = np.array([np.nanmean(tas[(yr == y) & (mo >= 6) & (mo <= 9)]) for y in years])
gdd = np.array([np.nansum(np.clip(tas[yr == y] - TB, 0, None)) for y in years])
syears = np.arange(Y0 + 1, Y1 + 1)
scd = np.array([float(np.sum(sd[(syr == y)] >= SD0)) for y in syears])
pom = np.array([np.nansum(pr[(syr == y) & ((mo >= 10) | (mo <= 5))]) for y in syears])
base = (years >= 1991) & (years <= 2020)
anom = jjas - jjas[base].mean()


def trend(x: FArr, y: FArr) -> tuple[float, float, float, float, float]:
    s, b, lo, hi = stats.theilslopes(y, x, 0.95)
    p = stats.kendalltau(x, y).pvalue
    return float(s), float(b), float(lo), float(hi), float(p)


def style(ax: Axes) -> None:
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(MultipleLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which="both", top=True, right=True, direction="in", pad=2)
    ax.tick_params(which="major", length=4.5)
    ax.tick_params(which="minor", length=2.5)
    ax.grid(axis="y", which="major", lw=0.5, color="0.85")
    ax.set_axisbelow(True)
    ax.set_xlim(Y0 - 1, Y1 + 1)


def ptxt(p: float) -> str:
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def add_trend(ax: Axes, x: FArr, y: FArr, unit: str, scale: float = 10.0) -> tuple[float, float]:
    s, b, lo, hi, p = trend(x, y)
    xm = np.median(x)
    ym = np.median(y)
    xx = np.linspace(x.min(), x.max(), 50)
    ax.fill_between(xx, ym + lo * (xx - xm), ym + hi * (xx - xm), color="0.6", alpha=0.25, lw=0)
    ax.plot(xx, b + s * xx, color="black", lw=2.0, zorder=4)
    ax.text(
        0.98,
        0.95,
        f"Theil–Sen {s * scale:+.{2 if abs(s * scale) < 10 else 0}f} {unit} per decade, {ptxt(p)}".replace(
            "-", "\u2212"
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
    )
    return s, p


fig, axs = plt.subplots(2, 2, figsize=(17.0 / 2.54, 10.6 / 2.54), constrained_layout=True)
typing.cast(typing.Any, fig.get_layout_engine()).set(w_pad=0.03, h_pad=0.03, wspace=0.02, hspace=0.02)
xf = years.astype(float)
xs = syears.astype(float)

ax = axs[0, 0]
ax.bar(years, anom, width=0.8, color=np.where(anom >= 0, OI["vermilion"], OI["blue"]), lw=0, zorder=2)
ax.axhline(0, color="0.3", lw=0.7)
st = add_trend(ax, xf, anom, "°C")
ax.set_ylabel("JJAS temp. anomaly (°C)")
ax.set_ylim(anom.min() - 0.3, anom.max() + 0.9)

ax = axs[0, 1]
ax.plot(years, gdd, color=OI["orange"], lw=1.2, marker="o", ms=3.2, mec="white", mew=0.5, zorder=3)
sg = add_trend(ax, xf, gdd, "°C d")
ax.set_ylabel("GDD above 5 °C (°C d)")
ax.set_ylim(gdd.min() - 0.08 * np.ptp(gdd), gdd.max() + 0.35 * np.ptp(gdd))

ax = axs[1, 0]
ax.plot(syears, scd, color=OI["sky"], lw=1.2, marker="s", ms=3.0, mec="white", mew=0.5, zorder=3)
ss = add_trend(ax, xs, scd, "days")
ax.set_ylabel("Snow-cover duration (d)")
ax.set_ylim(scd.min() - 0.08 * np.ptp(scd), scd.max() + 0.35 * np.ptp(scd))
ax.set_xlabel("Year (snow year ending in September)")

ax = axs[1, 1]
ax.bar(syears, pom, width=0.8, color=OI["green"], lw=0, zorder=2)
sp = add_trend(ax, xs, pom, "mm")
ax.set_ylabel("Oct–May precipitation (mm)")
ax.set_ylim(0, pom.max() * 1.25)
ax.set_xlabel("Year (snow year ending in September)")

for ax, tag in zip(axs.flat, "abcd"):
    style(ax)
    ax.text(0.02, 0.95, f"({tag})", transform=ax.transAxes, ha="left", va="top", fontsize=10, fontweight="bold")
axs[0, 0].set_xticklabels([])
axs[0, 1].set_xticklabels([])

fig.canvas.draw()
r = typing.cast(typing.Any, fig.canvas).get_renderer()
hits = 0
for ax in axs.flat:
    for txt in ax.texts:
        bb = txt.get_window_extent(renderer=r)
        for ln in ax.get_lines():
            d = np.column_stack(ln.get_data())
            if d.shape[0] > 2:
                p_ = ax.transData.transform(d)
                hits += int(
                    ((p_[:, 0] >= bb.x0) & (p_[:, 0] <= bb.x1) & (p_[:, 1] >= bb.y0) & (p_[:, 1] <= bb.y1)).sum()
                )
        for pch in ax.patches:
            vs = ax.transData.transform(pch.get_path().transformed(pch.get_patch_transform()).vertices)
            hits += int(((vs[:, 0] >= bb.x0) & (vs[:, 0] <= bb.x1) & (vs[:, 1] >= bb.y0) & (vs[:, 1] <= bb.y1)).sum())
print("covers_data hits:", hits)

os.makedirs(OUT, exist_ok=True)
fig.savefig(os.path.join(OUT, "fig_climate.pdf"), bbox_inches="tight", pad_inches=0.02)
fig.savefig(os.path.join(OUT, "fig_climate.png"), dpi=600, bbox_inches="tight", pad_inches=0.02)
fig.savefig(os.path.join(OUT, "fig_climate_preview.png"), dpi=110, bbox_inches="tight", pad_inches=0.02)

rows = ["year,jjas_t,jjas_anom,gdd,scd_snowyear,precip_oct_may"]
for i, y in enumerate(years):
    j = np.where(syears == y)[0]
    sc = f"{scd[j[0]]:.0f}" if len(j) else ""
    pm = f"{pom[j[0]]:.1f}" if len(j) else ""
    rows.append(f"{y},{jjas[i]:.3f},{anom[i]:.3f},{gdd[i]:.1f},{sc},{pm}")
with open(os.path.join(OUT, "climate_series.csv"), "w") as f:
    f.write("\n".join(rows) + "\n")
print("JJAS mean 1991-2020", round(float(jjas[base].mean()), 2))
for name, (s, p) in (("JJAS T degC", st), ("GDD degCd", sg), ("SCD days", ss), ("P Oct-May mm", sp)):
    print(f"{name}: {s * 10:+.3f} per decade, p={p:.4f}")
print("means: GDD", round(float(gdd.mean())), "SCD", round(float(scd.mean())), "P", round(float(pom.mean())))
print("matplotlib", mpl.__version__, "numpy", np.__version__, "scipy", stats.__name__)

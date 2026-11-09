#!/usr/bin/env python3

from __future__ import annotations

import os
import sys
import typing
from itertools import pairwise

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.text import Text
from matplotlib.transforms import Bbox

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
W, H = 170.0, 126.0

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Nimbus Sans", "Helvetica"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Nimbus Sans",
        "mathtext.it": "Nimbus Sans:italic",
        "mathtext.bf": "Nimbus Sans:bold",
        "mathtext.fallback": "stixsans",
        "pdf.fonttype": 42,
    }
)
assert "dejavu" not in findfont(FontProperties(family=plt.rcParams["font.sans-serif"])).lower()

SW = {
    "Python": ("#0072B2", "#DCEBF5"),
    "GRASS GIS": ("#009E73", "#D6F0E8"),
    "GMT": ("#E69F00", "#FBEDCF"),
    "Data": ("#555555", "#F2F2F2"),
}
HEAD = "#3A3A3A"
FS_BOX, FS_HEAD = 6.6, 7.4

fig = plt.figure(figsize=(W / 25.4, H / 25.4))
ax: Axes = fig.add_axes((0, 0, 1, 1))
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.set_axis_off()

boxes: list[tuple[Text, FancyBboxPatch]] = []
arrows: list[FancyArrowPatch] = []


def box(x: float, y: float, w: float, h: float, text: str, sw: str, bold_first: bool = False) -> FancyBboxPatch:
    edge, fill = SW[sw]
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", fc=fill, ec=edge, lw=1.0, zorder=3)
    ax.add_patch(p)
    lines = text.split("\n")
    if bold_first:
        t = ax.text(
            x + w / 2,
            y + h - 1.6,
            lines[0],
            ha="center",
            va="top",
            fontsize=FS_BOX,
            fontweight="bold",
            zorder=4,
        )
        boxes.append((t, p))
        t2 = ax.text(
            x + w / 2,
            y + 1.6,
            "\n".join(lines[1:]),
            ha="center",
            va="bottom",
            linespacing=1.1,
            fontsize=FS_BOX - 0.4,
            zorder=4,
            color="0.25",
        )
        boxes.append((t2, p))
    else:
        t = ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=FS_BOX, zorder=4, linespacing=1.15)
        boxes.append((t, p))
    return p


def lane(x: float, y: float, w: float, h: float, num: int, title: str, rq: str, items: list[tuple[str, str]]) -> None:
    ax.add_patch(
        FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.8", fc="white", ec="0.55", lw=0.8, zorder=1)
    )
    hd = FancyBboxPatch(
        (x, y + h - 7.0), w, 7.0, boxstyle="round,pad=0,rounding_size=1.8", fc=HEAD, ec=HEAD, lw=0.8, zorder=2
    )
    ax.add_patch(hd)
    t = ax.text(
        x + 2.2,
        y + h - 3.5,
        f"{num}  {title}",
        ha="left",
        va="center",
        fontsize=FS_HEAD,
        fontweight="bold",
        color="white",
        zorder=4,
    )
    boxes.append((t, hd))
    if rq:
        t = ax.text(
            x + w - 2.0,
            y + h - 3.5,
            rq,
            ha="right",
            va="center",
            fontsize=FS_HEAD - 0.6,
            fontweight="bold",
            color="#F0C060",
            zorder=4,
        )
        boxes.append((t, hd))
    bh, gap = 9.6, 1.35
    for k, (txt, sw) in enumerate(items):
        by = y + h - 7.0 - gap - (k + 1) * bh - k * gap
        box(x + 2.0, by, w - 4.0, bh, txt, sw)


def arrow(pts: list[tuple[float, float]], head: bool = True) -> None:
    for (x0, y0), (x1, y1) in pairwise(pts if not head else pts[:-1]):
        ln = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-", lw=1.0, color="0.2", zorder=2, shrinkA=0, shrinkB=0)
        ax.add_patch(ln)
        arrows.append(ln)
    if not head:
        return
    a = FancyArrowPatch(
        pts[-2],
        pts[-1],
        arrowstyle="-|>,head_length=2.6,head_width=1.5",
        lw=1.0,
        color="0.2",
        zorder=2,
        shrinkA=0,
        shrinkB=0,
        mutation_scale=1.0,
    )
    ax.add_patch(a)
    arrows.append(a)


ax.text(1.0, 122.6, "Open input data (no fieldwork)", ha="left", va="center", fontsize=FS_HEAD, fontweight="bold")
fig.canvas.draw()
lx = 93.0
for name, (edge, fill) in SW.items():
    label = "input data" if name == "Data" else name
    ax.add_patch(
        FancyBboxPatch((lx, 121.0), 5.0, 3.2, boxstyle="round,pad=0,rounding_size=0.6", fc=fill, ec=edge, lw=1.0)
    )
    lt = ax.text(lx + 6.2, 122.6, label, ha="left", va="center", fontsize=FS_BOX)
    ext = lt.get_window_extent(renderer=typing.cast(typing.Any, fig.canvas).get_renderer())
    lx = float(ax.transData.inverted().transform((ext.x1, ext.y0))[0]) + 5.0
IW, IY, IH = 31.0, 103.0, 15.0
IX = [1.0 + k * (IW + 3.25) for k in range(5)]
inputs = [
    "Landsat 5/7/8/9\nCollection 2 Level-2\n1985–2025, Jun–Sep, 30 m",
    "Sentinel-2 L2A\nSTAC: Earth Search, MPC\n2017–2025, 10–20 m",
    "AOI / protected-area\nmask\nvector polygon",
    "Copernicus GLO-30 DEM\nelevation, slope, aspect\n30 m",
    "ERA5-Land and ERA5\nOpen-Meteo archive API\ndaily, 1984–2025",
]
for x, txt in zip(IX, inputs):
    box(x, IY, IW, IH, txt, "Data", bold_first=True)

LW, LH = 51.0, 40.5
LX = [1.0, 56.5, 112.0]
UY, DY = 55.5, 7.5
lane(
    LX[0],
    UY,
    LW,
    LH,
    1,
    "Scene acquisition",
    "",
    [
        ("STAC search: 1985–2025, Jun–Sep,\nLandsat + Sentinel-2", "Python"),
        ("Cloud/snow screening (QA_PIXEL,\nSCL); clear fraction per scene", "Python"),
        ("Windowed COG reads (rasterio)\nover the analysis window", "Python"),
    ],
)
lane(
    LX[1],
    UY,
    LW,
    LH,
    2,
    "Preprocessing",
    "",
    [
        ("Reflectance scaling\n$\\rho = 2.75\\times10^{-5}\\,\\mathrm{DN} - 0.2$", "Python"),
        ("TM/ETM+ $\\rightarrow$ OLI and MSI $\\rightarrow$ OLI\nharmonisation (per-band OLS)", "Python"),
        ("NDVI, NDSI; snow, water\nand AOI masks", "Python"),
    ],
)
lane(
    LX[2],
    UY,
    LW,
    LH,
    3,
    "Phenology & composites",
    "",
    [
        ("Harmonic regression (order 2):\namplitude $A$, phase $\\varphi$", "Python"),
        ("Annual peak-season NDVI\ncomposites, 1985–2025", "Python"),
        ("Per-pixel NDVI cube\n($x$, $y$, year)", "Python"),
    ],
)
lane(
    LX[2],
    DY,
    LW,
    LH,
    4,
    "Terrain & climate covariates",
    "",
    [
        ("r.slope.aspect, r.geomorphon\n(ten landform classes)", "GRASS GIS"),
        ("r.sun: summer global\nirradiance (Jun–Sep)", "GRASS GIS"),
        ("GDD > 5 °C, snow-cover duration\n(ERA5-Land, lapse-rate adjusted)", "Python"),
    ],
)
lane(
    LX[1],
    DY,
    LW,
    LH,
    5,
    "Ecotone & trends",
    "RQ1, RQ2",
    [
        ("Logistic NDVI($z$) fit per year and\naspect stratum $\\rightarrow$ $z_{50}$, $s$", "Python"),
        ("Theil–Sen, Mann–Kendall\n(Hamed–Rao), Pettitt test", "Python"),
        ("Greening velocity\n$v = (\\partial\\mathrm{NDVI}/\\partial t)\\,/\\,|\\nabla\\mathrm{NDVI}|$", "Python"),
    ],
)
lane(
    LX[0],
    DY,
    LW,
    LH,
    6,
    "Attribution & mapping",
    "RQ3",
    [
        ("Gradient boosting with spatial\nblock cross-validation", "Python"),
        ("SHAP attribution of drivers\nof $z_{50}$ and NDVI trends", "Python"),
        ("Maps, 3-D views and swath\nprofiles (grdview, grdtrack)", "GMT"),
    ],
)

ib = IY
ut = UY + LH
arrow([(IX[0] + IW / 2, ib), (IX[0] + IW / 2, ut)])
arrow([(IX[1] + IW / 2 - 6, ib), (IX[1] + IW / 2 - 6, 100.0), (40.0, 100.0), (40.0, ut)])
arrow([(IX[2] + IW / 2, ib), (IX[2] + IW / 2, ut)])
bus = 99.5
arrow([(IX[3] + IW / 2, ib), (IX[3] + IW / 2, bus), (167.0, bus), (167.0, DY + LH - 3.5), (LX[2] + LW, DY + LH - 3.5)])
arrow([(IX[4] + IW / 2, ib), (IX[4] + IW / 2, bus)], head=False)
my = UY + LH / 2 - 3.0
arrow([(LX[0] + LW, my), (LX[1], my)])
arrow([(LX[1] + LW, my), (LX[2], my)])
cy = (UY + DY + LH) / 2
arrow([(LX[2] + 8.0, UY), (LX[2] + 8.0, cy), (LX[1] + LW / 2, cy), (LX[1] + LW / 2, DY + LH)])
dy = DY + LH / 2 - 3.0
arrow([(LX[2], dy), (LX[1] + LW, dy)])
arrow([(LX[1], dy), (LX[0] + LW, dy)])
arrow([(LX[2] + LW / 2, DY), (LX[2] + LW / 2, 3.2), (LX[0] + LW / 2, 3.2), (LX[0] + LW / 2, DY)])

fig.canvas.draw()
r = typing.cast(typing.Any, fig.canvas).get_renderer()
outside = 0
for t, p in boxes:
    tb = t.get_window_extent(renderer=r)
    pb = p.get_window_extent(renderer=r).padded(-1.5)
    if not (pb.x0 <= tb.x0 and tb.x1 <= pb.x1 and pb.y0 <= tb.y0 and tb.y1 <= pb.y1):
        outside += 1
        print("text outside box:", repr(t.get_text()))
texts = [t for t in ax.texts if t.get_text()]
tbs = [t.get_window_extent(renderer=r) for t in texts]
overlap = 0
for i in range(len(tbs)):
    for j in range(i + 1, len(tbs)):
        if tbs[i].overlaps(tbs[j]):
            overlap += 1
            print("overlap:", repr(texts[i].get_text()[:30]), "|", repr(texts[j].get_text()[:30]))
crossing = 0
for a in arrows:
    path = a.get_path().transformed(a.get_transform())
    ab = Bbox.from_extents(*path.get_extents().extents).padded(0.5)
    crossing += sum(1 for tb in tbs if tb.overlaps(ab))
print(f"text_outside_box: {outside}  text_overlaps: {overlap}  arrow_crosses_text: {crossing}")

os.makedirs(OUT, exist_ok=True)
fig.savefig(os.path.join(OUT, "fig_workflow.pdf"))
fig.savefig(os.path.join(OUT, "fig_workflow.png"), dpi=600)
fig.savefig(os.path.join(OUT, "fig_workflow_preview.png"), dpi=130)
print("matplotlib", mpl.__version__)

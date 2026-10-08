"""Charts for the presentation deck (ppt/index.html), Swiss style: one accent, no titles.

Run after analysis.py, from courses/SDST1016/project/:
    uv run --with pandas --with matplotlib scripts/deck_charts.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "ppt" / "images"
OUT.mkdir(parents=True, exist_ok=True)

PAPER, INK, GREY2, GREY3, ACCENT = "#fafaf8", "#0a0a0a", "#d4d4d2", "#737373", "#002FA7"
plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Arial Unicode MS", "DejaVu Sans"], "font.size": 15,
    "figure.facecolor": PAPER, "axes.facecolor": PAPER, "axes.edgecolor": GREY2,
    "axes.labelcolor": GREY3, "xtick.color": GREY3, "ytick.color": GREY3,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.grid": True, "grid.color": "#e0e0e0", "grid.linewidth": .8, "axes.axisbelow": True,
    "xtick.major.size": 0, "ytick.major.size": 0, "savefig.dpi": 200,
})
K = lambda x, pos=None: f"{x / 1000:,.0f}k"

flows = pd.read_csv(ROOT / "data/processed/daily_hk_residents_land.csv", parse_dates=["Date"]).set_index("Date")
br = pd.read_csv(ROOT / "data/processed/breaks_2024_2025.csv", parse_dates=["start", "end"])


def running(b):
    win = pd.date_range(b.start - pd.Timedelta(days=1), b.end + pd.Timedelta(days=1))
    return flows.loc[win, "net"].cumsum().values


# 3:1 canvas; the plot sits in the right 52% so the slide's title block can cover the left
fig = plt.figure(figsize=(15, 5))
ax = fig.add_axes([0.455, 0.13, 0.515, 0.80])

ords = np.array([running(b) for _, b in br[br.kind == "ordinary"].iterrows()])
x = np.arange(ords.shape[1])
lo, hi = np.percentile(ords, [10, 90], axis=0)
ax.fill_between(x, lo, hi, color=GREY2, alpha=.7, lw=0)
ax.plot(x, ords.mean(axis=0), color=INK, lw=3)

longs = br[br.kind == "long"].sort_values("peak_net_outflow")
for _, b in longs.iterrows():
    y = running(b)
    ax.plot(np.arange(len(y)), y, color=ACCENT, lw=1.8, alpha=.85)

top = longs.iloc[-1]
ax.plot(top.peak_day_offset, top.peak_net_outflow, "s", color=ACCENT, ms=9)
ax.annotate("Easter 2025  ·  peak net outflow 643k", (top.peak_day_offset, top.peak_net_outflow),
            xytext=(14, 0), textcoords="offset points", va="center", color=INK, fontsize=15)
ax.text(3.25, 430000, "Long weekends\neach line is one break, n = 10", color=ACCENT, fontsize=15, va="bottom")
ax.text(0.0, -165000, "Ordinary weekends (black)\nmean of 93, band = 10th–90th percentile", color=INK, fontsize=15, va="bottom")

ax.axhline(0, color=GREY3, lw=1)
ax.set_xticks(range(7), ["Eve", "Day 1", "Day 2", "Day 3", "Day 4", "Day 5", "Day 6"])
ax.set_xlim(-.15, 6.15)
ax.set_ylim(-175000, 700000)
ax.set_yticks([0, 200000, 400000, 600000])
ax.yaxis.set_major_formatter(K)
ax.grid(axis="x", visible=False)
ax.set_ylabel("Net outflow since the eve", fontsize=14)
fig.savefig(OUT / "08-cumulative-net-outflow.png")
print("wrote", OUT / "08-cumulative-net-outflow.png")

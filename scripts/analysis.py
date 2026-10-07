"""SDST1016 Group 3: Weekend Habit or Holiday Trip? HK residents' northbound travel on ordinary and long weekends.

Run from courses/SDST1016/project/:
    uv run --with pandas --with scipy --with matplotlib --with statsmodels scripts/analysis.py

Inputs (data/):
    immd_daily.csv          ImmD daily passenger traffic, 2021-01-01 onward
    holidays_1823.json      1823 public holiday iCal (2025 onward)
    2024 holidays are typed in below from gov.hk/en/about/abouthk/holiday/2024.htm
Outputs:
    data/processed/*.csv    daily series, break table, test results
    figures/*.png           charts for slides and report
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
DATA, OUT, FIG = ROOT / "data", ROOT / "data" / "processed", ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

# 7 land control points to the Mainland (HZMB goes to Zhuhai AND Macao, kept out of
# the main definition and tested as a sensitivity check)
LAND_SZ = ["Lo Wu", "Lok Ma Chau Spur Line", "Lok Ma Chau", "Shenzhen Bay",
           "Man Kam To", "Heung Yuen Wai", "Express Rail Link West Kowloon"]
HZMB = "Hong Kong-Zhuhai-Macao Bridge"

MAIN_START, MAIN_END = "2024-01-01", "2025-12-31"   # registered study period
OOS_START, OOS_END = "2026-01-01", "2026-10-06"     # out-of-sample check

HOLIDAYS_2024 = {
    "2024-01-01": "The first day of January", "2024-02-10": "Lunar New Year's Day",
    "2024-02-12": "The third day of Lunar New Year", "2024-02-13": "The fourth day of Lunar New Year",
    "2024-03-29": "Good Friday", "2024-03-30": "The day following Good Friday",
    "2024-04-01": "Easter Monday", "2024-04-04": "Ching Ming Festival", "2024-05-01": "Labour Day",
    "2024-05-15": "The Birthday of the Buddha", "2024-06-10": "Tuen Ng Festival",
    "2024-07-01": "Hong Kong Special Administrative Region Establishment Day",
    "2024-09-18": "The day following the Chinese Mid-Autumn Festival", "2024-10-01": "National Day",
    "2024-10-11": "Chung Yeung Festival", "2024-12-25": "Christmas Day",
    "2024-12-26": "The first weekday after Christmas Day",
}
HOLIDAYS_2023_TAIL = ["2023-12-25", "2023-12-26"]  # so the break spanning 2024-01-01 is built right


# ---------------------------------------------------------------- load and clean
def load_daily():
    df = pd.read_csv(DATA / "immd_daily.csv", encoding="utf-8-sig")
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")
    df["Control Point"] = df["Control Point"].replace({"Macau Ferry Terminal": "Macao Ferry Terminal"})
    assert not df.duplicated(["Date", "Control Point", "Arrival / Departure"]).any()
    assert (df[["Hong Kong Residents", "Mainland Visitors", "Other Visitors"]].sum(axis=1) == df["Total"]).all()
    return df


def resident_flows(df, points):
    sub = df[df["Control Point"].isin(points)]
    p = sub.pivot_table(index="Date", columns="Arrival / Departure",
                        values="Hong Kong Residents", aggfunc="sum")
    p = p.rename(columns={"Departure": "dep", "Arrival": "arr"})
    full = pd.date_range(p.index.min(), p.index.max())
    assert len(full) == len(p), "missing days"
    p["net"] = p["dep"] - p["arr"]
    return p


def load_holidays():
    with open(DATA / "holidays_1823.json", encoding="utf-8-sig") as f:
        ev = json.load(f)["vcalendar"][0]["vevent"]
    hol = {pd.Timestamp(e["dtstart"][0]): e["summary"] for e in ev}
    for d, name in HOLIDAYS_2024.items():
        hol.setdefault(pd.Timestamp(d), name)
    for d in HOLIDAYS_2023_TAIL:
        hol.setdefault(pd.Timestamp(d), "PH")
    return hol


# ---------------------------------------------------------------- breaks
def build_breaks(dates, hol):
    """A break = maximal run of consecutive days off (Sat, Sun, public holiday)."""
    off = pd.Series([(d.weekday() >= 5) or (d in hol) for d in dates], index=dates)
    breaks, start = [], None
    for d, is_off in off.items():
        if is_off and start is None:
            start = d
        if not is_off and start is not None:
            breaks.append((start, d - pd.Timedelta(days=1)))
            start = None
    rows = []
    for s, e in breaks:
        n = (e - s).days + 1
        days = pd.date_range(s, e)
        has_ph = any(d in hol for d in days)
        names = sorted({hol[d] for d in days if d in hol and hol[d] != "PH"})
        is_lny = any("Lunar New Year" in hol.get(d, "") for d in days)
        if n >= 3:
            kind = "long"
        elif n == 2 and s.weekday() == 5 and not has_ph:
            kind = "ordinary"
        elif n == 2 and s.weekday() == 5:
            kind = "ordinary_ph"      # holiday fell on Sat/Sun, still a plain 2-day weekend
        elif n == 2:
            kind = "midweek2"
        else:
            kind = "single"
        # clean = no public holiday within 5 days either side (the surrounding work weeks are normal)
        near = pd.date_range(s - pd.Timedelta(days=5), e + pd.Timedelta(days=5))
        clean = not any(d in hol for d in near)
        rows.append(dict(start=s, end=e, length=n, kind=kind, lny=is_lny, clean=clean,
                         holidays="; ".join(names), start_dow=s.day_name()))
    return pd.DataFrame(rows)


def break_metrics(br, flows):
    out = []
    for _, b in br.iterrows():
        eve = b.start - pd.Timedelta(days=1)
        win = pd.date_range(eve, b.end)
        if win[0] < flows.index.min() or win[-1] > flows.index.max():
            out.append(dict(peak_net_away=np.nan)); continue
        cum = flows.loc[win, "net"].cumsum()
        inside = pd.date_range(b.start, b.end)
        # day after the break: residents coming home, used for the return-day comparison
        after = b.end + pd.Timedelta(days=1)
        out.append(dict(
            peak_net_away=cum.max(),
            peak_day_offset=int(np.argmax(cum.values)),   # 0 = eve, 1 = first day off
            mean_daily_dep=flows.loc[inside, "dep"].mean(),
            total_dep=flows.loc[inside, "dep"].sum(),
            eve_dep=flows.loc[eve, "dep"],
            first_day_dep=flows.loc[b.start, "dep"],
            last_day_arr=flows.loc[b.end, "arr"],
            eve_dow=eve.day_name(),
        ))
    return pd.concat([br.reset_index(drop=True), pd.DataFrame(out)], axis=1)


# ---------------------------------------------------------------- stats helpers
def welch(a, b, alternative="greater"):
    a, b = np.asarray(a, float), np.asarray(b, float)
    t, p = stats.ttest_ind(a, b, equal_var=False, alternative=alternative)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    dfw = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    diff = a.mean() - b.mean()
    half = stats.t.ppf(0.975, dfw) * np.sqrt(va + vb)
    pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    mw = stats.mannwhitneyu(a, b, alternative=alternative)
    return dict(n_long=len(a), n_ord=len(b), mean_long=a.mean(), mean_ord=b.mean(),
                ratio=a.mean() / b.mean(), diff=diff, ci95_low=diff - half, ci95_high=diff + half,
                t=t, df=dfw, p_one_sided=p if alternative == "greater" else np.nan,
                p_two_sided=stats.ttest_ind(a, b, equal_var=False).pvalue,
                cohens_d=diff / pooled, mannwhitney_p=mw.pvalue)


def tost(a, b, margin_pct):
    """Two one-sided tests: is the difference in means within +/- margin_pct of mean(b)?"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = np.sqrt(va + vb)
    dfw = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    diff, m = a.mean() - b.mean(), margin_pct * b.mean()
    p = max(1 - stats.t.cdf((diff + m) / se, dfw), stats.t.cdf((diff - m) / se, dfw))
    half = stats.t.ppf(0.95, dfw) * se
    return dict(margin_pct=margin_pct, diff_pct=diff / b.mean(), ci90_low_pct=(diff - half) / b.mean(),
                ci90_high_pct=(diff + half) / b.mean(), tost_p=p)


def bootstrap_ratio(a, b, n=10000, seed=1016):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    r = [rng.choice(a, len(a)).mean() / rng.choice(b, len(b)).mean() for _ in range(n)]
    return np.percentile(r, [2.5, 97.5])


# ---------------------------------------------------------------- charts
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#a3a29c"
plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Arial Unicode MS", "DejaVu Sans"], "font.size": 12, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.titlesize": 15, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "savefig.dpi": 200, "savefig.bbox": "tight",
})
K = lambda x, pos=None: f"{x / 1000:,.0f}k"


def title(ax, main, sub):
    ax.set_title(main, color=INK, pad=26)
    ax.text(0, 1.02, sub, transform=ax.transAxes, color=INK2, fontsize=11, va="bottom")


def short_name(holidays):
    for key, name in [("Lunar", "Lunar New Year"), ("Good Friday", "Easter"), ("Christmas", "Christmas"),
                      ("Tuen Ng", "Tuen Ng"), ("Establishment", "July 1st"), ("Chung Yeung", "Chung Yeung"),
                      ("Ching Ming", "Ching Ming"), ("Buddha", "Buddha's Birthday"), ("Labour", "Labour Day"),
                      ("Mid-Autumn", "Mid-Autumn"), ("National", "National Day"), ("January", "New Year")]:
        if key in str(holidays):
            return name
    return "Long weekend"


def fig_weekly_rhythm(flows):
    d = flows.loc[MAIN_START:MAIN_END].copy()
    d["dow"] = d.index.dayofweek
    d["year"] = d.index.year
    fig, ax = plt.subplots(figsize=(10, 5.2))
    labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    x = np.arange(7)
    for col, color, name in [("dep", BLUE, "Leaving HK (departures)"), ("arr", ORANGE, "Coming back (arrivals)")]:
        m = d.groupby("dow")[col].median()
        ax.plot(x, m.values, color=color, lw=2.5, marker="o", ms=8, mec="#fcfcfb", mew=2, label=name)
        ax.annotate(K(m.iloc[5 if col == "dep" else 6]), (5 if col == "dep" else 6, m.iloc[5 if col == "dep" else 6]),
                    xytext=(0, 12), textcoords="offset points", ha="center", color=INK, fontweight="bold")
    ax.set_xticks(x, labels)
    ax.yaxis.set_major_formatter(K)
    ax.set_ylim(0, 420000)
    ax.set_ylabel("HK residents per day (median)")
    ax.legend(frameon=False, loc="upper left")
    title(ax, "Going north is already a weekly routine",
          "HK residents via the 7 land control points, median by day of week, 2024–2025")
    fig.savefig(FIG / "01_weekly_rhythm.png"); plt.close(fig)


def fig_cumulative_curves(m, flows):
    fig, ax = plt.subplots(figsize=(10, 5.6))
    ords = m[m.kind == "ordinary"]
    curves = []
    for _, b in ords.iterrows():
        win = pd.date_range(b.start - pd.Timedelta(days=1), b.end + pd.Timedelta(days=1))
        curves.append(flows.loc[win, "net"].cumsum().values)
    curves = np.array(curves)
    xo = np.arange(curves.shape[1])
    lo, hi = np.percentile(curves, [10, 90], axis=0)
    ax.fill_between(xo, lo, hi, color=GRAY, alpha=0.25, lw=0)
    ax.plot(xo, curves.mean(axis=0), color=INK2, lw=2.5, label=f"Ordinary weekend (mean of {len(ords)}, band = 10th–90th pct)")
    longs = m[m.kind == "long"].sort_values("start")
    for i, (_, b) in enumerate(longs.iterrows()):
        win = pd.date_range(b.start - pd.Timedelta(days=1), b.end + pd.Timedelta(days=1))
        y = flows.loc[win, "net"].cumsum().values
        ax.plot(np.arange(len(y)), y, color=BLUE, lw=1.6, alpha=0.75,
                label="Long weekend (one line each)" if i == 0 else None)
    top = longs.sort_values("peak_net_away").iloc[-1]
    ax.annotate(f"{short_name(top.holidays)} {top.start:%Y}: {K(top.peak_net_away)} away at peak",
                (top.peak_day_offset, top.peak_net_away), xytext=(-10, 6), textcoords="offset points",
                color=INK, fontsize=10.5, ha="right", va="bottom")
    ax.set_ylim(None, top.peak_net_away * 1.12)
    ax.axhline(0, color=MUTED, lw=1)
    maxlen = int(longs.length.max()) + 2
    ax.set_xticks(range(maxlen), ["Eve"] + [f"Day {i}" for i in range(1, maxlen)])
    ax.set_xlim(-0.2, maxlen - 0.8)
    ax.yaxis.set_major_formatter(K)
    ax.set_ylabel("HK residents still in the Mainland\n(cumulative departures − arrivals)")
    ax.legend(frameon=False, loc="upper right")
    title(ax, "On long weekends, people stay over instead of coming back the same day",
          "Running total of HK residents away, from the day before the break until the day after, 2024–2025")
    fig.savefig(FIG / "02_cumulative_away.png"); plt.close(fig)


def fig_peak_by_length(m):
    d = m[m.kind.isin(["single", "ordinary", "long"])].copy()
    d["grp"] = d.length.clip(upper=5)
    order = sorted(d.grp.unique())
    names = {1: "1 day\n(midweek holiday)", 2: "2 days\n(ordinary weekend)", 3: "3 days", 4: "4 days", 5: "5 days"}
    fig, ax = plt.subplots(figsize=(10, 5.4))
    rng = np.random.default_rng(3)
    for i, g in enumerate(order):
        v = d[d.grp == g].peak_net_away.values
        color = BLUE if g >= 3 else GRAY
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=46, color=color, alpha=0.8,
                   edgecolor="#fcfcfb", linewidth=1.2, zorder=3)
        ax.hlines(v.mean(), i - 0.28, i + 0.28, color=INK, lw=2.5, zorder=4)
        ax.annotate(f"{K(v.mean())}  (n={len(v)})", (i + 0.3, v.mean()), xytext=(4, 0),
                    textcoords="offset points", va="center", color=INK, fontsize=10.5)
    ax.set_xticks(range(len(order)), [names[g] for g in order])
    ax.set_xlim(-0.5, len(order) - 0.2)
    ax.yaxis.set_major_formatter(K)
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_ylabel("Peak HK residents away at once")
    title(ax, "Every extra day off adds more people staying north",
          "Peak number of HK residents in the Mainland at the same time, by length of break, 2024–2025 (bar = mean)")
    fig.savefig(FIG / "03_peak_by_length.png"); plt.close(fig)


def fig_peak_vs_daily(res_peak, res_daily, ci_peak, ci_daily):
    fig, ax = plt.subplots(figsize=(10, 4.2))
    rows = [("Daily departures during the break", res_daily, ci_daily),
            ("Peak residents away at once", res_peak, ci_peak)]
    for i, (lab, r, ci) in enumerate(rows):
        color = BLUE if r["p_one_sided"] < 0.05 else GRAY
        ax.hlines(i, ci[0], ci[1], color=color, lw=3, alpha=0.5)
        ax.plot(r["ratio"], i, "o", color=color, ms=12, mec="#fcfcfb", mew=2)
        ax.annotate(f"×{r['ratio']:.2f}   one-sided p = {r['p_one_sided']:.4f}", (ci[1], i),
                    xytext=(10, 0), textcoords="offset points", va="center", color=INK, fontsize=11)
    ax.axvline(1, color=MUTED, lw=1.2, ls="--")
    ax.text(1, -0.6, "no difference", color=INK2, fontsize=10, ha="center")
    ax.set_yticks([0, 1], [r[0] for r in rows])
    ax.set_ylim(-0.8, 1.6)
    ax.set_xlim(0.6, max(ci_peak[1], 3.5) + 1.2)
    ax.set_xlabel("Long weekend ÷ ordinary weekend (dot = ratio of means, line = 95% bootstrap CI)")
    ax.grid(axis="y", visible=False)
    title(ax, "Not more people going, but more people staying",
          "Long weekends vs ordinary weekends, HK residents via the 7 land control points, 2024–2025")
    fig.savefig(FIG / "04_peak_vs_daily.png"); plt.close(fig)


def fig_timeline(flows, m):
    d = flows.loc["2024-01-01":OOS_END, "dep"]
    wk = d.rolling(7, center=True).mean()
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.plot(d.index, d.values, color=GRAY, lw=0.7, alpha=0.8, label="Daily")
    ax.plot(wk.index, wk.values, color=INK2, lw=2, label="7-day average")
    for i, (_, b) in enumerate(m[m.kind == "long"].iterrows()):
        ax.axvspan(b.start, b.end + pd.Timedelta(days=1), color=BLUE, alpha=0.18, lw=0,
                   label="Long weekend" if i == 0 else None)
    ax.axvline(pd.Timestamp(OOS_START), color=MUTED, lw=1, ls="--")
    ax.text(pd.Timestamp(OOS_START), ax.get_ylim()[1] * 0.98, "  2026: out-of-sample check", color=INK2, fontsize=10, va="top")
    ax.yaxis.set_major_formatter(K)
    ax.set_ylim(0, None)
    ax.set_ylabel("HK residents leaving per day")
    ax.legend(frameon=False, loc="lower right", ncol=3)
    title(ax, "Two and a half years of HK residents heading north",
          "Daily departures through the 7 land control points, Jan 2024 – Oct 2026, long weekends shaded")
    fig.savefig(FIG / "05_timeline.png"); plt.close(fig)


def fig_eve_effect(m, flows):
    """Departures the day before a break vs the same weekday when no break follows."""
    longs = m[(m.kind == "long")].copy()
    rows = []
    for _, b in longs.iterrows():
        eve = b.start - pd.Timedelta(days=1)
        same = flows.loc[MAIN_START:MAIN_END]
        same = same[(same.index.dayofweek == eve.dayofweek)]
        # exclude days that are themselves eves of a break or inside one
        bad = set()
        for _, bb in m.iterrows():
            bad.update(pd.date_range(bb.start - pd.Timedelta(days=1), bb.end))
        same = same[~same.index.isin(bad)]
        if eve.dayofweek == 4:   # normal Fridays are always eves of a weekend: compare with ordinary-weekend Fridays
            fris = [o.start - pd.Timedelta(days=1) for _, o in m[m.kind == "ordinary"].iterrows()]
            same = flows.loc[[f for f in fris if MAIN_START <= str(f.date()) <= MAIN_END]]
        rows.append(dict(start=b.start, eve_dow=eve.day_name(), eve_dep=b.eve_dep,
                         baseline=same["dep"].median(), ratio=b.eve_dep / same["dep"].median()))
    return pd.DataFrame(rows)


def port_mix(raw, m):
    """Share of each land control point in HK-resident departures from eve to last day off."""
    dep = raw[(raw["Arrival / Departure"] == "Departure") & raw["Control Point"].isin(LAND_SZ + [HZMB])]
    dep = dep.pivot_table(index="Date", columns="Control Point", values="Hong Kong Residents", aggfunc="sum")
    rows = []
    for _, b in m[m.kind.isin(["long", "ordinary"])].iterrows():
        t = dep.loc[pd.date_range(b.start - pd.Timedelta(days=1), b.end)].sum()
        land = t[LAND_SZ]
        rows.append(dict(start=b.start, kind=b.kind, **(land / land.sum()).to_dict(),
                         hzmb_per_day=t[HZMB] / (b.length + 1)))
    return pd.DataFrame(rows)


def fig_port_mix(pm):
    short = {"Express Rail Link West Kowloon": "High-speed rail (West Kowloon)", "Lok Ma Chau Spur Line": "Lok Ma Chau Spur Line (Futian)"}
    g = pm.groupby("kind")[LAND_SZ].mean().T
    g = g.sort_values("ordinary")
    fig, ax = plt.subplots(figsize=(10, 5.2))
    y = np.arange(len(g))
    for i, (cp, r) in enumerate(g.iterrows()):
        hi = cp == "Express Rail Link West Kowloon"
        ax.hlines(i, r.ordinary, r.long, color=BLUE if hi else GRID, lw=3 if hi else 2)
        ax.plot(r.ordinary, i, "o", color=GRAY, ms=10, mec="#fcfcfb", mew=2, label="Ordinary weekend" if i == 0 else None)
        ax.plot(r.long, i, "o", color=BLUE, ms=10, mec="#fcfcfb", mew=2, label="Long weekend" if i == 0 else None)
        if hi:
            ax.annotate(f"{r.ordinary:.1%} → {r.long:.1%}", (max(r.long, r.ordinary), i), xytext=(12, 0),
                        textcoords="offset points", va="center", color=INK, fontweight="bold")
    ax.set_yticks(y, [short.get(c, c) for c in g.index])
    ax.xaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
    ax.set_xlim(0, None)
    ax.set_xlabel("Share of HK-resident departures (eve to last day off)")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, loc="lower right")
    title(ax, "Long weekends send people further: the high-speed rail share jumps",
          "Mix of the 7 land control points, mean over breaks, 2024–2025")
    fig.savefig(FIG / "06_port_mix.png"); plt.close(fig)


def main():
    raw = load_daily()
    hol = load_holidays()
    flows = resident_flows(raw, LAND_SZ)
    flows_hzmb = resident_flows(raw, LAND_SZ + [HZMB])
    flows.to_csv(OUT / "daily_hk_residents_land.csv")

    br = build_breaks(pd.date_range("2023-12-01", OOS_END), hol)
    m = break_metrics(br, flows)
    m_h = break_metrics(br, flows_hzmb)
    m["name"] = m.holidays.map(short_name).where(m.holidays.fillna("") != "", "")
    # a break belongs to the study period if it starts inside it (drops 2023-12-30 to 2024-01-01)
    main_m = m[(m.start >= MAIN_START) & (m.end <= MAIN_END)].copy()
    m.to_csv(OUT / "breaks_all.csv", index=False)
    main_m.to_csv(OUT / "breaks_2024_2025.csv", index=False)

    def split(df):
        return df[df.kind == "long"], df[df.kind == "ordinary"]

    results = {}
    L, O = split(main_m)
    results["main_peak"] = welch(L.peak_net_away, O.peak_net_away)
    results["main_daily_dep"] = welch(L.mean_daily_dep, O.mean_daily_dep)
    results["main_last_day_arr"] = welch(L.last_day_arr, O.last_day_arr)
    results["main_first_day_dep"] = welch(L.first_day_dep, O.first_day_dep)
    # sensitivity
    results["sens_clean_ordinary"] = welch(L.peak_net_away, O[O.clean].peak_net_away)
    results["sens_clean_daily_dep"] = welch(L.mean_daily_dep, O[O.clean].mean_daily_dep)
    results["sens_no_lny"] = welch(L[~L.lny].peak_net_away, O.peak_net_away)
    results["sens_ordinary_incl_ph"] = welch(L.peak_net_away, main_m[main_m.kind.isin(["ordinary", "ordinary_ph"])].peak_net_away)
    mh = m_h.loc[main_m.index]
    Lh, Oh = split(mh)
    results["sens_with_hzmb"] = welch(Lh.peak_net_away, Oh.peak_net_away)
    oos = m[(m.start >= OOS_START) & (m.end < pd.Timestamp(OOS_END))]
    Lo, Oo = split(oos)
    if len(Lo) >= 2:
        results["oos_2026_peak"] = welch(Lo.peak_net_away, Oo.peak_net_away)
        results["oos_2026_daily_dep"] = welch(Lo.mean_daily_dep, Oo.mean_daily_dep)
    pooled = pd.concat([main_m, oos])
    Lp, Op = split(pooled)
    results["pooled_2024_2026_peak"] = welch(Lp.peak_net_away, Op.peak_net_away)

    # supporting: departures on each public holiday itself vs ordinary-weekend daily departures
    ph_days = sorted(d for d in hol if pd.Timestamp(MAIN_START) <= d <= pd.Timestamp(MAIN_END))
    results["ph_day_dep"] = welch(flows.loc[ph_days, "dep"], O.mean_daily_dep)
    pd.DataFrame([tost(L.mean_daily_dep, O.mean_daily_dep, m) for m in (0.05, 0.10)]).to_csv(
        OUT / "tost_daily_dep.csv", index=False)

    pm = port_mix(raw, main_m)
    pm.to_csv(OUT / "port_mix.csv", index=False)
    xrl = "Express Rail Link West Kowloon"
    results["xrl_share"] = welch(pm[pm.kind == "long"][xrl], pm[pm.kind == "ordinary"][xrl])
    results["hzmb_per_day"] = welch(pm[pm.kind == "long"].hzmb_per_day, pm[pm.kind == "ordinary"].hzmb_per_day)

    res = pd.DataFrame(results).T
    res.to_csv(OUT / "tests.csv")

    ci_peak = bootstrap_ratio(L.peak_net_away, O.peak_net_away)
    ci_daily = bootstrap_ratio(L.mean_daily_dep, O.mean_daily_dep)

    # dose-response: peak ~ length, weekend-type breaks only (start on Fri/Sat/Sun or include a weekend)
    reg_df = main_m[main_m.kind.isin(["single", "ordinary", "long"])].copy()
    reg_df["year"] = reg_df.start.dt.year.astype(str)
    reg = smf.ols("peak_net_away ~ length + year", data=reg_df).fit(cov_type="HC1")
    with open(OUT / "regression_peak_on_length.txt", "w") as f:
        f.write(reg.summary().as_text())

    eve = fig_eve_effect(main_m, flows)
    eve.to_csv(OUT / "eve_effect.csv", index=False)

    by_len = main_m[main_m.kind.isin(["single", "ordinary", "long"])].groupby("length").agg(
        n=("peak_net_away", "size"), peak_mean=("peak_net_away", "mean"),
        peak_per_day=("peak_net_away", lambda s: s.mean()), daily_dep=("mean_daily_dep", "mean"))
    by_len["peak_per_day"] = by_len.peak_mean / by_len.index
    by_len.to_csv(OUT / "by_length.csv")

    d = flows.loc[MAIN_START:MAIN_END]
    dow = d.groupby(d.index.dayofweek)[["dep", "arr"]].median()
    yearly = d.groupby(d.index.year)["dep"].mean()
    dow.to_csv(OUT / "weekday_median.csv")

    fig_weekly_rhythm(flows)
    fig_cumulative_curves(main_m, flows)
    fig_peak_by_length(main_m)
    fig_peak_vs_daily(results["main_peak"], results["main_daily_dep"], ci_peak, ci_daily)
    fig_timeline(flows, pd.concat([main_m, oos]))
    fig_port_mix(pm)

    pd.set_option("display.width", 200, "display.max_columns", 30)
    print("== long weekends 2024-2025 ==")
    print(L[["start", "end", "length", "holidays", "lny", "peak_net_away", "peak_day_offset",
             "mean_daily_dep", "eve_dep", "eve_dow"]].to_string())
    print("\n== other non-ordinary breaks ==")
    print(main_m[~main_m.kind.isin(["long", "ordinary"])][["start", "length", "kind", "holidays", "peak_net_away"]].to_string())
    print("\n== tests ==")
    print(res[["n_long", "n_ord", "mean_long", "mean_ord", "ratio", "ci95_low", "ci95_high",
               "t", "df", "p_one_sided", "p_two_sided", "cohens_d", "mannwhitney_p"]].round(4).to_string())
    print("\nbootstrap ratio CI peak", ci_peak.round(2), "daily", ci_daily.round(2))
    print("\n== by length ==\n", by_len.round(0).to_string())
    print("\n== regression ==\n", reg.params.round(0).to_string(), "\n", reg.pvalues.round(4).to_string(), "\nR2", round(reg.rsquared, 3))
    print("\n== eve effect ==\n", eve.to_string())
    print("\n== weekday median 2024-25 ==\n", dow.round(0).to_string())
    print("\n== yearly mean dep ==\n", yearly.round(0).to_string())
    print("\n== 2026 long weekends ==\n", Lo[["start", "length", "holidays", "peak_net_away", "mean_daily_dep"]].to_string() if len(Lo) else "none")
    print("ordinary peak 2024/2025:", O.groupby(O.start.dt.year).peak_net_away.mean().round(0).to_dict())
    print("ordinary peak day offset counts:", O.peak_day_offset.value_counts().to_dict())


if __name__ == "__main__":
    main()

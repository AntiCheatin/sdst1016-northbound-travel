# Weekend Habit or Holiday Trip?

Hong Kong residents' northbound travel on ordinary and long weekends, 2024 to 2025. SDST1016 group project, Group 3.

**Finding:** long weekends (3+ days off) show a larger *peak cumulative net outflow* of Hong Kong residents through the seven land checkpoints than ordinary weekends: 390,348 vs 136,934, a ratio of 2.85 (Welch t = 5.52, one-sided p = 0.0002, n = 10 vs 93). The result is unchanged when both are compared over the same three-day window (eve plus the first two days off). Long weekends also show a different checkpoint mix, with the high-speed rail share of departures rising from 7.6% to 10.5%. We found insufficient evidence that mean daily departures are higher (ratio 1.02, one-sided p = 0.32; 90% CI of the difference −5.8% to +10.2%, equivalence not established).

**What the metric is not.** The peak cumulative net outflow is departures minus arrivals summed from a zero baseline on the day before each break. It is a net flow, not a count of people in the Mainland: the starting stock is unknown, the seven checkpoints are not a closed system, and the data count crossings rather than individuals. The patterns are consistent with longer stays on long weekends, but these aggregate counts cannot identify individual habits, length of stay or destinations.

Full write-up (Chinese): [RESULTS.md](RESULTS.md).

## Reproduce

```bash
uv run --with pandas --with scipy --with matplotlib --with statsmodels scripts/analysis.py
```

Runs in about 10 seconds and rewrites `data/processed/` and `figures/`.

## Data

| File | Source |
|---|---|
| `data/immd_daily.csv` | Immigration Department, [Statistics on Daily Passenger Traffic](https://www.immd.gov.hk/opendata/eng/transport/immigration_clearance/statistics_on_daily_passenger_traffic.csv), downloaded 2026-10-07 (to 2026-10-06) |
| `data/holidays_1823.json` | 1823 public holiday calendar, [en.json](https://www.1823.gov.hk/common/ical/en.json) (2025–2027) |
| `data/gov_holiday_2024.html` | [GovHK 2024 general holidays](https://www.gov.hk/en/about/abouthk/holiday/2024.htm), typed into the script |

## Method in one paragraph

A break is a run of consecutive Saturdays, Sundays and public holidays. For each break, start the day before it and add up HK-resident departures minus arrivals through the 7 land control points to Shenzhen, day by day; the maximum of that running total is the peak cumulative net outflow. Compare 10 long weekends with 93 ordinary weekends in 2024–2025 using Welch's two-sample t-test (one-sided), with a Mann-Whitney U test and bootstrap intervals as checks.

## Figures

| | |
|---|---|
| `01_weekly_rhythm.png` | Departures and arrivals by day of week |
| `02_cumulative_net_outflow.png` | Cumulative net outflow from the eve, every break |
| `03_peak_by_length.png` | Peak cumulative net outflow by length of break |
| `04_peak_vs_daily.png` | Peak vs daily departures, ratio with 95% CI |
| `05_timeline.png` | Daily departures, Jan 2024 – Oct 2026 |
| `06_port_mix.png` | Share of departures by control point |

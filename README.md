# Weekend Habit or Holiday Trip?

Hong Kong residents' northbound travel on ordinary and long weekends, 2024 to 2025. SDST1016 group project, Group 3.

**Finding:** northbound travel is mainly a weekend habit, and long weekends stretch it into a holiday trip. On long weekends (3+ days off), the peak number of Hong Kong residents in the Mainland at the same time is 2.85 times that of an ordinary weekend (Welch t = 5.52, one-sided p = 0.0002). Daily departures are not significantly higher (×1.02, p = 0.32; 90% CI of the difference −5.8% to +10.2%), and on public holidays themselves they are 4.8% lower (p = 0.50). Long weekends also raise the high-speed rail share of departures from 7.6% to 10.5%. The main result holds under five robustness checks and on out-of-sample 2026 data.

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

A break is a run of consecutive Saturdays, Sundays and public holidays. For each break, start the day before it and add up HK-resident departures minus arrivals through the 7 land control points to Shenzhen, day by day; the maximum of that running total is the peak number of residents away. Compare 10 long weekends with 93 ordinary weekends in 2024–2025 using Welch's two-sample t-test (one-sided), with a Mann-Whitney U test and bootstrap intervals as checks.

## Figures

| | |
|---|---|
| `01_weekly_rhythm.png` | Departures and arrivals by day of week |
| `02_cumulative_away.png` | Running total of residents away, every break |
| `03_peak_by_length.png` | Peak residents away by length of break |
| `04_peak_vs_daily.png` | Peak vs daily departures, ratio with 95% CI |
| `05_timeline.png` | Daily departures, Jan 2024 – Oct 2026 |
| `06_port_mix.png` | Share of departures by control point |

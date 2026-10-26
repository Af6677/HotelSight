"""Establish the dataset's calendar conventions before the dashboard derives them from dates.
Writes docs/date_conventions.json.  Run:  python scripts/check_date_conventions.py"""
import json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
d = pd.read_csv(ROOT / "experiment/data/hotel_bookings.csv")
arr = pd.to_datetime(d.arrival_date_year.astype(str) + "-" + d.arrival_date_month + "-" +
                     d.arrival_date_day_of_month.astype(str), format="%Y-%B-%d")
out = {}

# 1. Week number: compare candidate conventions with the stored value
cands = {"iso_8601": arr.dt.isocalendar().week.astype(int),
         "strftime_U_sunday_first": arr.dt.strftime("%U").astype(int),
         "strftime_W_monday_first": arr.dt.strftime("%W").astype(int)}
jan1 = pd.to_datetime(arr.dt.year.astype(str) + "-01-01")
cands["tsql_datepart_week_sunday_first (USED)"] = (arr.dt.dayofyear - 1 + (jan1.dt.weekday + 1) % 7) // 7 + 1
out["week_number_match_rate"] = {k: float((v == d.arrival_date_week_number).mean()) for k, v in cands.items()}
iso_y = arr.dt.isocalendar().year.astype(int)
mism = d.arrival_date_week_number != cands["iso_8601"]
out["iso_mismatches"] = int(mism.sum())
out["iso_mismatch_examples"] = (pd.DataFrame({"arrival": arr[mism].dt.date.astype(str), "stored_week": d.arrival_date_week_number[mism]})
                                .drop_duplicates().head(10).to_dict("records"))
out["iso_year_differs_from_calendar_year_rows"] = int((iso_y != arr.dt.year).sum())

# 2. Weekend nights: which nights count as 'weekend'?
nights = d.stays_in_weekend_nights + d.stays_in_week_nights
def count(starts_on):   # nights whose START date falls on the given weekdays (0=Mon)
    res = np.zeros(len(d), dtype=int); wd = arr.dt.weekday.values; n = nights.values
    for k in range(int(n.max())):
        res += ((k < n) & np.isin((wd + k) % 7, starts_on)).astype(int)
    return res
for name, days in {"nights_starting_fri_sat": [4, 5], "nights_starting_sat_sun": [5, 6],
                   "nights_starting_sun_mon (USED)": [6, 0]}.items():
    out[f"weekend_rule_{name}_match_rate"] = float((count(days) == d.stays_in_weekend_nights.values).mean())

# 3. Lead time and stay length vs status date for checked-out bookings
co = d.reservation_status == "Check-Out"
stay = (pd.to_datetime(d.reservation_status_date) - arr).dt.days
out["checked_out_status_date_equals_arrival_plus_nights_rate"] = float((stay[co] == nights[co]).mean())
out["range_lead_time"] = [int(d.lead_time.min()), int(d.lead_time.max())]
out["range_total_nights"] = [int(nights.min()), int(nights.max())]
out["range_weekend_nights"] = [int(d.stays_in_weekend_nights.min()), int(d.stays_in_weekend_nights.max())]
out["range_week_nights"] = [int(d.stays_in_week_nights.min()), int(d.stays_in_week_nights.max())]
out["arrival_date_range"] = [str(arr.min().date()), str(arr.max().date())]
(ROOT / "docs/date_conventions.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print(json.dumps(out, indent=2))

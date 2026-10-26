"""Date derivation follows the dataset's recorded conventions (checked on every row)."""
from datetime import date

import pandas as pd

from hotelsight.contract import dataset_weekend_nights, derive_from_dates, tsql_week_number
from hotelsight.paths import DATA


def test_rules_match_every_dataset_row():
    d = pd.read_csv(DATA, usecols=["arrival_date_year", "arrival_date_month", "arrival_date_day_of_month",
                                   "arrival_date_week_number", "stays_in_weekend_nights", "stays_in_week_nights"])
    arr = pd.to_datetime(d.arrival_date_year.astype(str) + "-" + d.arrival_date_month + "-" +
                         d.arrival_date_day_of_month.astype(str), format="%Y-%B-%d").dt.date
    nights = d.stays_in_weekend_nights + d.stays_in_week_nights
    weeks = [tsql_week_number(a) for a in arr]
    assert (pd.Series(weeks) == d.arrival_date_week_number).all()
    combos = pd.DataFrame({"a": arr, "n": nights}).drop_duplicates()
    lookup = {(a, n): dataset_weekend_nights(a, a + pd.Timedelta(days=int(n)).to_pytimedelta()) for a, n in combos.itertuples(index=False)}
    got = [lookup[(a, n)] for a, n in zip(arr, nights)]
    assert all(w == we and k == wk for (w, k), we, wk in zip(got, d.stays_in_weekend_nights, d.stays_in_week_nights))


def test_known_cases():
    assert tsql_week_number(date(2015, 7, 5)) == 28          # dataset value (ISO would give 27)
    assert tsql_week_number(date(2017, 1, 1)) == 1           # Sunday 1 Jan starts week 1
    assert tsql_week_number(date(2016, 1, 3)) == 2           # first Sunday of 2016 starts week 2
    assert dataset_weekend_nights(date(2016, 7, 2), date(2016, 7, 3)) == (0, 1)   # Saturday night -> 'week'
    assert dataset_weekend_nights(date(2016, 7, 3), date(2016, 7, 4)) == (1, 0)   # Sunday night -> 'weekend'
    assert dataset_weekend_nights(date(2016, 7, 4), date(2016, 7, 5)) == (1, 0)   # Monday night -> 'weekend'
    r = derive_from_dates(date(2026, 1, 1), date(2026, 3, 10), date(2026, 3, 17))
    assert r["lead_time"] == 68 and r["arrival_date_month"] == "March" and r["arrival_date_day_of_month"] == 10
    assert r["stays_in_weekend_nights"] + r["stays_in_week_nights"] == 7

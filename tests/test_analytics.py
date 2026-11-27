"""Hotel performance analytics on the historical bookings: totals add up, rules match the experiment's cleaning."""
import numpy as np
import pandas as pd
import pytest

from hotelsight import analytics


@pytest.fixture(scope="module")
def d():
    return analytics.load_bookings()


def test_cleaning_matches_the_experiment(d):
    assert len(d) == 119_208                                     # 119,390 minus the 182 invalid records
    assert d["cancelled"].mean() == pytest.approx(0.3708, abs=1e-4)


def test_summary_parts_add_up(d):
    s = analytics.summary(d)
    stayed = d[d["stayed"] == 1]
    assert s["room_revenue"] == pytest.approx((stayed["adr"] * stayed["nights"]).sum())
    assert s["adr"] == pytest.approx(s["room_revenue"] / s["rooms_sold"])
    assert s["revenue_lost"] == pytest.approx(d.loc[d["cancelled"] == 1, "value"].sum())
    # every booking is either stayed, cancelled or a no-show (no-shows are inside 'cancelled')
    assert int(d["stayed"].sum()) + s["cancelled"] == s["bookings"]


def test_monthly_totals_and_growth(d):
    m = analytics.monthly(d)
    assert m["room_revenue"].sum() == pytest.approx(analytics.summary(d)["room_revenue"])
    aug17 = m[m["month"] == pd.Timestamp("2017-08-01")].iloc[0]
    aug16 = m[m["month"] == pd.Timestamp("2016-08-01")].iloc[0]
    assert aug17["revenue_growth"] == pytest.approx(aug17["room_revenue"] / aug16["room_revenue"] - 1)
    assert np.isnan(m.iloc[0]["revenue_growth"])                  # July 2015 has no month a year earlier


def test_cancel_timing_and_pickup(d):
    t = analytics.cancel_timing(d)
    assert t["cancellations"].sum() == (d["status"] == "Canceled").sum()
    assert t["share"].sum() == pytest.approx(1)
    p = analytics.pickup(d)
    assert p["made"].min() >= d["month"].min() and p["made"].max() <= d["month"].max()


def test_rooms_sold_by_night_matches_room_nights(d):
    n = analytics.rooms_sold_by_night(d)
    assert n["rooms"].sum() == d.loc[d["stayed"] == 1, "nights"].sum()


def test_findings_come_from_the_data(d):
    f = {x["title"]: x for x in analytics.findings(d)}
    assert "Early bookings cancel the most" in f and "%" in f["Early bookings cancel the most"]["finding"]
    assert all(x["action"] for x in f.values())


def test_model_value_matches_the_sealed_test_results(d):
    v = analytics.model_value(d, one_copy=True)                 # the one-copy confusion matrix in Chapter 4
    assert (v["bookings"], v["cancellations"], v["caught_bookings"], v["false_alarms"]) == (13_085, 3_602, 2_578, 1_191)
    assert v["revenue_caught"] + v["missed_revenue"] == pytest.approx(v["revenue_lost"])
    every = analytics.model_value(d, one_copy=False)
    assert (every["bookings"], every["caught_bookings"], every["false_alarms"]) == (17_882, 5_459, 1_386)

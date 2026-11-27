"""Hotel insights: weekly figures, breakdowns, bands, reasons and the outcome check, on a small known history."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from hotelsight import history, insights

APP = Path(__file__).resolve().parents[1] / "app"


def _entry(ref, arrival, prob, lead, deposit="No Deposit", segment="Online TA", rate=100.0, nights=2,
           reason="Deposit type: No deposit", actual=None):
    rec = {"adr": rate, "stays_in_weekend_nights": 0, "stays_in_week_nights": nights, "lead_time": lead,
           "deposit_type": deposit, "market_segment": segment, "customer_type": "Transient", "country": "PRT",
           "hotel": "City Hotel"}
    return {"reference": ref, "hotel": "City Hotel", "arrival": arrival, "probability": prob,
            "top_raising": reason if prob >= 0.5 else "", "record": rec, "actual": actual}


@pytest.fixture()
def saved(tmp_path, monkeypatch):
    monkeypatch.setenv("HOTELSIGHT_HISTORY_DB", str(tmp_path / "h.db"))
    history.save([
        # week of Mon 3 Jul 2017
        _entry("A", "2017-07-03", 0.90, 200, reason="Lead time (days before arrival): 200", actual=1),
        _entry("B", "2017-07-05", 0.80, 5, actual=1),
        _entry("C", "2017-07-09", 0.20, 3, actual=0),
        # week of Mon 10 Jul 2017
        _entry("D", "2017-07-10", 0.60, 40, deposit="Non Refund", reason="Deposit type: Non-refundable (≥ full stay)",
               actual=0),
        _entry("E", "2017-07-12", 0.10, 400, segment="Direct", actual=0),
    ], "batch", "Test upload", 0.5, "abc")
    return insights.prepare(history.with_stay(history.load()))


def test_weekly_figures(saved):
    w = insights.weekly(saved)
    assert [str(x.date()) for x in w["week"]] == ["2017-07-03", "2017-07-10"]
    assert list(w["bookings"]) == [3, 2] and list(w["flagged"]) == [2, 1]
    assert list(w["expected_arrivals"]) == [1, 1]
    assert list(w["revenue_flagged"]) == [400.0, 200.0]          # 2 flagged × €100 × 2 nights, then 1 flagged
    assert list(w["actual_cancelled"]) == [2, 0]


def test_summary_and_bands(saved):
    s = insights.summary(saved)
    assert s["bookings"] == 5 and s["flagged"] == 3 and s["expected_arrivals"] == 2
    assert s["revenue"] == 1000.0 and s["revenue_flagged"] == 600.0 and s["nights_flagged"] == 6
    b = insights.bands(saved).set_index("band")["bookings"]
    assert list(b) == [2, 0, 1, 2]                               # 0.10/0.20 low, 0.60 high, 0.80/0.90 very high


def test_breakdown_keeps_lead_time_order(saved):
    b = insights.breakdown(saved, "lead_band")
    assert list(b["group"]) == ["0–7 days", "31–90 days", "181–365 days", "Over 365 days"]
    assert dict(zip(b["group"], b["share_flagged"]))["0–7 days"] == 0.5
    dep = insights.breakdown(saved, "deposit_type").set_index("group")
    assert dep.loc["Non-refundable (≥ full stay)", "share_flagged"] == 1.0


def test_reasons_and_actions(saved):
    r = insights.reasons(saved).set_index("reason")
    assert r.loc["Deposit type: No deposit", "bookings"] == 1
    assert "deposit" in r.loc["Deposit type: No deposit", "action"].lower()
    assert "agent or group organiser" in r.loc["Deposit type: Non-refundable (≥ full stay)", "action"]
    assert "30 days" in r.loc["Lead time (days before arrival)", "action"]
    assert abs(r["share_of_flagged"].sum() - 1) < 1e-9


def test_outcome_check(saved):
    c = insights.outcome_check(saved)
    assert (c["tp"], c["fp"], c["fn"], c["tn"]) == (2, 1, 0, 2)
    assert c["accuracy"] == 0.8 and abs(c["f1"] - 0.8) < 1e-9


def test_contact_list_and_attention_window(saved):
    c = insights.contact_list(saved)
    assert list(c["reference"]) == ["A", "B", "D"]               # soonest arrival first
    assert c["action"].str.len().gt(0).all()
    w, label = insights.attention_window(saved, today="2030-01-01")
    assert "Latest 4 weeks" in label and len(w) == 5
    w, label = insights.attention_window(saved, today="2017-07-10")
    assert label == "Arrivals from today" and set(w["reference"]) == {"D", "E"}


def test_action_list_page(saved):
    at = AppTest.from_file(str(APP / "views/actions.py"), default_timeout=180).run()
    assert not at.exception, [e.value for e in at.exception]
    assert next(m for m in at.metric if m.label == "Reservations to contact").value == "3"
    assert list(at.dataframe[0].value["Reference"]) == ["A", "B", "D"]
    at.selectbox(key="act_reason").select("Lead time (days before arrival)").run()
    assert list(at.dataframe[0].value["Reference"]) == ["A"]


def test_expected_performance_from_saved_predictions(saved):
    from hotelsight import analytics
    d = analytics.from_predictions(history.with_stay(history.load()))
    s = analytics.summary(d)
    assert s["bookings"] == 5 and s["cancelled"] == 3               # A, B, D flagged as likely to cancel
    assert s["room_revenue"] == 400.0                               # C and E expected to stay: €100 × 2 nights each
    assert s["revenue_lost"] == 600.0 and s["rooms_sold"] == 4
    assert analytics.findings(d, expected=True) is not None


def test_performance_page_switches_to_expected_results(saved):
    at = AppTest.from_file(str(APP / "views/performance_hotel.py"), default_timeout=180)
    at.session_state["perf_mode"] = "Current predicted bookings"
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert next(m for m in at.metric if m.label == "Expected room revenue").value == "€400.00"
    assert next(m for m in at.metric if m.label == "Likely to cancel").value == "60.0%"
    assert next(m for m in at.metric if m.label == "Predicted reservations").value == "5"
    actual = AppTest.from_file(str(APP / "views/performance_hotel.py"), default_timeout=180).run()
    assert next(m for m in actual.metric if m.label == "Bookings").value == "119,208"   # history is unchanged


def test_insights_page_with_a_selected_week(saved):
    at = AppTest.from_file(str(APP / "views/insights.py"), default_timeout=180).run()
    assert not at.exception, [e.value for e in at.exception]
    assert next(m for m in at.metric if m.label == "Arrivals").value == "5"
    week_box = next(s for s in at.selectbox if s.label == "Arrival week")
    week_box.select_index(2).run()                               # week of 10 Jul 2017
    assert not at.exception, [e.value for e in at.exception]
    arrivals = next(m for m in at.metric if m.label == "Arrivals")
    assert arrivals.value == "2" and "-1 vs previous week" in arrivals.delta
    assert next(m for m in at.metric if m.label == "Likely to cancel").value == "50.0%"


def test_overview_switches_to_current_predicted_bookings(saved):
    at = AppTest.from_file(str(APP / "views/home.py"), default_timeout=180)
    at.session_state["ov_mode"] = "Current predicted bookings"
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert next(m for m in at.metric if m.label == "Expected room revenue").value == "€400.00"
    likely = [m for m in at.metric if m.label == "Likely to cancel"]
    assert likely[0].value == "60.0%" and "vs booking history" in likely[0].delta
    history_view = AppTest.from_file(str(APP / "views/home.py"), default_timeout=180).run()
    assert any(m.label == "Room revenue" for m in history_view.metric)     # the default stays the dataset


def test_delete_selected_predictions(saved):
    ids = history.load().sort_values("reference")["id"].head(2).tolist()     # A and B
    assert history.delete_ids(ids) == 2
    left = history.load()
    assert len(left) == 3 and set(left["reference"]) == {"C", "D", "E"}
    assert history.delete_ids([]) == 0

"""Run every dashboard page headlessly with Streamlit's AppTest (charts are built, not drawn)."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app"
PAGES = ["streamlit_app.py", "views/home.py", "views/predict.py", "views/batch.py", "views/insights.py",
         "views/actions.py", "views/performance_hotel.py", "views/history.py", "views/evaluation.py",
         "views/explanations.py", "views/method.py"]


@pytest.mark.parametrize("page", PAGES)
def test_page_runs_without_errors(page):
    at = AppTest.from_file(str(APP / page), default_timeout=180).run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


def test_predict_default_booking():
    at = AppTest.from_file(str(APP / "views/predict.py"), default_timeout=180).run()
    at.button[0].click().run()
    assert not at.exception
    assert at.metric[0].label == "Cancellation probability" and at.metric[0].value.endswith("%")


def test_predict_example_matches_saved_prediction():
    at = AppTest.from_file(str(APP / "views/predict.py"), default_timeout=180).run()
    at.selectbox(key="example_choice").select("High risk – non-refundable, long lead time").run()
    at.button[0].click().run()
    assert not at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "Saved test prediction" in text and "✓" in text


def test_batch_sample_is_scored_and_saved_to_history():
    from hotelsight import history
    before = len(history.load())
    at = AppTest.from_file(str(APP / "views/batch.py"), default_timeout=300).run()
    at.button(key="use_sample").click().run()
    at.button(key="run_batch").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.metric[4].label == "Reservations scored" and at.metric[4].value == "200"
    assert len(history.load()) == before + 200
    hist = AppTest.from_file(str(APP / "views/history.py"), default_timeout=180).run()
    assert not hist.exception and not hist.error
    assert hist.metric[0].label == "With a known outcome"         # earlier tests may add saved outcomes too
    assert int(hist.metric[0].value.replace(",", "")) >= 200


def _saved(reference, arrival, probability=0.9):
    return {"reference": reference, "hotel": "City Hotel", "arrival": arrival, "probability": probability,
            "top_raising": "Deposit type: No deposit", "record": {"adr": 100.0, "stays_in_week_nights": 2}}


def test_overview_falls_back_to_the_latest_weeks_when_none_are_upcoming(tmp_path, monkeypatch):
    from hotelsight import history
    monkeypatch.setenv("HOTELSIGHT_HISTORY_DB", str(tmp_path / "h.db"))
    history.save([_saved("PAST-1", "2017-03-01")], "single", "Single prediction", 0.5, "abc")
    at = AppTest.from_file(str(APP / "views/home.py"), default_timeout=180).run()
    assert not at.exception, [e.value for e in at.exception]
    assert next(m for m in at.metric if m.label == "Guests to contact").value == "1"
    assert any("No upcoming arrivals" in c.value for c in at.caption)
    assert list(at.dataframe[0].value["Reference"]) == ["PAST-1"]


def test_overview_puts_the_soonest_upcoming_arrival_first(tmp_path, monkeypatch):
    from hotelsight import history
    monkeypatch.setenv("HOTELSIGHT_HISTORY_DB", str(tmp_path / "h.db"))
    history.save([_saved("LATER", "2099-06-01"), _saved("SOON", "2099-01-01"), _saved("PAST", "2017-03-01")],
                 "batch", "Upload", 0.5, "abc")
    at = AppTest.from_file(str(APP / "views/home.py"), default_timeout=180).run()
    assert not at.exception
    assert next(m for m in at.metric if m.label == "Guests to contact").value == "2"      # the two upcoming ones
    assert list(at.dataframe[0].value["Reference"]) == ["SOON", "LATER"]


def test_overview_shows_the_latest_month_of_hotel_performance():
    at = AppTest.from_file(str(APP / "views/home.py"), default_timeout=180).run()
    assert not at.exception
    assert next(m for m in at.metric if m.label == "Room revenue").delta.endswith("vs last year")


def test_predict_what_if_changes_the_score():
    at = AppTest.from_file(str(APP / "views/predict.py"), default_timeout=180).run()
    at.selectbox(key="example_choice").select("High risk – non-refundable, long lead time").run()
    at.button[0].click().run()
    before = at.metric[0].value
    at.selectbox(key="wi_field").select("deposit_type").run()
    at.selectbox(key="wi_val_deposit_type").select("No Deposit").run()
    assert not at.exception, [e.value for e in at.exception]
    after = next(m for m in at.metric if m.label == "Score after the change")
    assert after.value != before and "pts vs now" in after.delta


def test_predict_rejects_invalid_booking():
    at = AppTest.from_file(str(APP / "views/predict.py"), default_timeout=180).run()
    at.number_input(key="k_adults").set_value(0)          # form widgets submit together with the button
    at.button[0].click().run()
    assert any("at least one guest" in e.value for e in at.error)
    assert not at.metric                                   # no prediction shown


def test_demo_upcoming_arrivals_fill_the_overview(tmp_path, monkeypatch):
    monkeypatch.setenv("HOTELSIGHT_HISTORY_DB", str(tmp_path / "h.db"))
    at = AppTest.from_file(str(APP / "views/batch.py"), default_timeout=300).run()
    at.button(key="use_demo").click().run()
    at.button(key="run_batch").click().run()
    assert not at.exception, [e.value for e in at.exception]
    home = AppTest.from_file(str(APP / "views/home.py"), default_timeout=180).run()
    assert not home.exception
    assert any(c.value.startswith("Arrivals from today") for c in home.caption)
    assert next(m for m in home.metric if m.label == "Revenue flagged").delta.endswith("upcoming arrivals")
    assert any("200 saved predictions" in h.value for h in home.get("html") if hasattr(h, "value"))

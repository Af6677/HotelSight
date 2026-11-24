"""Batch upload (spreadsheet -> contract -> predictions) and the saved prediction history."""
import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from hotelsight import batch, history
from hotelsight.contract import country_names, derive_from_dates, input_ranges
from hotelsight.paths import ARM, ASSETS


def run(model, raw):
    parsed = batch.prepare(raw, model.categories, input_ranges(), country_names())
    return parsed, batch.predict(model, parsed, country_names())


def test_dataset_layout_reproduces_saved_predictions(model, paired):
    _, res = run(model, batch.sample_test_bookings(300))
    saved = paired.set_index("source_csv_row")[ARM]
    rows = res["Reference"].str.replace("CSV-", "").astype(int)
    assert res["Problems"].eq("").all()
    assert np.max(np.abs(res["Cancellation probability"].to_numpy() - saved.loc[rows].to_numpy())) < 1e-6
    assert set(res["Actual outcome"]) <= {"Cancelled", "Not cancelled"}
    cancelled = res["Actual outcome"].eq("Cancelled")
    flagged = res["Prediction"].eq("Likely to cancel")
    assert res["Correct?"].eq("Correct").eq(cancelled == flagged).all()


def test_template_round_trip_reproduces_examples(model):
    examples = json.loads((ASSETS / "example_bookings.json").read_text())
    raw = batch.read_table(batch.template(examples, model.categories), "template.xlsx")
    parsed, res = run(model, raw)
    assert parsed.layout == "template" and len(parsed.valid) == len(examples)
    for ex, p in zip(examples, res["Cancellation probability"]):
        assert abs(p - ex["saved_probability"]) < 1e-6, ex["title"]


def _row(**change):
    row = {"Booking reference": "A1", "Booking date": "01/01/2026", "Arrival date": "2026-03-10",
           "Departure date": "2026-03-13", "Hotel": "city hotel", "Meal package": "Bed & Breakfast (BB)",
           "Customer type": "Transient", "Booking source (market segment)": "Online travel agent",
           "Booked through a travel agent": "Yes", "Company booking": "no", "Days on waiting list": 0,
           "Adults": 2, "Children": "", "Babies": 0, "Guest country": "Portugal", "Returning guest": "No",
           "Previous cancellations": 0, "Previous completed bookings": 0, "Average daily rate (EUR)": "95",
           "Deposit type": "No Deposit", "Car parking spaces": 0, "Special requests": 1}
    return {**row, **change}


def test_labels_names_and_day_first_dates_match_the_form(model):
    _, res = run(model, pd.DataFrame([_row()]))
    rec = derive_from_dates(date(2026, 1, 1), date(2026, 3, 10), date(2026, 3, 13))
    rec.update({"hotel": "City Hotel", "meal": "BB", "customer_type": "Transient", "market_segment": "Online TA",
                "has_travel_agent": 1.0, "company_booking": 0.0, "days_in_waiting_list": 0, "adults": 2,
                "children": 0, "babies": 0, "country": "PRT", "is_repeated_guest": 0.0, "previous_cancellations": 0,
                "previous_bookings_not_canceled": 0, "adr": 95.0, "deposit_type": "No Deposit",
                "required_car_parking_spaces": 0, "total_of_special_requests": 1})
    assert res["Problems"][0] == ""
    assert abs(res["Cancellation probability"][0] - model.predict_proba(model.frame(rec))[0]) < 1e-9


def test_bad_rows_are_reported_not_scored(model):
    raw = pd.DataFrame([_row(), _row(Adults=0), _row(**{"Departure date": "2026-03-01"}),
                        _row(**{"Company booking": "maybe"}), _row(**{"Arrival date": "soon"})])
    parsed, res = run(model, raw)
    assert len(parsed.valid) == 1
    assert res["Prediction"].tolist()[1:] == ["Not scored"] * 4
    problems = " ".join(res["Problems"])
    for text in ["at least one guest", "departure date cannot be before", "Yes or No", "not a date"]:
        assert text in problems


def test_missing_columns_are_explained(model):
    with pytest.raises(ValueError, match="stay dates"):
        run(model, pd.DataFrame([{"hotel": "City Hotel"}]))
    with pytest.raises(ValueError, match="Missing columns: adr"):
        run(model, pd.DataFrame([_row()]).drop(columns=["Average daily rate (EUR)"]))


def test_fast_contributions_equal_shap_explainer(model, test_rows):
    rows = test_rows.iloc[:300]
    prob, fast = model.fast_contributions(rows)
    ex = model.explain(rows)
    assert np.max(np.abs(fast.to_numpy() - ex["contributions"].to_numpy())) < 1e-4
    assert np.max(np.abs(prob - ex["probability"])) < 1e-9


def test_history_save_load_delete(tmp_path, monkeypatch):
    monkeypatch.setenv("HOTELSIGHT_HISTORY_DB", str(tmp_path / "h.db"))
    assert history.load().empty
    entry = {"reference": "R1", "hotel": "City Hotel", "arrival": "2026-03-10", "probability": 0.7,
             "record": {"adr": 95.0}, "actual": None}
    a = history.save([entry, {**entry, "reference": "R2", "probability": 0.2}], "batch", "Upload", 0.5, "abc")
    history.save([entry], "single", "Single prediction", 0.5, "abc")
    h = history.load()
    assert len(h) == 3 and h["predicted_cancel"].tolist() == [1, 0, 1]     # newest first
    assert json.loads(h["inputs_json"][0]) == {"adr": 95.0}
    assert history.delete_batch(a) == 2 and len(history.load()) == 1
    assert history.clear() == 1 and history.load().empty

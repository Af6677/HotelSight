"""The dashboard pipeline reproduces the experiment's saved predictions exactly."""
import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from hotelsight.contract import DATE_DERIVED, MODEL_FIELDS, derive_from_dates
from hotelsight.paths import ARM, ARM_MODELS, ASSETS


def test_all_test_rows_reproduce_saved_probabilities(model, test_rows, paired):
    p = model.predict_proba(test_rows)
    assert len(p) == 17882
    assert np.max(np.abs(p - paired[ARM].to_numpy())) < 1e-6


def test_all_validation_rows_reproduce_saved_probabilities(model, cleaned):
    saved = pd.read_csv(ARM_MODELS / "xgboost/validation_predictions.csv", float_precision="round_trip")
    rows = cleaned.set_index("source_csv_row").loc[saved.source_csv_row].reset_index()
    assert np.max(np.abs(model.predict_proba(rows) - saved.probability.to_numpy())) < 1e-6


def _form_record(row):
    """Rebuild what a manager would type: three dates + the other inputs (agent/company as Yes/No)."""
    arrival = datetime.strptime(f"{row.arrival_date_year}-{row.arrival_date_month}-{row.arrival_date_day_of_month}",
                                "%Y-%B-%d").date()
    booking = arrival - timedelta(days=int(row.lead_time))
    departure = arrival + timedelta(days=int(row.stays_in_weekend_nights + row.stays_in_week_nights))
    rec = derive_from_dates(booking, arrival, departure)
    for f in MODEL_FIELDS:
        if f in DATE_DERIVED:
            continue
        if f == "has_travel_agent":
            rec[f] = 0.0 if pd.isna(row.agent) else 1.0
        elif f == "company_booking":
            rec[f] = 0.0 if pd.isna(row.company) else 1.0
        else:
            rec[f] = row[f]
    return rec


def test_form_path_reproduces_saved_probabilities(model, test_rows, paired):
    idx = np.random.default_rng(0).choice(len(test_rows), 400, replace=False)
    recs = [_form_record(test_rows.iloc[i]) for i in idx]
    p = model.predict_proba(model.frame(recs))
    assert np.max(np.abs(p - paired[ARM].to_numpy()[idx])) < 1e-6


def test_dashboard_examples_reproduce_saved_probabilities(model):
    for ex in json.loads((ASSETS / "example_bookings.json").read_text()):
        b, a, d = (datetime.fromisoformat(ex[k]).date() for k in ("booking_date", "arrival_date", "departure_date"))
        rec = {**derive_from_dates(b, a, d), **ex["inputs"]}
        p = model.predict_proba(model.frame(rec))[0]
        assert abs(p - ex["saved_probability"]) < 1e-6, ex["title"]

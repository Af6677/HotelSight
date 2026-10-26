"""Input validation, missing values and unseen categories."""
from datetime import date

import numpy as np

from hotelsight.contract import derive_from_dates, input_ranges, validate_dates, validate_record


def base_record():
    rec = derive_from_dates(date(2026, 1, 1), date(2026, 3, 10), date(2026, 3, 13))
    rec.update({"hotel": "City Hotel", "meal": "BB", "customer_type": "Transient", "market_segment": "Online TA",
                "has_travel_agent": 1.0, "company_booking": 0.0, "days_in_waiting_list": 0, "adults": 2,
                "children": 0, "babies": 0, "country": "PRT", "is_repeated_guest": 0.0, "previous_cancellations": 0,
                "previous_bookings_not_canceled": 0, "adr": 95.0, "deposit_type": "No Deposit",
                "required_car_parking_spaces": 0, "total_of_special_requests": 1})
    return rec


def test_valid_record_passes(model):
    v = validate_record(base_record(), input_ranges(), model.categories)
    assert v.ok and v.warnings == []


def test_date_errors():
    assert not validate_dates(date(2026, 3, 1), date(2026, 2, 1), date(2026, 2, 3)).ok    # booked after arrival
    assert not validate_dates(date(2026, 1, 1), date(2026, 2, 5), date(2026, 2, 3)).ok    # leaves before arriving
    v = validate_dates(date(2026, 1, 1), date(2026, 2, 5), date(2026, 2, 5))
    assert v.ok and v.warnings                                                             # zero-night: allowed, warned
    assert not validate_dates(None, date(2026, 2, 5), date(2026, 2, 6)).ok


def test_record_errors(model):
    for change in [{"adults": 0, "children": 0, "babies": 0}, {"adr": -1.0}, {"adults": 2.5},
                   {"previous_cancellations": -1}, {"has_travel_agent": 3}]:
        assert not validate_record({**base_record(), **change}, input_ranges(), model.categories).ok, change


def test_out_of_range_and_unseen_are_warnings(model):
    v = validate_record({**base_record(), "lead_time": 900, "country": "XYZ"}, input_ranges(), model.categories)
    assert v.ok and len(v.warnings) == 2


def test_unseen_categories_are_ignored_by_saved_encoder(model):
    rec = base_record()
    X = model.encode(model.frame({**rec, "country": "XYZ", "meal": "Brunch"}))
    names = np.array(model.encoded_names)
    assert X[0, np.char.startswith(names, "categorical__country_")].sum() == 0
    assert X[0, np.char.startswith(names, "categorical__meal_")].sum() == 0
    assert np.isfinite(model.predict_proba(model.frame({**rec, "country": "XYZ"}))).all()


def test_missing_values_follow_training_rules(model):
    rec = base_record()
    p = lambda r: float(model.predict_proba(model.frame(r))[0])
    assert p({**rec, "children": None}) == p({**rec, "children": 0})          # children -> 0 (experiment rule)
    median_adr = model.pre.named_transformers_["numeric"].named_steps["impute"].statistics_[
        list(model.pre.transformers_[0][2]).index("adr")]
    assert p({**rec, "adr": None}) == p({**rec, "adr": float(median_adr)})    # numeric -> training median
    assert np.isfinite(p({**rec, "meal": None, "country": None}))             # categorical -> 'Unknown'

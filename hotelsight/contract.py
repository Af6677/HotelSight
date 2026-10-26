"""The 24-input contract of the deployed model, friendly labels, date derivation and validation.

Field names, order and categories come from the saved experiment
(features.py and the fitted preprocessor). Nothing here changes encoding,
imputation or scaling: the saved preprocessor does all of that.

Dataset calendar conventions (verified on 100% of the 119,390 rows,
see scripts/check_date_conventions.py and docs/date_conventions.json):
  * arrival_date_week_number = SQL Server DATEPART(week, date): weeks start on
    Sunday and week 1 is the week containing 1 January (NOT ISO-8601).
  * stays_in_weekend_nights counts nights that BEGIN on a Sunday or a Monday;
    stays_in_week_nights counts all other nights. This is the dataset's recorded
    convention (a normal Saturday/Sunday rule matches only 50% of rows), so the
    dashboard reproduces it exactly instead of "correcting" it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, timedelta

from .paths import ASSETS

# ---------------------------------------------------------------- field contract
MODEL_FIELDS = [   # exact order of features.FEATURE_SETS["historical24"]
    "hotel", "meal", "customer_type", "market_segment", "has_travel_agent",
    "company_booking", "days_in_waiting_list", "lead_time", "arrival_date_month",
    "arrival_date_week_number", "arrival_date_day_of_month",
    "stays_in_weekend_nights", "stays_in_week_nights", "adults", "children",
    "babies", "country", "is_repeated_guest", "previous_cancellations",
    "previous_bookings_not_canceled", "adr", "deposit_type",
    "required_car_parking_spaces", "total_of_special_requests",
]
DATE_DERIVED = ["lead_time", "arrival_date_month", "arrival_date_week_number",
                "arrival_date_day_of_month", "stays_in_weekend_nights", "stays_in_week_nights"]

FRIENDLY_NAME = {
    "hotel": "Hotel", "meal": "Meal package", "customer_type": "Customer type",
    "market_segment": "Booking source (market segment)", "has_travel_agent": "Booked through a travel agent",
    "company_booking": "Company booking", "days_in_waiting_list": "Days on waiting list",
    "lead_time": "Lead time (days before arrival)", "arrival_date_month": "Arrival month",
    "arrival_date_week_number": "Arrival week number", "arrival_date_day_of_month": "Arrival day of month",
    "stays_in_weekend_nights": "Nights recorded as 'weekend' (Sun/Mon)",
    "stays_in_week_nights": "Other nights", "adults": "Adults", "children": "Children",
    "babies": "Babies", "country": "Guest country", "is_repeated_guest": "Returning guest",
    "previous_cancellations": "Previous cancellations", "previous_bookings_not_canceled": "Previous completed bookings",
    "adr": "Average daily rate (EUR)", "deposit_type": "Deposit type",
    "required_car_parking_spaces": "Car parking spaces", "total_of_special_requests": "Special requests",
}

CATEGORY_LABELS = {
    "hotel": {"City Hotel": "City Hotel", "Resort Hotel": "Resort Hotel"},
    "meal": {"BB": "Bed & Breakfast (BB)", "HB": "Half board (HB)",
             "FB": "Full board (FB)", "SC": "No meal package (SC)",
             "Undefined": "Not specified (recorded as 'Undefined')"},
    "customer_type": {"Transient": "Individual (Transient)",
                      "Transient-Party": "Linked to another booking (Transient-Party)",
                      "Contract": "Contract / allotment", "Group": "Group booking"},
    "market_segment": {"Online TA": "Online travel agent", "Offline TA/TO": "Offline agent / tour operator",
                       "Direct": "Direct with the hotel", "Groups": "Groups", "Corporate": "Corporate",
                       "Complementary": "Complimentary stay", "Aviation": "Aviation (airline crew)",
                       "Undefined": "Not specified (recorded as 'Undefined')"},
    "deposit_type": {"No Deposit": "No deposit", "Non Refund": "Non-refundable (≥ full stay)",
                     "Refundable": "Refundable (< full stay)"},
}
MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]


YES_NO_FIELDS = ["has_travel_agent", "company_booking", "is_repeated_guest"]


def display_value(field_name: str, value, countries: dict | None = None) -> str:
    """A model input value as a manager would read it (labels, Yes/No, euros)."""
    if value is None or value != value:                    # None or NaN
        return "not given"
    if field_name in YES_NO_FIELDS:
        return "Yes" if float(value) == 1 else "No"
    if field_name == "country":
        return (countries or {}).get(value, value)
    if field_name == "adr":
        return f"€{float(value):,.2f}"
    if field_name in CATEGORY_LABELS:
        return CATEGORY_LABELS[field_name].get(value, value)
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def country_names() -> dict:
    """Training-set country codes -> display names (built by scripts/build_assets.py)."""
    return json.loads((ASSETS / "country_names.json").read_text(encoding="utf-8"))


def input_ranges() -> dict:
    """Min/max of each numeric raw input in the retained TRAINING split."""
    return json.loads((ASSETS / "input_ranges.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------- date derivation
def tsql_week_number(d: date) -> int:
    """SQL Server DATEPART(week, d) with DATEFIRST 7 (Sunday-first weeks, week 1 contains 1 Jan)."""
    jan1 = date(d.year, 1, 1)
    sunday_based_offset = (jan1.weekday() + 1) % 7          # Sunday=0 ... Saturday=6
    return (d.timetuple().tm_yday - 1 + sunday_based_offset) // 7 + 1


def dataset_weekend_nights(arrival: date, departure: date) -> tuple[int, int]:
    """(weekend_nights, week_nights) using the dataset's convention: nights beginning Sun or Mon."""
    nights = (departure - arrival).days
    weekend = sum(1 for k in range(nights) if (arrival + timedelta(days=k)).weekday() in (6, 0))
    return weekend, nights - weekend


def derive_from_dates(booking_date: date, arrival_date: date, departure_date: date) -> dict:
    weekend, week = dataset_weekend_nights(arrival_date, departure_date)
    return {
        "lead_time": (arrival_date - booking_date).days,
        "arrival_date_month": MONTHS[arrival_date.month - 1],
        "arrival_date_week_number": tsql_week_number(arrival_date),
        "arrival_date_day_of_month": arrival_date.day,
        "stays_in_weekend_nights": weekend,
        "stays_in_week_nights": week,
    }


# ---------------------------------------------------------------- validation
@dataclass
class Validation:
    errors: list = field(default_factory=list)     # block prediction
    warnings: list = field(default_factory=list)   # allowed, but shown to the user

    @property
    def ok(self) -> bool:
        return not self.errors


COUNT_FIELDS = ["days_in_waiting_list", "adults", "children", "babies", "previous_cancellations",
                "previous_bookings_not_canceled", "required_car_parking_spaces", "total_of_special_requests"]


def validate_dates(booking_date, arrival_date, departure_date) -> Validation:
    v = Validation()
    if None in (booking_date, arrival_date, departure_date):
        v.errors.append("Please enter the booking date, arrival date and departure date.")
        return v
    if booking_date > arrival_date:
        v.errors.append("The booking date cannot be after the arrival date.")
    if departure_date < arrival_date:
        v.errors.append("The departure date cannot be before the arrival date.")
    if departure_date == arrival_date:
        v.warnings.append("Zero-night stay (departure on the arrival date). These exist in the training data (645 records) but are rare.")
    return v


def validate_record(rec: dict, ranges: dict | None = None, categories: dict | None = None) -> Validation:
    """Checks consistent with the training contract.

    Errors  = rules the training data was cleaned with, or impossible values.
    Warnings = values outside the range seen in the training split, or categories the
               saved encoder has never seen (it ignores them, so they add no information).
    """
    v = Validation()
    missing = [f for f in MODEL_FIELDS if f not in rec]
    if missing:
        v.errors.append("Missing inputs: " + ", ".join(missing)); return v
    for f in COUNT_FIELDS + ["lead_time", "stays_in_weekend_nights", "stays_in_week_nights"]:
        x = rec[f]
        if x is None:
            continue                                   # missing numeric -> saved imputer (children -> 0)
        if float(x) < 0:
            v.errors.append(f"{FRIENDLY_NAME[f]} cannot be negative.")
        elif float(x) != int(float(x)):
            v.errors.append(f"{FRIENDLY_NAME[f]} must be a whole number.")
    guests = sum(float(rec[f] or 0) for f in ["adults", "children", "babies"])
    if guests <= 0:
        v.errors.append("A booking needs at least one guest (zero-guest records were removed before training).")
    if rec["adr"] is not None and float(rec["adr"]) < 0:
        v.errors.append("The average daily rate cannot be negative (negative-ADR records were removed before training).")
    for f in ("has_travel_agent", "company_booking", "is_repeated_guest"):
        if rec[f] not in (0, 1, 0.0, 1.0):
            v.errors.append(f"{FRIENDLY_NAME[f]} must be Yes or No.")
    if ranges:
        for f, r in ranges.items():
            x = rec.get(f)
            if x is not None and not (r["min"] <= float(x) <= r["max"]):
                v.warnings.append(f"{FRIENDLY_NAME[f]} = {x:g} is outside the training range "
                                  f"({r['min']:g}–{r['max']:g}); the prediction is less reliable.")
    if categories:
        for f, known in categories.items():
            if str(rec.get(f)) not in known:
                v.warnings.append(f"{FRIENDLY_NAME[f]} '{rec.get(f)}' was not seen in training; "
                                  "the saved encoder ignores it, so this input contributes no information.")
    return v

"""One feature contract shared by training and raw-input prediction.

The historical CSV is not a verified booking-confirmation snapshot. The
general set is a conservative descriptor comparison, not proof of timing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CORE_FIELDS = [
    "hotel", "meal", "customer_type", "market_segment", "has_travel_agent",
    "company_booking", "days_in_waiting_list", "lead_time", "arrival_date_month",
    "arrival_date_week_number", "arrival_date_day_of_month",
    "stays_in_weekend_nights", "stays_in_week_nights", "adults", "children",
    "babies", "country", "is_repeated_guest", "previous_cancellations",
    "previous_bookings_not_canceled",
]
SNAPSHOT_FIELDS = [
    "adr", "deposit_type", "required_car_parking_spaces", "total_of_special_requests",
]
FEATURE_SETS = {"general20": CORE_FIELDS, "historical24": CORE_FIELDS + SNAPSHOT_FIELDS}
CATEGORICAL = ["hotel", "meal", "customer_type", "market_segment", "arrival_date_month", "country", "deposit_type"]
EXCLUDED = [
    "is_canceled", "reservation_status", "reservation_status_date",
    "assigned_room_type", "booking_changes", "arrival_date_year",
    "reserved_room_type", "distribution_channel", "agent", "company",
]


def present_identifier(series):
    """PMS NULL means not applicable; never test notna after string filling."""
    text = series.astype("string").str.strip().str.lower()
    absent = series.isna() | text.isin(["", "none", "null", "nan", "na"])
    return (~absent).astype(float)


def make_features(frame, feature_set="general20"):
    if feature_set not in FEATURE_SETS:
        raise ValueError("Unknown feature set: " + str(feature_set))
    frame = frame.copy()
    for source, flag in [("agent", "has_travel_agent"), ("company", "company_booking")]:
        if flag not in frame and source in frame:
            frame[flag] = present_identifier(frame[source])
    fields = FEATURE_SETS[feature_set]
    missing = sorted(set(fields) - set(frame))
    if missing:
        raise ValueError("Missing booking inputs: " + ", ".join(missing))
    x = frame[fields].copy()
    for field in fields:
        if field in CATEGORICAL:
            x[field] = x[field].map(lambda v: str(v).strip() if pd.notna(v) and str(v).strip() else "Unknown")
        else:
            x[field] = pd.to_numeric(x[field], errors="coerce").astype(float)
    # This fixed semantic assumption is disclosed, not a data-estimated fill.
    x["children"] = x["children"].fillna(0.)
    return x


def make_preprocessor(x):
    cats = [c for c in x if c in CATEGORICAL]
    nums = [c for c in x if c not in cats]
    return ColumnTransformer([
        ("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), nums),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)),
        ]), cats),
    ], sparse_threshold=0, verbose_feature_names_out=True)

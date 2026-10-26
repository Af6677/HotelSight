"""Page 5 – concise methodology and limitations."""
import sys as _sys, pathlib as _pl  # make app/common.py importable however the page is launched
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))

import streamlit as st

from common import load, page_header, section

plan = load("plan")
c = plan["cleaning"]
s = plan["splits"]["retained"]

page_header("About", "Method & limitations",
            "<b>HotelSight is an experimental prototype evaluated on historical hotel records.</b> "
            "It is a Final Year Project demonstration, not a validated live forecasting system.",
            chips=["Antonio et al. (2019) dataset", "Two Portuguese hotels, 2015–2017", "Seed 42"], icon_name="menu_book")

section("How a prediction is made")
FLOW = [("Booking form", "3 dates + 18 inputs"), ("Validation", "contract.py"),
        ("Feature construction", "experiment/features.py"), ("Saved preprocessor", "SHA-256 checked"),
        ("XGBoost", "frozen winner"), ("Probability + SHAP", "threshold 0.50")]
st.html('<div class="hs-flow">' + '<span class="arrow">→</span>'.join(
    f'<div class="node"><b>{a}</b><small>{b}</small></div>' for a, b in FLOW) + "</div>")

k = st.columns(4)
k[0].metric("Raw bookings", f"{c['raw_rows']:,}", border=True)
k[1].metric("After cleaning", f"{c['retained_rows']:,}", f"{c['removed_union']} removed", delta_color="off",
            delta_arrow="off", border=True)
k[2].metric("Train / validation / test", "70 / 15 / 15",
            f"{s['train']['rows'] // 1000}k · {s['validation']['rows'] // 1000}k · {s['test']['rows'] // 1000}k rows",
            delta_color="off", delta_arrow="off", border=True)
k[3].metric("Model candidates", "48 fits", "12 per arm", help="4 models × 3 configurations × 4 experiment arms",
            delta_color="off", delta_arrow="off", border=True)

tab_method, tab_limits = st.tabs([":material/science: Method", ":material/warning: Limitations"])

with tab_method:
    left, right = st.columns(2, gap="medium")
    with left.container(key="card_m1", height="stretch"):
        st.markdown(f"""
**Data.** Hotel Booking Demand dataset (Antonio et al., 2019): {c['raw_rows']:,} bookings from a city hotel and a
resort hotel in Portugal, arrivals July 2015 – August 2017.

**Cleaning.** {c['zero_guest_rows']} zero-guest records, {c['negative_adr_rows']} negative-ADR record and
{c['extreme_adr_5400_rows']} record with ADR 5,400 were removed, leaving {c['retained_rows']:,} records.
Outcome fields (`reservation_status`, `reservation_status_date`) and fields changed at check-in or later
(`assigned_room_type`, `booking_changes`) are not used. Anonymous agent/company IDs became Yes/No flags, and room type,
distribution channel and arrival year were left out. Missing country → "Unknown", missing children → 0.

**Inputs.** 24 inputs: 20 general booking descriptors, plus ADR, deposit type, parking spaces and special requests.

**Repeated records.** {c['retained_duplicate_copies']:,} records are exact copies of another record. Because the dataset
has no booking ID, it is unknown whether they are separate bookings. The deployed arm keeps them. Identical input
profiles are always kept in the same split, so no test booking has an identical twin in training.
""")
    with right.container(key="card_m2", height="stretch"):
        st.markdown(f"""
**Split.** Grouped, stratified 70 / 15 / 15 (seed 42): {s['train']['rows']:,} training,
{s['validation']['rows']:,} validation and {s['test']['rows']:,} test rows, each about 37% cancelled.
Encoding, scaling and imputation were learned on the training rows only.

**Models and tuning.** Logistic Regression, Random Forest, XGBoost and CatBoost, with class weighting from the
training class counts. Each model tried three fixed configurations chosen in advance; this is a limited manual
comparison, not an exhaustive search. Selection used validation F1. The threshold stayed at 0.50, and there was no
threshold or seed tuning.

**Evaluation.** The selected model was frozen, then scored once on the test set, both with every repeated copy and
with one copy per record. Sensitivity arms remove repeated records and/or the four timing-sensitive inputs.

**Explanations.** TreeSHAP on the deployed model, in log-odds. Local explanations reconstruct the model output; the
global importance is computed on the held-out test set.
""")

LIMITS = [
    ("Input timing is not verified", "The dataset was compiled relative to the day before arrival. Deposit type is "
     "derived from payments before arrival or cancellation, and ADR from lodging transactions. Parking and special "
     "requests can be added later. These values are therefore **not verified as known at booking confirmation**. "
     "Without ADR, deposit, parking and requests, test F1 falls from 81.0% to 73.5% (repeated records)."),
    ("Not a day-before-arrival forecast", "About 94% of cancelled bookings were already cancelled before the day before "
     "arrival, so the evaluation cannot be read as a prospective one-day-ahead forecast."),
    ("Historical classification, not live performance", "The scores measure how well the model separates historical "
     "cancelled and kept records. Performance on new bookings, other years or Malaysian hotels is unknown and would "
     "need new data and retraining."),
    ("Repeated records", "Whether they are genuine separate bookings is an assumption. Both evaluation populations are "
     "reported (81.0% vs 69.9% F1), and neither is \"the true\" score."),
    ("Test-set exposure", "The dataset and earlier test scores were inspected during development, so this is an "
     "exploratory result, not an independent external validation."),
    ("Probabilities are not calibrated", "Class weighting raises the scores. Use them to rank risk, not as exact odds."),
    ("Overfitting gap", "Training F1 (0.95) is well above validation F1 (0.80)."),
    ("SHAP shows associations, not causes", "It explains this model's behaviour. It does not show that changing an "
     "input would prevent a cancellation."),
    ("Quirks reproduced on purpose", "The dataset's week number follows SQL Server's Sunday-first convention, and its "
     "\"weekend nights\" are nights starting on Sunday or Monday. The dashboard copies these conventions so its inputs "
     "match the training data."),
]

with tab_limits:
    cols = st.columns(3, gap="small")
    for i, (title, body) in enumerate(LIMITS):
        with cols[i % 3].container(key=f"card_lim{i}"):
            st.markdown(f"**{title}**")
            st.caption(body)

st.caption("Source of the dataset definitions: Antonio, de Almeida & Nunes (2019), Data in Brief 22, 41–49.")

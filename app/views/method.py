"""Page 5 – concise methodology and limitations."""
import sys as _sys, pathlib as _pl  # make app/common.py importable however the page is launched
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))

import streamlit as st

from common import kpis, load, page_header, section

plan = load("plan")
c = plan["cleaning"]
s = plan["splits"]["retained"]

page_header("About", "Method & limitations",
            "HotelSight is an experimental prototype evaluated on historical hotel records. It is a Final Year "
            "Project demonstration, not a validated live forecasting system.")

section("How a prediction is made")
FLOW = [("Booking form", "3 dates + 18 inputs"), ("Validation", "contract.py"),
        ("Feature construction", "experiment/features.py"), ("Saved preprocessor", "SHA-256 checked"),
        ("XGBoost", "frozen winner"), ("Probability + SHAP", "threshold 0.50")]
st.html('<div class="hs-flow">' + '<span class="arrow">→</span>'.join(
    f'<div class="node"><b>{a}</b><small>{b}</small></div>' for a, b in FLOW) + "</div>")

k = kpis("method", ["database", "cleaning_services", "call_split", "model_training"])
k[0].metric("Raw bookings", f"{c['raw_rows']:,}", "Hotel Booking Demand", delta_color="off", delta_arrow="off",
            border=True)
k[1].metric("After cleaning", f"{c['retained_rows']:,}", f"{c['removed_union']} removed", delta_color="off",
            delta_arrow="off", border=True)
k[2].metric("Train / validation / test", "70 / 15 / 15",
            f"{s['train']['rows'] // 1000}k · {s['validation']['rows'] // 1000}k · {s['test']['rows'] // 1000}k rows",
            delta_color="off", delta_arrow="off", border=True)
k[3].metric("Model candidates", "48 fits", "12 per arm", help="4 models × 3 configurations × 4 experiment arms",
            delta_color="off", delta_arrow="off", border=True)

tab_method, tab_limits, tab_model = st.tabs([":material/science: Method", ":material/warning: Limitations",
                                             ":material/memory: Deployed model"])

with tab_method:
    left, right = st.columns(2, gap="medium")
    with left.container(key="card_m1", height="stretch"):
        st.markdown(f"""
**Data.** The study uses the Hotel Booking Demand dataset (Antonio et al., 2019). It has {c['raw_rows']:,} bookings
from a city hotel and a resort hotel in Portugal, with arrivals from July 2015 to August 2017.

**Cleaning.** {c['zero_guest_rows']} bookings with no guests, {c['negative_adr_rows']} booking with a negative room rate
and {c['extreme_adr_5400_rows']} booking with a room rate of 5,400 were removed. That left {c['retained_rows']:,}
bookings. Two fields that reveal the outcome (`reservation_status`, `reservation_status_date`) were not used. Two fields
that can change at or after check-in (`assigned_room_type`, `booking_changes`) were not used either. Agent and company
IDs became Yes or No flags. Room type, distribution channel and arrival year were left out. A missing country became
"Unknown" and a missing number of children became 0.

**Inputs.** The model uses 24 inputs. 20 are general booking details. The other 4 are room rate (ADR), deposit type,
parking spaces and special requests.

**Repeated records.** {c['retained_duplicate_copies']:,} bookings are exact copies of another booking. The dataset has
no booking ID, so it is unclear whether they are separate bookings. The model keeps them. Bookings with identical
details always stay in the same data split, so no test booking has an identical copy in the training data.
""")
    with right.container(key="card_m2", height="stretch"):
        st.markdown(f"""
**Split.** The data was split 70 / 15 / 15 into {s['train']['rows']:,} training, {s['validation']['rows']:,}
validation and {s['test']['rows']:,} test rows. About 37% of each part is cancelled bookings. Encoding, scaling and
filling in gaps were learned from the training rows only.

**Models and tuning.** Four models were compared: Logistic Regression, Random Forest, XGBoost and CatBoost. Each one
gave extra weight to cancelled bookings to balance the two classes. Each model tried three fixed settings chosen in
advance. This is a small manual comparison, not a full search. The best model was chosen by its validation F1 score.
The decision threshold stayed at 0.50 and was never tuned.

**Evaluation.** The chosen model was frozen and then scored once on the test set. It was scored two ways, counting
every repeated copy and counting one copy per record. Extra checks removed the repeated records, the four inputs whose
timing is uncertain, or both.

**Explanations.** SHAP shows why the model made each prediction. Values are in log-odds. For one booking, the SHAP
values add up exactly to the model output. The overall importance is calculated on the test set.
""")

LIMITS = [
    ("Input timing is not verified", "The dataset was prepared around the day before arrival. Deposit type comes from "
     "payments made before arrival or cancellation, and room rate (ADR) comes from billing records. Parking and special "
     "requests can be added later. So it is not certain that these values were **known when the booking was confirmed**."
     "Without ADR, deposit, parking and requests, test F1 drops from 81.0% to 73.5% (repeated records)."),
    ("Not a day-before-arrival forecast", "About 94% of cancelled bookings were cancelled before the day before "
     "arrival. So the results should not be read as a forecast made one day ahead."),
    ("Past data, not live results", "The scores show how well the model separates past cancelled and kept bookings. "
     "Its performance on new bookings, other years or Malaysian hotels is unknown. That would need new data and "
     "retraining."),
    ("Repeated records", "It cannot be confirmed that repeated records are real separate bookings. So results are shown both "
     "ways (F1 of 81.0% and 69.9%). Neither one is the single true score."),
    ("Test set already seen", "The dataset and earlier test scores were looked at during development. So this is an "
     "early result, not an independent outside check."),
    ("Probabilities are not exact", "The model gives extra weight to cancelled bookings. This helps it find them, but "
     "the percentages are not exact chances. Use them to rank bookings by risk."),
    ("Overfitting gap", "The model scores much higher on training data (F1 0.95) than on validation data (F1 0.80). "
     "This means it fits the training data more closely than new data."),
    ("SHAP shows links, not causes", "SHAP explains how this model behaves. It does not prove that changing an input "
     "would stop a cancellation."),
    ("Dataset rules copied on purpose", "The dataset counts week numbers the way SQL Server does, with weeks starting "
     "on Sunday. Its \"weekend nights\" are nights that start on a Sunday or Monday. The dashboard uses the same rules "
     "so its inputs match the training data."),
]

with tab_limits:
    cols = st.columns(3, gap="small")
    for i, (title, body) in enumerate(LIMITS):
        with cols[i % 3].container(key=f"card_lim{i}"):
            st.markdown(f"**{title}**")
            st.caption(body)

with tab_model:
    sel = load("selected")
    p = sel["manual_parameters"]
    rows = [("Model", f"{sel['model']}, configuration {sel['configuration']}"), ("Trees", p["n_estimators"]),
            ("Max depth", p["max_depth"]), ("Learning rate", p["learning_rate"]),
            ("Column sample", p["colsample_bytree"]), ("Inputs", 24), ("Decision threshold", "0.50"),
            ("Experiment arm", "<code>retained_historical24</code> (repeated records kept, 24 inputs)"),
            ("Model file SHA-256", f"<code>{sel['model_sha256']}</code>")]
    with st.container(key="card_deployed"):
        st.html('<dl class="hs-kv">' + "".join(f"<dt>{k}</dt><dd>{v}</dd>" for k, v in rows) + "</dl>")
        st.caption("The model and preprocessor files are checked against these frozen hashes every time they are loaded.")

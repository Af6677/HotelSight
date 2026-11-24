"""Check a reservation – enter one booking, get its cancellation probability and an individual SHAP explanation."""
import sys as _sys, pathlib as _pl  # make app/common.py importable however the page is launched
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))

import math
from datetime import date, timedelta
from html import escape

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from common import (CANCEL, STAY, convert_euro_text, diverging_legend, euros, friendly, get_countries, get_model,
                    get_ranges, icon, load, money, page_header, pct, section, show_chart, step, svg_img, waterfall_chart)
from hotelsight import history
from hotelsight.waterfall import waterfall_steps
from hotelsight.contract import (CATEGORY_LABELS, DATE_DERIVED, MODEL_FIELDS, derive_from_dates, display_value,
                                 validate_dates, validate_record)

model = get_model()
countries = get_countries()
ranges = get_ranges()
examples = load("examples")
YES_NO = ["Yes", "No"]
BLANK = "Blank reservation"
DATE_FORMAT = "DD/MM/YYYY"

page_header("Reservations", "Check a reservation",
            "Enter a reservation as it is currently recorded to see how likely it is to cancel, and why.")

# ------------------------------------------------------------------ form state
DEFAULTS = {
    "k_booking": date.today(), "k_arrival": date.today() + timedelta(days=69),
    "k_departure": date.today() + timedelta(days=72),
    "k_hotel": "City Hotel", "k_meal": "BB", "k_deposit": "No Deposit", "k_adr": 95.0,
    "k_market": "Online TA", "k_customer": "Transient", "k_agent": "Yes", "k_company": "No", "k_wait": 0,
    "k_adults": 2, "k_children": 0, "k_babies": 0, "k_country": "PRT", "k_repeat": "No",
    "k_prevc": 0, "k_prevok": 0, "k_parking": 0, "k_requests": 0, "k_ref": "", "k_save": True,
}
FIELD_KEY = {"hotel": "k_hotel", "meal": "k_meal", "deposit_type": "k_deposit", "adr": "k_adr",
             "market_segment": "k_market", "customer_type": "k_customer", "days_in_waiting_list": "k_wait",
             "adults": "k_adults", "children": "k_children", "babies": "k_babies", "country": "k_country",
             "previous_cancellations": "k_prevc", "previous_bookings_not_canceled": "k_prevok",
             "required_car_parking_spaces": "k_parking", "total_of_special_requests": "k_requests"}
YESNO_KEY = {"has_travel_agent": "k_agent", "company_booking": "k_company", "is_repeated_guest": "k_repeat"}

if "k_hotel" not in st.session_state:
    st.session_state.update(DEFAULTS)


def load_example():
    choice = st.session_state.example_choice
    st.session_state.pop("result", None)
    if choice == BLANK:
        st.session_state.update(DEFAULTS); st.session_state.example = None; return
    ex = next(e for e in examples if e["title"] == choice)
    st.session_state.example = ex
    st.session_state.k_ref = f"EXAMPLE-{ex['source_csv_row']}"
    st.session_state.k_booking = date.fromisoformat(ex["booking_date"])
    st.session_state.k_arrival = date.fromisoformat(ex["arrival_date"])
    st.session_state.k_departure = date.fromisoformat(ex["departure_date"])
    for f, k in FIELD_KEY.items():
        v = ex["inputs"][f]
        st.session_state[k] = v if f in ("hotel", "meal", "deposit_type", "market_segment", "customer_type", "country") \
            else (float(v) if f == "adr" else int(v))
    for f, k in YESNO_KEY.items():
        st.session_state[k] = "Yes" if ex["inputs"][f] == 1 else "No"


def label(field):
    return lambda v: CATEGORY_LABELS.get(field, {}).get(v, v)


def saved_value(rec, f):
    """Value text as stored in the prediction history (amounts stay in EUR)."""
    return display_value(f, rec[f], countries)


def shown_value(rec, f):
    """Value text for the screen: room rates follow the display currency chosen in the sidebar."""
    return convert_euro_text(saved_value(rec, f))


def matches_example(rec, ex):
    """True when the form still holds the example's inputs, so its saved prediction and outcome apply."""
    return ex is not None and all(
        np.isclose(float(rec[f]), float(ex["inputs"][f])) if not isinstance(ex["inputs"][f], str)
        else rec[f] == ex["inputs"][f] for f in ex["inputs"])


def dial(prob, threshold):
    """Semicircle risk dial: the arc fills to the probability; the tick marks the decision threshold."""
    colour = CANCEL if prob >= threshold else STAY

    def at(p, r):
        a = math.pi * (1 - p)
        return 100 + r * math.cos(a), 112 - r * math.sin(a)

    x, y = at(min(max(prob, 0.004), 1.0), 80)
    (t1x, t1y), (t2x, t2y), (lx, ly) = at(threshold, 67), at(threshold, 93), at(threshold, 98)
    svg = (f'<svg viewBox="0 0 200 134" xmlns="http://www.w3.org/2000/svg" font-family="Segoe UI, Arial, sans-serif">'
           f'<path d="M20 112 A80 80 0 0 1 180 112" fill="none" stroke="#EDF1F7" stroke-width="15"/>'
           f'<path d="M20 112 A80 80 0 0 1 {x:.2f} {y:.2f}" fill="none" stroke="{colour}" stroke-width="15"/>'
           f'<line x1="{t1x:.2f}" y1="{t1y:.2f}" x2="{t2x:.2f}" y2="{t2y:.2f}" stroke="#0F1B2D" stroke-width="2"/>'
           f'<text x="{lx:.1f}" y="{ly - 3:.1f}" text-anchor="middle" font-size="11" fill="#55637A">{threshold:.0%}</text>'
           f'<text x="100" y="104" text-anchor="middle" font-size="11" fill="#55637A">threshold</text>'
           f'<text x="20" y="129" text-anchor="middle" font-size="11" fill="#55637A">0%</text>'
           f'<text x="180" y="129" text-anchor="middle" font-size="11" fill="#55637A">100%</text></svg>')
    verdict = "Likely to cancel" if prob >= threshold else "Likely to stay"
    return f'<div class="hs-dial">{svg_img(svg, 190, 127, f"Risk dial: {prob:.1%}, {verdict}")}</div>'


def stay_summary(res) -> str:
    rec, (booked, arrive, depart) = res["rec"], res["dates"]
    nights = int(rec["stays_in_weekend_nights"] + rec["stays_in_week_nights"])
    counts = [(int(rec["adults"]), "adult", "adults"), (int(rec["children"] or 0), "child", "children"),
              (int(rec["babies"]), "baby", "babies")]
    guests = ", ".join(f"{n} {one if n == 1 else many}" for n, one, many in counts if n) or "0"

    def node(lbl, d, note):
        return (f'<div class="hs-tl-node"><div class="lbl">{lbl}</div><div class="date">{d:%d %b %Y}</div>'
                f'<div class="day">{note}</div></div>')

    facts = [("Guests", guests), ("Room rate", money(rec["adr"], 2)),
             ("Stay value", euros(rec["adr"] * nights) if nights else "Day use"),
             ("Deposit", shown_value(rec, "deposit_type").split(" (")[0]),
             ("Channel", shown_value(rec, "market_segment")), ("Meal plan", shown_value(rec, "meal").split(" (")[0])]
    return ('<div class="hs-timeline">' + node("Booked", booked, f'{int(rec["lead_time"])} days ahead')
            + '<div class="hs-tl-line"></div>' + node("Check-in", arrive, f"{arrive:%A}")
            + '<div class="hs-tl-line"></div>' + node("Check-out", depart, f'{nights} night{"s" * (nights != 1)}')
            + '</div><div class="hs-facts">'
            + "".join(f'<div class="hs-fact"><span>{k}</span><b>{escape(str(v))}</b></div>' for k, v in facts) + "</div>")


# ------------------------------------------------------------------ layout: form on the left, answer beside it
form_col, result_col = st.columns([1.16, 1], gap="medium")

with form_col:
    st.selectbox("Start from", [BLANK] + [e["title"] for e in examples], key="example_choice", on_change=load_example,
                 help="Examples are real reservations from the held-out test set, so the prediction can be checked "
                      "against the experiment's saved prediction and the real outcome.")
    with st.form("booking"):
        step("calendar_month", "Stay")
        c1, c2, c3 = st.columns(3)
        c1.date_input("Booked on", key="k_booking", format=DATE_FORMAT, help="Date the reservation was made.")
        c2.date_input("Check-in", key="k_arrival", format=DATE_FORMAT)
        c3.date_input("Check-out", key="k_departure", format=DATE_FORMAT)

        step("bed", "Room and rate")
        c1, c2 = st.columns(2)
        c1.selectbox("Property", model.categories["hotel"], key="k_hotel")
        c2.number_input("Room rate per night (EUR)", min_value=0.0, step=5.0, key="k_adr",
                        help="Average daily rate (ADR) in euros, as in the dataset. The model works in euros, so enter the rate in EUR. "
                             "The currency chosen in the sidebar only changes the amounts shown in the results.")
        c1, c2 = st.columns(2)
        c1.selectbox("Meal plan", model.categories["meal"], key="k_meal", format_func=label("meal"))
        c2.selectbox("Deposit", model.categories["deposit_type"], key="k_deposit", format_func=label("deposit_type"))

        step("storefront", "Booking channel")
        c1, c2 = st.columns(2)
        c1.selectbox("Channel", model.categories["market_segment"], key="k_market", format_func=label("market_segment"),
                     help="The dataset's market segment.")
        c2.selectbox("Customer type", model.categories["customer_type"], key="k_customer", format_func=label("customer_type"))
        c1, c2, c3 = st.columns(3)
        c1.radio("Travel agent?", YES_NO, key="k_agent", horizontal=True)
        c2.radio("Company booking?", YES_NO, key="k_company", horizontal=True)
        c3.number_input("Waiting list (days)", min_value=0, step=1, key="k_wait")

        step("group", "Guest")
        codes = sorted(countries, key=lambda c: countries[c].lower())
        c1, c2 = st.columns([2, 1])
        c1.selectbox("Guest country", codes, key="k_country", format_func=lambda c: f"{countries[c]} ({c})",
                     help="Only the 167 countries seen in training are listed. 'Not recorded' is the dataset's missing-country category.")
        c2.radio("Returning guest?", YES_NO, key="k_repeat", horizontal=True)
        c1, c2, c3 = st.columns(3)
        c1.number_input("Adults", min_value=0, step=1, key="k_adults")
        c2.number_input("Children", min_value=0, step=1, key="k_children")
        c3.number_input("Babies", min_value=0, step=1, key="k_babies")
        c1, c2, c3 = st.columns(3)
        c1.number_input("Past cancellations", min_value=0, step=1, key="k_prevc")
        c2.number_input("Past completed stays", min_value=0, step=1, key="k_prevok")
        c3.number_input("Car parking spaces", min_value=0, step=1, key="k_parking")

        step("room_service", "Requests and reference")
        c1, c2 = st.columns(2)
        c1.number_input("Special requests", min_value=0, step=1, key="k_requests",
                        help="For example twin beds or a high floor.")
        c2.text_input("Reservation reference", key="k_ref", placeholder="Optional, e.g. RSV-10234",
                      help="Shown in the prediction history so you can find this reservation again.")
        with st.container(key="form_actions"):
            c1, c2 = st.columns([1, 1.6], vertical_alignment="center")
            c1.toggle("Save to history", key="k_save", help="Keep this prediction on the Prediction history page.")
            submitted = c2.form_submit_button("Check cancellation risk", type="primary", width="stretch")

# ------------------------------------------------------------------ predict, then answer in the right-hand panel
ss = st.session_state
failed = False
with result_col, st.container(key="result_panel"):
    if submitted:
        dv = validate_dates(ss.k_booking, ss.k_arrival, ss.k_departure)
        if not dv.ok:
            for e in dv.errors: st.error(e, icon=":material/error:")
            ss.pop("result", None); failed = True
        else:
            rec = derive_from_dates(ss.k_booking, ss.k_arrival, ss.k_departure)
            rec.update({f: ss[k] for f, k in FIELD_KEY.items()})
            rec.update({f: 1.0 if ss[k] == "Yes" else 0.0 for f, k in YESNO_KEY.items()})
            rec = {f: rec[f] for f in MODEL_FIELDS}
            v = validate_record(rec, ranges, model.categories)
            if not v.ok:
                for e in v.errors: st.error(e, icon=":material/error:")
                ss.pop("result", None); failed = True
            else:
                ex = model.explain(model.frame(rec))
                ss.result = {"rec": rec, "warnings": dv.warnings + v.warnings, "prob": float(ex["probability"][0]),
                             "margin": float(ex["margin"][0]), "base": ex["base_log_odds"],
                             "contrib": ex["contributions"].iloc[0], "recon": float(ex["reconstruction_error"][0]),
                             "example": ss.get("example"), "saved": False, "reference": ss.k_ref.strip(),
                             "dates": (ss.k_booking, ss.k_arrival, ss.k_departure)}
                if ss.k_save:
                    c = ss.result["contrib"]
                    up, down = c.idxmax(), c.idxmin()
                    example = ss.get("example")
                    history.save([{"reference": ss.k_ref.strip() or None, "hotel": rec["hotel"],
                                   "arrival": ss.k_arrival.isoformat(), "probability": ss.result["prob"],
                                   "top_raising": f"{friendly(up)}: {saved_value(rec, up)}" if c[up] > 0 else "",
                                   "top_lowering": f"{friendly(down)}: {saved_value(rec, down)}" if c[down] < 0 else "",
                                   "actual": example["actual_cancelled"] if matches_example(rec, example) else None,
                                   "record": rec}],
                                 source="single", batch_name="Single prediction", threshold=model.threshold,
                                 model_sha256=model.record["model_sha256"])
                    ss.result["saved"] = True

    res = ss.get("result")
    if res:
        prob, rec, contrib = res["prob"], res["rec"], res["contrib"]
        for w in res["warnings"]:
            st.warning(w, icon=":material/warning:")

        def shown(f):
            return f'{friendly(f).split(" (")[0]}: {shown_value(rec, f)}'

        with st.container(key="card_risk"):
            example_badge = (f'<span class="hs-badge">{icon("verified")}Held-out test reservation</span>'
                             if matches_example(rec, res["example"]) else "")
            st.html(f'<div class="hs-res-head"><div><div class="hs-res-title">{escape(res["reference"] or "This reservation")}'
                    f'</div><div class="hs-res-sub">{escape(rec["hotel"])} · {escape(shown_value(rec, "customer_type"))}'
                    f'</div></div>{example_badge}</div>')
            c1, c2 = st.columns([1, 1], vertical_alignment="center")
            with c1:
                st.metric("Cancellation probability", pct(prob))
                st.html('<span class="hs-verdict hi">' + icon("warning") + "Likely to cancel</span>"
                        if prob >= model.threshold else
                        '<span class="hs-verdict lo">' + icon("check_circle") + "Likely to stay</span>")
            with c2:
                st.html(dial(prob, model.threshold))
            st.caption("A score for ranking reservations by risk (test ROC AUC 0.93), not a calibrated real-world "
                       "frequency." + (" Saved to the prediction history." if res.get("saved") else ""))

        with st.container(key="card_factors"):
            up = [f for f in contrib.sort_values(ascending=False).index[:3] if contrib[f] > 0]
            down = [f for f in contrib.sort_values().index[:3] if contrib[f] < 0]

            def items(fields):
                if not fields:
                    return "<li><span>None</span></li>"
                return "".join(f"<li><span>{escape(shown(f))}</span><b>{f'{contrib[f]:+.2f}'.replace('-', '−')}</b></li>"
                               for f in fields)

            st.html(f'<div class="hs-card-title">Main reasons<small>SHAP contribution, log-odds</small></div>'
                    f'<div class="hs-list-title up">{icon("arrow_upward")}Raising the risk</div>'
                    f'<ul class="hs-drivers">{items(up)}</ul>'
                    f'<div class="hs-list-title down">{icon("arrow_downward")}Lowering the risk</div>'
                    f'<ul class="hs-drivers">{items(down)}</ul>')

        with st.container(key="card_stay"):
            st.html('<div class="hs-card-title">Stay details</div>' + stay_summary(res))
    elif not failed:
        with st.container(key="card_empty"):
            st.html(f'<div class="hs-empty">{icon("hotel")}<b>Your result will appear here</b>'
                    '<span>Fill in the reservation, or load an example, then press Check cancellation risk.</span></div>')

# ---------------- full SHAP explanation, below the form and the answer
if res:
    section("Why the model gave this score", "Each input's contribution to this reservation's score")
    order = contrib.abs().sort_values(ascending=False).index
    top, rest = list(order[:10]), list(order[10:])
    chart_col, text_col = st.columns([1.7, 1], gap="medium")
    with chart_col.container(key="card_shap", height="stretch"):
        diverging_legend("Top 10 inputs, step by step", totals=True)
        steps = [(shown(f), float(contrib[f])) for f in top]
        if rest:
            steps.append((f"Other {len(rest)} inputs combined", float(contrib[rest].sum())))
        show_chart(waterfall_chart(waterfall_steps(res["base"], steps, "Average prediction (start)", "This booking's score")))
    with text_col.container(key="card_read", height="stretch"):
        st.html('<div class="hs-card-title">How to read this</div>')
        total = res["base"] + contrib.sum()
        st.markdown(
            f"The model starts from its average output on the training data, **{res['base']:+.3f} log-odds "
            f"({pct(1 / (1 + np.exp(-res['base'])))})**. Adding every input's contribution gives "
            f"**{total:+.3f} log-odds**, which converts to **{pct(1 / (1 + np.exp(-total)))}**.")
        st.caption(f"Reconstruction check: |base + sum of contributions − model output| = {res['recon']:.1e} log-odds "
                   f"{'✓' if res['recon'] < 1e-3 else '✗'}. Contributions add up in log-odds, not in percentage points. "
                   "These are associations the model learned from historical records, not proof that an input causes "
                   "a cancellation.")
        ex = res["example"]
        if ex is not None:
            st.html('<div class="hs-card-title" style="margin:6px 0 0">Check against the experiment</div>')
            if matches_example(rec, ex):
                diff = abs(prob - ex["saved_probability"])
                st.markdown(f"**Saved test prediction:** {pct(ex['saved_probability'], 2)}  \n"
                            f"**Difference:** {diff:.1e} {'✓' if diff < 1e-6 else '✗'}  \n"
                            f"**Actual outcome:** {'cancelled' if ex['actual_cancelled'] else 'not cancelled'}")
                st.caption(f"Held-out test reservation, CSV row {ex['source_csv_row']:,}.")
            else:
                st.caption("The example's inputs were edited, so there is no saved prediction to compare with.")

    with st.expander("Model inputs sent to the model (24, exact contract order)", icon=":material/table_view:"):
        rows = [{"Model input": f, "Description": friendly(f), "Value": rec[f],
                 "Source": "calculated from dates" if f in DATE_DERIVED else "form"} for f in MODEL_FIELDS]
        st.dataframe(pd.DataFrame(rows).astype({"Value": str}), hide_index=True, width="stretch")
        st.caption("Week number follows SQL Server DATEPART(week): weeks start on Sunday, and week 1 contains 1 January. "
                   "The dataset's 'weekend nights' are nights that begin on a Sunday or a Monday. Both rules reproduce "
                   "100% of the dataset's recorded values.")

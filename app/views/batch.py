"""Upload reservations – score a whole spreadsheet of bookings at once, and save the results to the history."""
import sys as _sys, pathlib as _pl  # make app/common.py importable however the page is launched
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))

import hashlib

import altair as alt
import pandas as pd
import streamlit as st

from common import (BLUE, NAVY, RED, base_chart, euros, get_countries, get_model, get_ranges, load, page_header, pct,
                    section)
from hotelsight import batch, history

model = get_model()
countries = get_countries()
ranges = get_ranges()
examples = load("examples")
ss = st.session_state
XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
OUTCOME = alt.Scale(domain=["Likely to cancel", "Likely to stay"], range=[RED, BLUE])

page_header("Reservations office · bulk upload", "Upload reservations",
            "Upload an Excel or CSV list of reservations. Every row is checked with the same rules as the single "
            "reservation form, scored by the deployed model, and explained by its main risk factors.",
            chips=["Excel (.xlsx) or CSV", f"Up to {batch.MAX_ROWS:,} reservations", "Same checks as the form",
                   "Room revenue at risk", "Saved to history"], icon_name="upload_file")


@st.cache_data
def template_bytes() -> bytes:
    return batch.template(examples, model.categories)


@st.cache_data
def guide() -> pd.DataFrame:
    return batch.read_table(template_bytes(), "template.xlsx", sheet="Column guide")


@st.cache_data
def sample(n: int = 200) -> pd.DataFrame:
    return batch.sample_test_bookings(n)


# ------------------------------------------------------------------ 1. the file
section("1 · Prepare your file", "One row per reservation")
c1, c2 = st.columns([1.3, 1], gap="medium")
with c1.container(key="card_template", height="stretch"):
    st.html('<div class="hs-card-title">Start from the template</div>')
    st.markdown("Fill in one row per reservation with the **booking, check-in and check-out dates** and the other "
                "reservation details. Column order does not matter, and both dataset codes (e.g. `BB`, `Online TA`) "
                "and the dashboard's labels are accepted. Files in the original *Hotel Booking Demand* layout also work.")
    st.download_button("Download Excel template", template_bytes(), file_name="HotelSight_reservation_template.xlsx",
                       mime=XLSX, icon=":material/download:", key="dl_template")
with c2.container(key="card_sample", height="stretch"):
    st.html('<div class="hs-card-title">No file to hand?</div>')
    st.markdown("Try **200 real reservations** from the held-out test set. They include the real outcome, so the "
                "results can be compared with what actually happened.")
    if st.button("Use 200 sample reservations", key="use_sample", icon=":material/dataset:"):
        ss.batch_use_sample = True
with st.expander("Column guide", icon=":material/view_column:"):
    st.dataframe(guide(), hide_index=True, width="stretch")

# ------------------------------------------------------------------ 2. upload + check
section("2 · Upload and check", "Rows with problems are listed and skipped")
up = st.file_uploader("Reservation file", type=["xlsx", "csv"], key="batch_file",
                      help=f"Excel workbook (.xlsx) or CSV, at most {batch.MAX_ROWS:,} reservations.")
src = None
if up is not None:
    data = up.getvalue()
    sheets = batch.sheet_names(data, up.name)
    sheet = st.selectbox("Sheet", sheets, key="batch_sheet") if len(sheets) > 1 else 0
    try:
        src = {"name": up.name, "key": hashlib.md5(data).hexdigest() + str(sheet),
               "df": batch.read_table(data, up.name, sheet)}
    except Exception as e:                                       # unreadable file: say so, don't crash
        st.error(f"Could not read this file: {e}")
elif ss.get("batch_use_sample"):
    src = {"name": "Sample – 200 held-out test reservations", "key": "sample", "df": sample()}

parsed = None
if src is not None:
    if ss.get("batch_parsed_key") != src["key"]:
        try:
            ss.batch_parsed = batch.prepare(src["df"], model.categories, ranges, countries)
        except ValueError as e:
            ss.batch_parsed = str(e)
        ss.batch_parsed_key = src["key"]
    parsed = ss.batch_parsed
    if isinstance(parsed, str):
        st.error(parsed, icon=":material/error:")
        st.caption("Tip: download the template above and copy your reservations into it.")
        parsed = None

if parsed is not None:
    bad = [r for r in parsed.rows if r.errors]
    warned = [r for r in parsed.valid if r.warnings]
    k = st.columns(4)
    k[0].metric("Rows read", f"{len(parsed.rows):,}", border=True)
    k[1].metric("Ready to score", f"{len(parsed.valid):,}", border=True)
    k[2].metric("Rows with problems", f"{len(bad):,}", border=True)
    k[3].metric("Rows with warnings", f"{len(warned):,}", border=True)
    st.caption(f"**{src['name']}** · " + ("Template layout: dates are converted to the model's date inputs."
                                         if parsed.layout == "template" else
                                         "Dataset layout: date inputs are read directly; agent/company IDs become Yes/No.")
               + (f" Ignored columns: {', '.join(parsed.ignored_columns)}." if parsed.ignored_columns else ""))
    if bad:
        with st.expander(f"{len(bad)} row(s) will be skipped", icon=":material/error:", expanded=len(bad) <= 5):
            st.dataframe(pd.DataFrame([{"Row": r.row, "Reference": r.reference, "Problem": " ".join(r.errors)}
                                       for r in bad]), hide_index=True, width="stretch")
    if warned:
        with st.expander(f"{len(warned)} row(s) have warnings (still scored)", icon=":material/warning:"):
            st.dataframe(pd.DataFrame([{"Row": r.row, "Reference": r.reference, "Warning": " ".join(r.warnings)}
                                       for r in warned]), hide_index=True, width="stretch")

    # ------------------------------------------------------------------ 3. predict
    section("3 · Score", "Every valid row is scored by the deployed model")
    c1, c2, c3 = st.columns([1.4, 1, 1], vertical_alignment="bottom")
    name = c1.text_input("Upload name", value=src["name"], key=f"batch_name_{src['key']}",
                         help="Shown in the prediction history.")
    save = c2.checkbox("Save results to history", value=True, key="batch_save")
    run = c3.button(f"Score {len(parsed.valid):,} reservations", type="primary", key="run_batch", width="stretch",
                    disabled=not parsed.valid, icon=":material/online_prediction:")
    if run:
        bar = st.progress(0.0, text="Scoring reservations…")
        results = batch.predict(model, parsed, countries,
                                progress=lambda f: bar.progress(f, text=f"Scoring reservations… {f:.0%}"))
        bar.empty()
        batch_id = None
        if save:
            scored = results[results["Cancellation probability"].notna()]
            records = {r.row: r for r in parsed.valid}
            batch_id = history.save(
                [{"reference": x["Reference"], "hotel": x["Hotel"], "arrival": x["Arrival"],
                  "probability": x["Cancellation probability"], "top_raising": x["Main factor raising risk"],
                  "top_lowering": x["Main factor lowering risk"], "actual": records[x["Row"]].actual,
                  "record": records[x["Row"]].record} for _, x in scored.iterrows()],
                source="batch", batch_name=name.strip() or src["name"], threshold=model.threshold,
                model_sha256=model.record["model_sha256"])
        ss.batch_result = {"key": src["key"], "results": results, "name": name, "saved": batch_id is not None}

# ------------------------------------------------------------------ results
res = ss.get("batch_result")
if parsed is not None and res and res["key"] == src["key"]:
    df = res["results"]
    scored = df[df["Cancellation probability"].notna()].copy()
    flagged = scored[scored["Prediction"] == "Likely to cancel"]
    n_cancel = len(flagged)
    section("Results", res["name"])
    if res["saved"]:
        st.success(f"Saved {len(scored):,} predictions to the history. Open **Prediction history** in the sidebar "
                   "to review them later.", icon=":material/history:")
    k = st.columns(4)
    k[0].metric("Reservations scored", f"{len(scored):,}",
                f"{len(df) - len(scored):,} not scored" if len(df) > len(scored) else "all rows scored",
                delta_color="off", delta_arrow="off", border=True)
    k[1].metric("Likely to cancel", f"{n_cancel:,}", f"{pct(n_cancel / max(len(scored), 1))} of scored",
                delta_color="off", delta_arrow="off", border=True)
    k[2].metric("Nights flagged", f"{int(flagged['Nights'].sum()):,}",
                f"of {int(scored['Nights'].sum()):,} room nights", delta_color="off", delta_arrow="off", border=True)
    k[3].metric("Revenue flagged", euros(flagged["Stay value (EUR)"].sum()),
                f"of {euros(scored['Stay value (EUR)'].sum())} booked", delta_color="off", delta_arrow="off",
                border=True, help="Room rate × nights for the reservations flagged as likely to cancel (EUR).")

    if "Actual outcome" in scored and (scored["Actual outcome"] != "").any():
        known = scored[scored["Actual outcome"] != ""]
        y, p = (known["Actual outcome"] == "Cancelled").to_numpy(), (known["Prediction"] == "Likely to cancel").to_numpy()
        tp = int((y & p).sum())
        f1 = 2 * tp / max(int(y.sum()) + int(p.sum()), 1)
        st.info(f"This file includes recorded outcomes for {len(known):,} reservations: the predictions agree with "
                f"{pct(float((y == p).mean()))} of them (F1 {f1:.3f}). Small files give noisy figures; the sealed "
                "test results are on the Test performance page.", icon=":material/fact_check:")

    left, right = st.columns([1.35, 1], gap="medium")
    with left.container(key="card_arrivals", height="stretch"):
        st.html('<div class="hs-card-title">Arrivals by month</div>')
        arr = scored.assign(When=pd.to_datetime(scored["Arrival"], errors="coerce", format="%Y-%m-%d")).dropna(subset=["When"])
        if len(arr):
            arr["Month"] = arr["When"].dt.to_period("M").dt.to_timestamp()
            st.altair_chart(base_chart(alt.Chart(arr).mark_bar(cornerRadiusTopLeft=2, cornerRadiusTopRight=2).encode(
                x=alt.X("yearmonth(Month):O", title="Arrival month",
                        axis=alt.Axis(format="%b %Y", formatType="time", labelAngle=-45)),
                y=alt.Y("count():Q", title="Reservations"),
                color=alt.Color("Prediction:N", scale=OUTCOME, legend=alt.Legend(orient="top", title=None)),
                order=alt.Order("Prediction:N"),
                tooltip=[alt.Tooltip("yearmonth(Month):O", title="Arrival month", format="%B %Y"),
                         "Prediction:N", alt.Tooltip("count():Q", title="Reservations")]).properties(height=260)),
                width="stretch")
        else:
            st.caption("Arrival dates need a year (template layout, or an arrival_date_year column) to be charted.")
    with right.container(key="card_dist", height="stretch"):
        st.html('<div class="hs-card-title">Spread of cancellation probabilities</div>')
        hist = alt.Chart(scored).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
            x=alt.X("Cancellation probability:Q", bin=alt.Bin(extent=[0, 1], step=0.05),
                    axis=alt.Axis(format="%"), title="Cancellation probability"),
            y=alt.Y("count():Q", title="Reservations"),
            color=alt.Color("Prediction:N", scale=OUTCOME, legend=alt.Legend(orient="top", title=None)),
            tooltip=[alt.Tooltip("count():Q", title="Reservations"), "Prediction:N"])
        rule = alt.Chart(pd.DataFrame({"x": [model.threshold]})).mark_rule(color=NAVY, strokeDash=[4, 3]).encode(x="x:Q")
        st.altair_chart(base_chart((hist + rule).properties(height=260)), width="stretch")
        st.caption(f"Dashed line: the {model.threshold:.2f} decision threshold.")

    money = {"Room rate (EUR)": st.column_config.NumberColumn(format="€%.2f"),
             "Stay value (EUR)": st.column_config.NumberColumn(format="€%.2f"),
             "Cancellation probability": st.column_config.ProgressColumn("Probability", format="%.1f%%", min_value=0,
                                                                         max_value=100),
             "Lead time (days)": st.column_config.NumberColumn(format="%d"),
             "Nights": st.column_config.NumberColumn(format="%d")}
    with st.container(key="card_top"):
        st.html('<div class="hs-card-title">Highest-risk reservations</div>')
        top = scored.nlargest(10, "Cancellation probability")
        st.dataframe(top[["Reference", "Hotel", "Arrival", "Nights", "Stay value (EUR)", "Cancellation probability",
                          "Main factor raising risk"]].assign(
                         **{"Cancellation probability": top["Cancellation probability"] * 100}).rename(
                         columns={"Hotel": "Property"}),
                     hide_index=True, width="stretch", column_config=money)

    section("All rows", "Filter, then download")
    view = st.segmented_control("Show", ["All rows", "Likely to cancel", "Likely to stay", "Not scored"],
                                default="All rows", key="batch_view")
    shown = df if view in (None, "All rows") else df[df["Prediction"] == view]
    st.dataframe(shown.assign(**{"Cancellation probability": shown["Cancellation probability"] * 100}),
                 hide_index=True, width="stretch", height=min(38 + 35 * len(shown), 460), column_config=money)

    summary = pd.DataFrame({"Item": ["Upload", "Rows read", "Scored", "Likely to cancel", "Not scored",
                                     "Room nights flagged", "Room revenue flagged (EUR)", "Threshold", "Model",
                                     "Model file SHA-256"],
                            "Value": [res["name"], len(df), len(scored), n_cancel, len(df) - len(scored),
                                      int(flagged["Nights"].sum()), round(float(flagged["Stay value (EUR)"].sum()), 2),
                                      model.threshold, "XGBoost (retained_historical24)", model.record["model_sha256"]]})
    d1, d2, _ = st.columns([1, 1, 2])
    d1.download_button("Download results (Excel)", batch.to_excel({"Predictions": df, "Summary": summary}),
                       file_name="HotelSight_reservation_risk.xlsx", mime=XLSX, icon=":material/download:",
                       width="stretch", key="dl_xlsx")
    d2.download_button("Download results (CSV)", df.to_csv(index=False).encode("utf-8-sig"),
                       file_name="HotelSight_reservation_risk.csv", mime="text/csv", icon=":material/download:",
                       width="stretch", key="dl_csv")
    st.caption("Probabilities come from a class-weighted model: use them to rank reservations by risk, not as exact "
               "odds. Room revenue flagged is room rate × nights of the flagged reservations, not a forecast of lost "
               "revenue. Main factors are the inputs with the largest SHAP contribution for each reservation "
               "(associations learned from historical records, not causes).")
elif src is None:
    st.info("Upload a file above, or use the sample reservations, to continue.", icon=":material/upload_file:")

"""Upload reservations – score a whole spreadsheet of bookings at once, and save the results to the history."""
import sys as _sys, pathlib as _pl  # make app/common.py importable however the page is launched
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))

import hashlib

import pandas as pd
import streamlit as st

from common import (convert_euro_text, get_countries, get_model, get_ranges, kpis, load, money_view, page_header, pct,
                    section, verdict_colours)
from hotelsight import batch, history

model = get_model()
countries = get_countries()
ranges = get_ranges()
examples = load("examples")
ss = st.session_state
XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

page_header("Predict", "Upload reservations",
            f"Score up to {batch.MAX_ROWS:,} reservations from an Excel or CSV file. Each row is checked with the same "
            "rules as the single reservation form.")


@st.cache_data
def template_bytes() -> bytes:
    return batch.template(examples, model.categories)


@st.cache_data
def guide() -> pd.DataFrame:
    return batch.read_table(template_bytes(), "template.xlsx", sheet="Column guide")


@st.cache_data
def sample(n: int = 200) -> pd.DataFrame:
    return batch.sample_test_bookings(n)


@st.cache_data
def demo(day: str, n: int = 200) -> pd.DataFrame:              # cached per day, so the dates move with today
    return batch.sample_upcoming_bookings(n, today=pd.Timestamp(day).date())


# ------------------------------------------------------------------ the file
c1, c2 = st.columns([1.7, 1], gap="medium")
with c1.container(key="card_upload", height="stretch"):
    st.html('<div class="hs-card-title">Reservation file</div>')
    up = st.file_uploader("Reservation file", type=["xlsx", "csv"], key="batch_file", label_visibility="collapsed")
    st.caption("One row per reservation with the booking, check-in and check-out dates and the other details. Column "
               "order does not matter, and dataset codes (`BB`, `Online TA`) or the dashboard's labels both work. "
               "Files in the original *Hotel Booking Demand* layout also work.")
with c2.container(key="card_start", height="stretch"):
    st.html('<div class="hs-card-title">Need a file?</div>')
    st.download_button("Download Excel template", template_bytes(), file_name="HotelSight_reservation_template.xlsx",
                       mime=XLSX, icon=":material/download:", key="dl_template", width="stretch")
    if st.button("Try 200 sample reservations", key="use_sample", icon=":material/dataset:", width="stretch"):
        ss.batch_use_sample = "test"
    if st.button("Demo: 200 upcoming arrivals", key="use_demo", icon=":material/event_upcoming:", width="stretch"):
        ss.batch_use_sample = "demo"
    st.caption("The sample comes from the held-out test set and includes real outcomes, so you can compare them. "
               "The demo uses the same kind of bookings moved to arrive in the next 8 weeks, with no outcome yet, "
               "to show how the hotel would use the risk outlook and action list.")
with st.expander("Column guide", icon=":material/view_column:"):
    st.dataframe(guide(), hide_index=True, width="stretch")

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
elif ss.get("batch_use_sample") == "demo":
    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    src = {"name": "Demo – 200 upcoming arrivals (no real outcomes)", "key": "demo" + today, "df": demo(today)}
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
    section("File check", "Rows with problems are listed and skipped")
    k = kpis("check", ["table_rows", "task_alt", "error", "warning"], {2: "coral" if bad else "slate", 3: "slate"})
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

    # ------------------------------------------------------------------ score
    c1, c2, c3 = st.columns([1.4, 1, 1], vertical_alignment="bottom")
    name = c1.text_input("Upload name", value=src["name"], key=f"batch_name_{src['key']}",
                         help="Shown in the prediction history.")
    save = c2.toggle("Save results to history", value=True, key="batch_save")
    run = c3.button(f"Score {len(parsed.valid):,} reservations", type="primary", key="run_batch", width="stretch",
                    disabled=not parsed.valid)
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
        ss.batch_result = {"key": src["key"], "results": results, "name": name, "saved": batch_id is not None,
                           "batch_id": batch_id}

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
    k = kpis("batch", ["fact_check", "event_busy", "error"], {1: "coral", 2: "slate"})
    k[0].metric("Reservations scored", f"{len(scored):,}",
                f"{len(df) - len(scored):,} not scored" if len(df) > len(scored) else "all rows scored",
                delta_color="off", delta_arrow="off", border=True)
    k[1].metric("Likely to cancel", f"{n_cancel:,}", f"{pct(n_cancel / max(len(scored), 1))} of scored",
                delta_color="off", delta_arrow="off", border=True)
    k[2].metric("Rows not scored", f"{len(df) - len(scored):,}", "see the Problems column",
                delta_color="off", delta_arrow="off", border=True)

    if "Actual outcome" in scored and (scored["Actual outcome"] != "").any():
        known = scored[scored["Actual outcome"] != ""]
        y, p = (known["Actual outcome"] == "Cancelled").to_numpy(), (known["Prediction"] == "Likely to cancel").to_numpy()
        tp = int((y & p).sum())
        f1 = 2 * tp / max(int(y.sum()) + int(p.sum()), 1)
        st.info(f"This file includes recorded outcomes for {len(known):,} reservations: the predictions agree with "
                f"{pct(float((y == p).mean()))} of them (F1 {f1:.3f}): {int((y == p).sum()):,} correct and "
                f"{int((y != p).sum()):,} wrong. See the Correct? column below. Small files give noisy figures; the "
                "sealed test results are on the Model evaluation page.", icon=":material/fact_check:")

    if res["saved"]:
        with st.container(key="card_batch_next"):
            st.html('<div class="hs-card-title">Next steps for this upload</div>')
            c1, c2, _ = st.columns([1, 1, 1.2])
            if c1.button("Analyse this upload", icon=":material/analytics:", key="go_outlook", width="stretch"):
                ss["ins_batch"] = res["batch_id"]
                st.switch_page("views/insights.py")
            if c2.button("Open its action list", icon=":material/checklist:", key="go_actions", width="stretch"):
                ss["act_batch"] = res["batch_id"]
                st.switch_page("views/actions.py")
            st.caption("The risk outlook shows when and where the risk falls and how much revenue is flagged. The "
                       "action list shows which guests to contact first.")

    money = {"Hotel": st.column_config.TextColumn("Property"),
             "Cancellation probability": st.column_config.ProgressColumn("Probability", format="%.1f%%", min_value=0,
                                                                         max_value=100),
             "Lead time (days)": st.column_config.NumberColumn(format="%d"),
             "Nights": st.column_config.NumberColumn(format="%d")}

    section("All rows", "Filter, then download")
    views = ["All rows", "Likely to cancel", "Likely to stay", "Not scored"]
    if "Correct?" in df:
        views.append("Wrong predictions")
    view = st.segmented_control("Show", views, default="All rows", key="batch_view")
    shown = (df if view in (None, "All rows") else df[df["Correct?"] == "Wrong"] if view == "Wrong predictions"
             else df[df["Prediction"] == view])
    shown_view, shown_money = money_view(
        shown.assign(**{"Cancellation probability": shown["Cancellation probability"] * 100,
                        "Main factor raising risk": shown["Main factor raising risk"].map(convert_euro_text),
                        "Main factor lowering risk": shown["Main factor lowering risk"].map(convert_euro_text)}),
        ["Room rate (EUR)", "Stay value (EUR)"])
    st.dataframe(verdict_colours(shown_view), hide_index=True, width="stretch", height=min(38 + 35 * len(shown), 460),
                 column_config={**money, **shown_money})

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

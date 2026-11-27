"""Prediction history – every saved prediction (find, inspect, export, delete) and how accurate the saved predictions
turned out to be when the real outcome is known. Risk analysis lives on the Risk outlook page."""
import sys as _sys, pathlib as _pl  # make app/common.py importable however the page is launched
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))

import json
from html import escape

import altair as alt
import pandas as pd
import streamlit as st

from common import (CANCEL, INK, base_chart, convert_euro_text, dash, friendly, get_countries, kpis, money_view,
                    page_header, pct, saved_predictions, section, show_chart, verdict_colours)
from hotelsight import batch, history, insights
from hotelsight.contract import MODEL_FIELDS, display_value

countries = get_countries()
ss = st.session_state
XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
SOURCE = {"single": "Single reservation", "batch": "File upload"}
off = {"delta_color": "off", "delta_arrow": "off", "border": True}


def delete_batch():
    n = history.delete_batch(ss.hist_delete_batch)
    ss.hist_notice = f"Deleted {n:,} saved prediction(s)."


def delete_selected(ids):
    n = history.delete_ids(ids)
    ss.hist_notice = f"Deleted {n:,} selected prediction(s)."
    ss.pop("hist_table", None)


def clear_all():
    n = history.clear()
    ss.hist_notice = f"Deleted all {n:,} saved predictions."
    ss.hist_confirm = False


page_header("Records", "Prediction history",
            "Every saved check and upload, newest first, and how accurate the predictions were once the real outcome "
            "was known.")
if ss.get("hist_notice"):
    st.success(ss.pop("hist_notice"), icon=":material/check_circle:")

h, data = saved_predictions()
if h is None:
    st.info("No saved predictions yet. Predictions are saved automatically from **Check a reservation** and "
            "**Upload reservations** (untick *Save to history* to skip it).", icon=":material/history:")
    st.stop()

h["Source"] = h["source"].map(SOURCE)
h["Prediction"] = h["predicted_cancel"].map({1: "Likely to cancel", 0: "Likely to stay"})
h["Actual outcome"] = h["actual_cancelled"].map({1: "Cancelled", 0: "Not cancelled"}).fillna("")
batches = (h.groupby("batch_id").agg(name=("batch_name", "first"), when=("created_at", "min"), n=("id", "size"))
           .sort_values("when", ascending=False))
batch_label = {b: f"{r['name']} · {r['when']:%d %b %Y %H:%M} · {r['n']:,} reservation{'s' * (r['n'] != 1)}"
               for b, r in batches.iterrows()}
ALL, ALL_SOURCES = "All checks and uploads", "All sources"

tab_records, tab_accuracy = st.tabs([":material/history: Saved predictions",
                                     ":material/fact_check: Prediction accuracy"])

# ------------------------------------------------------------------ records
with tab_records:
    with st.container(key="card_filters"):
        c1, c2, c3, c4, c5 = st.columns([1, 1.6, .9, 1, 1.5])
        source = c1.selectbox("Source", [ALL_SOURCES] + list(SOURCE.values()), key="hist_source")
        pick = c2.selectbox("Check or upload", [ALL] + list(batches.index), key="hist_batch",
                            format_func=lambda b: b if b == ALL else batch_label[b])
        outcome = c3.selectbox("Prediction", ["All", "Likely to cancel", "Likely to stay"], key="hist_pred")
        search = c4.text_input("Search reference", key="hist_search", placeholder="e.g. RSV-10234")
        first, last = h["created_at"].min().date(), h["created_at"].max().date()
        dates = c5.date_input("Saved between", (first, last), min_value=first, max_value=last, key="hist_dates",
                              format="DD/MM/YYYY")

    f = h if source == ALL_SOURCES else h[h["Source"] == source]
    if pick != ALL:
        f = f[f["batch_id"] == pick]
    if outcome != "All":
        f = f[f["Prediction"] == outcome]
    if search.strip():
        f = f[f["reference"].fillna("").str.contains(search.strip(), case=False, regex=False)]
    if isinstance(dates, tuple) and len(dates) == 2:
        f = f[(f["created_at"].dt.date >= dates[0]) & (f["created_at"].dt.date <= dates[1])]

    st.caption(f"Showing {len(f):,} of {len(h):,} saved predictions from {f['batch_id'].nunique():,} checks and "
               "uploads. Select a row to see the exact inputs.")
    if f.empty:
        st.warning("No saved predictions match these filters.", icon=":material/filter_alt_off:")
    else:
        table = pd.DataFrame({
            "Saved": f["created_at"], "Reference": dash(f["reference"]), "Property": f["hotel"], "Arrival": f["arrival"],
            "Nights": f["nights"], "Stay value (EUR)": f["stay_value"], "Probability": f["probability"] * 100,
            "Prediction": f["Prediction"], "Actual outcome": f["Actual outcome"],
            "Main factor raising risk": f["top_raising"], "Main factor lowering risk": f["top_lowering"],
            "Source": f["Source"], "Check or upload": f["batch_name"]})
        view, money_cfg = money_view(table.assign(**{
            "Main factor raising risk": table["Main factor raising risk"].map(convert_euro_text),
            "Main factor lowering risk": table["Main factor lowering risk"].map(convert_euro_text)}),
            ["Stay value (EUR)"])
        picked = st.dataframe(verdict_colours(view), hide_index=True, width="stretch",
                              height=min(38 + 35 * len(table), 460), on_select="rerun", selection_mode="multi-row",
                              key="hist_table",
                              column_config={"Saved": st.column_config.DatetimeColumn(format="D MMM YYYY, HH:mm"),
                                             "Nights": st.column_config.NumberColumn(format="%d"), **money_cfg,
                                             "Probability": st.column_config.ProgressColumn(
                                                 format="%.1f%%", min_value=0, max_value=100)})
        rows = picked.selection.rows if picked is not None else []
        c1, c2 = st.columns([3, 1], vertical_alignment="center")
        c1.caption("Tick rows to see their inputs or delete them." if not rows else
                   f"{len(rows):,} row{'s' * (len(rows) != 1)} selected. The inputs of the first one are shown below.")
        c2.button(f"Delete selected ({len(rows):,})", key="hist_delete_selected", icon=":material/delete:",
                  width="stretch", disabled=not rows, on_click=delete_selected,
                  args=(f.iloc[rows]["id"].tolist() if rows else [],))
        if rows:
            rec_row = f.iloc[rows[0]]
            rec = json.loads(rec_row["inputs_json"] or "{}")
            with st.container(key="card_detail"):
                st.html(f'<div class="hs-card-title">Inputs of {escape(rec_row["reference"] or "this prediction")} · '
                        f'{pct(rec_row["probability"])} · {rec_row["Prediction"]}</div>')
                st.dataframe(pd.DataFrame([{"Input": friendly(x), "Value": display_value(x, rec.get(x), countries),
                                            "Model field": x} for x in MODEL_FIELDS]),
                             hide_index=True, width="stretch")
                st.caption(f"Saved {rec_row['created_at']:%d %b %Y %H:%M} · threshold {rec_row['threshold']:.2f} · "
                           f"model SHA-256 {str(rec_row['model_sha256'])[:12]}…")

        export = table.assign(Probability=f["probability"].to_numpy())
        export = pd.concat([export.reset_index(drop=True),
                            pd.DataFrame([json.loads(x or "{}") for x in f["inputs_json"]])
                            .reindex(columns=MODEL_FIELDS)], axis=1)
        d1, d2, _ = st.columns([1, 1, 2])
        d1.download_button("Export shown (Excel)", batch.to_excel({"Prediction history": export}),
                           file_name="HotelSight_prediction_history.xlsx", mime=XLSX, icon=":material/download:",
                           width="stretch", key="hist_dl_xlsx")
        d2.download_button("Export shown (CSV)", export.to_csv(index=False).encode("utf-8-sig"),
                           file_name="HotelSight_prediction_history.csv", mime="text/csv",
                           icon=":material/download:", width="stretch", key="hist_dl_csv")

    with st.container(key="card_hist_delete"):
        st.html('<div class="hs-card-title">Delete saved predictions</div>')
        c1, c2 = st.columns([2, 1], vertical_alignment="bottom")
        c1.selectbox("Delete one check or upload", list(batches.index), format_func=batch_label.get,
                     key="hist_delete_batch")
        c2.button("Delete this upload", on_click=delete_batch, key="hist_delete_btn", width="stretch",
                  icon=":material/delete:")
        c1, c2 = st.columns([2, 1], vertical_alignment="bottom")
        c1.checkbox(f"I understand this permanently deletes all {len(h):,} saved predictions", key="hist_confirm")
        c2.button("Delete all history", on_click=clear_all, key="hist_clear_btn", width="stretch", type="primary",
                  disabled=not ss.get("hist_confirm"), icon=":material/delete_forever:")
        st.caption(f"Deleting is permanent. To delete single rows, tick them in the table above. History file: "
                   f"{history.db_path()}")

# ------------------------------------------------------------------ accuracy against real outcomes
with tab_accuracy:
    known = data.dropna(subset=["actual"])
    check = insights.outcome_check(data)
    if check is None:
        st.info("None of the saved predictions has a recorded outcome yet. Upload a file that includes the real "
                "outcome (for example the 200 sample reservations), or check again after the guests' arrival dates.",
                icon=":material/fact_check:")
    else:
        st.caption("Only saved predictions whose real outcome is known. A reservation saved more than once counts each "
                   "time, so this is a working check. The sealed experiment results are on the Model evaluation page.")
        k = kpis("hist_acc", ["fact_check", "verified", "balance", "notifications_off", "search_off"],
                 {3: "coral", 4: "coral"})
        k[0].metric("With a known outcome", f"{check['known']:,}", "saved predictions", **off)
        k[1].metric("Accuracy", pct(check["accuracy"]), f"{check['tp'] + check['tn']:,} correct", **off)
        k[2].metric("F1-score", f"{check['f1']:.3f}", "cancelled class", **off)
        k[3].metric("False alarms", f"{check['fp']:,}", "flagged but the guest stayed", **off)
        k[4].metric("Missed cancellations", f"{check['fn']:,}", "not flagged but cancelled", **off)

        left, right = st.columns(2, gap="medium")
        with left.container(key="card_acc_month", height="stretch"):
            st.html('<div class="hs-card-title">Predicted and actual cancellations by arrival month</div>')
            k2 = known.dropna(subset=["arrival"])
            km = (k2.assign(month=k2["arrival"].dt.to_period("M").dt.to_timestamp()).groupby("month")
                  .agg(n=("actual", "size"), predicted=("flagged", "mean"), actual=("actual", "mean"),
                       correct=("flagged", lambda s: float((s == k2.loc[s.index, "actual"]).mean())))
                  .reset_index())
            long = pd.concat([km.assign(Series="Predicted to cancel", share=km["predicted"]),
                              km.assign(Series="Actually cancelled", share=km["actual"])])
            show_chart(base_chart(alt.Chart(long).mark_line(point=True, strokeWidth=2).encode(
                x=alt.X("yearmonth(month):O", title="Arrival month",
                        axis=alt.Axis(format="%b %Y", formatType="time", labelOverlap="greedy", labelAngle=0)),
                y=alt.Y("share:Q", title="Share of reservations", axis=alt.Axis(format="%")),
                color=alt.Color("Series:N", scale=alt.Scale(domain=["Predicted to cancel", "Actually cancelled"],
                                                            range=[CANCEL, INK]),
                                legend=alt.Legend(orient="top", title=None)),
                tooltip=[alt.Tooltip("month:T", title="Arrival month", format="%B %Y"), "Series:N",
                         alt.Tooltip("share:Q", title="Share", format=".1%"),
                         alt.Tooltip("n:Q", title="Reservations")]).properties(height=240)))
            st.caption("When the two lines drift apart for several months, the model should be retrained.")
        with right.container(key="card_acc_band", height="stretch"):
            st.html('<div class="hs-card-title">Actual cancellation rate in each risk band</div>')
            kb = known.groupby("risk_band", observed=False).agg(n=("actual", "size"), rate=("actual", "mean"))
            kb = kb.reindex(insights.BAND_LABELS).reset_index().rename(columns={"risk_band": "band", "index": "band"})
            kb = kb.dropna(subset=["rate"])
            bars = alt.Chart(kb).mark_bar(color=CANCEL, cornerRadiusEnd=3, height={"band": .6}).encode(
                y=alt.Y("band:N", sort=insights.BAND_LABELS, title=None),
                x=alt.X("rate:Q", title="Actually cancelled", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
                tooltip=[alt.Tooltip("band:N", title="Risk band"), alt.Tooltip("n:Q", title="Reservations"),
                         alt.Tooltip("rate:Q", title="Actually cancelled", format=".1%")])
            text = bars.mark_text(align="left", dx=4, color=INK, fontSize=12).encode(text=alt.Text("rate:Q", format=".0%"))
            show_chart(base_chart((bars + text).properties(height=180)))
            st.caption("A reliable model shows more real cancellations in each higher band.")

        with st.container(key="card_acc_errors"):
            st.html('<div class="hs-card-title">Where the errors happen</div>')
            err = known.assign(kind=lambda x: (x["flagged"] == 1).map({True: "False alarm", False: "Missed"})
                               .where(x["flagged"] != x["actual"], "Correct"))
            by = (err.groupby("market_segment")["kind"].value_counts().unstack(fill_value=0)
                  .reindex(columns=["Correct", "False alarm", "Missed"], fill_value=0))
            by["Reservations"] = by.sum(axis=1)
            by["Error rate"] = (by["False alarm"] + by["Missed"]) / by["Reservations"] * 100
            by = by.sort_values("Reservations", ascending=False).reset_index().rename(
                columns={"market_segment": "Booking source"})
            st.dataframe(by, hide_index=True, width="stretch", column_config={
                "Error rate": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100)})
            st.caption("False alarm: flagged but the guest stayed. Missed: not flagged but the booking was cancelled.")

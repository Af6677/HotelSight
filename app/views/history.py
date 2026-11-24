"""Prediction history – every saved prediction (single reservations and file uploads): filter, inspect, export, delete."""
import sys as _sys, pathlib as _pl  # make app/common.py importable however the page is launched
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))

import json
from html import escape

import altair as alt
import pandas as pd
import streamlit as st

from common import (OUTCOME, base_chart, convert_euro_text, dash, euros, friendly, get_countries, kpis, money_view,
                    page_header, pct, section, show_chart, verdict_colours)
from hotelsight import batch, history
from hotelsight.contract import MODEL_FIELDS, display_value

countries = get_countries()
ss = st.session_state
XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
SOURCE = {"single": "Single reservation", "batch": "File upload"}


def delete_batch():
    n = history.delete_batch(ss.hist_delete_batch)
    ss.hist_notice = f"Deleted {n:,} saved prediction(s)."


def clear_all():
    n = history.clear()
    ss.hist_notice = f"Deleted all {n:,} saved predictions."
    ss.hist_confirm = False


h = history.load()
page_header("Reservations", "Prediction history",
            "Every saved check and upload, newest first. Select a row to see the exact inputs that were used.")
if ss.get("hist_notice"):
    st.success(ss.pop("hist_notice"), icon=":material/check_circle:")

if h.empty:
    st.info("No saved predictions yet. Predictions are saved automatically from **Check a reservation** and "
            "**Upload reservations** (untick *Save to history* to skip it).", icon=":material/history:")
    st.stop()

h = history.with_stay(h)
h["Source"] = h["source"].map(SOURCE)
h["Prediction"] = h["predicted_cancel"].map({1: "Likely to cancel", 0: "Likely to stay"})
h["Actual outcome"] = h["actual_cancelled"].map({1: "Cancelled", 0: "Not cancelled"}).fillna("")
batches = (h.groupby("batch_id").agg(name=("batch_name", "first"), when=("created_at", "min"), n=("id", "size"))
           .sort_values("when", ascending=False))
batch_label = {b: f"{r['name']} · {r['when']:%d %b %Y %H:%M} · {r['n']:,} reservation{'s' * (r['n'] != 1)}"
               for b, r in batches.iterrows()}
ALL = "All checks and uploads"
ALL_SOURCES = "All sources"

# ------------------------------------------------------------------ filters
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

k = kpis("hist", ["list_alt", "folder_open", "event_busy", "payments"], {2: "coral", 3: "coral"})
k[0].metric("Predictions shown", f"{len(f):,}", f"of {len(h):,} saved", delta_color="off", delta_arrow="off",
            border=True)
k[1].metric("Checks and uploads", f"{f['batch_id'].nunique():,}", "in this view", delta_color="off",
            delta_arrow="off", border=True)
n_cancel = int(f["predicted_cancel"].sum())
k[2].metric("Likely to cancel", f"{n_cancel:,}", f"{pct(n_cancel / max(len(f), 1))} of shown", delta_color="off",
            delta_arrow="off", border=True)
k[3].metric("Revenue flagged", euros(f.loc[f["predicted_cancel"] == 1, "stay_value"].sum()),
            "room rate × nights", delta_color="off", delta_arrow="off", border=True,
            help="Room rate × nights for the shown reservations flagged as likely to cancel, shown in the "
                 "currency chosen in the sidebar. Downloaded files stay in EUR.")

if f.empty:
    st.warning("No saved predictions match these filters.", icon=":material/filter_alt_off:")
else:
    left, right = st.columns(2, gap="medium")
    colour = alt.Color("Prediction:N", scale=OUTCOME, legend=alt.Legend(orient="top", title=None))
    with left.container(key="card_hist_dist", height="stretch"):
        st.html('<div class="hs-card-title">Spread of cancellation probabilities</div>')
        show_chart(base_chart(alt.Chart(f).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
            x=alt.X("probability:Q", bin=alt.Bin(extent=[0, 1], step=0.05), axis=alt.Axis(format="%", values=[0, .2, .4, .6, .8, 1]),
                    title="Cancellation probability"),
            y=alt.Y("count():Q", title="Predictions"), color=colour,
            tooltip=[alt.Tooltip("count():Q", title="Predictions"), "Prediction:N"]).properties(height=240)))
    with right.container(key="card_hist_hotel", height="stretch"):
        st.html('<div class="hs-card-title">Predictions by property</div>')
        show_chart(base_chart(alt.Chart(f.assign(hotel=f["hotel"].fillna("Unknown"))).mark_bar(
            stroke="#fff", strokeWidth=1, height={"band": 0.5}).encode(
            y=alt.Y("hotel:N", title=None), x=alt.X("count():Q", title="Predictions", stack="zero"), color=colour,
            tooltip=["hotel:N", "Prediction:N", alt.Tooltip("count():Q", title="Predictions")]).properties(height=240)))

    # ------------------------------------------------------------------ table + details
    section("Saved predictions", "Select a row to see the exact inputs")
    table = pd.DataFrame({
        "Saved": f["created_at"], "Reference": dash(f["reference"]), "Property": f["hotel"], "Arrival": f["arrival"],
        "Nights": f["nights"], "Stay value (EUR)": f["stay_value"], "Probability": f["probability"] * 100,
        "Prediction": f["Prediction"], "Actual outcome": f["Actual outcome"],
        "Main factor raising risk": f["top_raising"], "Main factor lowering risk": f["top_lowering"],
        "Source": f["Source"], "Check or upload": f["batch_name"]})
    view, money_cfg = money_view(table.assign(**{
        "Main factor raising risk": table["Main factor raising risk"].map(convert_euro_text),
        "Main factor lowering risk": table["Main factor lowering risk"].map(convert_euro_text)}), ["Stay value (EUR)"])
    picked = st.dataframe(verdict_colours(view), hide_index=True, width="stretch", height=min(38 + 35 * len(table), 420),
                          on_select="rerun", selection_mode="single-row", key="hist_table",
                          column_config={"Saved": st.column_config.DatetimeColumn(format="D MMM YYYY, HH:mm"),
                                         "Nights": st.column_config.NumberColumn(format="%d"),
                                         **money_cfg,
                                         "Probability": st.column_config.ProgressColumn(
                                             format="%.1f%%", min_value=0, max_value=100)})
    rows = picked.selection.rows if picked is not None else []
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
                        pd.DataFrame([json.loads(x or "{}") for x in f["inputs_json"]]).reindex(columns=MODEL_FIELDS)],
                       axis=1)
    d1, d2, _ = st.columns([1, 1, 2])
    d1.download_button("Export shown (Excel)", batch.to_excel({"Prediction history": export}),
                       file_name="HotelSight_prediction_history.xlsx", mime=XLSX, icon=":material/download:",
                       width="stretch", key="hist_dl_xlsx")
    d2.download_button("Export shown (CSV)", export.to_csv(index=False).encode("utf-8-sig"),
                       file_name="HotelSight_prediction_history.csv", mime="text/csv", icon=":material/download:",
                       width="stretch", key="hist_dl_csv")

# ------------------------------------------------------------------ manage
with st.expander("Manage saved history", icon=":material/delete:"):
    c1, c2 = st.columns([2, 1], vertical_alignment="bottom")
    c1.selectbox("Delete one check or upload", list(batches.index), format_func=batch_label.get,
                 key="hist_delete_batch")
    c2.button("Delete it", on_click=delete_batch, key="hist_delete_btn", width="stretch",
              icon=":material/delete:")
    st.divider()
    c1, c2 = st.columns([2, 1], vertical_alignment="bottom")
    c1.checkbox(f"I understand this permanently deletes all {len(h):,} saved predictions", key="hist_confirm")
    c2.button("Delete all history", on_click=clear_all, key="hist_clear_btn", width="stretch", type="primary",
              disabled=not ss.get("hist_confirm"), icon=":material/delete_forever:")
    st.caption(f"History file: {history.db_path()}")

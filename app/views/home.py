"""Overview – what needs attention, how reliable the model is, and where to start."""
import sys as _sys, pathlib as _pl  # make app/common.py importable however the page is launched
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))

from datetime import date
from html import escape

import pandas as pd
import streamlit as st

from common import convert_euro_text, dash, euros, friendly, icon, kpis, load, money_view, nav_link, pct, section
from hotelsight import history

split = load("plan")["splits"]["retained"]
t = load("test_metrics")
rep, one = t["repeated"], t["one_copy"]
off = {"delta_color": "off", "delta_arrow": "off", "border": True}

with st.container(key="hero"):
    st.html('<div class="hs-hero"><h1>Spot the reservations most likely to cancel</h1>'
            '<p>Score reservations one at a time or in bulk, see why each was flagged, and keep a record of every '
            'prediction.</p></div>')
    with st.container(horizontal=True, gap="small"):
        with st.container(key="hero_primary", width="content"):
            nav_link("views/predict.py", "Check a reservation")
        nav_link("views/batch.py", "Upload reservations")

# ------------------------------------------------------------------ saved predictions
h = history.load()
if h.empty:
    section("Saved predictions")
    st.html(f'<div class="hs-note">{icon("info")}<span>No predictions saved yet. Every reservation check and upload is '
            'saved automatically and summarised here.</span></div>')
    flagged = h
else:
    h = history.with_stay(h)
    flagged = h[h["predicted_cancel"] == 1]
    flagged = flagged.assign(when=pd.to_datetime(flagged["arrival"], errors="coerce", format="%Y-%m-%d"))
    upcoming = flagged[flagged["when"] >= pd.Timestamp(date.today())].sort_values("when")
    k = kpis("home", ["bookmark", "event_busy", "payments", "event_upcoming"], {1: "coral", 2: "coral", 3: "coral"})
    k[0].metric("Saved predictions", f"{len(h):,}", f"{h['batch_id'].nunique():,} checks and uploads", **off)
    k[1].metric("Likely to cancel", f"{len(flagged):,}", f"{pct(len(flagged) / len(h))} of saved", **off)
    k[2].metric("Revenue flagged", euros(flagged["stay_value"].sum()), "room rate × nights", **off,
                help="Sum of room rate × nights for the saved reservations flagged as likely to cancel. Shown in the "
                     "currency chosen in the sidebar.")
    k[3].metric("Upcoming at risk", f"{len(upcoming):,}", "flagged, arriving from today", **off)

left, right = st.columns([1.7, 1], gap="medium")
with left.container(key="card_home_recent", height="stretch"):
    if len(flagged):
        recent = upcoming.head(7) if len(upcoming) else flagged.head(7)
        st.html('<div class="hs-card-title">' + ("Upcoming arrivals at risk" if len(upcoming) else
                                                 "Recently flagged reservations") + '</div>')
        view, money_cfg = money_view(pd.DataFrame({
            "Reference": dash(recent["reference"]), "Property": recent["hotel"], "Arrival": recent["arrival"],
            "Stay value": recent["stay_value"], "Risk": recent["probability"] * 100,
            "Main reason": recent["top_raising"].map(convert_euro_text)}), ["Stay value"])
        st.dataframe(view, hide_index=True, width="stretch",
                     column_config={**money_cfg,
                                    "Risk": st.column_config.ProgressColumn(format="%.2f%%", min_value=0, max_value=100,
                                                                            width=132)})
        nav_link("views/history.py", "Open the prediction history")
    else:
        st.html('<div class="hs-card-title">Flagged reservations</div>')
        st.caption("Reservations flagged as likely to cancel will be listed here.")

with right.container(key="card_home_model", height="stretch"):
    bars = [("Accuracy", rep["accuracy"], pct(rep["accuracy"])), ("Cancellations caught", rep["recall"], pct(rep["recall"])),
            ("Flags that were right", rep["precision"], pct(rep["precision"])),
            ("Ranking quality (AUC)", rep["roc_auc"], f"{rep['roc_auc']:.3f}")]
    st.html(f'<div class="hs-card-title">How reliable is the model?</div><div class="hs-bars">'
            + "".join(f'<div class="hs-bar"><span>{n}</span><div class="track"><div class="fill" style="width:{100 * v:.1f}%">'
                      f'</div></div><b>{s}</b></div>' for n, v, s in bars) + '</div>')
    st.caption(f"Tested on {rep['rows']:,} past reservations it never saw. F1 is {pct(rep['f1'])}, or {pct(one['f1'])} when "
               "each repeated record is counted once.")
    nav_link("views/performance.py", "See the full test results")

left, right = st.columns([1.7, 1], gap="medium")
with left.container(key="card_home_drivers", height="stretch"):
    imp = load("global_shap").sort_values("mean_abs_shap_repeated", ascending=False).head(7)
    top = float(imp["mean_abs_shap_repeated"].max())
    rows = "".join(f'<div class="hs-bar"><span>{escape(friendly(r.input))}</span><div class="track"><div class="fill" '
                   f'style="width:{100 * r.mean_abs_shap_repeated / top:.1f}%"></div></div>'
                   f'<b>{r.mean_abs_shap_repeated:.2f}</b></div>' for r in imp.itertuples())
    st.html(f'<div class="hs-card-title">What the model pays most attention to</div><div class="hs-bars">{rows}</div>')
    st.caption("Average influence on the test set (mean |SHAP|, log-odds). These are associations learned from "
               "historical records, not causes.")
with right.container(key="card_home_how", height="stretch"):
    st.html('<div class="hs-card-title">How it works</div><div class="hs-steps">'
            '<div><span><b>Enter or upload reservations</b>The same 24 details the model was trained on.</span></div>'
            '<div><span><b>Score the risk</b>A probability of 0.50 or more is flagged as likely to cancel.</span></div>'
            '<div><span><b>Explain and record</b>See what raised or lowered the score; every prediction is kept.</span></div>'
            f'</div><div class="hs-note">{icon("info")}<span>Research prototype trained on two Portuguese hotels '
            f'(2015–2017, {split["train"]["rows"]:,} bookings). Accuracy on new or Malaysian reservations is unknown, so '
            'use the scores to prioritise follow-up.</span></div>')

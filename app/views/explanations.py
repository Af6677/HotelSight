"""Page 4 – global SHAP feature importance of the deployed model."""
import sys as _sys, pathlib as _pl  # make app/common.py importable however the page is launched
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))

import altair as alt
import streamlit as st

from common import (INK, STAY, base_chart, diverging_bars, diverging_legend, friendly, get_countries, kpis, load,
                    page_header,
                    show_chart)
from hotelsight.contract import CATEGORY_LABELS

summ = load("shap_summary")
page_header("Model evidence", "What drives predictions",
            f"SHAP values from the deployed model on {summ['rows_repeated']:,} held-out test rows, in log-odds of "
            "cancellation. These are associations, not causes.")

imp = load("global_shap")
pop = st.segmented_control("Test population", ["Repeated records", "One copy per record"], default="Repeated records")
col = "mean_abs_shap_repeated" if pop != "One copy per record" else "mean_abs_shap_one_copy"
df = imp[["input", col]].rename(columns={col: "Mean |SHAP|"}).sort_values("Mean |SHAP|", ascending=False)
df["Input"] = df["input"].map(friendly)
df["label"] = df["Mean |SHAP|"].map(lambda v: f"{v:.2f}")
order = list(df["Input"])

c = kpis("drivers", ["looks_one", "looks_two", "looks_3"])
for i, (_, r) in enumerate(df.head(3).iterrows()):
    c[i].metric(f"#{i + 1} driver", r["Input"].split(" (")[0], f"{r['Mean |SHAP|']:.2f} log-odds on average", delta_color="off",
                delta_arrow="off", border=True)

left, right = st.columns([1, 1], gap="medium")
with left.container(key="card_global", height="stretch"):
    st.html('<div class="hs-card-title">Average influence of each input</div>')
    bars = alt.Chart(df).mark_bar(cornerRadiusEnd=4, color=STAY, height={"band": 0.66}).encode(
        y=alt.Y("Input:N", sort=order, title=None),
        x=alt.X("Mean |SHAP|:Q", title="Average absolute contribution (log-odds)",
                scale=alt.Scale(domain=[0, 1.15 * float(df["Mean |SHAP|"].max())], nice=False)),   # room for labels
        tooltip=["Input", alt.Tooltip("Mean |SHAP|:Q", format=".3f")])
    text = alt.Chart(df).mark_text(align="left", dx=5, color=INK, fontSize=12).encode(
        y=alt.Y("Input:N", sort=order), x="Mean |SHAP|:Q", text="label:N")
    show_chart(base_chart((bars + text).properties(height=alt.Step(26))))
    st.caption("Longer bar = the input moves predictions more, on average, in either direction. It does not say whether "
               "it raises or lowers risk; use the explorer for that.")

with right.container(key="card_byvalue", height="stretch"):
    st.html('<div class="hs-card-title">How each input\'s values push the score</div>')
    by = load("global_shap_by_value")
    choice = st.selectbox("Input", list(df["input"]), format_func=friendly)
    sub = by[by["input"] == choice].sort_values("order").copy()
    countries = get_countries()

    def show(v):
        if choice in ("has_travel_agent", "company_booking", "is_repeated_guest"):
            return {"1.0": "Yes", "0.0": "No"}.get(v, v)
        if choice == "country":
            return f"{countries.get(v, v)} ({v})"
        return CATEGORY_LABELS.get(choice, {}).get(v, v)

    sub["Value"] = sub["value"].map(show)
    sub["Mean SHAP"] = sub["mean_shap"]
    sub["Test rows"] = sub["rows"]
    diverging_legend()
    show_chart(diverging_bars(sub, "Value", "Mean SHAP", "Average contribution (log-odds)",
                                   ["Value:N", alt.Tooltip("Mean SHAP:Q", format="+.3f"),
                                    alt.Tooltip("Test rows:Q", format=",")]))
    st.caption("Categories show the 12 most frequent values; numbers are split into equal-count bands. "
               "These are associations learned from historical records (for example, parking requests go with "
               "lower predicted risk). They are not evidence that changing an input would change the outcome.")

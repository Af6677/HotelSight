"""Page 2 – four-model comparison on the validation set (the deployed experiment arm only)."""
import sys as _sys, pathlib as _pl  # make app/common.py importable however the page is launched
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))

import altair as alt
import pandas as pd
import streamlit as st

from common import FADED, INK, SECOND, STAY, base_chart, icon, kpis, load, page_header, pct, section, show_chart

val = load("validation_comparison")
sel = load("selected")

page_header("Model evidence", "Model comparison",
            "Four models were trained on the same data, each with three fixed settings. The one with the highest "
            "<b>validation F1</b> was selected before the test set was scored.")

best = val[val["Model"] == sel["model"]].iloc[0]
runner = val[val["Model"] != sel["model"]].sort_values("F1-score", ascending=False).iloc[0]
c = kpis("compare", ["emoji_events", "target", "show_chart", "radar"])
c[0].metric("Selected model", sel["model"], f"configuration {sel['configuration']}", delta_color="off",
            delta_arrow="off", border=True)
c[1].metric("Validation F1", pct(best["F1-score"], 2), f"{100 * (best['F1-score'] - runner['F1-score']):+.2f} pts",
            help=f"Lead over the runner-up, {runner['Model']}.", border=True)
c[2].metric("Validation ROC AUC", f"{best['ROC AUC']:.4f}", "ranking quality", delta_color="off", delta_arrow="off",
            border=True)
c[3].metric("Validation recall", pct(best["Recall"], 2), f"precision {pct(best['Precision'], 2)}", delta_color="off",
            delta_arrow="off", border=True)

left, right = st.columns([1.15, 1], gap="medium")
with left.container(key="card_f1", height="stretch"):
    st.html('<div class="hs-card-title">Validation F1-score – selection metric</div>')
    df = val[["Model", "F1-score"]].copy()
    df["Selected"] = df["Model"].map(lambda m: "Selected" if m == sel["model"] else "Not selected")
    df["label"] = df["F1-score"].map(lambda v: f"{100 * v:.2f}%")
    order = list(df.sort_values("F1-score", ascending=False)["Model"])
    bars = alt.Chart(df).mark_bar(cornerRadiusEnd=4, height={"band": 0.65}).encode(
        y=alt.Y("Model:N", sort=order, title=None),
        x=alt.X("F1-score:Q", title="Validation F1-score", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
        color=alt.Color("Selected:N", scale=alt.Scale(domain=["Selected", "Not selected"], range=[STAY, FADED]),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=["Model", alt.Tooltip("F1-score:Q", format=".4f")])
    text = alt.Chart(df).mark_text(align="left", dx=6, color=INK, fontSize=13).encode(
        y=alt.Y("Model:N", sort=order), x="F1-score:Q", text="label:N")
    show_chart(base_chart((bars + text).properties(height=alt.Step(46))))

with right.container(key="card_gap", height="stretch"):
    st.html('<div class="hs-card-title">Training vs validation F1 – overfitting gap</div>')
    gap = pd.concat([val[["Model", "Train F1"]].rename(columns={"Train F1": "F1"}).assign(Set="Train"),
                     val[["Model", "F1-score"]].rename(columns={"F1-score": "F1"}).assign(Set="Validation")])
    rule = alt.Chart(val).mark_rule(color="#CBD5E1", strokeWidth=2).encode(
        y=alt.Y("Model:N", sort=order, title=None), x=alt.X("F1-score:Q", scale=alt.Scale(domain=[0.6, 1])), x2="Train F1:Q")
    dots = alt.Chart(gap).mark_circle(size=170, opacity=1).encode(
        y=alt.Y("Model:N", sort=order, title=None),
        x=alt.X("F1:Q", title="F1-score", scale=alt.Scale(domain=[0.6, 1]), axis=alt.Axis(format="%")),
        color=alt.Color("Set:N", scale=alt.Scale(domain=["Train", "Validation"], range=[SECOND, STAY]),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=["Model", "Set", alt.Tooltip("F1:Q", format=".4f")])
    show_chart(base_chart((rule + dots).properties(height=alt.Step(46))))
    st.caption("XGBoost and Random Forest fit the training data much more closely than they generalise "
               "(train F1 0.95 vs validation 0.80).")

section("All validation metrics", "Best configuration of each model, every copy counted")
ranked = val.drop(columns=["Settings"]).sort_values("F1-score", ascending=False).reset_index(drop=True)
st.dataframe(
    ranked.style.apply(lambda r: ["background-color:#E8EFFC;font-weight:600" if r["Model"] == sel["model"] else ""] * len(r),
                       axis=1),
    hide_index=True, width="stretch",
    column_config={c: st.column_config.NumberColumn(c, format="%.4f") for c in
                   ["F1-score", "Precision", "Recall", "Accuracy", "ROC AUC", "Train F1"]})
st.caption(f"Highest F1 first; the selected model, {sel['model']}, is highlighted. Validation set: 17,881 rows (repeated records kept), 37.1% cancelled. Train F1 is shown to make the "
           "gap between training and validation visible.")

section("One copy per record", "Same models, each identical record counted once")
st.html(f'<div class="hs-note">{icon("info")}<span>The dataset has 31,980 rows that are exact copies of another row, and it '
        'has no booking ID to show whether they are separate bookings. The table above counts every row, so each copy '
        'counts again (<b>repeated records</b>). The table below counts each identical record once '
        '(<b>one copy</b>). The models are the same, only the counting changes. Scores are lower here because the '
        'repeated copies are mostly easy to predict.</span></div>')
one = load("validation_comparison_one_copy").sort_values("F1-score", ascending=False).reset_index(drop=True)
st.dataframe(
    one.style.apply(lambda r: ["background-color:#E8EFFC;font-weight:600" if r["Model"] == sel["model"] else ""] * len(r),
                    axis=1),
    hide_index=True, width="stretch",
    column_config={c: st.column_config.NumberColumn(c, format="%.4f") for c in
                   ["F1-score", "Precision", "Recall", "Accuracy", "ROC AUC"]})
st.caption(f"Validation set: 13,084 rows (one copy per record), 27.5% cancelled. The selected model, {sel['model']}, "
           "is highlighted.")

with st.expander("All 12 configurations tried in this arm", icon=":material/tune:"):
    st.dataframe(load("trials"), hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(c, format="%.4f")
                                for c in ["Validation F1", "Validation AUC", "Train F1"]})
    st.caption("Tuning method: a limited manual comparison of three fixed configurations per model, chosen in advance "
               "(48 fits across the experiment's four arms). It does not establish each algorithm's best possible "
               "performance. Only the selected XGBoost weights are shipped; the other models can be regenerated with "
               "experiment/reproduce.py.")

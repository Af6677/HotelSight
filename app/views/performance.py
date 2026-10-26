"""Page 3 – the selected model's sealed test results, plus the sensitivity experiments."""
import sys as _sys, pathlib as _pl  # make app/common.py importable however the page is launched
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))

import altair as alt
import pandas as pd
import streamlit as st

from common import BLUE, INK, NAVY, SEQ, base_chart, load, page_header, pct, section

t = load("test_metrics")
page_header("Model evidence · sealed test set", "Test performance of the selected model",
            f"{t['model']} was frozen on validation F1 before any test prediction was scored. The same test rows are "
            "reported in two ways, because the dataset contains repeated identical records whose meaning cannot be "
            "verified (no booking ID).",
            chips=["Scored once, after selection", f"{t['repeated']['rows']:,} repeated-record rows",
                   f"{t['one_copy']['rows']:,} one-copy rows", "Threshold 0.50"], icon_name="fact_check")


def confusion(cm):
    labels = ["Not cancelled", "Cancelled"]
    df = pd.DataFrame([{"Actual": labels[i], "Predicted": labels[j], "Bookings": cm[i][j]} for i in range(2) for j in range(2)])
    total = df.Bookings.sum()
    df["text"] = df.Bookings.map(lambda v: f"{v:,}\n({100 * v / total:.1f}%)")
    cut = 0.45 * df.Bookings.max()
    heat = alt.Chart(df).mark_rect(cornerRadius=6, stroke="white", strokeWidth=3).encode(
        x=alt.X("Predicted:N", sort=labels, axis=alt.Axis(orient="top", labelAngle=0, title="Predicted")),
        y=alt.Y("Actual:N", sort=labels, title="Actual"),
        color=alt.Color("Bookings:Q", scale=alt.Scale(range=SEQ[:6]), legend=None),
        tooltip=["Actual", "Predicted", alt.Tooltip("Bookings:Q", format=",")])
    text = alt.Chart(df).mark_text(fontSize=15, fontWeight=600, lineBreak="\n").encode(
        x=alt.X("Predicted:N", sort=labels), y=alt.Y("Actual:N", sort=labels), text="text:N",
        color=alt.condition(alt.datum.Bookings > cut, alt.value("white"), alt.value(INK)))
    return base_chart((heat + text).properties(height=230))


def block(m, key, name, rows_note):
    with st.container(key=key, height="stretch"):
        st.html(f'<div class="hs-card-title">{name}</div>')
        st.caption(rows_note)
        st.altair_chart(confusion(m["confusion_matrix"]), width="stretch")
        tn, fp = m["confusion_matrix"][0]; fn, tp = m["confusion_matrix"][1]
        st.caption(f"Caught {tp:,} of {tp + fn:,} cancellations; missed {fn:,}. Wrongly flagged {fp:,} of {tn + fp:,} "
                   "bookings that were not cancelled.")


rep, one = t["repeated"], t["one_copy"]
c = st.columns(4)
off = {"delta_color": "off", "delta_arrow": "off", "border": True}
c[0].metric("F1-score", pct(rep["f1"], 2), "repeated records", help="Repeated-record test population", **off)
c[1].metric("Accuracy", pct(rep["accuracy"], 2), "repeated records", help="Repeated-record test population", **off)
c[2].metric("F1-score", pct(one["f1"], 2), "one copy per record", help="One copy per complete record", **off)
c[3].metric("Accuracy", pct(one["accuracy"], 2), "one copy per record", help="One copy per complete record", **off)

section("Confusion matrices", "Decision threshold 0.50")
left, right = st.columns(2, gap="medium")
with left:
    block(rep, "card_cm_rep", "Repeated-record test population",
          f"{rep['rows']:,} rows: every repeated copy counts (the deployed arm's evaluation population).")
with right:
    block(one, "card_cm_one", "One copy per complete record",
          f"{one['rows']:,} rows: the same test rows with each identical record counted once.")

st.info("Repeated records are not proven to be separate bookings, and one-copy records are not proven to be unique "
        "guests. Neither population is 'the true' score; report both.", icon=":material/info:")

section("All test metrics")
names = [("f1", "F1-score"), ("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"), ("roc_auc", "ROC AUC")]
table = pd.DataFrame({"Metric": [n for _, n in names],
                      f"Repeated records ({rep['rows']:,} rows)": [rep[k] for k, _ in names],
                      f"One copy per record ({one['rows']:,} rows)": [one[k] for k, _ in names]})
st.dataframe(table, hide_index=True, width="stretch",
             column_config={col: st.column_config.ProgressColumn(col, format="%.4f", min_value=0, max_value=1)
                            for col in table.columns[1:]})

section("Sensitivity experiments", "Same experiment, same test rows")
sens = load("sensitivity")
with st.container(key="card_sens"):
    long = pd.concat([sens[["Experiment arm", "Test F1 (repeated)"]].rename(columns={"Test F1 (repeated)": "F1"})
                      .assign(Population="Repeated records"),
                      sens[["Experiment arm", "Test F1 (one copy)"]].rename(columns={"Test F1 (one copy)": "F1"})
                      .assign(Population="One copy per record")])
    long["label"] = long["F1"].map(lambda v: f"{100 * v:.1f}%")
    arms = list(sens["Experiment arm"])
    enc = dict(y=alt.Y("Experiment arm:N", sort=arms, title=None),
               yOffset=alt.YOffset("Population:N", sort=["Repeated records", "One copy per record"]))
    bars = alt.Chart(long).mark_bar(cornerRadiusEnd=4).encode(
        x=alt.X("F1:Q", title="Test F1-score", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
        color=alt.Color("Population:N", scale=alt.Scale(domain=["Repeated records", "One copy per record"],
                                                        range=[BLUE, NAVY]), legend=alt.Legend(orient="top", title=None)),
        tooltip=["Experiment arm", "Population", alt.Tooltip("F1:Q", format=".4f")], **enc)
    text = alt.Chart(long).mark_text(align="left", dx=5, color=INK, fontSize=12).encode(
        x="F1:Q", text="label:N", **enc)
    st.altair_chart(base_chart((bars + text).properties(height=alt.Step(26))), width="stretch")
    with st.expander("Full sensitivity table", icon=":material/table_view:"):
        st.dataframe(sens, hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(c, format="%.4f") for c in sens.columns if c not in
                                    ("Experiment arm", "Selected model")})
st.caption("The 20-input arms remove ADR, deposit type, parking and special requests, whose values are not verified "
           "as known at booking confirmation. Dropping them lowers repeated-record test F1 from 81.0% to 73.5%. "
           "This shows how much of the result depends on these timing-uncertain inputs.")

with st.expander("Paired bootstrap of F1 differences (10,000 replicates)", icon=":material/bar_chart:"):
    st.dataframe(load("bootstrap"), hide_index=True, width="stretch",
                 column_config={"F1 difference (points)": st.column_config.NumberColumn(format="%+.2f")})
    st.caption("Exploratory and conditional on these fitted models. An interval that includes 0 does not show a clear "
               "difference; it does not prove the two are equal. Keeping repeated records gave a small gain on the "
               "repeated-record population but no clear gain on the one-copy population.")

st.caption("Test-set exposure: this dataset and earlier test scores were inspected during development (including "
           "earlier runs not used here). Within this experiment, test outcomes were scored once, after selection was "
           "frozen. The result is exploratory, not an independent external validation.")

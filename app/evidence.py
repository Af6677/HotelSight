"""Model evidence shown on the Model evaluation page: model selection, sealed test results and sensitivity."""
import altair as alt
import pandas as pd
import streamlit as st

from common import (FADED, INK, SECOND, SEQ, STAY, base_chart, icon, kpis, load, pct, section, show_chart)


def confusion(cm):
    labels = ["Not cancelled", "Cancelled"]
    df = pd.DataFrame([{"Actual": labels[i], "Predicted": labels[j], "Bookings": cm[i][j]} for i in range(2) for j in range(2)])
    total = df.Bookings.sum()
    df["text"] = df.Bookings.map(lambda v: f"{v:,}\n({100 * v / total:.1f}%)")
    cut = 0.6 * df.Bookings.max()              # white text only on the dark cells
    heat = alt.Chart(df).mark_rect(cornerRadius=4, stroke="white", strokeWidth=2).encode(
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
        show_chart(confusion(m["confusion_matrix"]))
        tn, fp = m["confusion_matrix"][0]; fn, tp = m["confusion_matrix"][1]
        st.caption(f"Caught {tp:,} of {tp + fn:,} cancellations; missed {fn:,}. Wrongly flagged {fp:,} of {tn + fp:,} "
                   "bookings that were not cancelled.")




def render_selection():
    """Four models, three fixed configurations each, chosen on validation F1 (formerly the Model comparison page)."""
    val = load("validation_comparison")
    sel = load("selected")
    st.caption("Four models were trained on the same data, each with three fixed settings. The one with the highest "
               "validation F1 was selected before the test set was scored.")
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


def render_test():
    """The selected model's sealed test results (formerly the Test performance page)."""
    t = load("test_metrics")
    st.caption(f"{t['model']} was frozen before any test prediction was scored. The same test rows are reported in "
               "two ways, because repeated identical records in the dataset cannot be verified (no booking ID). The "
               "one-copy result is the more cautious estimate.")
    rep, one = t["repeated"], t["one_copy"]
    c = kpis("perf", ["target", "check_circle", "target", "check_circle"], {2: "slate", 3: "slate"})
    off = {"delta_color": "off", "delta_arrow": "off", "border": True}
    c[0].metric("F1-score", pct(one["f1"], 2), "one copy per record", help="One copy per complete record", **off)
    c[1].metric("Accuracy", pct(one["accuracy"], 2), "one copy per record", help="One copy per complete record", **off)
    c[2].metric("F1-score", pct(rep["f1"], 2), "repeated records", help="Repeated-record test population", **off)
    c[3].metric("Accuracy", pct(rep["accuracy"], 2), "repeated records", help="Repeated-record test population", **off)

    section("Confusion matrices", "Decision threshold 0.50")
    left, right = st.columns(2, gap="medium")
    with left:
        block(one, "card_cm_one", "One copy per complete record",
              f"{one['rows']:,} rows, each identical record counted once.")
    with right:
        block(rep, "card_cm_rep", "Repeated-record test population",
              f"{rep['rows']:,} rows, every repeated copy counted.")

    st.info("Repeated records are not proven to be separate bookings, and one-copy records are not proven to be unique "
            "guests. Neither population is 'the true' score; report both.", icon=":material/info:")

    section("All test metrics")
    names = [("f1", "F1-score"), ("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"), ("roc_auc", "ROC AUC")]
    table = pd.DataFrame({"Metric": [n for _, n in names],
                          f"One copy per record ({one['rows']:,} rows)": [one[k] for k, _ in names],
                          f"Repeated records ({rep['rows']:,} rows)": [rep[k] for k, _ in names]})
    st.dataframe(table, hide_index=True, width="stretch",
                 column_config={col: st.column_config.ProgressColumn(col, format="%.4f", min_value=0, max_value=1)
                                for col in table.columns[1:]})


def render_sensitivity():
    """Sensitivity experiments: repeated records kept or removed, 24 or 20 inputs."""
    section("Sensitivity experiments", "Same experiment, same test rows")
    sens = load("sensitivity")
    with st.container(key="card_sens"):
        long = pd.concat([sens[["Experiment arm", "Test F1 (repeated)"]].rename(columns={"Test F1 (repeated)": "F1"})
                          .assign(Population="Repeated records"),
                          sens[["Experiment arm", "Test F1 (one copy)"]].rename(columns={"Test F1 (one copy)": "F1"})
                          .assign(Population="One copy per record")])
        long["label"] = long["F1"].map(lambda v: f"{100 * v:.1f}%")
        arms = list(sens["Experiment arm"])
        enc = dict(y=alt.Y("Experiment arm:N", sort=arms, title=None, axis=alt.Axis(labelLimit=340)),
                   yOffset=alt.YOffset("Population:N", sort=["Repeated records", "One copy per record"]))
        bars = alt.Chart(long).mark_bar(cornerRadiusEnd=4).encode(
            x=alt.X("F1:Q", title="Test F1-score", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
            color=alt.Color("Population:N", scale=alt.Scale(domain=["Repeated records", "One copy per record"],
                                                            range=[STAY, SECOND]), legend=alt.Legend(orient="top", title=None)),
            tooltip=["Experiment arm", "Population", alt.Tooltip("F1:Q", format=".4f")], **enc)
        text = alt.Chart(long).mark_text(align="left", dx=5, color=INK, fontSize=12).encode(
            x="F1:Q", text="label:N", **enc)
        show_chart(base_chart((bars + text).properties(height=alt.Step(26))))
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

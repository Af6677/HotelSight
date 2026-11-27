"""Hotel performance – how the two hotels performed in the historical bookings (descriptive, looks back)."""
import sys as _sys, pathlib as _pl  # make app/common.py importable however the page is launched
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))

import altair as alt
import pandas as pd
import streamlit as st

from common import (CANCEL, DATA_CURRENT, DATA_HISTORY, INK, SECOND, SEQ, STAY, base_chart, booking_history,
                    currency_code, euros, get_countries, icon, kpis, nav_link, page_header, pct, saved_predictions,
                    section, show_chart, to_display)
from hotelsight import analytics
from hotelsight.insights import LEAD_LABELS

ALL = "All"
MONTH_AXIS = alt.Axis(format="%b %Y", formatType="time", labelOverlap="greedy", labelAngle=0, labelSeparation=12)


def bookings() -> pd.DataFrame:
    return booking_history()


@st.cache_data(show_spinner=False)
def nights_sold() -> pd.DataFrame:
    return analytics.rooms_sold_by_night(bookings())


page_header("Analyse", "Hotel performance",
            "Revenue, demand and cancellations: the real results in the booking history, or the expected results of "
            "the reservations you have checked or uploaded.")

mode = st.segmented_control("Data", [DATA_HISTORY, DATA_CURRENT], default=DATA_HISTORY, key="perf_mode",
                            label_visibility="collapsed",
                            help="Booking history dataset: real results of past bookings. Current predicted bookings: "
                                 "expected results of the reservations you checked or uploaded.") or DATA_HISTORY
expected = mode == DATA_CURRENT


def L(actual_text, expected_text):
    """The label for the current data: real results or expected results."""
    return expected_text if expected else actual_text


if expected:
    saved, _ = saved_predictions()
    d_all = analytics.from_predictions(saved) if saved is not None else pd.DataFrame()
    if d_all.empty:
        st.info("No saved predictions yet. Check a reservation or upload a file, and their expected results appear "
                "here.", icon=":material/insights:")
        st.stop()
    st.html(f'<div class="hs-note">{icon("insights")}<span><b>Expected results</b> for the {len(d_all):,} reservations you '
            'checked or uploaded, from the model: reservations flagged as likely to cancel count as expected '
            'cancellations, the rest as expected stays. They update every time a prediction is saved. They are not '
            'real results, and a reservation saved more than once is counted once.</span></div>')
else:
    d_all = bookings()
    st.caption("Real results of 119,208 bookings at both hotels, with arrivals from July 2015 to August 2017.")
code = currency_code()

with st.container(key="card_perf_filters"):
    c1, c2, c3 = st.columns([1.2, .9, .9])
    hotel = c1.selectbox("Property", [ALL, "City Hotel", "Resort Hotel"], key="perf_hotel",
                         format_func=lambda h: "Both hotels" if h == ALL else h)
    years = sorted(int(y) for y in d_all["year"].dropna().unique())
    year = c2.selectbox("Arrival year", [ALL] + years, key="perf_year",
                        format_func=lambda y: "All years" if y == ALL else str(y))
    month = c3.selectbox("Arrival month", [ALL] + list(range(1, 13)), key="perf_month",
                         format_func=lambda m: "All months" if m == ALL else pd.Timestamp(2000, m, 1).strftime("%B"))

d_h = d_all if hotel == ALL else d_all[d_all["hotel"] == hotel]


def period(df, y):
    out = df if y == ALL else df[df["year"] == y]
    return out if month == ALL else out[out["arrival"].dt.month == month]


d = period(d_h, year)
if d.empty:
    st.warning("There are no reservations for this property and period. Try another month or year.",
               icon=":material/filter_alt_off:")
    st.stop()
prev = None
if year != ALL:                                      # same months one year earlier, for a fair comparison
    months = d["arrival"].dt.month.unique()
    p = d_h[(d_h["year"] == year - 1) & d_h["arrival"].dt.month.isin(months)]
    prev = analytics.summary(p) if len(p) else None
s = analytics.summary(d)


def change(now, before, kind="pct"):
    if prev is None or before is None or before != before or before == 0:
        return None
    if kind == "pts":
        return f"{(now - before) * 100:+.1f} pts vs last year"
    return f"{(now - before) / before:+.1%} vs last year"


off = {"delta_color": "off", "delta_arrow": "off", "border": True}
section(L("Performance summary", "Expected performance"),
        "Compared with the same months a year earlier" if prev else
        L("Stayed bookings for revenue, all bookings for cancellations",
          "Reservations not flagged for revenue, flagged reservations for risk"))
k = kpis("perf_a", ["payments", "sell", "bed", "event_available"])
k[0].metric(L("Room revenue", "Expected room revenue"), euros(s["room_revenue"]), change(s["room_revenue"], prev and prev["room_revenue"]),
            border=True, help=L("Room rate × nights of the bookings that stayed (checked out).",
                                "Room rate × nights of the reservations not flagged as likely to cancel."))
k[1].metric(L("Average daily rate (ADR)", "Expected ADR"), euros(s["adr"]), change(s["adr"], prev and prev["adr"]), border=True,
            help="Room revenue divided by rooms sold.")
k[2].metric(L("Rooms sold", "Expected rooms sold"), f"{int(s['rooms_sold']):,}", change(s["rooms_sold"], prev and prev["rooms_sold"]),
            border=True, help=L("Room nights of the bookings that stayed.",
                                "Room nights of the reservations not flagged as likely to cancel."))
k[3].metric(L("Bookings", "Predicted reservations"), f"{s['bookings']:,}", change(s["bookings"], prev and prev["bookings"]), border=True,
            help="All bookings with an arrival in this period, including cancelled ones.")
k = kpis("perf_b", ["event_busy", "money_off", "nights_stay", "schedule"], {0: "coral", 1: "coral"})
k[0].metric(L("Cancellation rate", "Likely to cancel"), pct(s["cancel_rate"]), change(s["cancel_rate"], prev and prev["cancel_rate"], "pts"),
            delta_color="inverse", border=True,
            help=L("Cancelled and no-show bookings as a share of all bookings.",
                   "Reservations flagged as likely to cancel (probability 50% or more)."))
k[1].metric(L("Revenue lost to cancellations", "Revenue at risk"), euros(s["revenue_lost"]),
            change(s["revenue_lost"], prev and prev["revenue_lost"]), delta_color="inverse", border=True,
            help=L("Booked room value (room rate × nights) of cancelled and no-show bookings.",
                   "Room rate × nights of the reservations flagged as likely to cancel. Not a forecast of lost "
                   "revenue."))
k[2].metric("Average length of stay", f"{s['avg_stay']:.1f} nights",
            L(f"no-shows {pct(s['no_show_rate'])}", "of expected stays"), **off)
k[3].metric("Average booking lead time", f"{s['avg_lead']:.0f} days", "booked before arrival", **off)

tabs = st.tabs([":material/trending_up: Revenue and demand", ":material/event_busy: Cancellations",
                ":material/hub: Channels and guests", ":material/lightbulb: Findings and actions"]
               + ([] if expected else [":material/savings: Model value"]))
tab_rev, tab_can, tab_mix, tab_find = tabs[:4]

# ------------------------------------------------------------------ model value (booking history only)
if not expected:
    with tabs[4]:
        st.caption("What HotelSight would have flagged in advance on the bookings it never saw in training (the "
                   "held-out test bookings in the filters above), using the frozen model's saved test predictions.")
        pop = st.segmented_control("Count", ["One copy per record", "Every record"], default="One copy per record",
                                   key="perf_value_pop", label_visibility="collapsed") or "One copy per record"
        v = analytics.model_value(d, one_copy=pop == "One copy per record")
        if v is None:
            st.info("No held-out test bookings fall in this selection. Choose a wider period.",
                    icon=":material/filter_alt_off:")
        else:
            k = kpis("perf_value", ["money_off", "savings", "percent", "notifications_off"],
                     {0: "coral", 3: "slate"})
            k[0].metric("Revenue actually lost", euros(v["revenue_lost"]), f"{v['cancellations']:,} cancellations",
                        **off, help="Room rate × nights of the test bookings that were really cancelled.")
            k[1].metric("Flagged in advance", euros(v["revenue_caught"]), f"{v['caught_bookings']:,} bookings",
                        **off, help="Revenue of the cancelled bookings that the model had flagged as likely to "
                                    "cancel, so the hotel could have acted before the cancellation.")
            k[2].metric("Share of lost revenue caught", pct(v["share_caught"]), "of the revenue lost", **off)
            k[3].metric("False alarms", euros(v["false_alarm_revenue"]), f"{v['false_alarms']:,} guests still came",
                        **off, help="Revenue of bookings flagged as likely to cancel whose guests still arrived. "
                                    "Contacting them costs staff time, so the hotel should act gently, for example "
                                    "with a reconfirmation rather than a cancellation.")
            left, right = st.columns(2, gap="medium")
            with left.container(key="card_value_month", height="stretch"):
                st.html('<div class="hs-card-title">Predicted and actual cancellation rate by month</div>')
                bm = v["by_month"]
                long = pd.concat([bm.assign(Series="Predicted to cancel", share=bm["predicted"]),
                                  bm.assign(Series="Actually cancelled", share=bm["actual"])])
                show_chart(base_chart(alt.Chart(long).mark_line(point=True, strokeWidth=2).encode(
                    x=alt.X("yearmonth(month):O", title="Arrival month", axis=MONTH_AXIS),
                    y=alt.Y("share:Q", title="Share of test bookings", axis=alt.Axis(format="%")),
                    color=alt.Color("Series:N", scale=alt.Scale(domain=["Predicted to cancel", "Actually cancelled"],
                                                                range=[CANCEL, INK]),
                                    legend=alt.Legend(orient="top", title=None)),
                    tooltip=[alt.Tooltip("month:T", title="Month", format="%B %Y"), "Series:N",
                             alt.Tooltip("share:Q", title="Share", format=".1%"),
                             alt.Tooltip("bookings:Q", title="Test bookings")]).properties(height=240)))
                st.caption("The model follows the real seasonal pattern of cancellations month by month.")
            with right.container(key="card_value_seg", height="stretch"):
                st.html('<div class="hs-card-title">Predicted and actual cancellation rate by channel</div>')
                bs = v["by_segment"][v["by_segment"]["bookings"] >= 30]
                long = pd.concat([bs.assign(Series="Predicted to cancel", share=bs["predicted"]),
                                  bs.assign(Series="Actually cancelled", share=bs["actual"])])
                show_chart(base_chart(alt.Chart(long).mark_bar(cornerRadiusEnd=3).encode(
                    y=alt.Y("segment:N", sort=list(bs["segment"]), title=None),
                    yOffset=alt.YOffset("Series:N", sort=["Predicted to cancel", "Actually cancelled"]),
                    x=alt.X("share:Q", title="Share of test bookings", axis=alt.Axis(format="%"),
                            scale=alt.Scale(domain=[0, 1])),
                    color=alt.Color("Series:N", scale=alt.Scale(domain=["Predicted to cancel", "Actually cancelled"],
                                                                range=[CANCEL, INK]),
                                    legend=alt.Legend(orient="top", title=None)),
                    tooltip=[alt.Tooltip("segment:N", title="Channel"), "Series:N",
                             alt.Tooltip("share:Q", title="Share", format=".1%"),
                             alt.Tooltip("bookings:Q", title="Test bookings")]).properties(height=alt.Step(14))))
                st.caption("Channels with at least 30 test bookings. Close bars mean the model judges that channel well.")
            st.caption(f"Based on {v['bookings']:,} held-out test bookings. The one-copy count matches the test results "
                       "on the Model evaluation page. These are past bookings, so this shows what the model would "
                       "have done, not a guarantee for future bookings.")

# ------------------------------------------------------------------ revenue and demand
with tab_rev:
    m = analytics.monthly(period(d_h, year) if month == ALL else d_h if year == ALL else d_h[d_h["year"] == year])
    m = m.assign(rev=to_display(m["room_revenue"]), adr_d=to_display(m["adr"]))
    left, right = st.columns(2, gap="medium")
    with left.container(key="card_perf_rev", height="stretch"):
        st.html(f'<div class="hs-card-title">{L("Room revenue", "Expected room revenue")} by month</div>')
        show_chart(base_chart(alt.Chart(m).mark_bar(color=STAY, cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
            x=alt.X("yearmonth(month):O", title="Arrival month", axis=MONTH_AXIS),
            y=alt.Y("rev:Q", title=f"{L('Room revenue', 'Expected room revenue')} ({code})"),
            tooltip=[alt.Tooltip("month:T", title="Month", format="%B %Y"),
                     alt.Tooltip("rev:Q", title=f"Room revenue ({code})", format=",.0f"),
                     alt.Tooltip("revenue_growth:Q", title="vs same month last year", format="+.1%"),
                     alt.Tooltip("rooms_sold:Q", title="Rooms sold", format=",")]).properties(height=250)))
        st.caption("Hover a bar for the change against the same month a year earlier (where that month exists).")
    with right.container(key="card_perf_adr", height="stretch"):
        st.html(f'<div class="hs-card-title">{L("Average daily rate", "Expected average daily rate")} by month</div>')
        show_chart(base_chart(alt.Chart(m).mark_line(point=True, color=SECOND, strokeWidth=2).encode(
            x=alt.X("yearmonth(month):O", title="Arrival month", axis=MONTH_AXIS),
            y=alt.Y("adr_d:Q", title=f"ADR ({code})"),
            tooltip=[alt.Tooltip("month:T", title="Month", format="%B %Y"),
                     alt.Tooltip("adr_d:Q", title=f"ADR ({code})", format=",.2f")]).properties(height=250)))
        st.caption(L("Prices rise strongly in the summer months, so pricing and deposit rules matter most then.",
                     "Room rates of the reservations not flagged as likely to cancel."))

    left, right = st.columns(2, gap="medium")
    with left.container(key="card_perf_nights", height="stretch"):
        st.html(f'<div class="hs-card-title">{L("Rooms sold", "Expected rooms sold")} per night</div>')
        n = analytics.rooms_sold_by_night(d_all) if expected else nights_sold()
        n = n if hotel == ALL else n[n["hotel"] == hotel]
        if year != ALL:
            n = n[n["night"].dt.year == year]
        if month != ALL:
            n = n[n["night"].dt.month == month]
        nd = n.groupby("night", as_index=False)["rooms"].sum()
        show_chart(base_chart(alt.Chart(nd).mark_area(color=STAY, opacity=.35, line={"color": STAY}).encode(
            x=alt.X("night:T", title="Night", axis=alt.Axis(format="%b %Y", labelOverlap="greedy")),
            y=alt.Y("rooms:Q", title="Rooms sold"),
            tooltip=[alt.Tooltip("night:T", title="Night", format="%a %d %b %Y"),
                     alt.Tooltip("rooms:Q", title="Rooms sold")]).properties(height=220)))
        if len(nd):
            top = nd.sort_values("rooms", ascending=False).head(5)
            st.caption("Busiest nights: " + " · ".join(f"{r.night:%a %d %b %Y} ({r.rooms} rooms)"
                                                      for r in top.itertuples()))
    with right.container(key="card_perf_pickup", height="stretch"):
        st.html('<div class="hs-card-title">Booking pickup by month the booking was made</div>')
        pk = analytics.pickup(d)
        keep_l, lost_l = L("Kept", "Expected to stay"), L("Later cancelled", "Likely to cancel")
        long = pd.concat([pk.assign(Series=keep_l, count=pk["bookings"] - pk["cancelled"]),
                          pk.assign(Series=lost_l, count=pk["cancelled"])])
        show_chart(base_chart(alt.Chart(long).mark_bar().encode(
            x=alt.X("yearmonth(made):O", title="Month the booking was made", axis=MONTH_AXIS),
            y=alt.Y("count:Q", title="Bookings made"),
            color=alt.Color("Series:N", scale=alt.Scale(domain=[keep_l, lost_l], range=[STAY, CANCEL]),
                            legend=alt.Legend(orient="top", title=None)),
            tooltip=[alt.Tooltip("made:T", title="Booked in", format="%B %Y"), "Series:N",
                     alt.Tooltip("count:Q", title="Bookings")]).properties(height=220)))
        st.caption("Counts only bookings with an arrival in the selected period, so the first and last months "
                   "are incomplete.")

    with st.container(key="card_perf_stay"):
        st.html('<div class="hs-card-title">Length of stay</div>')
        ls = analytics.by_group(d[d["stayed"] == 1], "stay_band", order=analytics.STAY_LABELS)
        ls = ls.assign(rev=to_display(ls["room_revenue"]))
        show_chart(base_chart(alt.Chart(ls).mark_bar(color=STAY, cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
            x=alt.X("group:N", sort=analytics.STAY_LABELS, title="Length of stay", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("share:Q", title="Share of stays", axis=alt.Axis(format="%")),
            tooltip=[alt.Tooltip("group:N", title="Stay"), alt.Tooltip("bookings:Q", title="Stays", format=","),
                     alt.Tooltip("share:Q", title="Share", format=".1%"),
                     alt.Tooltip("rev:Q", title=f"Room revenue ({code})", format=",.0f")]).properties(height=200)))

    report = analytics.monthly(d)
    if len(report):
        st.download_button(L("Download the monthly performance report (CSV)",
                             "Download the expected monthly performance (CSV)"),
                           report.rename(columns={"month": "Month", "bookings": "Bookings", "cancelled": "Cancelled",
                                                  "cancel_rate": "Cancellation rate", "room_revenue": "Room revenue (EUR)",
                                                  "rooms_sold": "Rooms sold", "adr": "ADR (EUR)",
                                                  "revenue_lost": "Revenue lost (EUR)",
                                                  "revenue_last_year": "Room revenue same month last year (EUR)",
                                                  "revenue_growth": "Revenue growth"})
                           .assign(Month=lambda x: x["Month"].dt.strftime("%Y-%m")).round(4)
                           .to_csv(index=False).encode("utf-8-sig"),
                           file_name=L("HotelSight_monthly_performance.csv",
                                       "HotelSight_expected_monthly_performance.csv"), mime="text/csv",
                           icon=":material/download:", key="perf_dl")

# ------------------------------------------------------------------ cancellations
with tab_can:
    left, right = st.columns(2, gap="medium")
    with left.container(key="card_perf_cmonth", height="stretch"):
        st.html(f'<div class="hs-card-title">{L("Cancellation rate and revenue lost", "Share likely to cancel and revenue at risk")} by month</div>')
        mm = analytics.monthly(d).assign(lost=lambda x: to_display(x["revenue_lost"]))
        show_chart(base_chart(alt.Chart(mm).mark_line(point=True, color=CANCEL, strokeWidth=2).encode(
            x=alt.X("yearmonth(month):O", title="Arrival month", axis=MONTH_AXIS),
            y=alt.Y("cancel_rate:Q", title=L("Cancellation rate", "Likely to cancel"), axis=alt.Axis(format="%")),
            tooltip=[alt.Tooltip("month:T", title="Month", format="%B %Y"),
                     alt.Tooltip("cancel_rate:Q", title=L("Cancellation rate", "Likely to cancel"), format=".1%"),
                     alt.Tooltip("lost:Q", title=f"{L('Revenue lost', 'Revenue at risk')} ({code})", format=",.0f"),
                     alt.Tooltip("bookings:Q", title="Bookings", format=",")]).properties(height=230)))
    with right.container(key="card_perf_timing", height="stretch"):
        st.html('<div class="hs-card-title">How many days before arrival guests cancel</div>')
        t = analytics.cancel_timing(d)
        if expected:
            st.caption("This needs the date each booking was really cancelled, so it is only shown for the booking "
                       "history.")
        else:
          show_chart(base_chart(alt.Chart(t).mark_bar(color=CANCEL, cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
            x=alt.X("band:N", sort=analytics.TIMING_LABELS, title="Days before arrival", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("share:Q", title="Share of cancellations", axis=alt.Axis(format="%")),
            tooltip=[alt.Tooltip("band:N", title="Cancelled"), alt.Tooltip("cancellations:Q", format=","),
                     alt.Tooltip("share:Q", title="Share", format=".1%")]).properties(height=230)))
          st.caption("Uses the date each booking was cancelled, which is only known afterwards. It guides the "
                     "cancellation deadline and is never used by the prediction model.")

    def rate_bars(df, title, key, order=None):
        df = df.assign(lost=to_display(df["revenue_lost"]))
        with st.container(key=key, height="stretch"):
            st.html(f'<div class="hs-card-title">{title}</div>')
            bars = alt.Chart(df).mark_bar(color=CANCEL, cornerRadiusEnd=3, height={"band": .6}).encode(
                y=alt.Y("group:N", sort=order or list(df["group"]), title=None),
                x=alt.X("cancel_rate:Q", title=L("Cancellation rate", "Likely to cancel"), axis=alt.Axis(format="%"),
                        scale=alt.Scale(domain=[0, 1])),
                tooltip=[alt.Tooltip("group:N", title="Group"), alt.Tooltip("bookings:Q", format=","),
                         alt.Tooltip("cancel_rate:Q", title=L("Cancellation rate", "Likely to cancel"), format=".1%"),
                         alt.Tooltip("lost:Q", title=f"{L('Revenue lost', 'Revenue at risk')} ({code})", format=",.0f")])
            text = bars.mark_text(align="left", dx=4, color=INK, fontSize=12).encode(
                text=alt.Text("cancel_rate:Q", format=".0%"))
            show_chart(base_chart((bars + text).properties(height=max(120, 34 * len(df)))))

    left, right = st.columns(2, gap="medium")
    with left:
        rate_bars(analytics.by_group(d, "lead_band", order=LEAD_LABELS), "By booking lead time", "card_perf_lead",
                  LEAD_LABELS)
    with right:
        rate_bars(analytics.by_group(d, "deposit"), "By deposit type", "card_perf_dep")
    st.caption(L("Hover a bar for the number of bookings and the revenue lost. No-shows count as cancellations "
                 f"({pct(s['no_show_rate'])} of bookings in this period).",
                 "Hover a bar for the number of reservations and the revenue at risk."))

# ------------------------------------------------------------------ channels and guests
with tab_mix:
    seg = analytics.by_group(d, "segment")
    left, right = st.columns([1, 1.6], gap="medium")
    with left.container(key="card_perf_donut", height="stretch"):
        st.html('<div class="hs-card-title">Booking channel mix</div>')
        top_seg = seg.head(5).copy()
        if len(seg) > 5:
            top_seg.loc[len(top_seg)] = {"group": "Other", "bookings": seg["bookings"].iloc[5:].sum(),
                                         "share": seg["share"].iloc[5:].sum()}
        show_chart(base_chart(alt.Chart(top_seg).mark_arc(innerRadius=58, stroke="#fff", strokeWidth=2).encode(
            theta=alt.Theta("bookings:Q"),
            color=alt.Color("group:N", sort=list(top_seg["group"]), legend=alt.Legend(orient="bottom", title=None,
                                                                                    columns=2),
                            scale=alt.Scale(range=SEQ[::-1][:len(top_seg)])),
            tooltip=[alt.Tooltip("group:N", title="Channel"), alt.Tooltip("bookings:Q", format=","),
                     alt.Tooltip("share:Q", title="Share", format=".1%")]).properties(height=260)))
    with right.container(key="card_perf_seg", height="stretch"):
        st.html('<div class="hs-card-title">Channel performance</div>')
        table = pd.DataFrame({"Channel": seg["group"], "Bookings": seg["bookings"], "Share": seg["share"] * 100,
                              L("Cancellation rate", "Likely to cancel"): seg["cancel_rate"] * 100,
                              f"ADR ({code})": to_display(seg["adr"]),
                              f"{L('Room revenue', 'Expected room revenue')} ({code})": to_display(seg["room_revenue"]),
                              f"{L('Revenue lost', 'Revenue at risk')} ({code})": to_display(seg["revenue_lost"])})
        money = {c: st.column_config.NumberColumn(format="localized") for c in table.columns if c.endswith(f"({code})")}
        st.dataframe(table, hide_index=True, width="stretch", column_config={
            **money, f"ADR ({code})": st.column_config.NumberColumn(format="%.2f"),
            "Share": st.column_config.NumberColumn(format="%.1f%%"),
            L("Cancellation rate", "Likely to cancel"): st.column_config.ProgressColumn(format="%.0f%%", min_value=0,
                                                                                       max_value=100)})

    left, right = st.columns(2, gap="medium")
    with left.container(key="card_perf_agents", height="stretch"):
        st.html('<div class="hs-card-title">Top travel agents</div>')
        ag = analytics.agents(d)
        if expected:
            st.caption("Saved predictions keep only whether a travel agent was used, not which one, so agents are "
                       "shown for the booking history only.")
        elif ag.empty:
            st.caption("No travel agent has at least 50 bookings in this period.")
        else:
            st.dataframe(pd.DataFrame({"Agent": ag["group"], "Bookings": ag["bookings"],
                                       "Cancellation rate": ag["cancel_rate"] * 100,
                                       f"Revenue lost ({code})": to_display(ag["revenue_lost"])}),
                         hide_index=True, width="stretch", column_config={
                    "Cancellation rate": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100),
                    f"Revenue lost ({code})": st.column_config.NumberColumn(format="localized")})
            st.caption("Agent IDs are anonymous in the dataset. Agents with a high rate are candidates for stricter "
                       "terms or earlier release dates.")
    with right.container(key="card_perf_country", height="stretch"):
        st.html('<div class="hs-card-title">Top guest countries</div>')
        names = get_countries()
        ct = analytics.by_group(d, "country", top=8)
        ct["name"] = ct["group"].map(lambda c: names.get(c, c))
        show_chart(base_chart(alt.Chart(ct).mark_bar(color=STAY, cornerRadiusEnd=3, height={"band": .6}).encode(
            y=alt.Y("name:N", sort=list(ct["name"]), title=None),
            x=alt.X("bookings:Q", title="Bookings"),
            tooltip=[alt.Tooltip("name:N", title="Country"), alt.Tooltip("bookings:Q", format=","),
                     alt.Tooltip("cancel_rate:Q", title=L("Cancellation rate", "Likely to cancel"), format=".1%")])
            .properties(height=max(150, 30 * len(ct)))))
        st.caption(L("Hover a bar for each country's cancellation rate.",
                     "Hover a bar for the share likely to cancel in each country."))

    left, right = st.columns(2, gap="medium")
    with left:
        rate_bars(analytics.by_group(d, "guest"), "Returning and first-time guests", "card_perf_guest")
    with right:
        rate_bars(analytics.by_group(d, "customer"), "By customer type", "card_perf_cust")

# ------------------------------------------------------------------ findings and actions
TONE = {"risk": ("warning", "hs-find risk"), "good": ("thumb_up", "hs-find good"), "info": ("insights", "hs-find")}
with tab_find:
    st.caption(L("Generated from the bookings in the filters above. Each finding is a pattern in past data, so "
                 "the actions are suggestions for management to judge, not proven causes.",
                 "Generated from your predicted reservations in the filters above, so they change as you add "
                 "predictions. They are patterns in the model's predictions, not proven causes."))
    f = analytics.findings(d, expected=expected)
    if not f:
        st.caption("Not enough reservations in this selection for findings yet.")
    cols = st.columns(2, gap="medium")
    for i, item in enumerate(f):
        ic, cls = TONE[item["kind"]]
        with cols[i % 2].container(key=f"card_find_{i}"):
            st.html(f'<div class="{cls}"><div class="hs-card-title">{icon(ic)} {item["title"]}</div>'
                    f'<p>{item["finding"]}</p><p><b>Suggested action:</b> {item["action"]}</p></div>')
    st.html(f'<div class="hs-note">{icon("info")}<span>Occupancy rate, RevPAR and available rooms need each hotel\'s '
            'number of rooms, which the dataset does not include, so they are not shown. Rooms sold is shown instead.'
            '</span></div>')
    if expected:
        nav_link("views/actions.py", "Open the action list for these reservations", ":material/checklist:")

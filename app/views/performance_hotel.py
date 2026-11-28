"""Hotel performance – how the two hotels performed in the historical bookings (descriptive, looks back)."""
import sys as _sys, pathlib as _pl  # make app/common.py importable however the page is launched
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))

import altair as alt
import pandas as pd
import streamlit as st

from common import (data_stamp, CANCEL, DATA_CURRENT, DATA_HISTORY, INK, SECOND, SEQ, STAY, base_chart, booking_history,
                    chart_card, currency_code, euros, get_countries, icon, kpis, money_short, nav_link, page_header, pct,
                    saved_predictions, section, show_chart, takeaway, to_display)
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
data_stamp()

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

tab_find, tab_rev, tab_can, tab_mix, *rest = st.tabs(
    [":material/lightbulb: Key findings", ":material/trending_up: Revenue", ":material/event_busy: Cancellations",
     ":material/hub: Channels and guests"] + ([] if expected else [":material/savings: Model value"]))
BEST, OTHER = "Highlighted", "Other"


def rate_bars(df, order=None, height=None):
    """Horizontal bars of the cancellation rate per group, each labelled with its percentage."""
    df = df.assign(lost=to_display(df["revenue_lost"]))
    bars = alt.Chart(df).mark_bar(color=CANCEL, cornerRadiusEnd=3, height={"band": .6}).encode(
        y=alt.Y("group:N", sort=order or list(df["group"]), title=None),
        x=alt.X("cancel_rate:Q", title=L("Cancelled", "Likely to cancel"), axis=alt.Axis(format="%"),
                scale=alt.Scale(domain=[0, 1])),
        tooltip=[alt.Tooltip("group:N", title="Group"), alt.Tooltip("bookings:Q", title="Bookings", format=","),
                 alt.Tooltip("cancel_rate:Q", title=L("Cancelled", "Likely to cancel"), format=".1%"),
                 alt.Tooltip("lost:Q", title=f"{L('Revenue lost', 'Revenue at risk')} ({code})", format=",.0f")])
    text = bars.mark_text(align="left", dx=4, color=INK, fontSize=12.5, fontWeight=600).encode(
        text=alt.Text("cancel_rate:Q", format=".0%"))
    show_chart(base_chart((bars + text).properties(height=height or max(120, 36 * len(df)))))


# ------------------------------------------------------------------ key findings (first: the plain summary)
TONE = {"risk": ("warning", "hs-find risk"), "good": ("thumb_up", "hs-find good"), "info": ("insights", "hs-find")}
with tab_find:
    f = analytics.findings(d, expected=expected)
    st.caption(L("What the bookings in the filters above say, in plain words, with a suggested action for each. "
                 "These are patterns in past data, not proven causes.",
                 "What your predicted reservations say, in plain words. They change as you add predictions."))
    if not f:
        st.caption("Not enough reservations in this selection for findings yet.")
    cols = st.columns(2, gap="medium")
    for i, item in enumerate(f):
        ic, cls = TONE[item["kind"]]
        with cols[i % 2].container(key=f"card_find_{i}"):
            st.html(f'<div class="{cls}"><div class="hs-card-title">{icon(ic)} {item["title"]}</div>'
                    f'<p>{item["finding"]}</p><p><b>Suggested action:</b> {item["action"]}</p></div>')
    st.html(f'<div class="hs-note">{icon("info")}<span>Occupancy rate and RevPAR need each hotel\'s number of rooms, '
            'which the dataset does not include, so rooms sold is shown instead.</span></div>')
    if expected:
        nav_link("views/actions.py", "Open the action list for these reservations", ":material/checklist:")

# ------------------------------------------------------------------ revenue
with tab_rev:
    m = analytics.monthly(period(d_h, year) if month == ALL else d_h if year == ALL else d_h[d_h["year"] == year])
    m = m.assign(rev=to_display(m["room_revenue"]), adr_d=to_display(m["adr"]))
    if len(m):
        best, worst = m.loc[m["room_revenue"].idxmax()], m.loc[m["room_revenue"].idxmin()]
        hi, lo = m.loc[m["adr"].idxmax()], m.loc[m["adr"].idxmin()]
        m["tag"] = (m["month"] == best["month"]).map({True: BEST, False: OTHER})
        left, right = st.columns(2, gap="medium")
        with left:
            with chart_card("card_perf_rev", L("When do we earn the most?", "When do we expect to earn the most?"),
                            f"<b>{best['month']:%B %Y}</b> brought the most room revenue "
                            f"(<b>{money_short(best['room_revenue'])}</b>), <b>{worst['month']:%B %Y}</b> the least "
                            f"({money_short(worst['room_revenue'])})."):
                show_chart(base_chart(alt.Chart(m).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
                    x=alt.X("yearmonth(month):O", title=None, axis=MONTH_AXIS),
                    y=alt.Y("rev:Q", title=f"{L('Room revenue', 'Expected room revenue')} ({code})",
                            axis=alt.Axis(format="~s")),
                    color=alt.Color("tag:N", scale=alt.Scale(domain=[BEST, OTHER], range=["#1E40AF", "#93B4EA"]),
                                    legend=None),
                    tooltip=[alt.Tooltip("month:T", title="Month", format="%B %Y"),
                             alt.Tooltip("rev:Q", title=f"Room revenue ({code})", format=",.0f"),
                             alt.Tooltip("revenue_growth:Q", title="vs same month last year", format="+.1%"),
                             alt.Tooltip("rooms_sold:Q", title="Rooms sold", format=",")]).properties(height=240)))
                st.caption("Darker bar: the best month. Hover a bar for the change against the same month last year.")
        with right:
            with chart_card("card_perf_adr", L("How do room prices change through the year?",
                                               "What room prices do we expect?"),
                            f"Rooms were most expensive in <b>{hi['month']:%B %Y}</b> (<b>{euros(hi['adr'])}</b> a night) "
                            f"and cheapest in <b>{lo['month']:%B %Y}</b> ({euros(lo['adr'])})."):
                show_chart(base_chart(alt.Chart(m).mark_line(point=True, color=SECOND, strokeWidth=2.5).encode(
                    x=alt.X("yearmonth(month):O", title=None, axis=MONTH_AXIS),
                    y=alt.Y("adr_d:Q", title=f"Average price per night ({code})"),
                    tooltip=[alt.Tooltip("month:T", title="Month", format="%B %Y"),
                             alt.Tooltip("adr_d:Q", title=f"Average price per night ({code})", format=",.2f")])
                    .properties(height=240)))
                st.caption("This is the average daily rate (ADR): room revenue divided by rooms sold.")

    with st.expander("More detail: busiest nights, when bookings are made and length of stay",
                     icon=":material/expand_more:"):
        n = analytics.rooms_sold_by_night(d_all) if expected else nights_sold()
        n = n if hotel == ALL else n[n["hotel"] == hotel]
        if year != ALL:
            n = n[n["night"].dt.year == year]
        if month != ALL:
            n = n[n["night"].dt.month == month]
        nd = n.groupby("night", as_index=False)["rooms"].sum()
        if len(nd):
            peak = nd.loc[nd["rooms"].idxmax()]
            with chart_card("card_perf_nights", "Which nights are the busiest?",
                            f"The busiest night was <b>{peak['night']:%A %d %B %Y}</b> with <b>{int(peak['rooms'])} "
                            f"rooms</b> {L('sold', 'expected')}. Plan staff and prices for nights like this."):
                show_chart(base_chart(alt.Chart(nd).mark_area(color=STAY, opacity=.3, line={"color": STAY}).encode(
                    x=alt.X("night:T", title=None, axis=alt.Axis(format="%b %Y", labelOverlap="greedy")),
                    y=alt.Y("rooms:Q", title=L("Rooms sold", "Expected rooms sold")),
                    tooltip=[alt.Tooltip("night:T", title="Night", format="%a %d %b %Y"),
                             alt.Tooltip("rooms:Q", title="Rooms")]).properties(height=200)))
        pk = analytics.pickup(d)
        if len(pk):
            top_pk = pk.loc[pk["bookings"].idxmax()]
            keep_l, lost_l = L("Kept", "Expected to stay"), L("Later cancelled", "Likely to cancel")
            long = pd.concat([pk.assign(Series=keep_l, count=pk["bookings"] - pk["cancelled"]),
                              pk.assign(Series=lost_l, count=pk["cancelled"])])
            with chart_card("card_perf_pickup", "When are bookings made?",
                            f"Most bookings were made in <b>{top_pk['made']:%B %Y}</b> "
                            f"({int(top_pk['bookings']):,}), and <b>{top_pk['cancelled'] / top_pk['bookings']:.0%}</b> "
                            f"of those were {L('later cancelled', 'flagged as likely to cancel')}."):
                show_chart(base_chart(alt.Chart(long).mark_bar().encode(
                    x=alt.X("yearmonth(made):O", title="Month the booking was made", axis=MONTH_AXIS),
                    y=alt.Y("count:Q", title="Bookings made"),
                    color=alt.Color("Series:N", scale=alt.Scale(domain=[keep_l, lost_l], range=[STAY, CANCEL]),
                                    legend=alt.Legend(orient="top", title=None)),
                    tooltip=[alt.Tooltip("made:T", title="Booked in", format="%B %Y"), "Series:N",
                             alt.Tooltip("count:Q", title="Bookings")]).properties(height=200)))
                st.caption("This is booking pickup. The first and last months are incomplete.")
        ls = analytics.by_group(d[d["stayed"] == 1], "stay_band", order=analytics.STAY_LABELS)
        if len(ls):
            common = ls.loc[ls["share"].idxmax()]
            ls = ls.assign(rev=to_display(ls["room_revenue"]))
            with chart_card("card_perf_stay", "How long do guests stay?",
                            f"The most common stay is <b>{common['group']}</b> (<b>{common['share']:.0%}</b> of stays). "
                            f"The average stay is <b>{s['avg_stay']:.1f} nights</b>."):
                bars = alt.Chart(ls).mark_bar(color=STAY, cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
                    x=alt.X("group:N", sort=analytics.STAY_LABELS, title=None, axis=alt.Axis(labelAngle=0)),
                    y=alt.Y("share:Q", title="Share of stays", axis=alt.Axis(format="%")),
                    tooltip=[alt.Tooltip("group:N", title="Stay"), alt.Tooltip("bookings:Q", title="Stays", format=","),
                             alt.Tooltip("rev:Q", title=f"Room revenue ({code})", format=",.0f")])
                labels = bars.mark_text(dy=-7, color=INK, fontSize=12.5, fontWeight=600).encode(
                    text=alt.Text("share:Q", format=".0%"))
                show_chart(base_chart((bars + labels).properties(height=200)))

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
    mm = analytics.monthly(d)
    left, right = st.columns(2, gap="medium")
    with left:
        if len(mm):
            top_m, low_m = mm.loc[mm["cancel_rate"].idxmax()], mm.loc[mm["cancel_rate"].idxmin()]
            mm = mm.assign(lost=to_display(mm["revenue_lost"]))
            with chart_card("card_perf_cmonth", L("When do cancellations peak?", "When is the risk highest?"),
                            f"<b>{top_m['month']:%B %Y}</b> had the highest rate (<b>{top_m['cancel_rate']:.0%}</b>), "
                            f"<b>{low_m['month']:%B %Y}</b> the lowest ({low_m['cancel_rate']:.0%}).", "risk"):
                show_chart(base_chart(alt.Chart(mm).mark_line(point=True, color=CANCEL, strokeWidth=2.5).encode(
                    x=alt.X("yearmonth(month):O", title=None, axis=MONTH_AXIS),
                    y=alt.Y("cancel_rate:Q", title=L("Bookings cancelled", "Likely to cancel"),
                            axis=alt.Axis(format="%")),
                    tooltip=[alt.Tooltip("month:T", title="Month", format="%B %Y"),
                             alt.Tooltip("cancel_rate:Q", title=L("Cancelled", "Likely to cancel"), format=".1%"),
                             alt.Tooltip("lost:Q", title=f"{L('Revenue lost', 'Revenue at risk')} ({code})",
                                         format=",.0f"),
                             alt.Tooltip("bookings:Q", title="Bookings", format=",")]).properties(height=230)))
    with right:
        if expected:
            with chart_card("card_perf_timing", "How early do guests cancel?"):
                st.caption("This needs the date each booking was really cancelled, so it is only shown for the "
                           "booking history dataset.")
        else:
            t = analytics.cancel_timing(d)
            days = d["cancel_days_before"].dropna()
            if len(days):
                top_t = t.loc[t["share"].idxmax()]
                with chart_card("card_perf_timing", "How early do guests cancel?",
                                f"Most cancellations come <b>{top_t['band']}</b> before arrival "
                                f"(<b>{top_t['share']:.0%}</b>). Half cancel <b>{days.median():.0f} days or more</b> "
                                "ahead, so set the free-cancellation deadline before that."):
                    bars = alt.Chart(t).mark_bar(color=CANCEL, cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
                        x=alt.X("band:N", sort=analytics.TIMING_LABELS, title="Days before arrival",
                                axis=alt.Axis(labelAngle=0)),
                        y=alt.Y("share:Q", title="Share of cancellations", axis=alt.Axis(format="%")),
                        tooltip=[alt.Tooltip("band:N", title="Cancelled"),
                                 alt.Tooltip("cancellations:Q", title="Cancellations", format=",")])
                    labels = bars.mark_text(dy=-7, color=INK, fontSize=12.5, fontWeight=600).encode(
                        text=alt.Text("share:Q", format=".0%"))
                    show_chart(base_chart((bars + labels).properties(height=230)))

    lead = analytics.by_group(d, "lead_band", order=LEAD_LABELS)
    dep = analytics.by_group(d, "deposit")
    left, right = st.columns(2, gap="medium")
    with left:
        if len(lead) >= 2:
            a, b = lead.iloc[0], lead.iloc[-1]
            with chart_card("card_perf_lead", "Do bookings made far ahead cancel more?",
                            f"Bookings made <b>{b['group'].replace('Over', 'more than')}</b> ahead "
                            f"{L('cancelled', 'are likely to cancel')} <b>{b['cancel_rate']:.0%}</b> of the time, against "
                            f"<b>{a['cancel_rate']:.0%}</b> for bookings made <b>{a['group']}</b> ahead.", "risk"):
                rate_bars(lead, LEAD_LABELS)
    with right:
        if len(dep):
            nd_ = dep[dep["group"] == "No deposit"]
            worst_dep = dep.loc[dep["cancel_rate"].idxmax()]
            note = (f"Bookings with <b>no deposit</b> {L('cancelled', 'are likely to cancel')} "
                    f"<b>{nd_.iloc[0]['cancel_rate']:.0%}</b> of the time. " if len(nd_) else "")
            with chart_card("card_perf_dep", "Do deposits stop cancellations?",
                            note + f"The highest rate is for <b>{worst_dep['group'].lower()}</b> "
                                   f"(<b>{worst_dep['cancel_rate']:.0%}</b>), mostly group and agent business."):
                rate_bars(dep)
    st.caption("Each bar shows the share of bookings in that group that "
               + L(f"were cancelled (no-shows count as cancellations, {pct(s['no_show_rate'])} of bookings).",
                   "are flagged as likely to cancel."))

# ------------------------------------------------------------------ channels and guests
with tab_mix:
    seg = analytics.by_group(d, "segment")
    if len(seg):
        big = seg.iloc[0]
        rel = seg[seg["share"] >= .03]
        safest, riskiest = rel.loc[rel["cancel_rate"].idxmin()], rel.loc[rel["cancel_rate"].idxmax()]
        left, right = st.columns([1, 1.25], gap="medium")
        with left:
            top_seg = seg.head(5).copy()
            if len(seg) > 5:
                top_seg.loc[len(top_seg)] = {"group": "Other", "bookings": seg["bookings"].iloc[5:].sum(),
                                             "share": seg["share"].iloc[5:].sum()}
            with chart_card("card_perf_donut", "Where do our bookings come from?",
                            f"<b>{big['group']}</b> brings the most bookings (<b>{big['share']:.0%}</b>)."):
                show_chart(base_chart(alt.Chart(top_seg).mark_arc(innerRadius=58, stroke="#fff", strokeWidth=2).encode(
                    theta=alt.Theta("bookings:Q"),
                    color=alt.Color("group:N", sort=list(top_seg["group"]),
                                    legend=alt.Legend(orient="bottom", title=None, columns=2),
                                    scale=alt.Scale(range=SEQ[::-1][:len(top_seg)])),
                    tooltip=[alt.Tooltip("group:N", title="Channel"), alt.Tooltip("bookings:Q", format=","),
                             alt.Tooltip("share:Q", title="Share", format=".1%")]).properties(height=250)))
        with right:
            with chart_card("card_perf_seg", "Which channels are reliable?",
                            f"<b>{safest['group']}</b> is the most reliable ({safest['cancel_rate']:.0%}), "
                            f"<b>{riskiest['group'].lower()}</b> the riskiest (<b>{riskiest['cancel_rate']:.0%}</b>).",
                            "risk"):
                rate_bars(rel)
        with st.expander("Full channel table", icon=":material/table_view:"):
            table = pd.DataFrame({"Channel": seg["group"], "Bookings": seg["bookings"], "Share": seg["share"] * 100,
                                  L("Cancelled", "Likely to cancel"): seg["cancel_rate"] * 100,
                                  f"ADR ({code})": to_display(seg["adr"]),
                                  f"{L('Room revenue', 'Expected room revenue')} ({code})": to_display(seg["room_revenue"]),
                                  f"{L('Revenue lost', 'Revenue at risk')} ({code})": to_display(seg["revenue_lost"])})
            money = {c: st.column_config.NumberColumn(format="localized") for c in table.columns if c.endswith(f"({code})")}
            st.dataframe(table, hide_index=True, width="stretch", column_config={
                **money, f"ADR ({code})": st.column_config.NumberColumn(format="%.2f"),
                "Share": st.column_config.NumberColumn(format="%.1f%%"),
                L("Cancelled", "Likely to cancel"): st.column_config.ProgressColumn(format="%.0f%%", min_value=0,
                                                                                    max_value=100)})

    names = get_countries()
    ct = analytics.by_group(d, "country", top=8)
    guest = analytics.by_group(d, "guest")
    left, right = st.columns(2, gap="medium")
    with left:
        if len(ct):
            ct["name"] = ct["group"].map(lambda c: names.get(c, c))
            first = ct.iloc[0]
            with chart_card("card_perf_country", "Where do our guests come from?",
                            f"Most guests come from <b>{first['name']}</b> (<b>{first['share']:.0%}</b> of bookings)."):
                bars = alt.Chart(ct).mark_bar(color=STAY, cornerRadiusEnd=3, height={"band": .6}).encode(
                    y=alt.Y("name:N", sort=list(ct["name"]), title=None),
                    x=alt.X("bookings:Q", title="Bookings"),
                    tooltip=[alt.Tooltip("name:N", title="Country"), alt.Tooltip("bookings:Q", format=","),
                             alt.Tooltip("cancel_rate:Q", title=L("Cancelled", "Likely to cancel"), format=".1%")])
                show_chart(base_chart(bars.properties(height=max(150, 30 * len(ct)))))
    with right:
        if {"Returning guest", "First-time guest"} <= set(guest["group"]):
            g = guest.set_index("group")
            with chart_card("card_perf_guest", "Do returning guests cancel less?",
                            f"Yes. Returning guests {L('cancelled', 'are likely to cancel')} "
                            f"<b>{g.loc['Returning guest', 'cancel_rate']:.0%}</b> of the time, first-time guests "
                            f"<b>{g.loc['First-time guest', 'cancel_rate']:.0%}</b>.", "good"):
                rate_bars(guest, height=110)

    with st.expander("More detail: customer types and travel agents", icon=":material/expand_more:"):
        cust = analytics.by_group(d, "customer")
        if len(cust):
            worst_c = cust.loc[cust["cancel_rate"].idxmax()]
            with chart_card("card_perf_cust", "Which type of customer cancels most?",
                            f"<b>{worst_c['group']}</b> has the highest rate (<b>{worst_c['cancel_rate']:.0%}</b>).",
                            "risk"):
                rate_bars(cust)
        if expected:
            st.caption("Travel agents are shown for the booking history dataset only, because saved predictions keep "
                       "only whether an agent was used, not which one.")
        else:
            ag = analytics.agents(d)
            if len(ag):
                worst_a = ag.loc[ag["cancel_rate"].idxmax()]
                with chart_card("card_perf_agents", "Which travel agents cancel most?",
                                f"<b>{worst_a['group']}</b> cancels <b>{worst_a['cancel_rate']:.0%}</b> of its "
                                "bookings. Agents like this are candidates for stricter terms.", "risk"):
                    st.dataframe(pd.DataFrame({"Agent": ag["group"], "Bookings": ag["bookings"],
                                               "Cancelled": ag["cancel_rate"] * 100,
                                               f"Revenue lost ({code})": to_display(ag["revenue_lost"])}),
                                 hide_index=True, width="stretch", column_config={
                            "Cancelled": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100),
                            f"Revenue lost ({code})": st.column_config.NumberColumn(format="localized")})
                    st.caption("Agent IDs are anonymous in the dataset.")

# ------------------------------------------------------------------ model value (booking history only)
if rest:
    with rest[0]:
        pop = st.segmented_control("Count", ["One copy per record", "Every record"], default="One copy per record",
                                   key="perf_value_pop", label_visibility="collapsed",
                                   help="One copy per record counts identical rows once, the more cautious result."
                                   ) or "One copy per record"
        v = analytics.model_value(d, one_copy=pop == "One copy per record")
        if v is None:
            st.info("No held-out test bookings fall in this selection. Choose a wider period.",
                    icon=":material/filter_alt_off:")
        else:
            takeaway(f"Had the hotel used HotelSight, it would have been warned about <b>{money_short(v['revenue_caught'])}"
                     f"</b> of the <b>{money_short(v['revenue_lost'])}</b> lost to cancellations "
                     f"(<b>{v['share_caught']:.0%}</b>), before the guests cancelled.", "good")
            k = kpis("perf_value", ["money_off", "savings", "percent", "notifications_off"],
                     {0: "coral", 3: "slate"})
            k[0].metric("Revenue actually lost", euros(v["revenue_lost"]), f"{v['cancellations']:,} cancellations",
                        **off, help="Room rate × nights of the test bookings that were really cancelled.")
            k[1].metric("Warned in advance", euros(v["revenue_caught"]), f"{v['caught_bookings']:,} bookings",
                        **off, help="Revenue of the cancelled bookings that the model had flagged as likely to "
                                    "cancel, so the hotel could have acted before the cancellation.")
            k[2].metric("Share of lost revenue caught", pct(v["share_caught"]), "of the revenue lost", **off)
            k[3].metric("False alarms", euros(v["false_alarm_revenue"]), f"{v['false_alarms']:,} guests still came",
                        **off, help="Bookings flagged as likely to cancel whose guests still arrived. Act gently on "
                                    "them, for example with a reconfirmation.")
            bm = v["by_month"]
            long = pd.concat([bm.assign(Series="Predicted to cancel", share=bm["predicted"]),
                              bm.assign(Series="Actually cancelled", share=bm["actual"])])
            gap = float((bm["predicted"] - bm["actual"]).abs().mean())
            with chart_card("card_value_month", "Did the model follow the real pattern?",
                            f"Yes. Month by month, the predicted and real cancellation rates stay close (on average "
                            f"<b>{gap * 100:.0f} points</b> apart).", "good"):
                show_chart(base_chart(alt.Chart(long).mark_line(point=True, strokeWidth=2.5).encode(
                    x=alt.X("yearmonth(month):O", title=None, axis=MONTH_AXIS),
                    y=alt.Y("share:Q", title="Share of bookings", axis=alt.Axis(format="%")),
                    color=alt.Color("Series:N", scale=alt.Scale(domain=["Predicted to cancel", "Actually cancelled"],
                                                                range=[CANCEL, INK]),
                                    legend=alt.Legend(orient="top", title=None)),
                    strokeDash=alt.StrokeDash("Series:N", scale=alt.Scale(
                        domain=["Predicted to cancel", "Actually cancelled"], range=[[1, 0], [5, 3]]), legend=None),
                    tooltip=[alt.Tooltip("month:T", title="Month", format="%B %Y"), "Series:N",
                             alt.Tooltip("share:Q", title="Share", format=".1%"),
                             alt.Tooltip("bookings:Q", title="Test bookings")]).properties(height=240)))
            st.caption(f"Based on {v['bookings']:,} past bookings the model never saw in training. This shows what it "
                       "would have done, not a guarantee for future bookings.")

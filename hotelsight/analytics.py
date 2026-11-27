"""Hotel performance analytics from the historical Hotel Booking Demand data (Antonio et al., 2019).

This module only looks back at what happened, so it may use outcome fields such as reservation_status and
reservation_status_date (for example, to measure how many days before arrival guests cancelled). The prediction
model never uses those fields. The same three invalid-record rules as the experiment are applied.

The dataset has no room capacity, so occupancy rate, RevPAR and available rooms cannot be calculated and are not
shown. "Rooms sold" means room nights of bookings that were not cancelled.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .contract import CATEGORY_LABELS, MONTHS
from .insights import LEAD_BINS, LEAD_LABELS
from .paths import DATA

TIMING_BINS = [-1, 1, 7, 30, 90, 180, np.inf]
TIMING_LABELS = ["0–1 days", "2–7 days", "8–30 days", "31–90 days", "91–180 days", "Over 180 days"]
STAY_BINS = [-1, 1, 2, 3, 5, 7, np.inf]
STAY_LABELS = ["1 night", "2 nights", "3 nights", "4–5 nights", "6–7 nights", "8+ nights"]


def load_bookings(path=DATA) -> pd.DataFrame:
    """The cleaned historical bookings with the fields the performance views need."""
    raw = pd.read_csv(path, keep_default_na=False, na_values=[""], low_memory=False)   # keeps "NA" (Namibia)
    num = lambda c: pd.to_numeric(raw[c], errors="coerce")                # noqa: E731
    guests = num("adults").fillna(0) + num("children").fillna(0) + num("babies").fillna(0)
    adr = num("adr")
    keep = ~((guests == 0) | (adr < 0) | (adr == 5400))                   # the experiment's three removal rules
    raw["source_csv_row"] = np.arange(2, len(raw) + 2)                    # same row id as the experiment
    r = raw[keep].reset_index(drop=True)
    arrival = pd.to_datetime(r["arrival_date_year"].astype(str) + "-" + r["arrival_date_month"] + "-"
                             + r["arrival_date_day_of_month"].astype(str), format="%Y-%B-%d")
    nights = pd.to_numeric(r["stays_in_weekend_nights"]) + pd.to_numeric(r["stays_in_week_nights"])
    lead = pd.to_numeric(r["lead_time"])
    status_date = pd.to_datetime(r["reservation_status_date"], errors="coerce")
    d = pd.DataFrame({
        "hotel": r["hotel"], "arrival": arrival, "month": arrival.dt.to_period("M").dt.to_timestamp(),
        "year": arrival.dt.year, "booked": arrival - pd.to_timedelta(lead, unit="D"),
        "lead_time": lead, "nights": nights, "weekend_nights": pd.to_numeric(r["stays_in_weekend_nights"]),
        "adr": pd.to_numeric(r["adr"]), "cancelled": pd.to_numeric(r["is_canceled"]).astype(int),
        "status": r["reservation_status"],
    })
    d["value"] = d["adr"] * d["nights"]
    d["stayed"] = (d["status"] == "Check-Out").astype(int)
    d["no_show"] = (d["status"] == "No-Show").astype(int)
    d["cancel_days_before"] = np.where(d["status"] == "Canceled", (arrival - status_date).dt.days, np.nan)
    d["lead_band"] = pd.cut(d["lead_time"], LEAD_BINS, labels=LEAD_LABELS)
    d["stay_band"] = pd.cut(d["nights"], STAY_BINS, labels=STAY_LABELS)
    for field, col in [("deposit_type", "deposit"), ("market_segment", "segment"), ("customer_type", "customer")]:
        d[col] = r[field].map(lambda v, f=field: CATEGORY_LABELS[f].get(v, v))
    d["country"] = r["country"].replace({"NULL": "Unknown"})
    d["guest"] = np.where(pd.to_numeric(r["is_repeated_guest"]) == 1, "Returning guest", "First-time guest")
    d["agent"] = r["agent"].astype(str).str.strip().replace({"NULL": "", "nan": ""})
    d["requests"] = pd.to_numeric(r["total_of_special_requests"])
    d["room_changed"] = (r["reserved_room_type"] != r["assigned_room_type"]).astype(int)
    d["source_csv_row"] = r["source_csv_row"].to_numpy()
    return d


def model_value(d: pd.DataFrame, one_copy: bool = True) -> dict | None:
    """What the deployed model would have flagged in advance, on its held-out test bookings in d.

    Uses the frozen model's saved test predictions (no refitting). Revenue is room rate × nights. 'Caught' means
    revenue of bookings that were really cancelled and that the model had flagged as likely to cancel."""
    from .paths import ARM, PAIRED_TEST
    p = pd.read_csv(PAIRED_TEST, usecols=["source_csv_row", "one_copy", "actual", ARM])
    if one_copy:
        p = p[p["one_copy"]]
    t = d.merge(p, on="source_csv_row", how="inner")
    if t.empty:
        return None
    t["flagged"] = (t[ARM] >= 0.5).astype(int)
    real = t["actual"] == 1
    lost = float(t.loc[real, "value"].sum())
    caught = float(t.loc[real & (t["flagged"] == 1), "value"].sum())
    g = t.groupby("month")
    by_month = pd.DataFrame({"bookings": g.size(), "actual": g["actual"].mean(),
                             "predicted": g["flagged"].mean()}).reset_index()
    seg = t.groupby("segment")
    by_segment = pd.DataFrame({"bookings": seg.size(), "actual": seg["actual"].mean(),
                               "predicted": seg["flagged"].mean()}).reset_index().sort_values("bookings",
                                                                                               ascending=False)
    return {"bookings": len(t), "cancellations": int(real.sum()),
            "caught_bookings": int((real & (t["flagged"] == 1)).sum()),
            "revenue_lost": lost, "revenue_caught": caught, "share_caught": caught / lost if lost else 0.0,
            "false_alarm_revenue": float(t.loc[~real & (t["flagged"] == 1), "value"].sum()),
            "false_alarms": int((~real & (t["flagged"] == 1)).sum()),
            "missed_revenue": lost - caught, "by_month": by_month, "by_segment": by_segment.reset_index(drop=True)}


def summary(d: pd.DataFrame) -> dict:
    """Headline figures. Revenue and ADR use stayed bookings only; revenue lost uses cancelled and no-show bookings."""
    stayed, lost = d[d["stayed"] == 1], d[d["cancelled"] == 1]
    sold = float(stayed["nights"].sum())
    revenue = float(stayed["value"].sum())
    n = len(d)
    return {"bookings": n, "cancelled": len(lost), "cancel_rate": len(lost) / n if n else 0.0,
            "room_revenue": revenue, "rooms_sold": sold, "adr": revenue / sold if sold else float("nan"),
            "revenue_lost": float(lost["value"].sum()), "no_show_rate": float(d["no_show"].mean()) if n else 0.0,
            "avg_stay": float(stayed["nights"].mean()) if len(stayed) else float("nan"),
            "avg_lead": float(d["lead_time"].mean()) if n else float("nan")}


def monthly(d: pd.DataFrame) -> pd.DataFrame:
    """One row per arrival month, with the change in room revenue against the same month a year earlier."""
    rows = []
    for m, g in d.groupby("month"):
        s = summary(g)
        rows.append({"month": m, **{k: s[k] for k in ("bookings", "cancelled", "cancel_rate", "room_revenue",
                                                          "rooms_sold", "adr", "revenue_lost")}})
    out = pd.DataFrame(rows).sort_values("month").reset_index(drop=True)
    if out.empty:
        return out
    prev = out.set_index("month")["room_revenue"]
    last_year = out["month"].map(lambda m: prev.get(m - pd.DateOffset(years=1), np.nan))
    out["revenue_last_year"] = last_year
    out["revenue_growth"] = (out["room_revenue"] - last_year) / last_year
    return out


def by_group(d: pd.DataFrame, column: str, top: int | None = None, order: list | None = None) -> pd.DataFrame:
    """Bookings, share of bookings, cancellation rate, ADR, room revenue and revenue lost per group."""
    x = d.copy()
    if top is not None:
        keep = x[column].value_counts().head(top).index
        x[column] = x[column].where(x[column].isin(keep), "Other")
    rows = []
    for key, g in x.groupby(column, observed=True):
        s = summary(g)
        rows.append({"group": str(key), "bookings": s["bookings"], "cancel_rate": s["cancel_rate"], "adr": s["adr"],
                     "room_revenue": s["room_revenue"], "revenue_lost": s["revenue_lost"]})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["share"] = out["bookings"] / out["bookings"].sum()
    if order:
        out["o"] = out["group"].map({k: i for i, k in enumerate(order)})
        return out.sort_values("o").drop(columns="o").reset_index(drop=True)
    return out.sort_values("bookings", ascending=False).reset_index(drop=True)


def cancel_timing(d: pd.DataFrame) -> pd.DataFrame:
    """How many days before arrival the cancelled bookings were cancelled (no-shows are not included)."""
    days = d["cancel_days_before"].dropna()
    band = pd.cut(days, TIMING_BINS, labels=TIMING_LABELS)
    out = band.value_counts().reindex(TIMING_LABELS).fillna(0).astype(int).rename("cancellations").reset_index()
    out.columns = ["band", "cancellations"]
    out["share"] = out["cancellations"] / max(int(out["cancellations"].sum()), 1)
    return out


def agents(d: pd.DataFrame, top: int = 10, min_bookings: int = 50) -> pd.DataFrame:
    """Travel agents with the most bookings (IDs are anonymous in the dataset)."""
    a = d[d["agent"] != ""]
    out = by_group(a, "agent")
    if out.empty:
        return out
    out = out[out["bookings"] >= min_bookings].head(top).reset_index(drop=True)
    out["group"] = "Agent " + out["group"].str.replace(r"\.0$", "", regex=True)
    return out


def pickup(d: pd.DataFrame) -> pd.DataFrame:
    """Booking pickup: bookings made in each month (by booking date), and how many of them were later cancelled."""
    b = d.assign(made=d["booked"].dt.to_period("M").dt.to_timestamp())
    g = b.groupby("made")
    out = pd.DataFrame({"bookings": g.size(), "cancelled": g["cancelled"].sum()}).reset_index()
    first, last = d["month"].min(), d["month"].max()    # outside the arrival period the data is incomplete
    return out[(out["made"] >= first) & (out["made"] <= last)].reset_index(drop=True)


def rooms_sold_by_night(d: pd.DataFrame) -> pd.DataFrame:
    """Rooms sold on each night (stayed bookings only), per property."""
    s = d[(d["stayed"] == 1) & (d["nights"] > 0)]
    rep = np.repeat(np.arange(len(s)), s["nights"].to_numpy().astype(int))
    offset = np.concatenate([np.arange(n) for n in s["nights"].to_numpy().astype(int)]) if len(s) else np.array([], int)
    nights = s["arrival"].to_numpy()[rep] + offset.astype("timedelta64[D]")
    out = pd.DataFrame({"night": nights, "hotel": s["hotel"].to_numpy()[rep]})
    return out.groupby(["night", "hotel"]).size().rename("rooms").reset_index()


def from_predictions(h: pd.DataFrame) -> pd.DataFrame:
    """Saved predictions (history.with_stay output) in the same shape as load_bookings, with the model's EXPECTED
    outcome: a reservation flagged as likely to cancel counts as an expected cancellation, the rest as expected stays.

    A reservation saved more than once (same reference and arrival) is counted once. Fields that only exist after
    the outcome (cancellation date, no-show, travel agent ID) are left empty.
    """
    import json
    recs = pd.DataFrame([json.loads(x or "{}") for x in h["inputs_json"]], index=h.index)

    def num(c):
        return pd.to_numeric(recs[c], errors="coerce") if c in recs else pd.Series(np.nan, index=recs.index)

    def text(c):
        return recs[c].astype(str) if c in recs else pd.Series("", index=recs.index)

    arrival = pd.to_datetime(h["arrival"], errors="coerce", format="%Y-%m-%d")
    ref = h["reference"].fillna("").astype(str).str.strip()
    dup = (ref != "") & pd.DataFrame({"r": ref, "a": arrival}).duplicated()
    keep = arrival.notna() & ~dup
    recs, h, arrival = recs[keep], h[keep], arrival[keep]
    lead = num("lead_time")
    weekend, week = num("stays_in_weekend_nights").fillna(0), num("stays_in_week_nights").fillna(0)
    flagged = h["predicted_cancel"].astype(int)
    d = pd.DataFrame({
        "hotel": h["hotel"].fillna(text("hotel")), "arrival": arrival,
        "month": arrival.dt.to_period("M").dt.to_timestamp(), "year": arrival.dt.year,
        "booked": arrival - pd.to_timedelta(lead.fillna(0), unit="D"), "lead_time": lead,
        "nights": weekend + week, "weekend_nights": weekend, "adr": num("adr"),
        "cancelled": flagged, "status": np.where(flagged == 1, "Likely to cancel", "Expected to stay"),
        "probability": h["probability"].astype(float)})
    d["value"] = d["adr"] * d["nights"]
    d["stayed"] = 1 - d["cancelled"]
    d["no_show"] = 0
    d["cancel_days_before"] = np.nan
    d["lead_band"] = pd.cut(d["lead_time"], LEAD_BINS, labels=LEAD_LABELS)
    d["stay_band"] = pd.cut(d["nights"], STAY_BINS, labels=STAY_LABELS)
    for field, c in [("deposit_type", "deposit"), ("market_segment", "segment"), ("customer_type", "customer")]:
        d[c] = text(field).map(lambda v, f=field: CATEGORY_LABELS[f].get(v, v))
    d["country"] = text("country").replace({"": "Unknown", "nan": "Unknown"})
    d["guest"] = np.where(num("is_repeated_guest") == 1, "Returning guest", "First-time guest")
    d["agent"] = ""
    d["requests"] = num("total_of_special_requests")
    d["room_changed"] = 0
    return d.reset_index(drop=True)


def findings(d: pd.DataFrame, expected: bool = False) -> list[dict]:
    """Data-backed findings, each with what it means for the hotel and a suggested action.

    With expected=True the data are saved predictions, so the wording speaks of expected, not past, cancellations."""
    def rate(x):
        return (f"are flagged as likely to cancel {x:.0%} of the time" if expected
                else f"were cancelled {x:.0%} of the time")

    small = 10 if expected else 100
    lost = "at risk" if expected else "lost"
    out = []
    lead = by_group(d, "lead_band", order=LEAD_LABELS).set_index("group")
    if {"0–7 days", "181–365 days"} <= set(lead.index):
        lo, hi = lead.loc["0–7 days", "cancel_rate"], lead.loc["181–365 days", "cancel_rate"]
        out.append({"title": "Early bookings carry the most risk" if expected else "Early bookings cancel the most",
                    "finding": f"Bookings made 181–365 days ahead {rate(hi)}, against {lo:.0%} for bookings made "
                               "within a week.",
                    "action": "Ask for a deposit or offer a prepaid rate on bookings made far in advance, and "
                              "reconfirm them about 30 days before arrival.", "kind": "risk"})
    timing = d["cancel_days_before"].dropna()
    if len(timing):
        med, within30 = float(timing.median()), float((timing <= 30).mean())
        out.append({"title": "When guests cancel",
                    "finding": f"Half of all cancellations happened {med:.0f} or more days before arrival, and "
                               f"{within30:.0%} happened in the last 30 days.",
                    "action": "Set the free-cancellation deadline and the reconfirmation date around this pattern, "
                              "so rooms released early can still be resold.", "kind": "info"})
    seg = by_group(d, "segment")
    seg = seg[seg["bookings"] >= 0.05 * len(d)]
    if len(seg):
        worst = seg.sort_values("revenue_lost", ascending=False).iloc[0]
        best = seg.sort_values("cancel_rate").iloc[0]
        out.append({"title": f"{worst['group']} carries the largest revenue {'at risk' if expected else 'loss'}",
                    "finding": f"{worst['group']} bookings {rate(worst['cancel_rate'])}, the largest share of revenue "
                               f"{lost}. The most reliable source is {best['group'].lower()} "
                               f"({best['cancel_rate']:.0%}).",
                    "action": f"Reconfirm {worst['group'].lower()} bookings directly with the guest and encourage "
                              "direct bookings, for example with a best-rate offer.", "kind": "risk"})
    dep = by_group(d, "deposit").set_index("group")
    nr = CATEGORY_LABELS["deposit_type"]["Non Refund"]
    if nr in dep.index and dep.loc[nr, "bookings"] >= small:
        out.append({"title": "Non-refundable bookings still carry risk" if expected
                    else "Non-refundable bookings still cancel",
                    "finding": f"Non-refundable bookings {rate(dep.loc[nr, 'cancel_rate'])}, mostly group and agent "
                               "business.",
                    "action": "Review non-refundable group and agent contracts, and check release dates with the "
                              "organisers.", "kind": "risk"})
    guest = by_group(d, "guest").set_index("group")
    if {"Returning guest", "First-time guest"} <= set(guest.index):
        out.append({"title": "Returning guests are reliable",
                    "finding": f"Returning guests {rate(guest.loc['Returning guest', 'cancel_rate'])}, first-time "
                               f"guests {guest.loc['First-time guest', 'cancel_rate']:.0%}.",
                    "action": "Grow repeat business with a loyalty offer, and give first-time guests a personal "
                              "reconfirmation.", "kind": "good"})
    m = monthly(d)
    if len(m) >= 3:
        peak = m.sort_values("room_revenue", ascending=False).iloc[0]
        low = m.sort_values("room_revenue").iloc[0]
        lead_in = "Expected room revenue is highest in " if expected else "Room revenue was highest in "
        out.append({"title": "Peak and low months",
                    "finding": f"{lead_in}{peak['month']:%B %Y} and lowest in {low['month']:%B %Y}.",
                    "action": "Use the peak months for stricter deposit rules and careful overbooking, and the low "
                              "months for promotions.", "kind": "info"})
    return out

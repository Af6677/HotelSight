"""Hotel insights built from saved predictions.

Weekly figures, where the risk comes from, risk bands, the most common reasons and a suggested action for each.
Pure pandas on the prediction history: the model is not called and nothing is refitted.
Probabilities come from a class-weighted model, so counts of flagged reservations are used, never sums of
probabilities as an "expected number of cancellations".
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .contract import display_value

LEAD_BINS = [-1, 7, 30, 90, 180, 365, np.inf]
LEAD_LABELS = ["0–7 days", "8–30 days", "31–90 days", "91–180 days", "181–365 days", "Over 365 days"]
BAND_EDGES = [0, .30, .50, .70, 1.000001]
BAND_LABELS = ["Low (below 30%)", "Watch (30–50%)", "High (50–70%)", "Very high (70% and above)"]

BREAKDOWNS = {                      # label shown to the manager -> column in prepare()
    "Lead time": "lead_band",
    "Deposit type": "deposit_type",
    "Booking source": "market_segment",
    "Customer type": "customer_type",
    "Guest country (top 10)": "country",
    "Property": "hotel",
}

# Main factor (the friendly name before ':' in top_raising) -> what a hotel can do about it.
# Deposit, source and customer type depend on the value, so they are matched on the full text first.
ACTIONS_BY_TEXT = {
    "Deposit type: No deposit": "Ask for a deposit or a card guarantee, or offer a prepaid rate.",
    "Deposit type: Non-refundable": "Reconfirm with the agent or group organiser. In this data most "
                                    "non-refundable bookings were still cancelled.",
    "Booking source (market segment): Online travel agent": "Contact the guest directly to reconfirm, and invite "
                                                            "them to book direct next time.",
    "Booking source (market segment): Groups": "Check the group's release date and rooming list with the organiser.",
    "Booking source (market segment): Offline agent": "Check the allotment release date with the agent or tour "
                                                     "operator.",
}
ACTIONS_BY_FACTOR = {
    "Lead time (days before arrival)": "Send a reconfirmation about 30 days before arrival.",
    "Guest country": "Send a reconfirmation in the guest's language before arrival.",
    "Special requests": "Contact the guest to personalise the stay, for example room or bed preference.",
    "Car parking spaces": "Mention parking or transfer options in the reconfirmation.",
    "Previous cancellations": "Ask for a deposit or card guarantee for this guest.",
    "Customer type": "Check the contract or group terms for this booking.",
    "Average daily rate (EUR)": "Offer a choice between a flexible rate and a prepaid rate.",
    "Booking source (market segment)": "Reconfirm the booking with the guest or the booking channel.",
    "Deposit type": "Ask for a deposit or a card guarantee.",
}
DEFAULT_ACTION = "Reconfirm the booking with the guest before arrival."


def suggested_action(reason: str) -> str:
    """Suggested hotel action for a 'Factor: value' reason text."""
    if not isinstance(reason, str) or not reason.strip():
        return DEFAULT_ACTION
    for text, action in ACTIONS_BY_TEXT.items():
        if reason.startswith(text):
            return action
    return ACTIONS_BY_FACTOR.get(reason.split(":")[0].strip(), DEFAULT_ACTION)


def risk_band(probability) -> pd.Categorical:
    return pd.cut(pd.Series(probability, dtype=float), BAND_EDGES, labels=BAND_LABELS, right=False)


def prepare(h: pd.DataFrame, countries: dict | None = None) -> pd.DataFrame:
    """history.with_stay() output -> one tidy row per saved prediction for the insight views."""
    recs = pd.DataFrame([json.loads(x or "{}") for x in h["inputs_json"]], index=h.index)
    col = lambda name: recs[name] if name in recs else pd.Series(np.nan, index=h.index)   # noqa: E731
    out = pd.DataFrame(index=h.index)
    out["reference"] = h["reference"] if "reference" in h else None
    out["arrival"] = pd.to_datetime(h["arrival"], errors="coerce", format="%Y-%m-%d")
    out["week"] = out["arrival"] - pd.to_timedelta(out["arrival"].dt.weekday, unit="D")   # Monday of arrival week
    out["probability"] = h["probability"].astype(float)
    out["flagged"] = h["predicted_cancel"].astype(int)
    out["actual"] = pd.to_numeric(h["actual_cancelled"], errors="coerce")
    out["nights"] = pd.to_numeric(h["nights"], errors="coerce")
    out["stay_value"] = pd.to_numeric(h["stay_value"], errors="coerce")
    out["adr"] = pd.to_numeric(col("adr"), errors="coerce")
    out["lead_time"] = pd.to_numeric(col("lead_time"), errors="coerce")
    out["lead_band"] = pd.cut(out["lead_time"], LEAD_BINS, labels=LEAD_LABELS)
    for field in ("hotel", "deposit_type", "market_segment", "customer_type", "country"):
        values = col(field) if field != "hotel" else h["hotel"].where(h["hotel"].notna(), col("hotel"))
        out[field] = [display_value(field, v, countries) if isinstance(v, str) and v else "Not given" for v in values]
    out["risk_band"] = risk_band(out["probability"]).to_numpy()
    out["reason"] = h["top_raising"].fillna("").astype(str).str.strip()
    out["reason_factor"] = out["reason"].str.split(":").str[0].str.strip()
    return out


def summary(df: pd.DataFrame) -> dict:
    """Headline figures for a set of saved predictions."""
    flagged = df[df["flagged"] == 1]
    n = len(df)
    return {"bookings": n, "flagged": len(flagged), "share_flagged": len(flagged) / n if n else 0.0,
            "expected_arrivals": n - len(flagged),
            "revenue": float(df["stay_value"].sum()), "revenue_flagged": float(flagged["stay_value"].sum()),
            "nights_flagged": float(flagged["nights"].sum()),
            "avg_rate": float(df["adr"].mean()) if df["adr"].notna().any() else float("nan")}


def weekly(df: pd.DataFrame) -> pd.DataFrame:
    """One row per arrival week (Monday start): bookings, flagged, share, revenue, expected arrivals, outcomes."""
    d = df.dropna(subset=["week"])
    if d.empty:
        return pd.DataFrame(columns=["week", "bookings", "flagged", "share_flagged", "expected_arrivals",
                                     "revenue", "revenue_flagged", "nights_flagged", "known", "actual_cancelled"])
    g = d.groupby("week")
    flagged = d[d["flagged"] == 1].groupby("week")
    known = d.dropna(subset=["actual"]).groupby("week")
    w = pd.DataFrame({"bookings": g.size(), "flagged": g["flagged"].sum(), "revenue": g["stay_value"].sum(),
                      "revenue_flagged": flagged["stay_value"].sum(), "nights_flagged": flagged["nights"].sum(),
                      "known": known.size(), "actual_cancelled": known["actual"].sum()}).fillna(0)
    w["share_flagged"] = w["flagged"] / w["bookings"]
    w["expected_arrivals"] = w["bookings"] - w["flagged"]
    for c in ("bookings", "flagged", "known", "actual_cancelled", "expected_arrivals"):
        w[c] = w[c].astype(int)
    return w.reset_index().sort_values("week").reset_index(drop=True)


def breakdown(df: pd.DataFrame, column: str, top: int | None = None) -> pd.DataFrame:
    """Share flagged and revenue flagged per group of one input. With top, the smaller groups become 'Other'."""
    d = df.copy()
    if top is not None:
        keep = d[column].value_counts().head(top).index
        d[column] = d[column].where(d[column].isin(keep), "Other")
    g = d.groupby(column, observed=True, sort=False)
    out = pd.DataFrame({"bookings": g.size(), "flagged": g["flagged"].sum(),
                        "revenue_flagged": d[d["flagged"] == 1].groupby(column, observed=True)["stay_value"].sum()})
    out = out.fillna({"revenue_flagged": 0.0})
    out["share_flagged"] = out["flagged"] / out["bookings"]
    out = out.reset_index().rename(columns={column: "group"})
    out["group"] = out["group"].astype(str)
    if column == "lead_band":                                   # keep the natural order of lead time
        out["order"] = out["group"].map({k: i for i, k in enumerate(LEAD_LABELS)})
        return out.sort_values("order").drop(columns="order").reset_index(drop=True)
    return out.sort_values(["share_flagged", "bookings"], ascending=False).reset_index(drop=True)


def bands(df: pd.DataFrame) -> pd.DataFrame:
    """Reservations and booked revenue in each risk band (all four bands, even when empty)."""
    g = df.groupby("risk_band", observed=False)
    out = pd.DataFrame({"bookings": g.size(), "revenue": g["stay_value"].sum()}).reindex(BAND_LABELS).fillna(0)
    out["bookings"] = out["bookings"].astype(int)
    return out.reset_index().rename(columns={"risk_band": "band", "index": "band"})


def reasons(df: pd.DataFrame, top: int = 8) -> pd.DataFrame:
    """Most common main reasons among flagged reservations, with the share, revenue and a suggested action."""
    f = df[(df["flagged"] == 1) & (df["reason"] != "")]
    if f.empty:
        return pd.DataFrame(columns=["reason", "bookings", "share_of_flagged", "revenue_flagged", "action"])
    key = f["reason"].where(f["reason"].map(lambda r: any(r.startswith(t) for t in ACTIONS_BY_TEXT)),
                            f["reason_factor"]).rename("reason")
    g = f.groupby(key)
    out = pd.DataFrame({"bookings": g.size(), "revenue_flagged": g["stay_value"].sum()}).reset_index()
    out["share_of_flagged"] = out["bookings"] / len(f)
    out["action"] = out["reason"].map(suggested_action)
    return out.sort_values("bookings", ascending=False).head(top).reset_index(drop=True)


def attention_window(df: pd.DataFrame, today=None, weeks: int = 4) -> tuple[pd.DataFrame, str]:
    """The saved predictions that need attention now: arrivals from today, or else the latest weeks with arrivals."""
    today = pd.Timestamp(today if today is not None else pd.Timestamp.today()).normalize()
    d = df.dropna(subset=["arrival"])
    upcoming = d[d["arrival"] >= today]
    if len(upcoming):
        return upcoming, "Arrivals from today"
    if d.empty:
        return d, "No saved arrival dates"
    last = d["week"].max()
    start = last - pd.Timedelta(weeks=weeks - 1)
    end = last + pd.Timedelta(days=6)
    return d[d["week"] >= start], f"No upcoming arrivals. Latest {weeks} weeks saved: {start:%d %b %Y} – {end:%d %b %Y}"


def contact_list(df: pd.DataFrame) -> pd.DataFrame:
    """Flagged reservations, soonest arrival first and then highest risk, each listed once, with a suggested action."""
    f = df[df["flagged"] == 1]
    named = f["reference"].notna() & (f["reference"].astype(str).str.strip() != "")
    f = pd.concat([f[named].drop_duplicates(["reference", "arrival"]), f[~named]])
    f = f.sort_values(["arrival", "probability"], ascending=[True, False], na_position="last")
    return f.assign(action=f["reason"].map(suggested_action))


def outcome_check(df: pd.DataFrame) -> dict | None:
    """Predicted against actual for saved predictions whose real outcome is known."""
    k = df.dropna(subset=["actual"])
    if k.empty:
        return None
    y, p = k["actual"].astype(int).to_numpy(), k["flagged"].to_numpy()
    tp, fp = int(((y == 1) & (p == 1)).sum()), int(((y == 0) & (p == 1)).sum())
    fn, tn = int(((y == 1) & (p == 0)).sum()), int(((y == 0) & (p == 0)).sum())
    return {"known": len(k), "predicted_share": float(p.mean()), "actual_share": float(y.mean()),
            "accuracy": (tp + tn) / len(k), "f1": 2 * tp / max(2 * tp + fp + fn, 1),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}

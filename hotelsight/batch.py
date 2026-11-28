"""Many bookings at once: read a spreadsheet, map it onto the 24-input contract, validate, predict.

Two layouts are accepted:
  * "template" – one row per booking with three dates (booking, arrival, departure) and the other
    inputs, as in the downloadable template. The date inputs are derived exactly as on the form.
  * "dataset"  – the Hotel Booking Demand layout, which already has lead time, arrival month, week
    number, day of month and the two night counts. Agent/company IDs become Yes/No flags with the
    experiment's own rule, and an `is_canceled` column is kept as the actual outcome.
Every row goes through the same validation as the form; rows with errors are reported, not scored.
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from datetime import date, datetime

import numpy as np
import pandas as pd

from .contract import (CATEGORY_LABELS, DATE_DERIVED, FRIENDLY_NAME, MODEL_FIELDS, MONTHS, YES_NO_FIELDS,
                       derive_from_dates, display_value, validate_dates, validate_record)
from .paths import DATA, MEMBERSHIP
from .pipeline import experiment_features

MAX_ROWS = 10_000
DATE_COLUMNS = ["booking_date", "arrival_date", "departure_date"]
TEMPLATE_COLUMNS = ["booking_reference", *DATE_COLUMNS] + [f for f in MODEL_FIELDS if f not in DATE_DERIVED]
CATEGORICAL_FIELDS = ["hotel", "meal", "customer_type", "market_segment", "country", "deposit_type"]
NUMERIC_FIELDS = [f for f in MODEL_FIELDS if f not in CATEGORICAL_FIELDS + YES_NO_FIELDS + ["arrival_date_month"]]


def normalise(name) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


ALIASES = {
    **{normalise(label): f for f, label in FRIENDLY_NAME.items()},
    **{f: f for f in MODEL_FIELDS + DATE_COLUMNS + ["agent", "company", "arrival_date_year"]},
    **{a: "booking_reference" for a in ["booking_reference", "reference", "booking_ref", "booking_id", "booking_no",
                                        "booking_number", "reservation_id", "reservation_no", "id"]},
    "check_in_date": "arrival_date", "check_in": "arrival_date", "checkin_date": "arrival_date",
    "check_out_date": "departure_date", "check_out": "departure_date", "checkout_date": "departure_date",
    "reservation_date": "booking_date", "date_booked": "booking_date",
    "is_canceled": "actual_cancelled", "is_cancelled": "actual_cancelled", "actual_cancelled": "actual_cancelled",
    "cancelled": "actual_cancelled", "canceled": "actual_cancelled",
}


@dataclass
class Row:
    row: int                                  # spreadsheet row number (header is row 1)
    reference: str
    record: dict | None = None
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    actual: int | None = None
    arrival: str = ""


@dataclass
class Parsed:
    layout: str
    rows: list
    ignored_columns: list

    @property
    def valid(self) -> list:
        return [r for r in self.rows if not r.errors]


# ---------------------------------------------------------------- reading
def read_table(data: bytes, filename: str, sheet: str | int = 0) -> pd.DataFrame:
    """CSV or Excel (.xlsx) -> DataFrame of raw cell values (strings, numbers or dates)."""
    name = filename.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(data), dtype=str, keep_default_na=False, encoding_errors="replace")
    elif name.endswith((".xlsx", ".xlsm")):
        df = pd.read_excel(io.BytesIO(data), sheet_name=sheet, dtype=object)
    else:
        raise ValueError("Please upload an Excel workbook (.xlsx) or a CSV file.")
    df = df.dropna(how="all")
    df = df.loc[~df.apply(lambda r: all(_blank(v) for v in r), axis=1)]
    if len(df) > MAX_ROWS:
        raise ValueError(f"The file has {len(df):,} bookings. Please upload at most {MAX_ROWS:,} at a time.")
    return df


def sheet_names(data: bytes, filename: str) -> list:
    if not filename.lower().endswith((".xlsx", ".xlsm")):
        return []
    return pd.ExcelFile(io.BytesIO(data)).sheet_names


# pandas' default missing-value markers, which is how the experiment read the dataset (e.g. children "NA")
MISSING_TEXT = {"", "#n/a", "#n/a n/a", "#na", "-1.#ind", "-1.#qnan", "-nan", "1.#ind", "1.#qnan", "<na>", "n/a",
                "na", "null", "nan", "none"}


def _blank(v) -> bool:
    if v is None or v is pd.NaT:
        return True
    if isinstance(v, float):
        return bool(np.isnan(v))
    return isinstance(v, str) and v.strip().lower() in MISSING_TEXT


# ---------------------------------------------------------------- value parsing
def _number(v, name, errors):
    if _blank(v):
        return None
    try:
        return float(str(v).replace(",", "")) if isinstance(v, str) else float(v)
    except (TypeError, ValueError):
        errors.append(f"{name} must be a number (got '{v}').")
        return None


def _flag(v, name, errors):
    text = str(v).strip().lower() if not _blank(v) else ""
    if text in ("yes", "y", "true", "1", "1.0"):
        return 1.0
    if text in ("no", "n", "false", "0", "0.0"):
        return 0.0
    errors.append(f"{name} must be Yes or No" + (f" (got '{v}')." if text else " (it is empty)."))
    return None


def _date(v, name, errors):
    if _blank(v):
        errors.append(f"{name} is empty.")
        return None
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    text = str(v).strip()
    try:
        return datetime.strptime(text[:10], "%Y-%m-%d").date()
    except ValueError:
        pass
    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%d %b %Y", "%d %B %Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    errors.append(f"{name} '{v}' is not a date. Use YYYY-MM-DD (for example 2026-03-14) or DD/MM/YYYY.")
    return None


def _month(v, errors):
    if _blank(v):
        return None
    text = str(v).strip()
    for m in MONTHS:
        if text.lower() in (m.lower(), m[:3].lower()):
            return m
    try:
        n = int(float(text))
        if 1 <= n <= 12:
            return MONTHS[n - 1]
    except ValueError:
        pass
    errors.append(f"Arrival month '{v}' is not a month name or number.")
    return None


def _lookups(categories: dict, countries: dict) -> dict:
    """Accept the dataset code or the dashboard label, case-insensitively, for every category."""
    out = {}
    for f in CATEGORICAL_FIELDS:
        m = {str(c).lower(): str(c) for c in categories.get(f, [])}
        m.update({label.lower(): code for code, label in CATEGORY_LABELS.get(f, {}).items()})
        if f == "country":
            m.update({name.lower(): code for code, name in countries.items()})
        out[f] = m
    return out


def _category(v, lookup):
    if _blank(v):
        return "Unknown"                       # the experiment's fixed rule for missing categories
    text = str(v).strip()
    return lookup.get(text.lower(), text)      # unseen values are kept and flagged by validate_record


# ---------------------------------------------------------------- mapping + validation
def prepare(raw: pd.DataFrame, categories: dict, ranges: dict, countries: dict) -> Parsed:
    """Rename columns, work out the layout, then build and validate one record per row."""
    renamed, ignored = {}, []
    for col in raw.columns:
        target = ALIASES.get(normalise(col))
        if target and target not in renamed.values():
            renamed[col] = target
        else:
            ignored.append(str(col))
    df = raw.rename(columns=renamed)[list(renamed.values())]
    cols = set(df.columns)

    if set(DATE_COLUMNS) <= cols:
        layout = "template"
    elif set(DATE_DERIVED) <= cols:
        layout = "dataset"
    else:
        missing = [c for c in DATE_COLUMNS if c not in cols]
        raise ValueError("Could not find the stay dates. Add the columns " + ", ".join(missing) +
                         " (or download the template to see the expected layout).")
    needed = [f for f in MODEL_FIELDS if f not in DATE_DERIVED]
    missing = [f for f in needed if f not in cols and not (f == "has_travel_agent" and "agent" in cols)
               and not (f == "company_booking" and "company" in cols)]
    if missing:
        raise ValueError("Missing columns: " + ", ".join(f"{m} ({FRIENDLY_NAME[m]})" for m in missing) + ".")

    lookups = _lookups(categories, countries)
    rows = []
    for i, (_, r) in enumerate(df.iterrows()):
        ref = r.get("booking_reference")
        out = Row(row=i + 2, reference=f"Row {i + 2}" if _blank(ref) else str(ref).strip())
        errs = out.errors
        rec = {}
        if layout == "template":
            b, a, d = (_date(r[c], FRIENDLY_NAME.get(c, c.replace("_", " ").capitalize()), errs) for c in DATE_COLUMNS)
            if None not in (b, a, d):
                dv = validate_dates(b, a, d)
                errs += dv.errors
                out.warnings += dv.warnings
                if dv.ok:
                    rec.update(derive_from_dates(b, a, d))
                    out.arrival = a.isoformat()
        else:
            for f in DATE_DERIVED:
                rec[f] = _month(r[f], errs) if f == "arrival_date_month" else _number(r[f], FRIENDLY_NAME[f], errs)
            year = _number(r.get("arrival_date_year"), "Arrival year", []) if "arrival_date_year" in cols else None
            if rec.get("arrival_date_month") and rec.get("arrival_date_day_of_month"):
                day = int(rec["arrival_date_day_of_month"])
                out.arrival = (f"{int(year)}-{MONTHS.index(rec['arrival_date_month']) + 1:02d}-{day:02d}" if year
                               else f"{day} {rec['arrival_date_month']}")
        for f in needed:
            if f in YES_NO_FIELDS and f not in cols:              # dataset layout: agent / company IDs
                source = r["agent" if f == "has_travel_agent" else "company"]
                rec[f] = float(experiment_features.present_identifier(pd.Series([None if _blank(source) else source]))[0])
            elif f in YES_NO_FIELDS:
                rec[f] = _flag(r[f], FRIENDLY_NAME[f], errs)
            elif f in CATEGORICAL_FIELDS:
                rec[f] = _category(r[f], lookups[f])
            else:
                rec[f] = _number(r[f], FRIENDLY_NAME[f], errs)
        if "actual_cancelled" in cols and not _blank(r["actual_cancelled"]):
            out.actual = 1 if str(r["actual_cancelled"]).strip().lower() in ("1", "1.0", "yes", "y", "true") else 0
        if not errs:
            rec = {f: rec[f] for f in MODEL_FIELDS}
            v = validate_record(rec, ranges, categories)
            errs += v.errors
            out.warnings += v.warnings
            out.record = rec
        rows.append(out)
    return Parsed(layout=layout, rows=rows, ignored_columns=ignored)


# ---------------------------------------------------------------- prediction
def predict(model, parsed: Parsed, countries: dict, chunk: int = 1000, progress=None) -> pd.DataFrame:
    """Score every valid row (with TreeSHAP for the main factors) and return one result row per input row."""
    valid = parsed.valid
    scored = {}
    for start in range(0, len(valid), chunk):
        part = valid[start:start + chunk]
        prob, contrib = model.fast_contributions(model.frame([r.record for r in part]))
        for j, r in enumerate(part):
            c = contrib.iloc[j]
            up, down = c.idxmax(), c.idxmin()
            scored[r.row] = {
                "probability": float(prob[j]),
                "raise": f"{FRIENDLY_NAME[up]}: {display_value(up, r.record[up], countries)}" if c[up] > 0 else "",
                "lower": f"{FRIENDLY_NAME[down]}: {display_value(down, r.record[down], countries)}" if c[down] < 0 else "",
            }
        if progress:
            progress(min(start + chunk, len(valid)) / max(len(valid), 1))
    out = []
    for r in parsed.rows:
        s = scored.get(r.row)
        rec = r.record or {}
        nights = (rec.get("stays_in_weekend_nights") or 0) + (rec.get("stays_in_week_nights") or 0) if rec else None
        rate = rec.get("adr")
        item = {"Row": r.row, "Reference": r.reference, "Hotel": rec.get("hotel", ""), "Arrival": r.arrival,
                "Lead time (days)": rec.get("lead_time"), "Nights": nights, "Room rate (EUR)": rate,
                "Stay value (EUR)": rate * nights if rate is not None and nights is not None else None,
                "Cancellation probability": s["probability"] if s else None,
                "Prediction": ("Likely to cancel" if s["probability"] >= model.threshold else "Likely to stay") if s else "Not scored",
                "Main factor raising risk": s["raise"] if s else "",
                "Main factor lowering risk": s["lower"] if s else "",
                "Problems": " ".join(r.errors), "Warnings": " ".join(r.warnings)}
        if any(x.actual is not None for x in parsed.rows):
            item["Actual outcome"] = {1: "Cancelled", 0: "Not cancelled"}.get(r.actual, "")
        out.append(item)
    df = pd.DataFrame(out)
    if "Actual outcome" in df:                 # put the prediction and what really happened side by side
        where = df.columns.get_loc("Prediction") + 1
        df.insert(where, "Actual outcome", df.pop("Actual outcome"))
        expected = df["Prediction"].map({"Likely to cancel": "Cancelled", "Likely to stay": "Not cancelled"})
        checked = expected.notna() & df["Actual outcome"].ne("")
        df.insert(where + 1, "Correct?", np.where(~checked, "", np.where(expected == df["Actual outcome"], "Correct", "Wrong")))
    return df


# ---------------------------------------------------------------- files for download
def to_excel(sheets: dict) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xl:
        for name, df in sheets.items():
            df.to_excel(xl, sheet_name=name[:31], index=False)
            ws = xl.sheets[name[:31]]
            for col in ws.columns:                                   # readable column widths
                width = max(len(str(c.value)) if c.value is not None else 0 for c in col[:200])
                ws.column_dimensions[col[0].column_letter].width = min(max(width + 2, 10), 60)
            ws.freeze_panes = "A2"
    return buf.getvalue()


def template(examples: list, categories: dict) -> bytes:
    """Excel template: the dashboard's example bookings plus a sheet explaining every column."""
    rows = []
    for ex in examples:
        inp = ex["inputs"]
        row = {"booking_reference": f"EXAMPLE-{ex['source_csv_row']}", "booking_date": ex["booking_date"],
               "arrival_date": ex["arrival_date"], "departure_date": ex["departure_date"]}
        for f in TEMPLATE_COLUMNS[4:]:
            v = inp[f]
            row[f] = ("Yes" if v == 1 else "No") if f in YES_NO_FIELDS else (int(v) if isinstance(v, float) and
                                                                              f != "adr" and v.is_integer() else v)
        rows.append(row)
    guide = []
    for c in TEMPLATE_COLUMNS:
        if c == "booking_reference":
            desc, allowed = "Your own booking ID (optional)", "Any text"
        elif c in DATE_COLUMNS:
            desc, allowed = c.replace("_", " ").capitalize(), "Date: YYYY-MM-DD, DD/MM/YYYY or an Excel date"
        elif c in YES_NO_FIELDS:
            desc, allowed = FRIENDLY_NAME[c], "Yes or No"
        elif c == "country":
            desc, allowed = FRIENDLY_NAME[c], "ISO 3-letter code (e.g. PRT, GBR, MYS) or the country name"
        elif c in CATEGORICAL_FIELDS:
            desc, allowed = FRIENDLY_NAME[c], ", ".join(str(x) for x in categories.get(c, []))
        elif c == "adr":
            desc, allowed = FRIENDLY_NAME[c], "Number of euros per night, 0 or more"
        else:
            desc, allowed = FRIENDLY_NAME[c], "Whole number, 0 or more"
        guide.append({"Column": c, "Meaning": desc, "Allowed values": allowed})
    notes = pd.DataFrame({"Notes": [
        "One row per booking. Keep the column names in row 1; the column order does not matter.",
        "Lead time, arrival month, week number, day of month and night counts are calculated from the three dates.",
        "Rows with problems (for example a missing date or zero guests) are listed but not scored.",
        f"At most {MAX_ROWS:,} bookings per file.",
        "The Hotel Booking Demand dataset layout (lead_time, arrival_date_month, ... , agent, company) is also accepted.",
    ]})
    return to_excel({"Bookings": pd.DataFrame(rows, columns=TEMPLATE_COLUMNS), "Column guide": pd.DataFrame(guide),
                     "Notes": notes})


def sample_upcoming_bookings(n: int = 200, seed: int = 7, today: date | None = None, weeks: int = 8) -> pd.DataFrame:
    """Demonstration only: real held-out test bookings moved to arrive in the next `weeks` weeks, without an outcome.

    Every booking detail is kept (lead time, nights, guests, rate, deposit and so on). Only the arrival date changes,
    and the fields the dataset derives from it (year, month, day, week number, weekend and week nights) are worked out
    again with the dataset's own rules. The real outcome is removed, because these arrivals have not happened yet."""
    from datetime import timedelta
    from .contract import dataset_weekend_nights, tsql_week_number
    today = today or date.today()
    out = sample_test_bookings(n, seed)
    offsets = np.random.default_rng(seed).integers(1, weeks * 7 + 1, size=len(out))
    for i, off in enumerate(offsets):
        arrival = today + timedelta(days=int(off))
        nights = int(float(out.at[i, "stays_in_weekend_nights"] or 0)) + int(float(out.at[i, "stays_in_week_nights"] or 0))
        weekend, week = dataset_weekend_nights(arrival, arrival + timedelta(days=nights))
        out.at[i, "arrival_date_year"] = str(arrival.year)
        out.at[i, "arrival_date_month"] = MONTHS[arrival.month - 1]
        out.at[i, "arrival_date_day_of_month"] = str(arrival.day)
        out.at[i, "arrival_date_week_number"] = str(tsql_week_number(arrival))
        out.at[i, "stays_in_weekend_nights"] = str(weekend)
        out.at[i, "stays_in_week_nights"] = str(week)
    out["booking_reference"] = out["booking_reference"].str.replace("CSV-", "DEMO-", regex=False)
    return out.drop(columns=[c for c in ("is_canceled",) if c in out]).reset_index(drop=True)


def sample_test_bookings(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Held-out test bookings in the dataset layout (with their real outcome), for demonstrations."""
    data = pd.read_csv(DATA, dtype=str, keep_default_na=False)
    data.insert(0, "source_csv_row", np.arange(2, len(data) + 2))
    test = pd.read_csv(MEMBERSHIP, usecols=["source_csv_row", "split"]).query("split == 'test'")
    rows = data[data.source_csv_row.isin(test.source_csv_row)].sample(n, random_state=seed).sort_values("source_csv_row")
    rows.insert(0, "booking_reference", "CSV-" + rows.pop("source_csv_row").astype(str))
    return rows.drop(columns=["reservation_status", "reservation_status_date", "assigned_room_type",
                              "booking_changes", "reserved_room_type", "distribution_channel"]).reset_index(drop=True)

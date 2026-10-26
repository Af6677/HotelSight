"""Build the dashboard's derived assets from the frozen experiment (no training).

Outputs in assets/:
  country_names.json        training-set country codes -> display names
  input_ranges.json         numeric input ranges in the retained TRAINING split
  example_bookings.json     real held-out test bookings expressed as dashboard form inputs
  global_shap_importance.csv mean |SHAP| per raw input (test set, deployed model)
  global_shap_by_value.csv  mean SHAP per category / value band
  shap_summary.json         base value, additivity check, rows used

Run:  python scripts/build_assets.py
"""
import json, sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(1, str(ROOT / "experiment"))
from hotelsight.paths import ASSETS, DATA, MEMBERSHIP, PAIRED_TEST, ARM
from hotelsight.contract import MODEL_FIELDS, derive_from_dates
from hotelsight.pipeline import DeployedModel
from experiment import clean                      # the experiment's own cleaning

ASSETS.mkdir(exist_ok=True)
m = DeployedModel()
d, _ = clean(DATA)
d = d.merge(pd.read_csv(MEMBERSHIP)[["source_csv_row", "split"]], on="source_csv_row", validate="one_to_one")
train = d[d.split == "train"]
x_train = m.features(train)

# ---- 1. country names
import pycountry
names, manual = {}, {"Unknown": "Not recorded", "CN": "CN (code as recorded in the dataset)",
                     "TMP": "Timor-Leste (old code TMP)"}
for code in m.categories["country"]:
    if code in manual:
        names[code] = manual[code]; continue
    c = pycountry.countries.get(alpha_3=code)
    names[code] = f"{getattr(c, 'common_name', None) or c.name}" if c else f"{code} (code as recorded)"
(ASSETS / "country_names.json").write_text(json.dumps(names, indent=1, ensure_ascii=False), encoding="utf-8")
print("countries:", len(names), "unmatched:", [k for k, v in names.items() if "as recorded" in v])

# ---- 2. numeric input ranges in the retained training split
num = [f for f in MODEL_FIELDS if f not in m.categories and f not in ("has_travel_agent", "company_booking", "is_repeated_guest")]
ranges = {f: {"min": float(x_train[f].min()), "max": float(x_train[f].max()), "median": float(x_train[f].median())} for f in num}
(ASSETS / "input_ranges.json").write_text(json.dumps(ranges, indent=1), encoding="utf-8")

# ---- 3. example bookings from the held-out test set, as dashboard form inputs
paired = pd.read_csv(PAIRED_TEST, float_precision="round_trip")
test = d.set_index("source_csv_row").loc[paired.source_csv_row].reset_index()
test["saved_probability"] = paired[ARM].values
ok = test.country.isin([c for c in m.categories["country"] if c not in ("Unknown", "CN")]) & \
     ((test.stays_in_weekend_nights + test.stays_in_week_nights) > 0) & paired.one_copy.values
picks = {
    "High risk – non-refundable, long lead time": test[ok & (test.is_canceled == 1) & (test.deposit_type == "Non Refund")].sort_values("lead_time").iloc[len(test) // 400],
    "Low risk – direct booking with special requests": test[ok & (test.is_canceled == 0) & (test.market_segment == "Direct") & (test.total_of_special_requests >= 2) & (test.saved_probability < 0.05)].iloc[0],
    "Borderline – online travel agent": test[ok & (test.market_segment == "Online TA")].assign(gap=lambda t: (t.saved_probability - 0.5).abs()).sort_values("gap").iloc[0],
    "Missed cancellation (model error)": test[ok & (test.is_canceled == 1) & (test.saved_probability < 0.3)].iloc[0],
}
examples = []
for title, r in picks.items():
    arrival = datetime.strptime(f"{int(r.arrival_date_year)}-{r.arrival_date_month}-{int(r.arrival_date_day_of_month)}", "%Y-%B-%d").date()
    booking = arrival - timedelta(days=int(r.lead_time))
    departure = arrival + timedelta(days=int(r.stays_in_weekend_nights + r.stays_in_week_nights))
    derived = derive_from_dates(booking, arrival, departure)
    rec = m.features(pd.DataFrame([r]))[MODEL_FIELDS].iloc[0].to_dict()
    assert all(derived[k] == rec[k] for k in derived), (title, derived, {k: rec[k] for k in derived})
    examples.append({"title": title, "source_csv_row": int(r.source_csv_row), "actual_cancelled": int(r.is_canceled),
                     "saved_probability": float(r.saved_probability),
                     "booking_date": booking.isoformat(), "arrival_date": arrival.isoformat(), "departure_date": departure.isoformat(),
                     "inputs": {k: (v.item() if hasattr(v, "item") else v) for k, v in rec.items() if k not in derived}})
(ASSETS / "example_bookings.json").write_text(json.dumps(examples, indent=1), encoding="utf-8")
print("examples:", [(e["title"], round(e["saved_probability"], 3), e["actual_cancelled"]) for e in examples])

# ---- 4. global SHAP on the held-out test set (deployed model)
ex = m.explain(test)
contrib, one = ex["contributions"], paired.one_copy.values
assert np.allclose(ex["probability"], test.saved_probability, atol=1e-6)
imp = pd.DataFrame({"input": MODEL_FIELDS,
                    "mean_abs_shap_repeated": contrib.abs().mean().values,
                    "mean_abs_shap_one_copy": contrib[one].abs().mean().values,
                    "mean_shap_repeated": contrib.mean().values}).sort_values("mean_abs_shap_repeated", ascending=False)
imp.to_csv(ASSETS / "global_shap_importance.csv", index=False)

raw = m.features(test)
rows = []
for f in MODEL_FIELDS:
    s, v = contrib[f], raw[f]
    if f in m.categories or f in ("has_travel_agent", "company_booking", "is_repeated_guest"):
        g = pd.DataFrame({"value": v.astype(str), "shap": s}).groupby("value").shap.agg(["mean", "size"])
        g = g.sort_values("size", ascending=False).head(12)
        rows += [{"input": f, "value": k, "mean_shap": float(r["mean"]), "rows": int(r["size"]), "order": i} for i, (k, r) in enumerate(g.iterrows())]
    else:
        bins = pd.qcut(v.rank(method="first"), q=min(8, v.nunique()), labels=False)
        g = pd.DataFrame({"bin": bins, "v": v, "shap": s}).groupby("bin").agg(lo=("v", "min"), hi=("v", "max"), mean=("shap", "mean"), size=("shap", "size"))
        rows += [{"input": f, "value": (f"{r.lo:g}" if r.lo == r.hi else f"{r.lo:g}–{r.hi:g}"), "mean_shap": float(r["mean"]), "rows": int(r["size"]), "order": int(b)} for b, r in g.iterrows()]
pd.DataFrame(rows).to_csv(ASSETS / "global_shap_by_value.csv", index=False)

summary = {"computed_utc": datetime.now(timezone.utc).isoformat(), "arm": ARM, "explainer": "shap.TreeExplainer (tree_path_dependent)",
           "units": "log-odds of cancellation", "base_log_odds": ex["base_log_odds"],
           "base_probability": float(1 / (1 + np.exp(-ex["base_log_odds"]))),
           "rows_repeated": int(len(test)), "rows_one_copy": int(one.sum()),
           "max_reconstruction_error_log_odds": float(ex["reconstruction_error"].max()),
           "max_probability_vs_saved_error": float(np.abs(ex["probability"] - test.saved_probability).max())}
(ASSETS / "shap_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
print(json.dumps(summary, indent=1)); print(imp.head(10).round(4).to_string(index=False))

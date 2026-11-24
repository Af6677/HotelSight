"""Read-only access to the frozen experiment's recorded results.

Every number the dashboard shows comes from these files, and all of them
belong to the same experiment (plan.json / selection_before_test.json /
test_results.json). Nothing is recomputed from a different model.
"""
from __future__ import annotations

import json

import pandas as pd

from .paths import ARM, ARM_MODELS, ASSETS, EXPERIMENT, PAIRED_COMPARISONS, PLAN, SELECTION, TEST_RESULTS

MODEL_SLUGS = ["logistic_regression", "random_forest", "xgboost", "catboost"]
ARM_LABEL = {"retained_historical24": "Repeated records kept · 24 inputs (DEPLOYED)",
             "retained_general20": "Repeated records kept · 20 inputs",
             "deduplicated_historical24": "One copy per record · 24 inputs",
             "deduplicated_general20": "One copy per record · 20 inputs"}
METRICS = ["f1", "precision", "recall", "accuracy", "roc_auc"]
METRIC_LABEL = {"f1": "F1-score", "precision": "Precision", "recall": "Recall", "accuracy": "Accuracy", "roc_auc": "ROC AUC"}


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def plan() -> dict:
    return _read(PLAN)


def selected() -> dict:
    return _read(SELECTION)["winners"][ARM]


def validation_comparison() -> pd.DataFrame:
    """Best configuration of each of the four models in the deployed arm (validation set)."""
    rows = []
    for slug in MODEL_SLUGS:
        b = _read(ARM_MODELS / slug / "best.json")
        rows.append({"Model": b["model"], "Configuration": b["configuration"],
                     **{METRIC_LABEL[k]: b["validation"][k] for k in METRICS},
                     "Train F1": b["train"]["f1"], "Settings": json.dumps(b["manual_parameters"])})
    return pd.DataFrame(rows)


def validation_comparison_one_copy() -> pd.DataFrame:
    """The same four models and settings, with each identical validation record counted once."""
    rows = []
    for slug in MODEL_SLUGS:
        b = _read(ARM_MODELS / slug / "best.json")
        rows.append({"Model": b["model"], "Configuration": b["configuration"],
                     **{METRIC_LABEL[k]: b["one_copy_validation"][k] for k in METRICS}})
    return pd.DataFrame(rows)


def trials() -> pd.DataFrame:
    """All 12 configurations tried in the deployed arm (3 per model)."""
    t = _read(EXPERIMENT / "results" / "retained" / "historical24_trials.json")
    return pd.DataFrame([{"Model": r["model"], "Configuration": r["configuration"],
                          "Settings": json.dumps(r["manual_parameters"]),
                          "Validation F1": r["validation"]["f1"], "Validation AUC": r["validation"]["roc_auc"],
                          "Train F1": r["train"]["f1"]} for r in t])


def test_metrics() -> dict:
    """Sealed test metrics of the deployed model: repeated-record and one-copy populations."""
    r = _read(TEST_RESULTS)["results"][ARM]
    return {"repeated": r["retained_test"], "one_copy": r["one_copy_test"], "model": r["model"]}


def sensitivity() -> pd.DataFrame:
    """All four frozen arm winners, scored on identical test rows (same experiment)."""
    res = _read(TEST_RESULTS)["results"]
    rows = []
    for arm in ["retained_historical24", "deduplicated_historical24", "retained_general20", "deduplicated_general20"]:
        r = res[arm]
        rows.append({"Experiment arm": ARM_LABEL[arm], "Selected model": r["model"],
                     "Validation F1": r["validation"]["f1"],
                     "Test F1 (repeated)": r["retained_test"]["f1"],
                     "Test accuracy (repeated)": r["retained_test"]["accuracy"],
                     "Test F1 (one copy)": r["one_copy_test"]["f1"],
                     "Test accuracy (one copy)": r["one_copy_test"]["accuracy"]})
    return pd.DataFrame(rows)


def bootstrap() -> pd.DataFrame:
    names = {"duplicate_policy_general20": "Keep vs remove repeats (20 inputs)",
             "duplicate_policy_historical24": "Keep vs remove repeats (24 inputs)",
             "extra_features_deduplicated": "24 vs 20 inputs (repeats removed)",
             "extra_features_retained": "24 vs 20 inputs (repeats kept)"}
    pops = {"retained": "Repeated records", "one_copy": "One copy per record"}
    return pd.DataFrame([{"Comparison": names[c["comparison"]], "Test population": pops[c["population"]],
                          "F1 difference (points)": 100 * c["f1_difference"],
                          "95% interval": f"{100 * c['lower_95']:+.2f} to {100 * c['upper_95']:+.2f}",
                          "Interval includes 0": "Yes" if c["interval_includes_zero"] else "No"}
                         for c in _read(PAIRED_COMPARISONS)])


def global_shap() -> pd.DataFrame:
    return pd.read_csv(ASSETS / "global_shap_importance.csv")


def global_shap_by_value() -> pd.DataFrame:
    return pd.read_csv(ASSETS / "global_shap_by_value.csv", keep_default_na=False)


def shap_summary() -> dict:
    return _read(ASSETS / "shap_summary.json")


def examples() -> list:
    return _read(ASSETS / "example_bookings.json")

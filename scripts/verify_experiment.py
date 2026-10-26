"""Verify the frozen `retained_historical24` experiment before the dashboard uses it.

Checks (all executed, results written to docs/verification_report.json):
 1. File hashes of the selected model and preprocessor match the frozen selection record.
 2. The preprocessor loads without version warnings in this environment.
 3. Raw CSV -> shared feature construction -> saved preprocessing -> saved model
    reproduces ALL saved validation probabilities and ALL saved test probabilities.
 4. Recomputed metrics equal the sealed test_results.json values
    (repeated-record and one-copy test populations).
 5. The four-model validation comparison is recomputed from each model's saved
    validation probabilities and matches its best.json record.

Run:  python scripts/verify_experiment.py
"""
import hashlib, json, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, joblib
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiment"
sys.path.insert(0, str(EXP))
from features import make_features            # the experiment's own shared feature contract
from experiment import clean, load_model       # the experiment's own cleaning and model loader

ARM = "retained_historical24"
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text(encoding="utf-8"))
report, ok = {}, True

def check(name, passed, detail):
    global ok
    ok &= bool(passed)
    report[name] = {"passed": bool(passed), **detail}
    print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")

def metrics(y, p):
    yhat = (p >= 0.5).astype(int)
    return {"f1": f1_score(y, yhat), "accuracy": accuracy_score(y, yhat), "precision": precision_score(y, yhat),
            "recall": recall_score(y, yhat), "roc_auc": roc_auc_score(y, p),
            "confusion_matrix": confusion_matrix(y, yhat, labels=[0, 1]).tolist()}

sel = read(EXP / "selection_before_test.json")["winners"][ARM]
check("1_hashes", sha(EXP / sel["model_path"]) == sel["model_sha256"] and
      sha(EXP / sel["preprocessor_path"]) == sel["preprocessor_sha256"],
      {"model": sel["model"], "model_path": sel["model_path"], "preprocessor_path": sel["preprocessor_path"]})

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    pre = joblib.load(EXP / sel["preprocessor_path"])
    model = load_model(sel["slug"], EXP / sel["model_path"], threads=2)
check("2_load_without_warnings", len(caught) == 0, {"warnings": [str(w.message) for w in caught]})

d, _ = clean(EXP / "data/hotel_bookings.csv")
mem = pd.read_csv(EXP / "splits/membership.csv")
d = d.merge(mem[["source_csv_row", "split"]], on="source_csv_row", validate="one_to_one")
predict = lambda frame: model.predict_proba(np.asarray(pre.transform(make_features(frame, "historical24")), dtype=np.float32))[:, 1]

saved_val = pd.read_csv(EXP / "models/retained/historical24/xgboost/validation_predictions.csv", float_precision="round_trip")
val = d.set_index("source_csv_row").loc[saved_val.source_csv_row].reset_index()
pv = predict(val)
err_v = float(np.max(np.abs(pv - saved_val.probability)))
check("3a_validation_probabilities_reproduced", err_v < 1e-6 and (val.is_canceled.values == saved_val.actual.values).all(),
      {"rows": len(val), "max_abs_error": err_v})

paired = pd.read_csv(EXP / "results/paired_test_predictions.csv", float_precision="round_trip")
test = d.set_index("source_csv_row").loc[paired.source_csv_row].reset_index()
pt = predict(test)
err_t = float(np.max(np.abs(pt - paired[ARM])))
check("3b_test_probabilities_reproduced", err_t < 1e-6 and (test.is_canceled.values == paired.actual.values).all(),
      {"rows": len(test), "max_abs_error": err_t})

sealed = read(EXP / "test_results.json")["results"][ARM]
y, one = paired.actual.values, paired.one_copy.values
m_rep, m_one = metrics(y, pt), metrics(y[one], pt[one])
diff = max(abs(m_rep[k] - sealed["retained_test"][k]) for k in ["f1", "accuracy", "precision", "recall", "roc_auc"])
diff = max(diff, max(abs(m_one[k] - sealed["one_copy_test"][k]) for k in ["f1", "accuracy", "precision", "recall", "roc_auc"]))
check("4_sealed_test_metrics_match", diff < 1e-9 and m_rep["confusion_matrix"] == sealed["retained_test"]["confusion_matrix"],
      {"repeated_record_test": {k: round(v, 4) if isinstance(v, float) else v for k, v in m_rep.items()},
       "one_copy_test": {k: round(v, 4) if isinstance(v, float) else v for k, v in m_one.items()},
       "rows_repeated": int(len(y)), "rows_one_copy": int(one.sum()), "max_metric_difference": diff})

rows, worst = [], 0.0
for slug in ["logistic_regression", "random_forest", "xgboost", "catboost"]:
    folder = EXP / "models/retained/historical24" / slug
    best = read(folder / "best.json")
    vp = pd.read_csv(folder / "validation_predictions.csv", float_precision="round_trip")
    m = metrics(vp.actual.values, vp.probability.values)
    worst = max(worst, max(abs(m[k] - best["validation"][k]) for k in ["f1", "accuracy", "precision", "recall", "roc_auc"]))
    rows.append({"model": best["model"], "configuration": best["configuration"],
                 **{k: round(m[k], 4) for k in ["f1", "precision", "recall", "accuracy", "roc_auc"]},
                 "weights_in_package": (folder / Path(best["model_path"]).name).exists()})
check("5_validation_comparison_recomputed", worst < 1e-9, {"max_metric_difference": worst, "models": rows})

(ROOT / "docs").mkdir(exist_ok=True)
(ROOT / "docs/verification_report.json").write_text(json.dumps({"all_passed": ok, "arm": ARM, "checks": report}, indent=2), encoding="utf-8")
print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED"); sys.exit(0 if ok else 1)

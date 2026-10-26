"""Reconstruct Claude's splits and recompute supplied prediction metrics.

python audit.py --source source --data hotel_bookings.csv
The saved preprocessing requires pandas 3.x; audit without --load-model to
inspect predictions and reconstructed splits under the training CSV's pandas.
"""
import argparse
import hashlib
import importlib.metadata
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from threadpoolctl import threadpool_limits


def metrics(y, p):
    pred = np.asarray(p) >= .5
    return {k: float(f(y, pred)) for k, f in [("f1", f1_score), ("accuracy", accuracy_score), ("precision", precision_score), ("recall", recall_score)]} | {
        "auc": float(roc_auc_score(y, p)), "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(), "rows": len(y)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True); ap.add_argument("--data", required=True)
    ap.add_argument("--output", default="audit_verification.json"); ap.add_argument("--load-model", action="store_true")
    args = ap.parse_args(); source = Path(args.source).resolve()
    sys.path.insert(0, str(source)); import prep
    report = {"packages": {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scikit-learn", "xgboost", "joblib"]}, "scenarios": {}}
    for scenario in ["simple", "grouped", "strict"]:
        directory = source / f"results_{scenario}"; r = {}
        d = prep.load_and_clean(args.data, dedup=scenario == "strict", log=lambda x: None)
        Xtr, Xva, Xte, ytr, yva, yte = prep.split(d) if scenario == "strict" else prep.split_grouped(d, simple=scenario == "simple")
        r["splits"] = {n: {"rows": len(y), "cancelled": int(y.sum())} for n, y in [("train", ytr), ("validation", yva), ("test", yte)]}
        h = {n: pd.util.hash_pandas_object(x, index=False) for n, x in [("train", Xtr), ("validation", Xva), ("test", Xte)]}
        r["raw_predictor_overlap_rows"] = {"validation_in_train": int(h["validation"].isin(set(h["train"])).sum()), "test_in_train": int(h["test"].isin(set(h["train"])).sum()), "test_in_validation": int(h["test"].isin(set(h["validation"])).sum())}
        r["input_fields"] = list(Xtr)
        hist = pd.read_csv(directory / "tuning_log.csv")
        r["configuration_counts"] = hist.groupby("model").size().to_dict()
        r["tuning_seconds"] = hist.groupby("model").seconds.sum().to_dict()
        r["best_logged_validation_f1"] = hist.groupby("model").val_f1.max().to_dict()
        for stage, X, y, name in [("validation", Xva, yva, "validation_predictions.csv"), ("test", Xte, yte, "test_predictions_selected.csv")]:
            if not (directory / name).exists(): continue
            p = pd.read_csv(directory / name, float_precision="round_trip")
            assert np.array_equal(p.row_id, X.index), (scenario, stage, "row IDs differ")
            assert np.array_equal(p.actual, y), (scenario, stage, "targets differ")
            if stage == "validation":
                r["validation_recomputed"] = {n: metrics(p.actual, p["prob_" + n.replace(" ", "_").lower()]) for n in ["Logistic Regression", "Random Forest", "XGBoost", "CatBoost"]}
            else:
                assert np.array_equal((p.probability >= .5).astype(int), p.predicted)
                r["test_recomputed"] = metrics(p.actual, p.probability)
        # Refit preprocessing only, to independently check the implementation.
        cat = prep.SIMPLE_CAT if scenario == "simple" else prep.CAT_COLS
        num = prep.SIMPLE_NUM if scenario == "simple" else prep.NUM_COLS
        fresh = prep.make_preprocessor(cat, num).fit(Xtr)
        Z = {n: fresh.transform(x) for n, x in [("train", Xtr), ("validation", Xva), ("test", Xte)]}
        hs = {n: pd.util.hash_pandas_object(pd.DataFrame(z), index=False) for n, z in Z.items()}
        r["encoded_predictor_overlap_rows"] = {"validation_in_train": int(hs["validation"].isin(set(hs["train"])).sum()), "test_in_train": int(hs["test"].isin(set(hs["train"])).sum()), "test_in_validation": int(hs["test"].isin(set(hs["validation"])).sum())}
        if scenario == "simple" and args.load_model:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                pp = joblib.load(directory / "preprocessing.joblib")
                model = joblib.load(directory / "model_xgboost.joblib")
                lr = joblib.load(directory / "model_logistic_regression.joblib")
            r["loading_warnings"] = [str(x.message) for x in w]
            pre = pp["pre"]; r["saved_model_parameters"] = model.get_params()
            r["encoded_features"] = len(pre.get_feature_names_out())
            r["saved_training_median_matches"] = bool(np.allclose(pp["num_median"], Xtr[num].median(), atol=1e-12))
            r["saved_preprocessor_matches_fresh_fit"] = all(np.array_equal(pre.transform(x), Z[n]) for n, x in [("train", Xtr), ("validation", Xva), ("test", Xte)])
            with threadpool_limits(limits=4):
                model.set_params(n_jobs=4)
                pv = model.predict_proba(Z["validation"])[:, 1]; pt = model.predict_proba(Z["test"])[:, 1]
                plr = lr.predict_proba(Z["validation"])[:, 1]
            sv = pd.read_csv(directory / "validation_predictions.csv", float_precision="round_trip")
            st = pd.read_csv(directory / "test_predictions_selected.csv", float_precision="round_trip")
            r["saved_model_reproduction"] = {"xgboost_validation_max_probability_error": float(np.max(np.abs(pv - sv.prob_xgboost))), "xgboost_test_max_probability_error": float(np.max(np.abs(pt - st.probability))), "lr_validation_max_probability_error": float(np.max(np.abs(plr - sv.prob_logistic_regression))), "test_metrics": metrics(yte, pt)}
        report["scenarios"][scenario] = r
        Path(args.output).write_text(json.dumps(report, indent=2, default=str))
        print(scenario, json.dumps(r, default=str), flush=True)
    # The revised "day-before-arrival" description requires an at-risk cohort.
    raw = pd.read_csv(args.data)
    valid = raw[["adults", "children", "babies"]].fillna(0).sum(axis=1).gt(0) & raw.adr.ge(0) & raw.adr.lt(5000)
    raw = raw.loc[valid].copy()
    months = {s: i+1 for i, s in enumerate(["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"])}
    arrival = pd.to_datetime(dict(year=raw.arrival_date_year, month=raw.arrival_date_month.map(months), day=raw.arrival_date_day_of_month))
    status = pd.to_datetime(raw.reservation_status_date); c = raw.is_canceled.eq(1)
    report["prediction_landmark_check"] = {"cancelled_bookings": int(c.sum()), "status_before_day_before_arrival": int((c & (status < arrival-pd.Timedelta(days=1))).sum()), "status_on_or_before_day_before_arrival": int((c & (status <= arrival-pd.Timedelta(days=1))).sum()), "already_known_fraction": float((status[c] <= arrival[c]-pd.Timedelta(days=1)).mean()), "interpretation": "A day-before-arrival prospective cancellation task must restrict to bookings still active then and predict subsequent cancellations. The supplied evaluation includes all historical cancellations, most already recorded by that point."}
    report["missing_packaged_files"] = [str(Path(f"results_{s}") / name) for s in ["simple", "grouped", "strict"] for name in ["splits.joblib", "preprocessing.joblib", "model_logistic_regression.joblib", "model_random_forest.joblib", "model_xgboost.joblib", "model_catboost.joblib"] if not (source / f"results_{s}" / name).exists()]
    report["dataset_sha256"] = hashlib.sha256(Path(args.data).read_bytes()).hexdigest()
    Path(args.output).write_text(json.dumps(report, indent=2, default=str))


if __name__ == "__main__": main()

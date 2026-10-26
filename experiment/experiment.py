"""Controlled 2x2 FYP experiment; no test-driven refitting in this directory.

python experiment.py prepare --data hotel_bookings.csv
python experiment.py train --policy deduplicated --threads 4
python experiment.py train --policy retained --threads 4
python experiment.py select
python experiment.py evaluate --threads 4

All four models run three fixed manual configurations in each arm (48 fits).
Same group memberships and validation objectives within each duplicate policy.
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from features import CORE_FIELDS, FEATURE_SETS, make_features, make_preprocessor

ROOT = Path(__file__).resolve().parent
SEED = 42
THRESHOLD = .5
PARTS = ["train", "validation", "test"]
POLICIES = ["deduplicated", "retained"]
MODELS = {"logistic_regression": "Logistic Regression", "random_forest": "Random Forest", "xgboost": "XGBoost", "catboost": "CatBoost"}
CONFIGS = {
    "logistic_regression": [
        {"C": .1, "solver": "lbfgs"}, {"C": 1., "solver": "lbfgs"}, {"C": 10., "solver": "lbfgs"},
    ],
    "random_forest": [
        {"n_estimators": 250, "max_depth": 20, "min_samples_split": 5, "min_samples_leaf": 2, "max_features": "log2"},
        {"n_estimators": 400, "max_depth": 40, "min_samples_split": 5, "min_samples_leaf": 1, "max_features": "log2"},
        {"n_estimators": 400, "max_depth": 40, "min_samples_split": 5, "min_samples_leaf": 1, "max_features": .5},
    ],
    "xgboost": [
        {"n_estimators": 400, "learning_rate": .1, "max_depth": 6, "colsample_bytree": .5},
        {"n_estimators": 600, "learning_rate": .1, "max_depth": 10, "colsample_bytree": .5},
        {"n_estimators": 600, "learning_rate": .1, "max_depth": 10, "colsample_bytree": .8},
    ],
    "catboost": [
        {"iterations": 350, "learning_rate": .1, "depth": 6, "l2_leaf_reg": 3.},
        {"iterations": 500, "learning_rate": .1, "depth": 8, "l2_leaf_reg": 3.},
        {"iterations": 500, "learning_rate": .1, "depth": 8, "l2_leaf_reg": 6.},
    ],
}


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=lambda v: v.item() if isinstance(v, np.generic) else str(v)), encoding="utf-8")
    tmp.replace(path)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def clean(path):
    raw = pd.read_csv(path)
    original = list(raw)
    if not raw.is_canceled.isin([0, 1]).all(): raise ValueError("Expected a binary target without missing values")
    raw["source_csv_row"] = np.arange(2, len(raw) + 2)
    zero = raw[["adults", "children", "babies"]].fillna(0).sum(axis=1).eq(0)
    negative = raw.adr.lt(0)
    extreme = raw.adr.eq(5400)
    d = raw.loc[~(zero | negative | extreme)].copy().reset_index(drop=True)
    d["full_record_id"] = d.groupby(original, sort=False, dropna=False).ngroup().to_numpy()
    d["one_copy"] = ~d.duplicated(original)
    x = make_features(d, "general20")
    d["profile_group"] = x.groupby(CORE_FIELDS, sort=False, dropna=False).ngroup().to_numpy()
    audit = {"raw_rows": len(raw), "zero_guest_rows": int(zero.sum()), "negative_adr_rows": int(negative.sum()),
             "extreme_adr_5400_rows": int(extreme.sum()), "removed_union": int((zero | negative | extreme).sum()),
             "raw_duplicate_copies": int(raw.duplicated(original).sum()), "retained_rows": len(d),
             "deduplicated_rows": int(d.one_copy.sum()), "retained_duplicate_copies": int((~d.one_copy).sum()),
             "zero_night_rows_retained": int((d.stays_in_week_nights + d.stays_in_weekend_nights).eq(0).sum()),
             "cancelled_share_retained": float(d.is_canceled.mean()), "cancelled_share_deduplicated": float(d.loc[d.one_copy, "is_canceled"].mean()),
             "outlier_policy": "ADR 5400 exclusion fixed before fitting to match the supplied Claude brief; an extreme value is not proven erroneous. No other data-derived outlier threshold.",
             "missing_policy": "Country -> Unknown and children -> 0 are fixed semantic assumptions. PMS NULL agent/company -> absent flags. Other numeric medians and all encoding/scaling are learned on the respective training set only."}
    return d, audit


def grouped_split(d):
    """Exact one-copy stratified quotas; balance repeated-copy quotas approximately.

    No outcome is part of the grouping key. Class counts are used only to
    stratify the partition, as in ordinary stratified sampling.
    """
    unique = d.loc[d.one_copy].copy()
    y = unique.is_canceled.to_numpy(dtype=int)
    tr, hold = train_test_split(np.arange(len(unique)), test_size=.30, random_state=SEED, stratify=y)
    va, te = train_test_split(hold, test_size=.5, random_state=SEED, stratify=y[hold])
    quotas = np.array([np.bincount(y[i], minlength=2) for i in [tr, va, te]])
    remaining = quotas.copy()
    fractions = np.array([.7, .15, .15])
    extra_total = np.bincount(d.loc[~d.one_copy, "is_canceled"], minlength=2)
    extra_target = fractions[:, None] * extra_total
    extra_assigned = np.zeros((3, 2), dtype=int)
    groups = list(d.groupby("profile_group", sort=False).indices.values())
    complex_groups = [g for g in groups if len(g) > 1]
    rng = np.random.default_rng(SEED)
    rng.shuffle(complex_groups)
    complex_groups.sort(key=len, reverse=True)
    mapping = {}
    all_y = d.is_canceled.to_numpy(dtype=int)
    one = d.one_copy.to_numpy()
    ids = d.profile_group.to_numpy()
    for g in complex_groups:
        counts = np.bincount(all_y[g[one[g]]], minlength=2)
        extra = np.bincount(all_y[g[~one[g]]], minlength=2)
        feasible = np.flatnonzero((remaining >= counts).all(axis=1))
        if not len(feasible): raise RuntimeError("Grouped stratification cannot satisfy the fixed quotas")
        scores = []
        for s in feasible:
            # Smallest incremental squared discrepancy; no model scores used.
            before = extra_assigned[s] - extra_target[s]
            after = before + extra
            delta_extra = np.sum((after ** 2 - before ** 2) / np.maximum(extra_target[s], 1))
            assigned = quotas[s] - remaining[s]
            delta_unique = np.sum(((assigned + counts - quotas[s]) ** 2 - (assigned - quotas[s]) ** 2) / np.maximum(quotas[s], 1))
            scores.append(delta_extra + .1 * delta_unique)
        minimum = min(scores)
        tied = feasible[np.isclose(scores, minimum, atol=1e-12, rtol=0)]
        s = int(rng.choice(tied))
        mapping[int(ids[g[0]])] = s
        remaining[s] -= counts
        extra_assigned[s] += extra
    single = np.array([g[0] for g in groups if len(g) == 1], dtype=int)
    for label in [0, 1]:
        pool = single[all_y[single] == label].copy(); rng.shuffle(pool)
        assert len(pool) == remaining[:, label].sum()
        start = 0
        for s in range(3):
            n = int(remaining[s, label])
            for i in pool[start:start+n]: mapping[int(ids[i])] = s
            start += n
    part = np.array([PARTS[mapping[int(g)]] for g in ids])
    for s, name in enumerate(PARTS):
        assert np.array_equal(np.bincount(all_y[(part == name) & one], minlength=2), quotas[s])
    return part


def prepare(data):
    if (ROOT / "plan.json").exists():
        print("Frozen plan already exists; not changing data or groups.", flush=True); return
    d, audit = clean(data)
    d["split"] = grouped_split(d)
    splits = {}
    hashes = {}
    for policy in POLICIES:
        keep = np.ones(len(d), dtype=bool) if policy == "retained" else d.one_copy.to_numpy()
        splits[policy] = {}
        for part in PARTS:
            v = d.loc[keep & d.split.eq(part)]
            splits[policy][part] = {"rows": len(v), "share": len(v) / int(keep.sum()),
                                    "class_counts": np.bincount(v.is_canceled, minlength=2).tolist(),
                                    "cancellation_share": float(v.is_canceled.mean()), "groups": int(v.profile_group.nunique())}
    membership = d[["source_csv_row", "profile_group", "full_record_id", "one_copy", "split"]]
    (ROOT / "splits").mkdir(parents=True, exist_ok=True)
    membership.to_csv(ROOT / "splits/membership.csv", index=False)
    overlap = {}
    for kind in ["profile_group", "full_record_id"]:
        groups = {p: set(d.loc[d.split.eq(p), kind]) for p in PARTS}
        overlap[kind] = {a + "_" + b: len(groups[a] & groups[b]) for i, a in enumerate(PARTS) for b in PARTS[i+1:]}
        assert not any(overlap[kind].values())
    (ROOT / "prepared").mkdir(exist_ok=True)
    for policy in POLICIES:
        for family in FEATURE_SETS:
            work = ROOT / "prepared" / policy / family; work.mkdir(parents=True, exist_ok=True)
            keep = np.ones(len(d), dtype=bool) if policy == "retained" else d.one_copy.to_numpy()
            train = keep & d.split.eq("train").to_numpy()
            x = make_features(d, family)
            pre = make_preprocessor(x.loc[train]); pre.fit(x.loc[train])
            joblib.dump(pre, work / "preprocessor.joblib", compress=3)
            save(work / "features.json", {"raw_input_fields": FEATURE_SETS[family], "model_fields": list(x), "encoded_feature_names": list(pre.get_feature_names_out()),
                                          "country_policy": "No top-country grouping. One-hot vocabulary learned on training only; unseen categories ignored."})
            for part in ["train", "validation"]:
                ix = keep & d.split.eq(part).to_numpy()
                path = work / (part + ".npz")
                np.savez_compressed(path, X=np.asarray(pre.transform(x.loc[ix]), dtype=np.float32), y=d.loc[ix, "is_canceled"].to_numpy(dtype=int),
                                    source_csv_row=d.loc[ix, "source_csv_row"].to_numpy(), one_copy=d.loc[ix, "one_copy"].to_numpy())
                hashes[str(path.relative_to(ROOT))] = sha(path)
            # Raw model form input, sampled from training only.
            x.loc[train].head(5).to_csv(work / "example_inputs.csv", index=False)
    env = {"python": platform.python_version(), "platform": platform.platform(),
           "packages": {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scipy", "scikit-learn", "joblib", "threadpoolctl", "xgboost", "catboost"]}}
    save(ROOT / "environment.json", env)
    plan = {"created_utc": now(), "seed": SEED, "threshold": THRESHOLD, "target": "is_canceled", "positive_class": 1,
            "candidate_fits": 48, "configurations": CONFIGS, "models": MODELS, "feature_sets": FEATURE_SETS,
            "cleaning": audit, "splits": splits, "overlap": overlap, "dataset_sha256": sha(data), "prepared_hashes": hashes,
            "group_definition": "All 20 raw general predictor values, including derived agent/company presence flags; no target, reservation status, status date or other excluded field. Groups built before any fitted preprocessing.",
            "split_method": "Fixed seed 42, exact stratified 70/15/15 quotas for one-copy records. Allocate entire general-input groups, approximately balance duplicate-copy class counts, restore repeated rows into their group's original split. Both policies share all group memberships.",
            "selection": "Select each model configuration and each arm's overall winner using its own validation F1; AUC breaks exact F1 ties. Deduplicated uses one-copy validation; retained uses repeated-record validation. Do not compare these objectives as if they were identical.",
            "comparison": "Freeze all four arm winners before scoring any test predictions. Score each frozen winner on the identical retained test population and the identical one-copy subset. Paired group bootstrap is descriptive, conditional on these fitted winners.",
            "imbalance": "Training class frequencies only. LR/RF class_weight balanced; XGBoost negative/positive training ratio; CatBoost auto_class_weights Balanced. No resampling.",
            "tuning": "Three fixed manual configurations per model per arm, small changes or small combinations. Fixed training rounds; no early stopping, automated search, threshold tuning or seed search. This is a finite manual comparison, not exhaustive convergence of the search.",
            "timing_limit": "The CSV does not provide booking-confirmation snapshots. general20 removes deposit/ADR/request snapshots but retained general descriptors may also be amended later. historical24 is a historical sensitivity arm, not a verified booking-confirmation model.",
            "duplicate_assumption": "No booking ID establishes that repeated rows are genuine separate bookings or group bookings. Retention and deduplication are alternative weighting assumptions, both reported.",
            "test_exposure": "This dataset and its test performance have been inspected in earlier experiments. This is an exploratory rerun; repartitioning does not create independent external validation. Test outcomes will not tune this run.",
            "not_claude_reproduction": "Claude's ZIP was unavailable. This implements an explicit new comparison; preprocessing, groups and split assignments can differ from Claude's reported 80.7% run."}
    save(ROOT / "plan.json", plan)
    print("PREPARED", json.dumps({"cleaning": audit, "splits": splits, "overlap": overlap}), flush=True)


def metrics(y, p):
    y = np.asarray(y, dtype=int); p = np.asarray(p, dtype=float)
    pred = (p >= THRESHOLD).astype(int)
    return {"f1": float(f1_score(y, pred, zero_division=0)), "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)), "accuracy": float(accuracy_score(y, pred)),
            "roc_auc": float(roc_auc_score(y, p)), "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(), "rows": len(y), "threshold": THRESHOLD}


def model_for(slug, config, ratio, threads):
    if slug == "logistic_regression":
        return LogisticRegression(**config, class_weight="balanced", max_iter=3000, random_state=SEED)
    if slug == "random_forest":
        return RandomForestClassifier(**config, class_weight="balanced", random_state=SEED, n_jobs=threads)
    if slug == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(**config, objective="binary:logistic", tree_method="hist", scale_pos_weight=ratio,
                             subsample=1., gamma=0., min_child_weight=1., reg_lambda=1., reg_alpha=0.,
                             random_state=SEED, n_jobs=threads, eval_metric="logloss")
    from catboost import CatBoostClassifier
    return CatBoostClassifier(**config, auto_class_weights="Balanced", loss_function="Logloss", random_seed=SEED,
                             thread_count=threads, verbose=False, allow_writing_files=False)


def persist(model, slug, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if slug in ["xgboost", "catboost"]: model.save_model(str(path))
    else: joblib.dump(model, path, compress=3)


def load_model(slug, path, threads=4):
    if slug == "xgboost":
        from xgboost import XGBClassifier
        m = XGBClassifier(); m.load_model(str(path)); m.set_params(n_jobs=threads); return m
    if slug == "catboost":
        from catboost import CatBoostClassifier
        m = CatBoostClassifier(); m.load_model(str(path)); return m
    m = joblib.load(path)
    if slug == "random_forest": m.n_jobs = threads
    return m


def train(policy, threads):
    if (ROOT / "selection_before_test.json").exists(): raise RuntimeError("Winners already frozen; further training blocked")
    plan = read(ROOT / "plan.json")
    assert plan["configurations"] == CONFIGS
    for family in FEATURE_SETS:
        work = ROOT / "prepared" / policy / family
        for part in ["train", "validation"]:
            p = work / (part + ".npz"); assert sha(p) == plan["prepared_hashes"][str(p.relative_to(ROOT))]
        with np.load(work / "train.npz") as a: xtr, ytr = a["X"], a["y"]
        with np.load(work / "validation.npz") as a: xva, yva, rows, one = a["X"], a["y"], a["source_csv_row"], a["one_copy"]
        ratio = float(np.sum(ytr == 0) / np.sum(ytr == 1))
        history_path = ROOT / "results" / policy / (family + "_trials.json")
        history = read(history_path) if history_path.exists() else []
        done = {(r["slug"], r["configuration"]) for r in history}
        for slug, label in MODELS.items():
            for number, config in enumerate(CONFIGS[slug], 1):
                if (slug, number) in done: continue
                print("TRAIN", policy, family, slug, number, flush=True)
                started = now(); t0 = time.perf_counter()
                model = model_for(slug, config, ratio, threads)
                with threadpool_limits(limits=threads), warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    model.fit(xtr, ytr)
                    if slug == "random_forest": model.n_jobs = 1
                    pv = np.asarray(model.predict_proba(xva)[:, 1], dtype=float)
                    pt = np.asarray(model.predict_proba(xtr)[:, 1], dtype=float)
                r = {"policy": policy, "family": family, "slug": slug, "model": label, "configuration": number,
                     "manual_parameters": config, "effective_parameters": model.get_params(), "training_positive_weight_ratio": ratio,
                     "started_utc": started, "finished_utc": now(), "seconds": time.perf_counter()-t0,
                     "train": metrics(ytr, pt), "validation": metrics(yva, pv), "one_copy_validation": metrics(yva[one], pv[one]),
                     "threshold": THRESHOLD, "warnings": [str(w.message) for w in caught],
                     "iterations": np.asarray(getattr(model, "n_iter_", [])).tolist()}
                previous = [v for v in history if v["slug"] == slug]
                score = (r["validation"]["f1"], r["validation"]["roc_auc"])
                if not previous or score > max((v["validation"]["f1"], v["validation"]["roc_auc"]) for v in previous):
                    directory = ROOT / "models" / policy / family / slug
                    suffix = {"xgboost": ".ubj", "catboost": ".cbm"}.get(slug, ".joblib")
                    path = directory / (slug + suffix)
                    persist(model, slug, path)
                    prepath = work / "preprocessor.joblib"
                    best = {**r, "model_path": str(path.relative_to(ROOT)), "preprocessor_path": str(prepath.relative_to(ROOT)), "model_sha256": sha(path), "preprocessor_sha256": sha(prepath)}
                    save(directory / "best.json", best)
                    pd.DataFrame({"source_csv_row": rows, "actual": yva, "probability": pv, "predicted": (pv >= THRESHOLD).astype(int), "one_copy": one}).to_csv(directory / "validation_predictions.csv", index=False)
                history.append(r); save(history_path, history)
                print("DONE", policy, family, slug, number, "F1", round(r["validation"]["f1"], 6), "seconds", round(r["seconds"], 1), flush=True)


def select(threads):
    if (ROOT / "selection_before_test.json").exists():
        print("Selection already frozen", flush=True); return
    winners = {}; rows = []; checks = []
    for policy in POLICIES:
        for family in FEATURE_SETS:
            history = read(ROOT / "results" / policy / (family + "_trials.json"))
            expected = {(s, i) for s in MODELS for i in [1, 2, 3]}
            assert len(history) == 12 and {(r["slug"], r["configuration"]) for r in history} == expected
            with np.load(ROOT / "prepared" / policy / family / "validation.npz") as a: x, y = a["X"], a["y"]
            candidates = []
            for slug in MODELS:
                directory = ROOT / "models" / policy / family / slug
                best = read(directory / "best.json")
                assert sha(ROOT / best["model_path"]) == best["model_sha256"]
                with threadpool_limits(limits=threads): p = load_model(slug, ROOT / best["model_path"], threads).predict_proba(x)[:, 1]
                stored = pd.read_csv(directory / "validation_predictions.csv", float_precision="round_trip")
                assert np.allclose(p, stored.probability, atol=1e-12, rtol=0)
                recalculated = metrics(y, p)
                for key in ["f1", "accuracy", "precision", "recall", "roc_auc"]: assert abs(recalculated[key]-best["validation"][key]) < 1e-12
                checks.append({"policy": policy, "family": family, "model": best["model"], "saved_predictions_reproduced": True})
                candidates.append(best)
                rows.append({"policy": policy, "family": family, "model": best["model"], "configuration": best["configuration"], **{k: best["validation"][k] for k in ["f1", "precision", "recall", "accuracy", "roc_auc"]}})
            winners[policy + "_" + family] = max(candidates, key=lambda r: (r["validation"]["f1"], r["validation"]["roc_auc"]))
    save(ROOT / "selection_before_test.json", {"frozen_utc": now(), "winners": winners})
    save(ROOT / "results/selection_verification.json", checks)
    pd.DataFrame(rows).to_csv(ROOT / "results/validation_comparison.csv", index=False)
    print("WINNERS FROZEN", json.dumps({k: {"model": v["model"], "validation_f1": v["validation"]["f1"]} for k, v in winners.items()}), flush=True)


def evaluate(data, threads):
    if (ROOT / "test_results.json").exists():
        print("Test results sealed; no repeated evaluation or refitting.", flush=True); return
    plan = read(ROOT / "plan.json"); assert sha(data) == plan["dataset_sha256"]
    selection = read(ROOT / "selection_before_test.json")
    d, _ = clean(data)
    membership = pd.read_csv(ROOT / "splits/membership.csv")
    d = d.merge(membership[["source_csv_row", "split"]], on="source_csv_row", validate="one_to_one")
    test = d.loc[d.split.eq("test")].reset_index(drop=True)
    y = test.is_canceled.to_numpy(dtype=int); one = test.one_copy.to_numpy()
    table = test[["source_csv_row", "profile_group", "full_record_id", "one_copy"]].copy(); table["actual"] = y
    results = {}; checks = []
    for arm, r in selection["winners"].items():
        pre = joblib.load(ROOT / r["preprocessor_path"])
        model = load_model(r["slug"], ROOT / r["model_path"], threads)
        # Raw-input inference must reproduce validation probabilities first.
        directory = (ROOT / r["model_path"]).parent
        saved = pd.read_csv(directory / "validation_predictions.csv", float_precision="round_trip").head(64)
        raw = d.set_index("source_csv_row").loc[saved.source_csv_row]
        with threadpool_limits(limits=threads):
            pv = model.predict_proba(np.asarray(pre.transform(make_features(raw, r["family"])), dtype=np.float32))[:, 1]
            assert np.allclose(pv, saved.probability, atol=1e-12, rtol=0)
            p = np.asarray(model.predict_proba(np.asarray(pre.transform(make_features(test, r["family"])), dtype=np.float32))[:, 1], dtype=float)
        results[arm] = {"model": r["model"], "configuration": r["configuration"], "validation": r["validation"], "retained_test": metrics(y, p), "one_copy_test": metrics(y[one], p[one])}
        table[arm] = p
        checks.append({"arm": arm, "raw_validation_rows_reproduced": len(saved), "maximum_probability_error": float(np.max(np.abs(pv - saved.probability)))})
    table.to_csv(ROOT / "results/paired_test_predictions.csv", index=False)
    save(ROOT / "results/raw_inference_verification.json", checks)
    save(ROOT / "test_results.json", {"evaluated_utc": now(), "results": results, "test_exposure": plan["test_exposure"]})
    save(ROOT / "evaluation_seal.json", {"selection_sha256": sha(ROOT / "selection_before_test.json"), "results_sha256": sha(ROOT / "test_results.json"), "no_further_tuning": True})
    print("TEST RESULTS", json.dumps(results), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "train", "select", "evaluate"])
    parser.add_argument("--data", default=str(ROOT / "data/hotel_bookings.csv"))
    parser.add_argument("--policy", choices=POLICIES)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.action == "prepare": prepare(args.data)
    elif args.action == "train":
        if not args.policy: parser.error("--policy is required for train")
        train(args.policy, args.threads)
    elif args.action == "select": select(args.threads)
    else: evaluate(args.data, args.threads)

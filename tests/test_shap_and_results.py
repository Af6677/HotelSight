"""SHAP correctness and that every displayed metric belongs to the deployed experiment."""
import json

import numpy as np
import pandas as pd
import xgboost as xgb

from hotelsight import results
from hotelsight.contract import MODEL_FIELDS
from hotelsight.paths import ARM, ASSETS, TEST_RESULTS


def test_local_shap_reconstructs_model_output(model, test_rows):
    sample = test_rows.sample(2000, random_state=1)
    ex = model.explain(sample)
    assert ex["reconstruction_error"].max() < 1e-4                       # log-odds
    assert np.max(np.abs(ex["probability_from_shap"] - ex["probability"])) < 1e-5
    assert list(ex["contributions"].columns) == MODEL_FIELDS
    np.testing.assert_allclose(ex["contributions"].sum(axis=1), ex["encoded"].sum(axis=1), atol=1e-5)


def test_shap_matches_xgboost_native_treeshap(model, test_rows):
    sample = test_rows.sample(500, random_state=2)
    X = model.encode(sample)
    native = model.model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)
    ex = model.explain(sample)
    np.testing.assert_allclose(ex["encoded"].to_numpy(), native[:, :-1], atol=1e-4)
    np.testing.assert_allclose(ex["base_log_odds"], native[0, -1], atol=1e-4)


def test_global_shap_assets(model, test_rows):
    imp = pd.read_csv(ASSETS / "global_shap_importance.csv")
    assert sorted(imp.input) == sorted(MODEL_FIELDS) and (imp.mean_abs_shap_repeated >= 0).all()
    s = json.loads((ASSETS / "shap_summary.json").read_text())
    assert s["arm"] == ARM and s["rows_repeated"] == 17882 and s["max_reconstruction_error_log_odds"] < 1e-4
    top = model.explain(test_rows.iloc[:3000])["contributions"].abs().mean().sort_values(ascending=False).index[:3]
    assert set(top) <= set(imp.input.head(5))


def test_displayed_metrics_come_from_the_sealed_experiment():
    sealed = json.loads(TEST_RESULTS.read_text())["results"][ARM]
    t = results.test_metrics()
    assert t["repeated"] == sealed["retained_test"] and t["one_copy"] == sealed["one_copy_test"]
    assert round(t["repeated"]["f1"], 4) == 0.8102 and round(t["repeated"]["accuracy"], 4) == 0.8570
    assert round(t["one_copy"]["f1"], 4) == 0.6995


def test_validation_comparison_selects_the_deployed_model():
    v = results.validation_comparison()
    assert list(v.Model) == ["Logistic Regression", "Random Forest", "XGBoost", "CatBoost"]
    assert v.loc[v["F1-score"].idxmax(), "Model"] == results.selected()["model"] == "XGBoost"
    assert len(results.trials()) == 12
    sens = results.sensitivity()
    assert len(sens) == 4 and round(sens.loc[2, "Test F1 (repeated)"], 4) == 0.7353

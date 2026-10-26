"""Model loading: hashes, version warnings, feature contract."""
import json
import warnings

from hotelsight.contract import MODEL_FIELDS
from hotelsight.paths import EXPERIMENT, SELECTION
from hotelsight.pipeline import DeployedModel, experiment_features, sha256


def test_frozen_hashes_match():
    rec = json.loads(SELECTION.read_text())["winners"]["retained_historical24"]
    assert rec["model"] == "XGBoost"
    assert sha256(EXPERIMENT / rec["model_path"]) == rec["model_sha256"]
    assert sha256(EXPERIMENT / rec["preprocessor_path"]) == rec["preprocessor_sha256"]


def test_loads_without_version_warnings():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        DeployedModel()
    assert [str(w.message) for w in caught] == []


def test_feature_contract_is_the_experiment_contract(model):
    spec = json.loads((EXPERIMENT / "prepared/retained/historical24/features.json").read_text())
    assert MODEL_FIELDS == experiment_features.FEATURE_SETS["historical24"] == spec["model_fields"]
    assert model.encoded_names == spec["encoded_feature_names"]
    assert len(model.encoded_names) == 218 and len(MODEL_FIELDS) == 24


def test_threshold_is_050(model):
    assert model.threshold == 0.5
    assert list(model.classify([0.4999, 0.5, 0.9])) == [0, 1, 1]

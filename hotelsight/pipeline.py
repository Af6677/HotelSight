"""Booking record -> shared feature construction -> saved preprocessing -> saved model.

The feature construction is the experiment's own `features.make_features`
(imported from ../experiment/features.py, not copied), so the dashboard and the
experiment cannot drift apart. The model file and preprocessor are checked
against the SHA-256 hashes frozen in selection_before_test.json before use.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json

import joblib
import numpy as np
import pandas as pd

from .contract import MODEL_FIELDS
from .paths import ARM, EXPERIMENT, FAMILY, PLAN, SELECTION


def _load_experiment_features():
    spec = importlib.util.spec_from_file_location("hotelsight_experiment_features", EXPERIMENT / "features.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


experiment_features = _load_experiment_features()


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class DeployedModel:
    """The frozen retained_historical24 XGBoost winner and its train-fitted preprocessor."""

    def __init__(self, threads: int = 2):
        from xgboost import XGBClassifier

        self.record = json.loads(SELECTION.read_text(encoding="utf-8"))["winners"][ARM]
        model_path = EXPERIMENT / self.record["model_path"]
        pre_path = EXPERIMENT / self.record["preprocessor_path"]
        if sha256(model_path) != self.record["model_sha256"]:
            raise RuntimeError(f"Model file hash does not match the frozen selection: {model_path}")
        if sha256(pre_path) != self.record["preprocessor_sha256"]:
            raise RuntimeError(f"Preprocessor hash does not match the frozen selection: {pre_path}")
        self.threshold = float(json.loads(PLAN.read_text(encoding="utf-8"))["threshold"])   # 0.50
        self.pre = joblib.load(pre_path)
        self.model = XGBClassifier()
        self.model.load_model(str(model_path))
        self.model.set_params(n_jobs=threads)
        self.encoded_names = list(self.pre.get_feature_names_out())
        self.owner = self._encoded_owner()
        self.categories = {}
        for name, trans, cols in self.pre.transformers_:
            if name == "categorical":
                for col, cats in zip(cols, trans.named_steps["encode"].categories_):
                    self.categories[col] = [str(c) for c in cats]
        self._explainer = None

    # ------------------------------------------------------------- pipeline steps
    def _encoded_owner(self) -> list:
        """Raw input that each encoded column belongs to (numeric first, then one-hot blocks)."""
        owner = []
        for name, trans, cols in self.pre.transformers_:
            if name == "numeric":
                owner += list(cols)
            elif name == "categorical":
                for col, cats in zip(cols, trans.named_steps["encode"].categories_):
                    owner += [col] * len(cats)
        assert len(owner) == len(self.encoded_names)
        return owner

    @staticmethod
    def frame(records) -> pd.DataFrame:
        """One dict or a list of dicts -> DataFrame in the exact contract order."""
        if isinstance(records, dict):
            records = [records]
        return pd.DataFrame(records)[MODEL_FIELDS]

    def features(self, raw: pd.DataFrame) -> pd.DataFrame:
        return experiment_features.make_features(raw, FAMILY)

    def encode(self, raw: pd.DataFrame) -> np.ndarray:
        # float32 exactly as in training (experiment.py: np.asarray(pre.transform(x), dtype=np.float32))
        return np.asarray(self.pre.transform(self.features(raw)), dtype=np.float32)

    def predict_proba(self, raw: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self.encode(raw))[:, 1]

    def predict_margin(self, raw: pd.DataFrame) -> np.ndarray:
        return self.model.predict(self.encode(raw), output_margin=True)

    def classify(self, probability) -> np.ndarray:
        return (np.asarray(probability) >= self.threshold).astype(int)

    # ------------------------------------------------------------- SHAP
    @property
    def explainer(self):
        """TreeSHAP (tree_path_dependent). Its background is the training data, carried in the
        trees' node cover statistics, so no separate background sample is needed.
        Output units: log-odds of cancellation."""
        if self._explainer is None:
            import shap
            self._explainer = shap.TreeExplainer(self.model)
        return self._explainer

    def explain(self, raw: pd.DataFrame) -> dict:
        X = self.encode(raw)
        sv = np.asarray(self.explainer.shap_values(X))
        if sv.ndim == 3:
            sv = sv[..., -1]
        base = float(np.ravel(self.explainer.expected_value)[-1])
        enc = pd.DataFrame(sv, columns=self.encoded_names)
        grouped = enc.T.groupby(self.owner, sort=False).sum().T[MODEL_FIELDS]    # one-hot blocks summed
        margin = self.model.predict(X, output_margin=True)
        prob = self.model.predict_proba(X)[:, 1]
        recon = base + grouped.sum(axis=1).to_numpy()
        return {"base_log_odds": base, "contributions": grouped, "encoded": enc,
                "margin": margin, "probability": prob,
                "reconstruction_error": np.abs(recon - margin),                     # log-odds units
                "probability_from_shap": 1 / (1 + np.exp(-recon))}


    def fast_contributions(self, raw: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
        """(probability, per-input contributions) via XGBoost's native TreeSHAP (pred_contribs).
        Same values as `explain` (checked in tests), much faster for a whole spreadsheet."""
        from xgboost import DMatrix
        X = self.encode(raw)
        sv = self.model.get_booster().predict(DMatrix(X), pred_contribs=True)[:, :-1]     # last column = bias
        grouped = pd.DataFrame(sv, columns=self.encoded_names).T.groupby(self.owner, sort=False).sum().T[MODEL_FIELDS]
        return self.model.predict_proba(X)[:, 1], grouped


def to_probability(log_odds):
    return 1 / (1 + np.exp(-np.asarray(log_odds, dtype=float)))

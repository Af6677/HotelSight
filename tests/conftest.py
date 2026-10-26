import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / "experiment"))

# Tests never write to the real prediction history.
os.environ["HOTELSIGHT_HISTORY_DB"] = str(Path(tempfile.mkdtemp(prefix="hotelsight_test_")) / "predictions.db")


@pytest.fixture(scope="session")
def model():
    from hotelsight.pipeline import DeployedModel
    return DeployedModel()


@pytest.fixture(scope="session")
def cleaned():
    """The experiment's own cleaning + frozen split membership."""
    from experiment import clean
    from hotelsight.paths import DATA, MEMBERSHIP
    d, _ = clean(DATA)
    return d.merge(pd.read_csv(MEMBERSHIP)[["source_csv_row", "split"]], on="source_csv_row", validate="one_to_one")


@pytest.fixture(scope="session")
def paired():
    from hotelsight.paths import PAIRED_TEST
    return pd.read_csv(PAIRED_TEST, float_precision="round_trip")


@pytest.fixture(scope="session")
def test_rows(cleaned, paired):
    return cleaned.set_index("source_csv_row").loc[paired.source_csv_row].reset_index()

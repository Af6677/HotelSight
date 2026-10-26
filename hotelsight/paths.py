"""All file locations, relative to the package root (no absolute paths)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiment"          # frozen controlled experiment (unchanged)
ASSETS = ROOT / "assets"                  # derived, reproducible dashboard assets
DOCS = ROOT / "docs"

ARM = "retained_historical24"             # the ONLY experiment arm used by the dashboard
FAMILY = "historical24"
POLICY = "retained"

SELECTION = EXPERIMENT / "selection_before_test.json"
TEST_RESULTS = EXPERIMENT / "test_results.json"
PLAN = EXPERIMENT / "plan.json"
PAIRED_TEST = EXPERIMENT / "results" / "paired_test_predictions.csv"
PAIRED_COMPARISONS = EXPERIMENT / "results" / "paired_comparisons.json"
MEMBERSHIP = EXPERIMENT / "splits" / "membership.csv"
DATA = EXPERIMENT / "data" / "hotel_bookings.csv"
ARM_MODELS = EXPERIMENT / "models" / POLICY / FAMILY

# Test report

Environment for executed tests:
- Python 3.12.3 on Linux, in a **fresh virtual environment created from the delivered ZIP**
- Extracted into a folder whose path contains spaces
- Packages installed with `pip install -r requirements.txt`; `pip check` reported no broken requirements

## Executed – passed

| Check | Command | Result |
|---|---|---|
| Selected model and preprocessor hashes match the frozen selection | `scripts/verify_experiment.py` | PASS |
| Preprocessor and model load with no version warnings | `scripts/verify_experiment.py` | PASS |
| All 17,881 saved validation probabilities reproduced from the raw CSV | `scripts/verify_experiment.py` | PASS (max error 0.0) |
| All 17,882 saved test probabilities reproduced from the raw CSV | `scripts/verify_experiment.py` | PASS (max error 0.0) |
| Sealed test metrics recomputed: F1 0.8102 / accuracy 0.8570 (repeated), F1 0.6995 (one copy) | `scripts/verify_experiment.py` | PASS (difference 0.0) |
| Four-model validation metrics recomputed from saved probabilities | `scripts/verify_experiment.py` | PASS |
| Experiment files unchanged (88 SHA-256 checksums) | checksum check | PASS |
| Week number = SQL Server DATEPART(week); weekend nights = nights starting Sun/Mon | `scripts/check_date_conventions.py` | 100% of 119,390 rows |
| 30 automated tests (list below) | `python -m pytest` | **30 passed** |
| Dashboard server starts, health endpoint `ok`, app page served | `scripts/launch_check.py` | PASS |

### Automated tests (pytest)

- **Artifacts:** frozen hashes; loading without version warnings; feature contract equals the experiment's (24 inputs, 218 encoded columns); threshold 0.50.
- **Prediction consistency:**
  - all 17,882 test rows and all 17,881 validation rows reproduce the saved probabilities;
  - 400 random test bookings rebuilt **through the form path** (three dates + Yes/No flags) reproduce them;
  - the 4 dashboard example bookings reproduce them.
- **Date rules:** match every dataset row; known edge cases.
- **Validation and missing values:**
  - a valid record passes;
  - date errors are caught (booked after arrival, departure before arrival, missing date), and a zero-night stay gives a warning;
  - record errors are caught: zero guests, negative ADR, non-integer counts, negative counts, invalid Yes/No;
  - out-of-range and unseen values give warnings;
  - unseen country and meal values are ignored by the saved encoder;
  - missing children → 0, missing ADR → the training median, missing categories → 'Unknown'.
- **SHAP:**
  - local SHAP reconstructs the model's log-odds (< 1e-4) and probability (2,000 rows);
  - it equals XGBoost's native TreeSHAP (`pred_contribs`, 500 rows);
  - the global SHAP assets are consistent.
- **Results:** every displayed test metric equals `test_results.json`; the validation comparison selects XGBoost; 12 trials; 4 sensitivity arms.
- **App (Streamlit AppTest):**
  - all 6 page scripts run without exceptions or errors;
  - the default booking gives a probability;
  - the example booking matches its saved prediction (✓);
  - an invalid booking (0 guests) is rejected and no prediction is shown.

## Executed in the development environment (not part of the package)

- **Headless Chromium** (Playwright) opened every page of the running dashboard:
  - all charts rendered, with no Vega or JavaScript errors after fixes;
  - the example booking "High risk – non-refundable, long lead time" gave 99.98%, matching the saved test prediction (difference 0.0);
  - screenshots are in `docs/screenshots/`.
- **Chart colours** were checked with a palette validator: blue/red diverging pair, all checks passed.

## Reviewed but not executed

- `setup_windows.bat`, `run_dashboard.bat`, `run_tests.bat` and the PowerShell commands in README.md: checked by reading only (Windows CRLF line endings, quoted paths, explicit `.venv\Scripts\python.exe`). **They were not run on Windows.**

## Not tested

- **Installation and running on Windows.** All pinned packages publish Python 3.12 Windows wheels; `nvidia-nccl-cu13` is restricted to Linux by an environment marker. This has not been run on a Windows machine.
- **Full retraining** with `experiment/reproduce.py`, which reruns all 48 fits. This was not rerun here; the frozen run's records are used.
- **Usability testing with participants** (Table 3.7). This must be done by the student.
- **Browsers other than Chromium, and dark mode.** The app forces a light theme.
- **Predictive performance on new, recent or Malaysian bookings.** No such data is available.

# HotelSight – Hotel Booking Cancellation Prediction (FYP prototype)

Afiq Fakhri bin Yusoff · Bachelor of Information System (Hons.) Intelligent System Engineering · UiTM

HotelSight is an **experimental prototype evaluated on historical hotel records**. A hotel manager enters a
booking and gets a cancellation probability, a classification at the 0.50 threshold, and a SHAP explanation.
It is **not** a validated live forecasting system and has **not** been evaluated on Malaysian hotels.

## The model and its results (one experiment only)

The dashboard uses one frozen, verified experiment arm: **`retained_historical24`** from the controlled
experiment in `experiment/`. In that arm, repeated records are kept and grouped, there are 24 inputs, and the
selected model is XGBoost configuration 2. Every number shown in the dashboard comes from this experiment.

| Selected model: XGBoost | F1 | Accuracy | Precision | Recall | ROC AUC |
|---|---|---|---|---|---|
| Validation (17,881 rows, repeated records) | 0.8029 | 0.8516 | 0.7912 | 0.8149 | 0.9302 |
| **Test, repeated records (17,882 rows)** | **0.8102** | **0.8570** | 0.7975 | 0.8234 | 0.9317 |
| **Test, one copy per record (13,085 rows)** | **0.6995** | 0.8307 | 0.6840 | 0.7157 | 0.8906 |

The 20-input sensitivity model, without ADR, deposit type, parking and special requests, scores 0.7353
(repeated records) and 0.6024 (one copy).

## Quick start on Windows

1. Install **Python 3.12** from python.org, with the **py launcher** ticked.
2. Extract the ZIP, for example to `C:\FYP\HotelSight_FYP`.
3. Double-click **`setup_windows.bat`**. It creates `.venv`, installs exact versions and verifies the model.
4. Double-click **`run_dashboard.bat`**. The dashboard opens at <http://localhost:8501>.
5. Optional: double-click **`run_tests.bat`** to run all checks.

### Or in a VS Code PowerShell terminal (opened in the extracted folder)

```powershell
cd "C:\FYP\HotelSight_FYP"
py -3.12 -m venv ".venv"
& ".\.venv\Scripts\python.exe" -m pip install --upgrade pip
& ".\.venv\Scripts\python.exe" -m pip install -r "requirements.txt"
& ".\.venv\Scripts\python.exe" "scripts\verify_experiment.py"
& ".\.venv\Scripts\python.exe" -m streamlit run "app\streamlit_app.py"
```

The `&` and quotes keep paths with spaces working, and you don't need to activate the environment. Stop the
dashboard with `Ctrl+C` in the terminal. Run the tests with:

```powershell
& ".\.venv\Scripts\python.exe" -m pytest
& ".\.venv\Scripts\python.exe" "scripts\launch_check.py"
```

**Troubleshooting**
- *"py is not recognized" / Python 3.12 not found:* reinstall Python 3.12 with the py launcher.
- *Port 8501 in use:* add `--server.port 8502` to the streamlit command.
- *Always launch from the package folder*, so `.streamlit\config.toml` (the light theme) is picked up.

## What's in the package

| Path | Contents |
|---|---|
| `app/` | Streamlit dashboard: `streamlit_app.py` plus 8 pages in `app/views/` |
| `hotelsight/` | `contract.py` (24-input contract, labels, date rules, validation), `pipeline.py` (hash-checked model loading, prediction, SHAP), `results.py` (read-only access to the frozen results) |
| `experiment/` | **Frozen controlled experiment, unchanged**: data, split membership, 4 preprocessors, selected models, 48 trial records, sealed test results, bootstrap, its own README and reproduction scripts, and the audit of earlier runs |
| `assets/` | Derived dashboard assets: country names, training input ranges, example bookings, global SHAP. Rebuild with `scripts/build_assets.py` |
| `scripts/` | `verify_experiment.py`, `check_date_conventions.py`, `build_assets.py`, `launch_check.py` |
| `tests/` | 56 automated tests (pytest) |
| `docs/` | `THESIS_AMENDMENTS.md/.docx` (Chapter 3 and 4 changes), `TEST_REPORT.md`, `verification_report.json`, `date_conventions.json`, `screenshots/` |
| `archive/` | Earlier exploratory runs (superseded, **not used by the dashboard**) |
| `requirements.txt` / `requirements-top.txt` | Exact versions, full set and top-level packages |
| `*.bat` | Windows setup, launch and test scripts |

### Model files included and omitted

Included, and all the dashboard needs: the selected XGBoost model
`experiment/models/retained/historical24/xgboost/xgboost.ubj` and its preprocessor
`experiment/prepared/retained/historical24/preprocessor.joblib`. Both are checked against the SHA-256 values
frozen in `experiment/selection_before_test.json` every time they are loaded. The winners of the
other three arms (two 20-input CatBoost models and the deduplicated 24-input XGBoost) are also included.

**Omitted** to keep the download small: the non-selected model weights, including Random Forest at about
130 MB each, and the cached training/validation arrays. Each one, with its size and SHA-256, is listed in
`experiment/omitted_files.json`. Their validation predictions and metrics are included, so the four-model
comparison can be checked without them. To regenerate every model from scratch in a new folder, leaving
the frozen results unchanged:

```powershell
& ".\.venv\Scripts\python.exe" "experiment\reproduce.py" --out "fresh_run" --threads 4
```

This reruns all 48 fits, which can take a long time on a laptop. Exact bit-for-bit identity across operating
systems is not guaranteed.

## How a prediction is made

Booking form → validation (`hotelsight/contract.py`) → the experiment's own feature construction
(`experiment/features.py`, imported, not copied) → saved preprocessor → saved XGBoost → probability and
class (threshold 0.50) → TreeSHAP explanation in log-odds, checked to add up to the model output.

The dashboard asks for three dates and derives the date inputs using the **dataset's own conventions**,
both verified on 100% of rows (`docs/date_conventions.json`):
- **Week number** follows SQL Server `DATEPART(week)`: weeks start on Sunday, and week 1 contains 1 January. ISO weeks would match only 46% of rows.
- **"Weekend nights"** are nights that begin on a **Sunday or Monday**. A Saturday/Sunday rule matches only 50% of rows.

## Limitations

- **Input timing is not verified.** The dataset was compiled relative to the day before arrival. ADR, deposit
  type, parking and special requests are not verified as known at booking confirmation. Without them, F1 drops
  from 0.81 to 0.74.
- **Not a day-before-arrival forecast.** 93.9% of cancelled bookings were already cancelled before that day.
- **Historical data only:** two Portuguese hotels, 2015–2017. Performance on future or Malaysian bookings is unknown.
- **Repeated records** are not proven to be separate bookings, so both test populations are reported.
- **Test-set exposure:** the dataset and earlier test scores were inspected during development. The results are exploratory.
- **Probabilities are not calibrated.** Class weighting raises them, so use them to rank risk.
- **Overfitting gap:** train F1 0.95 vs validation F1 0.80.
- **Limited tuning:** three fixed configurations per model.
- **SHAP shows associations, not causes.**
- **Not yet done:** usability testing with participants.

## Sources

Antonio, N., de Almeida, A., & Nunes, L. (2019). Hotel booking demand datasets. *Data in Brief*, 22, 41–49.
https://doi.org/10.1016/j.dib.2018.11.126

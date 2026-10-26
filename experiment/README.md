# HotelSight controlled model comparison

This package answers two questions with actual training: what changes when repeated records are retained, and what changes when four timing-sensitive historical fields are included?

Read `HotelSight_Controlled_Comparison.md` for the new results and `audit/Claude_System_Audit.md` for the uploaded Claude ZIP audit. These are exploratory experiments on a dataset already inspected during development. The prediction timestamp is not verified as booking confirmation. The 24-field historical model is not certified for prospective use in Malaysian hotels.

## Actual experiment

- Logistic Regression, Random Forest, XGBoost and CatBoost all ran three manually specified configurations in each of four arms: **48 fitted candidates**.
- General20 and Historical24 have the same 20 base fields; Historical24 adds ADR, deposit type, parking and special requests.
- Both duplicate policies use the same booking-profile groups and split assignments. No target or outcome field enters the group key.
- Stratified 70/15/15 class quotas are exact for one-copy records; repeated-record class quotas are almost exact. See plan.json for actual shares and counts.
- All four models within an arm share training-only preprocessing. Model class weights use that arm's training class counts only.
- Threshold 0.50, seed 42; validation F1 selects configurations and each arm's winner. All selections were frozen before evaluating test predictions.
- The four frozen winners are scored on identical test rows, both with repetitions and with one copy per complete record.
- There is no room-type input or anonymous agency/company ID. Agent and company presence are binary inputs.

## Open on Windows in VS Code

Extract the ZIP, then open the `HotelSight_Controlled_Experiment` folder in VS Code. Open a PowerShell terminal in that folder. Run one line at a time. Use Python **3.12**, matching the tested Python major/minor and the current XGBoost requirement.

```powershell
py -3.12 -m venv .venv_compare
.\.venv_compare\Scripts\python.exe -m pip install -r requirements.txt
```

No virtual-environment activation is required with these explicit interpreter paths.

Generate predictions using the trained historical model and five example bookings:

```powershell
.\.venv_compare\Scripts\python.exe predict.py --policy retained --family historical24 --input prepared/retained/historical24/example_inputs.csv --output sample_predictions.csv
```

The more conservative 20-field candidate can be used similarly:

```powershell
.\.venv_compare\Scripts\python.exe predict.py --policy retained --family general20 --input prepared/retained/general20/example_inputs.csv --output general_predictions.csv
```

`historical24` is explicitly a historical sensitivity model. Missing confirmation-time evidence is not fixed by entering the same fields into a form.

## Reproduce all four arms from scratch

This creates a new folder, keeping the frozen results unchanged. It reruns all 48 fits, fits every preprocessor, creates every best model, freezes winners, evaluates them and rebuilds the report.

```powershell
.\.venv_compare\Scripts\python.exe reproduce.py --out fresh_run --threads 4
```

The finite candidate lists are in experiment.py. Three configurations per model is a matched trial-count budget, not a claim that every algorithm is fully optimised. Runtime can differ by algorithm.

The recorded run used Python 3.12.14 on Linux. Exact Python packages are pinned in requirements.txt and environment.json. The code uses relative artifact paths and native XGBoost/CatBoost model formats. Windows instructions have not been executed on a Windows host here; raw-input inference and the CLI were executed in the recorded environment. Bitwise training identity across platforms is not promised.

## Package contents

| File/directory | Purpose |
|---|---|
| features.py | Shared feature contract and preprocessing |
| experiment.py | Preparation, grouped splitting, training, selection and evaluation |
| reproduce.py | Complete fresh-directory replay |
| predict.py | Raw-input inference for each frozen winner |
| report.py | Matched tables, figure and paired bootstrap |
| plan.json; analysis_plan.json | Frozen design, seed, configuration lists and planned comparisons |
| data/hotel_bookings.csv | Exact supplied public dataset used in the experiment |
| splits/membership.csv | Original CSV row, raw-input group, full-record ID, one-copy flag and split |
| prepared/*/*/preprocessor.joblib | Four train-fitted preprocessors |
| prepared/*/*/features.json | Raw and encoded feature names |
| prepared/*/*/example_inputs.csv | Training-only examples for inference |
| models/*/*/*/best.json | Best configuration and metrics for all 16 model/arm combinations |
| models/*/*/*/validation_predictions.csv | Validation probabilities from those best configurations |
| selected model files | The four frozen winner objects, sufficient for raw-input prediction |
| results | All 48 trials, 16 validation comparisons, model reload checks, raw-input checks, paired test probabilities and bootstrap differences |
| selection_before_test.json | All four selections frozen before test scoring |
| test_results.json; evaluation_seal.json | Full test metrics and checksums binding the evaluation to selection |
| environment.json; requirements.txt | Recorded package versions |
| omitted_files.json | Hashes/sizes of reproducible non-selected weights and numeric caches omitted from compact delivery |
| audit | Independent audit code, verification JSON, audit report and original uploaded ZIP |
| file_checksums.json | SHA-256 of every delivered file except this checksum file |

The compact package includes all trial evidence and the **four selected winners**. Large non-selected model weights and regenerable numerical training caches are omitted explicitly; `reproduce.py` rebuilds them. Source validation reload checks were completed for all 16 best models before packaging.

This deliverable covers the comparative experiment and the uploaded package audit. It contains no Streamlit dashboard and no new SHAP computation. The original Claude ZIP contains its existing SHAP figures and source code; the audit distinguishes figure inspection from rerunning SHAP.

## Reading the scores

An F1 score on repeated records answers a different question from an F1 score with one copy per record. The same fitted model can score above 80% under one weighting and near 70% under the other. Compare training policies in the same evaluation column.

Paired intervals are exploratory and conditional on the fitted models. They do not remove earlier test exposure or establish future/Malaysian performance. Choose a duplicate policy for its unit of analysis and record its assumptions in Chapters 3 and 4.

## Reproduce the Claude audit

Extract audit/FYP_model_results_original.zip into audit/source. Use a separate environment for its saved preprocessing: our successful verification used pandas 3.0.6, scikit-learn 1.9.1, narwhals 2.27.1, numpy 2.3.5, scipy 1.17.0, xgboost 3.4.1 and joblib 1.5.3. These are verification versions, not a recovered complete original environment. Keep them separate from the newly trained comparison models' requirements.

```powershell
py -3.12 -m venv .venv_audit
.\.venv_audit\Scripts\python.exe -m pip install pandas==3.0.6 scikit-learn==1.9.1 narwhals==2.27.1 numpy==2.3.5 scipy==1.17.0 xgboost==3.4.1 joblib==1.5.3 threadpoolctl==3.6.0
.\.venv_audit\Scripts\python.exe audit/audit.py --source audit/source --data data/hotel_bookings.csv --load-model --output audit/audit_verification_repeated.json
```

The audit only reconstructs inputs, loads supplied objects and recomputes predictions. It does not retrain Claude's full search. Running Claude's evaluate.py still requires artifacts missing from its uploaded ZIP.

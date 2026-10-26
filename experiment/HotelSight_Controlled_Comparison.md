# HotelSight: controlled comparison of duplicate policies and inputs

## What was actually run

All 48 candidate fits completed: four models × three hand-picked configurations × two duplicate policies × two predictor sets. These are actual training and prediction results. Seed 42 and a classification threshold of 0.50 were fixed. No seed search, oversampling, threshold tuning or test-driven refitting was performed.

This is a new controlled experiment, not a reproduction of Claude's exact scores. Its groups, country encoding and tuning budget differ from Claude's run. CatBoost receives the same one-hot/scaled matrix as the other models in this experiment; Claude used CatBoost's native categorical handling.

## Dataset and evaluation populations

The supplied CSV contains 119,390 records. Matching the brief's cleaning cohort removes 180 zero-guest records, one negative ADR and the one ADR of 5,400. This leaves 119,208 retained records or 87,228 full records after deduplication. There are 31,980 repeated copies beyond the first. Zero-night records were retained, as in Claude's experiment. An extreme ADR is not proven erroneous; its exclusion is an explicit fixed assumption.

Groups are defined using the 20 raw general inputs, without the target or removed outcome fields. Every group has exactly the same train/validation/test assignment across both policies and both input sets. One-copy class quotas are stratified 70/15/15; repeated-copy class counts are balanced approximately. Group membership and class counts are included in the files.

| Policy | Training rows | Validation rows | Test rows | Cancellation proportion |
|---|---:|---:|---:|---:|
| deduplicated | 61,059 | 13,084 | 13,085 | 27.52% in training |
| retained | 83,445 | 17,881 | 17,882 | 37.08% in training |

Preprocessing is fitted separately on each arm's training rows and shared unchanged by its four models. The country vocabulary, numerical medians, scaling and encoding never learn from validation or test rows. Countries are not frequency-grouped in this experiment. Agent/company flags are derived before filling missing identifiers; missing IDs cannot accidentally become 'present'.

## Predictor choices

**General20** uses hotel, meal, customer type, market segment, travel-agent presence, company-booking presence, waiting-list days, lead time, arrival month/week/day, weekend/weekday nights, adults/children/babies, country, returning-guest indicator and previous cancelled/non-cancelled booking counts.

**Historical24** adds ADR, deposit type, parking spaces and special requests. These are useful historical predictors, but their CSV values are not verified as available at initial confirmation. Even General20 is a conservative proxy comparison: the CSV lacks confirmation-time snapshots, and several ordinary booking descriptors can be amended later. Neither arm is certified for prospective booking-confirmation prediction or for Malaysian hotels.

No anonymous room codes or raw agent/company identifiers are needed in either form. Arrival year and distribution channel are omitted consistently.

## Validation comparison: all four models

Each model's configuration and each arm's winner were selected by validation F1, with AUC breaking exact ties. Both policies use the same validation groups, but deduplication counts one record while retention counts every copy. These validation objectives should not be treated as identical.

| Policy | Inputs | Model | Validation F1 | Precision | Recall | Accuracy | ROC AUC |
|---|---|---|---:|---:|---:|---:|---:|
| deduplicated | general20 | Logistic Regression | 55.93% | 46.73% | 69.65% | 69.80% | 0.7747 |
| deduplicated | general20 | Random Forest | 60.20% | 54.38% | 67.43% | 75.47% | 0.8280 |
| deduplicated | general20 | XGBoost | 62.25% | 53.84% | 73.76% | 75.37% | 0.8370 |
| deduplicated | general20 | CatBoost | 62.27% | 50.44% | 81.37% | 72.87% | 0.8331 |
| deduplicated | historical24 | Logistic Regression | 62.22% | 52.50% | 76.37% | 74.48% | 0.8297 |
| deduplicated | historical24 | Random Forest | 68.48% | 66.39% | 70.70% | 82.08% | 0.8826 |
| deduplicated | historical24 | XGBoost | 69.80% | 64.87% | 75.53% | 82.01% | 0.8934 |
| deduplicated | historical24 | CatBoost | 69.39% | 60.55% | 81.26% | 80.27% | 0.8915 |
| retained | general20 | Logistic Regression | 69.29% | 66.58% | 72.22% | 76.26% | 0.8419 |
| retained | general20 | Random Forest | 72.24% | 70.25% | 74.34% | 78.82% | 0.8786 |
| retained | general20 | XGBoost | 73.05% | 72.52% | 73.59% | 79.87% | 0.8899 |
| retained | general20 | CatBoost | 73.79% | 70.29% | 77.66% | 79.55% | 0.8900 |
| retained | historical24 | Logistic Regression | 74.19% | 71.91% | 76.62% | 80.24% | 0.8844 |
| retained | historical24 | Random Forest | 79.05% | 83.97% | 74.68% | 85.33% | 0.9266 |
| retained | historical24 | XGBoost | 80.29% | 79.12% | 81.49% | 85.16% | 0.9302 |
| retained | historical24 | CatBoost | 80.03% | 77.04% | 83.26% | 84.59% | 0.9288 |

All 48 trial configurations, effective parameters, class weights, training/validation metrics, timestamps, durations and warnings are saved in results/*/*_trials.json. The budget was three fixed configurations per model rather than an open-ended automated search. This limited manual comparison does not establish each algorithm's maximum achievable performance.

## Matched test comparison

All four winners were frozen before any test metrics were computed. The same held-out rows and the same one-copy subset were then used for every winner.

| Training policy / inputs | Selected model | One-copy test F1 | One-copy accuracy | Repeated-record test F1 | Repeated-record accuracy |
|---|---|---:|---:|---:|---:|
| Dedup / 20 fields | CatBoost | 60.81% | 71.76% | 72.51% | 75.79% |
| Retained / 20 fields | CatBoost | 60.24% | 76.23% | 73.53% | 79.30% |
| Dedup / 24 fields | XGBoost | 69.69% | 81.98% | 80.08% | 84.41% |
| Retained / 24 fields | XGBoost | 69.95% | 83.07% | 81.02% | 85.70% |

The accompanying figure is comparison.png in the experiment package.

Changing columns from one-copy to repeated-record evaluation leaves the trained model unchanged. Any resulting F1 difference is an evaluation-weighting effect, not a training improvement. Deduplicated records are not guaranteed unique guests or independent bookings; missing identifiers prevent that conclusion.

## Paired uncertainty checks

The same predictor-profile groups were resampled for every model in 10,000 bootstrap replicates, seed 20261022. Intervals are conditional on these fitted models and are exploratory. They do not repair earlier test exposure, identify a population of true independent guests or account for training/search variability. No multiplicity adjustment is applied.

| Comparison (first minus second) | Evaluation | F1 difference, percentage points | 95% interval, percentage points |
|---|---|---:|---:|
| duplicate_policy_general20 | retained | +1.02 | [+0.25, +1.74] |
| duplicate_policy_historical24 | retained | +0.94 | [+0.35, +1.56] |
| extra_features_deduplicated | retained | +7.58 | [+6.40, +9.05] |
| extra_features_retained | retained | +7.50 | [+6.16, +9.15] |
| duplicate_policy_general20 | one_copy | -0.57 | [-1.46, +0.31] |
| duplicate_policy_historical24 | one_copy | +0.26 | [-0.43, +0.95] |
| extra_features_deduplicated | one_copy | +8.88 | [+7.85, +9.90] |
| extra_features_retained | one_copy | +9.71 | [+8.52, +10.89] |

An interval containing zero does not demonstrate a clear difference; it does not prove equivalence. An interval excluding zero supports a conditional difference on this held-out population, subject to the exploratory limitations above.

## What to use in the FYP

Choose the duplicate policy according to the intended unit of analysis. Retention preserves the observed frequency of repeated rows, under an explicit assumption that those repetitions should count. Deduplication gives each complete record equal weight. Keep both as a sensitivity comparison. A higher retained-record F1 alone is not a reason to declare a better predictor.

For the stated booking-confirmation application, establish when each input was recorded. Present Historical24 as a retrospective sensitivity result unless confirmation-time values can be verified. General20 is a more conservative starting point, not a substitute for a timing audit. Update Chapters 3 and 4 to match the implemented group split, missing-value rules, predictors, finite manual tuning budget and test-exposure disclosure. Report the achieved scores without requiring 80%.

The dataset and earlier test scores were already inspected during project development. These new split assignments do not make the data an independent fresh external test. Within this run, test outcomes were withheld until selection was frozen; the findings remain exploratory. A prospective day-before-arrival task would also need to exclude bookings already cancelled by that landmark and define subsequent cancellation as its target.

## Reproduction and files

README.md contains exact Windows commands. The package contains the dataset, the four selected winner models, all 16 best-model metadata records, four fitted preprocessors, 48 trial records, frozen selection, split membership, paired probabilities, bootstrap differences, environment versions, hashes and raw-input prediction code. Check RAW inference verification and model reload checks in results/. Large non-selected model weights and regenerable training caches are omitted from the compact delivery; omitted_files.json lists their hashes. reproduce.py recreates every fitted candidate in a new folder. This is an experiment package; no Streamlit dashboard or new SHAP computation is included.

## Sources

- Antonio et al. (2019), original dataset definitions and extraction timing: https://pmc.ncbi.nlm.nih.gov/articles/PMC6297060/
- scikit-learn 1.8, train-only preprocessing: https://scikit-learn.org/1.8/common_pitfalls.html

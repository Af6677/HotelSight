# Proposed thesis amendments – for discussion with Dr. Mazura binti Mat Din

These amendments make Chapters 3 and 4 describe what was actually implemented and evaluated.
Every number below comes from the frozen controlled experiment in `experiment/`, the
`retained_historical24` arm, and was re-verified by `scripts/verify_experiment.py`.
**Please agree these changes with your supervisor before editing the thesis.**

> Earlier exploratory runs (validation F1 0.80–0.82, test F1 0.797–0.807, kept in `archive/`)
> used different splits and preprocessing. They are **superseded** and should not be mixed with
> these results. Mention them only in the test-exposure disclosure (Section 3.5.2 below).

---

## Related wording in Chapters 1 and 2 (short)

| Where | Current wording | Proposed change |
|---|---|---|
| Abstract, 1.6, 3.6.4 | the system predicts "at booking confirmation" | "The system is an experimental prototype evaluated on historical hotel records. It estimates cancellation risk from the booking details recorded in the dataset; the dataset does not verify that every input was known at booking confirmation." |
| 1.6 Research Scope | "31 candidate predictor variables" | Keep, and add: "After cleaning and leakage review, 24 inputs were used (Section 3.3.4)." |
| 1.6 Research Scope | – | Add: "Results describe historical classification performance on two Portuguese hotels (2015–2017). They do not establish performance for future bookings or for Malaysian hotels." |
| 3.5.3 | "These two explanation levels address the **third** research gap" | Change to "**second** research gap" (explainability is gap 2 in Section 2.6.2). |

---

## Chapter 3 amendments

### 3.3.1 Dataset Description – add after the class distribution paragraph

> After cleaning (Section 3.3.4), 119,208 records remain, 37.08% of them cancelled. The file
> contains 31,980 records that are exact copies of an earlier record (31,994 before cleaning).
> Because the published dataset has no booking identifier, it cannot be determined whether
> these repeated records are separate bookings (for example, rooms in one group booking) or
> duplicated rows. When each identical record is counted once, 87,228 records remain, 27.52% of
> them cancelled.

### Table 3.2 – update the "Variable Role" column

| Variable | New role |
|---|---|
| reservation_status, reservation_status_date | Removed – outcome leakage |
| assigned_room_type | Removed – assigned at check-in |
| booking_changes | Removed – counted until check-in or cancellation |
| arrival_date_year | Removed – only 2015–2017, cannot take values for new bookings |
| reserved_room_type | Removed – anonymised letter codes a manager cannot interpret |
| distribution_channel | Removed – overlaps with market_segment |
| agent, company | Replaced by Yes/No flags `has_travel_agent`, `company_booking` (IDs are anonymised) |
| days_in_waiting_list | Predictor – recorded before confirmation (Antonio et al., 2019) |
| adr, deposit_type, required_car_parking_spaces, total_of_special_requests | Predictor – **timing not verified** (see 3.3.4); tested in a sensitivity experiment |
| all other candidate predictors | Predictor |

### 3.3.4 Data Pre-processing – replace the second, third and fourth paragraphs

**Cleaning.**
> Records with no guests (180), a negative ADR (1) and the single ADR of 5,400 (1) were removed,
> leaving 119,208 records. The ADR of 5,400 is treated as an outlier by a fixed rule; it is not
> proven to be an error. Missing country values (488) were set to "Unknown", and missing children
> counts (4) to 0. Missing agent and company identifiers mean "not applicable" in the source
> system (Antonio et al., 2019), so they became the flags `has_travel_agent` and
> `company_booking`.

**Leakage and timing review.**
> `reservation_status` and `reservation_status_date` reveal the outcome and were removed.
> `assigned_room_type` is set at check-in and `booking_changes` is counted until check-in or
> cancellation, so both were removed. The dataset documentation states that values were taken
> from the property-management system relative to the day before arrival, that deposit type is
> derived from payments made before arrival or cancellation, and that ADR is derived from lodging
> transactions. Parking spaces and special requests can also be added after booking. These four
> inputs are therefore **not verified as known at booking confirmation**. They were kept in the
> main model and their effect is measured separately in a 20-input sensitivity experiment
> (Section 4.5). The evaluation also cannot be read as a day-before-arrival forecast: 93.9% of
> cancelled bookings already had a cancellation date earlier than the day before arrival.

**Final inputs.**
> 24 inputs: hotel, meal, customer_type, market_segment, has_travel_agent, company_booking,
> days_in_waiting_list, lead_time, arrival_date_month, arrival_date_week_number,
> arrival_date_day_of_month, stays_in_weekend_nights, stays_in_week_nights, adults, children,
> babies, country, is_repeated_guest, previous_cancellations, previous_bookings_not_canceled,
> adr, deposit_type, required_car_parking_spaces and total_of_special_requests.

**Repeated records and splitting.**
> To stop identical bookings from appearing in both training and test data, records with
> identical values on the 20 general inputs form one group. Each group is assigned as a whole to
> the training, validation or test set (70/15/15, seed 42). The split is stratified so that the
> one-copy records follow exact 70/15/15 class quotas and the repeated copies are balanced
> approximately. No outcome or excluded field is part of the grouping key, and no group appears in
> more than one set. Two weighting policies use the same groups: *retained* (every repeated copy
> kept: 83,445 / 17,881 / 17,882 rows) and *deduplicated* (one copy per complete record:
> 61,059 / 13,084 / 13,085 rows).

**Training-only preprocessing.**
> Median imputation and standard scaling of numeric inputs, and one-hot encoding of categorical
> inputs (unseen categories ignored), were fitted on the training rows only and then applied
> unchanged to the validation and test rows. All four models used this same encoded matrix.

**Class imbalance.**
> Class weights were computed from the training class counts: `class_weight="balanced"` for
> Logistic Regression and Random Forest, `scale_pos_weight` = 1.697 for XGBoost and
> `auto_class_weights="Balanced"` for CatBoost. No resampling was used.

### 3.4.3 Manual Hyperparameter Configuration – replace the first paragraph

> Each model was given three configurations, chosen in advance by hand as small changes or
> small combinations of the hyperparameters in Table 3.4 (Table 3.4a). Each configuration was
> trained once and evaluated on the validation set. The decision threshold stayed at 0.50, the
> random seed stayed at 42, and the number of training rounds was fixed, with no early stopping,
> automated search, threshold tuning or seed search. This is a limited manual comparison with an
> equal budget for every model (12 fits per experiment arm, 48 fits in total). It does not
> establish each algorithm's best possible performance.

**New Table 3.4a – configurations actually compared (main arm)**

| Model | Varied | Fixed |
|---|---|---|
| Logistic Regression | C = 0.1 / 1 / 10 | solver lbfgs, L2 penalty, class weight balanced |
| Random Forest | trees 250/400/400, max depth 20/40/40, min samples leaf 2/1/1, max features log2/log2/0.5 | min samples split 5, class weight balanced |
| XGBoost | rounds 400/600/600, max depth 6/10/10, column subsample 0.5/0.5/0.8 | learning rate 0.1, subsample 1.0, gamma 0, scale_pos_weight 1.697 |
| CatBoost | iterations 350/500/500, depth 6/8/8, L2 3/3/6 | learning rate 0.1, auto class weights Balanced |

### 3.5.2 Model Comparison and Selection – add at the end

> Selection used validation F1, with AUC breaking exact ties. All selections were frozen before
> any test prediction was scored, and the frozen winners were then scored once on the test set.
> The test set is reported in two ways: (a) **repeated records**, where every repeated copy
> counts, and (b) **one copy per complete record**. Repeated records are not proven to be
> separate bookings, and one-copy records are not proven to be unique guests, so both are
> reported. A paired group bootstrap (10,000 replicates) gives exploratory 95% intervals for F1
> differences between arms.
>
> **Test-set exposure.** The dataset, and test scores from earlier exploratory runs with
> different splits, were inspected during development. The final experiment repartitioned the
> data and withheld its own test outcomes until selection was frozen. Its results are therefore
> exploratory evidence, not an independent external validation.

### 3.5.3 SHAP Explainability Analysis – replace

> SHAP values were computed with TreeSHAP (`shap.TreeExplainer`, tree-path-dependent) on the
> deployed XGBoost model. Its background is the training data, as encoded in the trees' node
> statistics. Contributions are in **log-odds of cancellation**; the base value plus all
> contributions reconstructs the model's log-odds output (maximum error 2 × 10⁻⁵ on the 17,882
> test rows), and the probability is obtained with the logistic function. One-hot columns are
> summed back to the 24 inputs. Global importance is the mean absolute SHAP value over the test
> set. Local explanations show the largest contributions for a single booking. SHAP describes
> associations learned by the model; it does not show that an input causes a cancellation.

### 3.6.2 / 3.6.4 Dashboard – replace the input description

> The dashboard collects the 24 inputs with plain-English labels. The user enters three dates
> (booking, arrival and departure), and the system derives lead time, arrival month, week number,
> day of month and the two night counts, reproducing the dataset's conventions. The week number
> follows SQL Server `DATEPART(week)`, with Sunday-first weeks and week 1 containing 1 January.
> The dataset's "weekend nights" are nights beginning on a Sunday or Monday. Both rules reproduce
> 100% of the dataset's recorded values. Country names are shown while country codes are passed
> to the model. Travel-agent and company involvement are Yes/No questions, and ADR is entered in
> EUR as in the dataset. The pipeline is: booking form → validation → shared feature construction
> → saved preprocessing → saved model → probability and classification (threshold 0.50) → SHAP
> explanation. Pages: prediction with local explanation, four-model validation comparison, test
> performance, global SHAP importance, and methodology and limitations.

### 3.6.5 System Testing – add

> Automated tests (pytest, 30 tests) check model loading and file hashes, reproduction of all
> saved validation and test predictions, the date rules on all 119,390 records, input validation,
> missing and unseen values, SHAP additivity and agreement with XGBoost's own TreeSHAP, and that
> every page runs without errors. Functional test cases FT01–FT04 (Table 3.6) correspond to these
> automated tests (see Chapter 4). Usability testing with participants (Table 3.7) is carried out
> separately.

---

## Chapter 4 – proposed structure and results

### 4.1 Data preparation results
Use the cleaning counts (3.3.4) and this split table:

| Population | Training | Validation | Test | Cancelled |
|---|---|---|---|---|
| Repeated records kept | 83,445 | 17,881 | 17,882 | 37.08% |
| One copy per record | 61,059 | 13,084 | 13,085 | 27.52% |

### 4.2 Exploratory data analysis
*(Still to be produced from your EDA notebook. It is not part of this package.)*

### 4.3 Model comparison (validation set, repeated records, 24 inputs)

| Model | Config | F1 | Precision | Recall | Accuracy | AUC | Train F1 |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 1 | 0.7419 | 0.7191 | 0.7662 | 0.8024 | 0.8844 | 0.7412 |
| Random Forest | 2 | 0.7905 | 0.8397 | 0.7468 | 0.8533 | 0.9266 | 0.9504 |
| **XGBoost (selected)** | 2 | **0.8029** | 0.7912 | 0.8149 | 0.8516 | 0.9302 | 0.9533 |
| CatBoost | 2 | 0.8003 | 0.7704 | 0.8326 | 0.8459 | 0.9288 | 0.8592 |

Discussion points: the ensembles beat Logistic Regression, which is consistent with Chapter 2. The
top three are within 1.3 F1 points. XGBoost and Random Forest show a large train–validation gap,
so they overfit the training data.

### 4.4 Test results of the selected model (XGBoost)

| Test population | Rows | F1 | Accuracy | Precision | Recall | AUC |
|---|---|---|---|---|---|---|
| Repeated records | 17,882 | **0.8102** | **0.8570** | 0.7975 | 0.8234 | 0.9317 |
| One copy per record | 13,085 | **0.6995** | 0.8307 | 0.6840 | 0.7157 | 0.8906 |

Confusion matrix (repeated records): TN 9,866 · FP 1,386 · FN 1,171 · TP 5,459.
Confusion matrix (one copy): TN 8,292 · FP 1,191 · FN 1,024 · TP 2,578.

### 4.5 Sensitivity experiments (same test rows)

| Arm | Model | Test F1 (repeated) | Test F1 (one copy) |
|---|---|---|---|
| Repeated kept · 24 inputs (deployed) | XGBoost | 0.8102 | 0.6995 |
| One copy · 24 inputs | XGBoost | 0.8008 | 0.6969 |
| Repeated kept · 20 inputs | CatBoost | 0.7353 | 0.6024 |
| One copy · 20 inputs | CatBoost | 0.7251 | 0.6081 |

Paired bootstrap of F1 differences, in percentage points:
- **Keeping repeated records (24 inputs):** +0.94 [+0.35, +1.56] on repeated records; +0.26 [−0.43, +0.95] on one copy per record. That is a small gain on one population and no clear difference on the other.
- **Adding the four timing-uncertain inputs (repeats kept):** +7.50 [+6.16, +9.15] on repeated records; +9.71 [+8.52, +10.89] on one copy per record.

Interpretation: much of the score depends on ADR, deposit type, parking and special requests,
whose availability at booking confirmation is not verified.

### 4.6 SHAP results
Global (mean |SHAP|, repeated records): country 1.28, deposit type 1.01, market segment 0.81,
special requests 0.69, lead time 0.68, parking 0.56, customer type 0.55, ADR 0.52. Use the
dashboard's "What drives predictions" page and one local explanation as figures. Describe them as
associations.

### 4.7 System implementation
Use the dashboard screenshots in `docs/screenshots/`.

### 4.8 Functional testing

| Test case | Function | Automated test | Result |
|---|---|---|---|
| FT01 | Enter valid booking data | `test_predict_default_booking` | Pass |
| FT02 | Leave a required field empty / invalid | `test_predict_rejects_invalid_booking`, `test_date_errors`, `test_record_errors` | Pass |
| FT03 | Generate a prediction | `test_form_path_reproduces_saved_probabilities`, `test_predict_example_matches_saved_prediction` | Pass |
| FT04 | Display SHAP explanation | `test_local_shap_reconstructs_model_output`, `test_page_runs_without_errors` | Pass |

Usability results: *from your participant sessions (not part of this package).*

### 4.9 Discussion and limitations
Cover these points:
- input timing is unverified;
- the evaluation is not a day-before-arrival forecast;
- historical Portuguese data, so performance on future or Malaysian bookings is unknown;
- the repeated-record assumption;
- test-set exposure;
- probabilities are not calibrated because of class weighting;
- the overfitting gap;
- the limited three-configuration tuning budget;
- SHAP shows associations, not causes.

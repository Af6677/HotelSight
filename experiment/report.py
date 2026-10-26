"""Produce the paired comparison, figures and report after frozen evaluation."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from experiment import ROOT, read, save


def pct(v): return f"{100*v:.2f}%"


def paired_bootstrap():
    pred = pd.read_csv(ROOT / "results/paired_test_predictions.csv", float_precision="round_trip")
    plan = read(ROOT / "analysis_plan.json")
    g, _ = pd.factorize(pred.profile_group, sort=True)
    n = int(g.max()+1)
    arms = [c for c in pred if c.startswith(("retained_", "deduplicated_"))]
    counts = {}
    for pop in ["retained", "one_copy"]:
        keep = np.ones(len(pred), dtype=bool) if pop == "retained" else pred.one_copy.to_numpy()
        for arm in arms:
            positive = pred[arm].to_numpy() >= .5; y = pred.actual.to_numpy().astype(bool)
            counts[pop, arm] = np.stack([np.bincount(g, weights=(keep & mask).astype(float), minlength=n) for mask in [positive & y, positive & ~y, ~positive & y]], axis=1)
    pairs = [("duplicate_policy_general20", "retained_general20", "deduplicated_general20"),
             ("duplicate_policy_historical24", "retained_historical24", "deduplicated_historical24"),
             ("extra_features_deduplicated", "deduplicated_historical24", "deduplicated_general20"),
             ("extra_features_retained", "retained_historical24", "retained_general20")]
    rng = np.random.default_rng(plan["bootstrap_seed"])
    samples = {(pop, label): [] for pop in ["retained", "one_copy"] for label, _, _ in pairs}
    def f1(c): return np.divide(2*c[..., 0], 2*c[..., 0]+c[..., 1]+c[..., 2], out=np.zeros(c.shape[:-1]), where=(2*c[..., 0]+c[..., 1]+c[..., 2]) != 0)
    for start in range(0, plan["bootstrap_iterations"], 100):
        b = min(100, plan["bootstrap_iterations"]-start)
        weights = rng.multinomial(n, np.full(n, 1/n), size=b)
        scores = {key: f1(weights @ value) for key, value in counts.items()}
        for pop in ["retained", "one_copy"]:
            for label, a, b in pairs: samples[pop, label].extend((scores[pop, a]-scores[pop, b]).tolist())
    results = []
    for pop in ["retained", "one_copy"]:
        for label, a, b in pairs:
            delta = np.asarray(samples[pop, label]); low, high = np.quantile(delta, [.025, .975])
            results.append({"population": pop, "comparison": label, "model_a": a, "model_b": b,
                            "f1_difference": float(f1(counts[pop, a].sum(axis=0))-f1(counts[pop, b].sum(axis=0))),
                            "lower_95": float(low), "upper_95": float(high), "interval_includes_zero": bool(low <= 0 <= high),
                            "bootstrap_groups": n, "replicates": plan["bootstrap_iterations"]})
    pd.DataFrame({p+"__"+label: values for (p, label), values in samples.items()}).to_csv(ROOT / "results/bootstrap_differences.csv", index=False)
    save(ROOT / "results/paired_comparisons.json", results)
    return results


def main():
    results = read(ROOT / "test_results.json")["results"]
    plan = read(ROOT / "plan.json"); validation = pd.read_csv(ROOT / "results/validation_comparison.csv")
    pair_path = ROOT / "results/paired_comparisons.json"
    pair = read(pair_path) if pair_path.exists() else paired_bootstrap()
    order = ["deduplicated_general20", "retained_general20", "deduplicated_historical24", "retained_historical24"]
    names = {"deduplicated_general20": "Dedup / 20 fields", "retained_general20": "Retained / 20 fields",
             "deduplicated_historical24": "Dedup / 24 fields", "retained_historical24": "Retained / 24 fields"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    colors = ["#476e8c", "#19806d", "#476e8c", "#19806d"]
    for ax, pop, title in zip(axes, ["one_copy_test", "retained_test"], ["Same test records, one copy each", "Same test records, repetitions counted"]):
        values = [results[a][pop]["f1"] for a in order]
        bars = ax.bar(range(4), values, color=colors, width=.65)
        ax.bar_label(bars, labels=[pct(v) for v in values], padding=4, fontsize=10)
        ax.set_xticks(range(4), [names[a].replace(" / ", "\n") for a in order])
        ax.set_ylim(0, 1); ax.set_title(title, pad=16, fontsize=12); ax.set_ylabel("Cancellation F1")
        ax.axhline(.8, ls="--", color="#9b5144", linewidth=1)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Duplicate weighting changes the evaluation question", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, .94])
    fig.savefig(ROOT / "comparison.png", dpi=180); plt.close(fig)
    lines = ["# HotelSight: controlled comparison of duplicate policies and inputs", "",
             "## What was actually run", "",
             "All 48 candidate fits completed: four models × three hand-picked configurations × two duplicate policies × two predictor sets. These are actual training and prediction results. Seed 42 and a classification threshold of 0.50 were fixed. No seed search, oversampling, threshold tuning or test-driven refitting was performed.", "",
             "This is a new controlled experiment, not a reproduction of Claude's exact scores. Its groups, country encoding and tuning budget differ from Claude's run. CatBoost receives the same one-hot/scaled matrix as the other models in this experiment; Claude used CatBoost's native categorical handling.", "",
             "## Dataset and evaluation populations", "",
             f"The supplied CSV contains {plan['cleaning']['raw_rows']:,} records. Matching the brief's cleaning cohort removes 180 zero-guest records, one negative ADR and the one ADR of 5,400. This leaves {plan['cleaning']['retained_rows']:,} retained records or {plan['cleaning']['deduplicated_rows']:,} full records after deduplication. There are {plan['cleaning']['retained_duplicate_copies']:,} repeated copies beyond the first. Zero-night records were retained, as in Claude's experiment. An extreme ADR is not proven erroneous; its exclusion is an explicit fixed assumption.", "",
             "Groups are defined using the 20 raw general inputs, without the target or removed outcome fields. Every group has exactly the same train/validation/test assignment across both policies and both input sets. One-copy class quotas are stratified 70/15/15; repeated-copy class counts are balanced approximately. Group membership and class counts are included in the files.", "",
             "| Policy | Training rows | Validation rows | Test rows | Cancellation proportion |", "|---|---:|---:|---:|---:|"]
    for policy in ["deduplicated", "retained"]:
        s = plan["splits"][policy]
        lines.append(f"| {policy} | {s['train']['rows']:,} | {s['validation']['rows']:,} | {s['test']['rows']:,} | {pct(s['train']['cancellation_share'])} in training |")
    lines += ["", "Preprocessing is fitted separately on each arm's training rows and shared unchanged by its four models. The country vocabulary, numerical medians, scaling and encoding never learn from validation or test rows. Countries are not frequency-grouped in this experiment. Agent/company flags are derived before filling missing identifiers; missing IDs cannot accidentally become 'present'.", "",
              "## Predictor choices", "",
              "**General20** uses hotel, meal, customer type, market segment, travel-agent presence, company-booking presence, waiting-list days, lead time, arrival month/week/day, weekend/weekday nights, adults/children/babies, country, returning-guest indicator and previous cancelled/non-cancelled booking counts.", "",
              "**Historical24** adds ADR, deposit type, parking spaces and special requests. These are useful historical predictors, but their CSV values are not verified as available at initial confirmation. Even General20 is a conservative proxy comparison: the CSV lacks confirmation-time snapshots, and several ordinary booking descriptors can be amended later. Neither arm is certified for prospective booking-confirmation prediction or for Malaysian hotels.", "",
              "No anonymous room codes or raw agent/company identifiers are needed in either form. Arrival year and distribution channel are omitted consistently.", "",
              "## Validation comparison: all four models", "",
              "Each model's configuration and each arm's winner were selected by validation F1, with AUC breaking exact ties. Both policies use the same validation groups, but deduplication counts one record while retention counts every copy. These validation objectives should not be treated as identical.", "",
              "| Policy | Inputs | Model | Validation F1 | Precision | Recall | Accuracy | ROC AUC |", "|---|---|---|---:|---:|---:|---:|---:|"]
    for r in validation.to_dict("records"):
        lines.append(f"| {r['policy']} | {r['family']} | {r['model']} | {pct(r['f1'])} | {pct(r['precision'])} | {pct(r['recall'])} | {pct(r['accuracy'])} | {r['roc_auc']:.4f} |")
    lines += ["", "All 48 trial configurations, effective parameters, class weights, training/validation metrics, timestamps, durations and warnings are saved in results/*/*_trials.json. The budget was three fixed configurations per model rather than an open-ended automated search. This limited manual comparison does not establish each algorithm's maximum achievable performance.", "",
              "## Matched test comparison", "",
              "All four winners were frozen before any test metrics were computed. The same held-out rows and the same one-copy subset were then used for every winner.", "",
              "| Training policy / inputs | Selected model | One-copy test F1 | One-copy accuracy | Repeated-record test F1 | Repeated-record accuracy |", "|---|---|---:|---:|---:|---:|"]
    for a in order:
        r = results[a]; u = r["one_copy_test"]; t = r["retained_test"]
        lines.append(f"| {names[a]} | {r['model']} | {pct(u['f1'])} | {pct(u['accuracy'])} | {pct(t['f1'])} | {pct(t['accuracy'])} |")
    lines += ["", "The accompanying figure is comparison.png in the experiment package.", "", "Changing columns from one-copy to repeated-record evaluation leaves the trained model unchanged. Any resulting F1 difference is an evaluation-weighting effect, not a training improvement. Deduplicated records are not guaranteed unique guests or independent bookings; missing identifiers prevent that conclusion.", "",
              "## Paired uncertainty checks", "",
              "The same predictor-profile groups were resampled for every model in 10,000 bootstrap replicates, seed 20261022. Intervals are conditional on these fitted models and are exploratory. They do not repair earlier test exposure, identify a population of true independent guests or account for training/search variability. No multiplicity adjustment is applied.", "",
              "| Comparison (first minus second) | Evaluation | F1 difference, percentage points | 95% interval, percentage points |", "|---|---|---:|---:|"]
    for r in pair:
        lines.append(f"| {r['comparison']} | {r['population']} | {100*r['f1_difference']:+.2f} | [{100*r['lower_95']:+.2f}, {100*r['upper_95']:+.2f}] |")
    lines += ["", "An interval containing zero does not demonstrate a clear difference; it does not prove equivalence. An interval excluding zero supports a conditional difference on this held-out population, subject to the exploratory limitations above.", "",
              "## What to use in the FYP", "",
              "Choose the duplicate policy according to the intended unit of analysis. Retention preserves the observed frequency of repeated rows, under an explicit assumption that those repetitions should count. Deduplication gives each complete record equal weight. Keep both as a sensitivity comparison. A higher retained-record F1 alone is not a reason to declare a better predictor.", "",
              "For the stated booking-confirmation application, establish when each input was recorded. Present Historical24 as a retrospective sensitivity result unless confirmation-time values can be verified. General20 is a more conservative starting point, not a substitute for a timing audit. Update Chapters 3 and 4 to match the implemented group split, missing-value rules, predictors, finite manual tuning budget and test-exposure disclosure. Report the achieved scores without requiring 80%.", "",
              "The dataset and earlier test scores were already inspected during project development. These new split assignments do not make the data an independent fresh external test. Within this run, test outcomes were withheld until selection was frozen; the findings remain exploratory. A prospective day-before-arrival task would also need to exclude bookings already cancelled by that landmark and define subsequent cancellation as its target.", "",
              "## Reproduction and files", "",
              "README.md contains exact Windows commands. The package contains the dataset, the four selected winner models, all 16 best-model metadata records, four fitted preprocessors, 48 trial records, frozen selection, split membership, paired probabilities, bootstrap differences, environment versions, hashes and raw-input prediction code. Check RAW inference verification and model reload checks in results/. Large non-selected model weights and regenerable training caches are omitted from the compact delivery; omitted_files.json lists their hashes. reproduce.py recreates every fitted candidate in a new folder. This is an experiment package; no Streamlit dashboard or new SHAP computation is included.", "",
              "## Sources", "",
              "- Antonio et al. (2019), original dataset definitions and extraction timing: https://pmc.ncbi.nlm.nih.gov/articles/PMC6297060/", "- scikit-learn 1.8, train-only preprocessing: https://scikit-learn.org/1.8/common_pitfalls.html", ""]
    (ROOT / "HotelSight_Controlled_Comparison.md").write_text("\n".join(lines), encoding="utf-8")
    save(ROOT / "results/summary.json", {a: {"model": results[a]["model"], "validation_f1": results[a]["validation"]["f1"], "one_copy_test": results[a]["one_copy_test"], "retained_test": results[a]["retained_test"]} for a in order})
    print(json.dumps(read(ROOT / "results/summary.json"), indent=2))


if __name__ == "__main__": main()

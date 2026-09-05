"""Confidence intervals on the differences between models.

    python -m src.experiments.run_bootstrap

Fits every model, collects per-user held-out errors and per-user precision, then resamples
users 1000 times to put an interval around each pairwise difference. Writes
``results/bootstrap_ci.csv``.

This exists because the main results table is a set of point estimates, and two of the
claims the report leans on are close enough that a reader is entitled to ask whether they
are real:

*   Matrix factorisation beats item-kNN on RMSE by about 0.02.
*   Most-popular beats every personalised model on precision@10.

An interval that excludes zero turns each of those from an observation into a finding. One
that includes zero is just as useful, and considerably more useful than not knowing.
"""

import argparse

import numpy as np
import pandas as pd

from src.config import load_config
from src.data.splitting import load_splits
from src.evaluation.bootstrap import DEFAULT_CONFIDENCE
from src.evaluation.bootstrap import DEFAULT_ITERATIONS
from src.evaluation.bootstrap import accuracy_samples
from src.evaluation.bootstrap import align_per_user
from src.evaluation.bootstrap import bootstrap_user_indices
from src.evaluation.bootstrap import compare_models
from src.evaluation.bootstrap import percentile_interval
from src.evaluation.bootstrap import ranking_samples
from src.evaluation.harness import predict_holdout
from src.evaluation.metrics import per_user_accuracy
from src.evaluation.ranking import precision_at_k
from src.evaluation.ranking import relevant_items_by_user
from src.experiments.registry import MODEL_ORDER
from src.experiments.registry import build_model
from src.experiments.run_main import load_items

OUTPUT_NAME = "bootstrap_ci.csv"
REPORT_K = 10


def per_user_precision(model, relevant, users, k):
    """precision@k for each user, NaN where the user has no relevant held-out item."""
    values = []
    for user_id in users:
        liked = relevant.get(int(user_id))
        if liked is None or len(liked) == 0:
            values.append(float("nan"))
            continue
        recommendations = model.recommend(user_id, k)
        values.append(precision_at_k(recommendations, liked, k))
    return np.array(values, dtype=np.float64)


def collect(config, train, validation, test, cache):
    """Per-user accuracy tables and precision vectors for every model."""
    items = load_items(config)
    threshold = float(config.section("evaluation")["relevance_threshold"])
    relevant = relevant_items_by_user(test, threshold)

    accuracy_tables = {}
    fitted_models = {}
    for name in MODEL_ORDER:
        model = build_model(name, config, items, validation=validation, cache=cache)
        model.fit(train)
        predictions = predict_holdout(model, test)
        accuracy_tables[name] = per_user_accuracy(predictions)
        fitted_models[name] = model
        print("  fitted " + name)
    return accuracy_tables, fitted_models, relevant


def build_precision_table(models, relevant, users, k):
    vectors = {}
    for name, model in models.items():
        vectors[name] = per_user_precision(model, relevant, users, k)
        print("  scored " + name)
    return vectors


def summarise(name, metric, samples, confidence):
    lower, upper = percentile_interval(samples, confidence)
    usable = samples[~np.isnan(samples)]
    if usable.size == 0:
        point = float("nan")
    else:
        point = float(np.mean(usable))
    return {
        "kind": "point",
        "model_a": name,
        "model_b": "",
        "metric": metric,
        "difference": point,
        "lower": lower,
        "upper": upper,
        "excludes_zero": "",
    }


def difference_rows(metric, samples_by_model, confidence):
    frame = compare_models(samples_by_model, confidence)
    rows = []
    for _, row in frame.iterrows():
        rows.append({
            "kind": "difference",
            "model_a": row["model_a"],
            "model_b": row["model_b"],
            "metric": metric,
            "difference": row["difference"],
            "lower": row["lower"],
            "upper": row["upper"],
            "excludes_zero": bool(row["excludes_zero"]),
        })
    return rows


def print_report(frame, metric, note):
    selected = frame[(frame["metric"] == metric) & (frame["kind"] == "difference")]
    selected = selected.reindex(
        selected["difference"].abs().sort_values(ascending=False).index
    )
    print("")
    print(metric + " differences, " + note)
    print("-" * 82)
    print("  " + "a".ljust(14) + "b".ljust(14) + "a - b".rjust(9)
          + "95% interval".rjust(22) + "  real?")
    for _, row in selected.head(12).iterrows():
        if row["excludes_zero"]:
            verdict = "yes"
        else:
            verdict = "no"
        interval = ("[" + format(row["lower"], ".4f") + ", "
                    + format(row["upper"], ".4f") + "]")
        print("  " + str(row["model_a"]).ljust(14) + str(row["model_b"]).ljust(14)
              + format(row["difference"], ".4f").rjust(9)
              + interval.rjust(22) + "  " + verdict)
    print("")


def main():
    parser = argparse.ArgumentParser(
        description="Bootstrap confidence intervals on model differences."
    )
    parser.add_argument("--config", default=None, help="path to a config file")
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE)
    parser.add_argument("--k", type=int, default=REPORT_K)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    config.ensure_output_dirs()
    train, validation, test = load_splits(config.path("processed_dir"))

    accuracy_tables, models, relevant = collect(
        config, train, validation, test, not args.no_cache
    )
    aligned, shared_users = align_per_user(accuracy_tables)
    print("comparing over " + str(len(shared_users)) + " users")

    precision_vectors = build_precision_table(models, relevant, shared_users, args.k)

    rng = config.fresh_rng()
    indices = bootstrap_user_indices(len(shared_users), args.iterations, rng)

    rows = []
    rmse_samples = {}
    precision_samples = {}
    for name in MODEL_ORDER:
        rmse_samples[name] = accuracy_samples(aligned[name], indices, "rmse")
        precision_samples[name] = ranking_samples(precision_vectors[name], indices)
        rows.append(summarise(name, "rmse", rmse_samples[name], args.confidence))
        rows.append(summarise(
            name, "precision_at_" + str(args.k), precision_samples[name], args.confidence
        ))

    rows.extend(difference_rows("rmse", rmse_samples, args.confidence))
    rows.extend(difference_rows(
        "precision_at_" + str(args.k), precision_samples, args.confidence
    ))

    frame = pd.DataFrame(rows)
    output_path = config.path("results_dir") / OUTPUT_NAME
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    print("wrote " + str(output_path) + " (" + str(len(frame)) + " rows)")

    print_report(frame, "rmse", "negative means a is more accurate")
    print_report(frame, "precision_at_" + str(args.k), "positive means a ranks better")


if __name__ == "__main__":
    main()

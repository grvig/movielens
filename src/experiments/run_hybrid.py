"""Fit the density-weighted hybrid and trace the accuracy-coverage frontier.

    python -m src.experiments.run_hybrid

Writes two files:

``results/hybrid_weights.csv``
    The density-weighted hybrid the project plan specified. For each history bin, the
    content weight that maximises validation precision@10, plus the fitted curve
    ``w(n) = sigmoid(a + b * log(1 + n))``.

``results/hybrid_frontier.csv``
    A sweep of fixed content weights from 0 to 1 on validation, giving RMSE, precision and
    coverage at each. Content-based reaches roughly three times the catalogue that matrix
    factorisation does, so this is what a coverage target costs in accuracy.

Both run on validation. The frontier is descriptive rather than a selection over test, and
the one fixed weight that gets registered as a model is chosen here, on validation, by a
stated rule: the largest weight whose RMSE stays within ``RMSE_TOLERANCE`` of the pure
collaborative end. That buys as much coverage as possible for a bounded accuracy cost.
"""

import argparse

import numpy as np
import pandas as pd

from src.config import load_config
from src.data.splitting import load_splits
from src.evaluation.harness import evaluate_model
from src.evaluation.harness import make_run_id
from src.evaluation.harness import results_frame
from src.models.content import ContentBased
from src.models.hybrid import WEIGHT_GRID
from src.models.hybrid import WeightedHybrid
from src.models.hybrid import component_fingerprint
from src.models.hybrid import weight_for_history
from src.models.mf import MatrixFactorization
from src.experiments.run_main import load_items

WEIGHTS_OUTPUT = "hybrid_weights.csv"
FRONTIER_OUTPUT = "hybrid_frontier.csv"
REPORT_K = 10
RMSE_TOLERANCE = 0.02


def build_hybrid(config, items, weight_mode, fixed_weight=0.5, validation=None):
    return WeightedHybrid(
        config,
        ContentBased(config, items, validation=validation),
        MatrixFactorization(config, validation=validation, cache=True),
        weight_mode=weight_mode,
        fixed_weight=fixed_weight,
        validation=validation,
    )


def run_density(config, items, train, validation):
    model = build_hybrid(config, items, "density", validation=validation)
    model.fit(train)
    frame = pd.DataFrame(model.curve_points)
    frame["fitted_intercept"] = model.intercept
    frame["fitted_slope"] = model.slope
    fitted = []
    for median_history in frame["median_history"]:
        fitted.append(weight_for_history(median_history, model.intercept, model.slope))
    frame["fitted_weight"] = fitted
    return model, frame


def run_frontier(config, items, train, validation, run_id):
    rows = []
    for weight in WEIGHT_GRID:
        model = build_hybrid(config, items, "fixed", fixed_weight=weight,
                             validation=validation)
        model.fit(train)
        evaluated = evaluate_model(
            model, train, validation, config, "val",
            variant="w=" + format(weight, ".1f"), run_id=run_id
        )
        frame = results_frame(evaluated)
        rows.append({
            "weight": weight,
            "rmse": value_of(frame, "rmse"),
            "precision_at_10": value_of(frame, "precision_at_k", REPORT_K),
            "recall_at_10": value_of(frame, "recall_at_k", REPORT_K),
            "coverage_at_10": value_of(frame, "coverage", REPORT_K),
            "popularity_percentile_at_10": value_of(
                frame, "popularity_percentile", REPORT_K
            ),
        })
        print("  w=" + format(weight, ".1f")
              + "  rmse " + format(rows[-1]["rmse"], ".4f")
              + "  p@10 " + format(rows[-1]["precision_at_10"], ".4f")
              + "  cov@10 " + format(rows[-1]["coverage_at_10"], ".4f"))
    return pd.DataFrame(rows)


def value_of(frame, metric, k=None):
    selected = frame[frame["metric"] == metric]
    if k is not None:
        selected = selected[selected["k"] == k]
    return float(selected["value"].iloc[0])


def choose_frontier_weight(frontier, tolerance=RMSE_TOLERANCE):
    """Largest content weight whose RMSE stays within ``tolerance`` of the pure CF end."""
    baseline = float(frontier[frontier["weight"] == 0.0]["rmse"].iloc[0])
    allowed = frontier[frontier["rmse"] <= baseline + tolerance]
    if len(allowed) == 0:
        return 0.0
    return float(allowed["weight"].max())


def main():
    parser = argparse.ArgumentParser(description="Fit and sweep the hybrid weight.")
    parser.add_argument("--config", default=None, help="path to a config file")
    args = parser.parse_args()

    config = load_config(args.config)
    config.ensure_output_dirs()
    train, validation, test = load_splits(config.path("processed_dir"))
    items = load_items(config)
    run_id = make_run_id()
    results_dir = config.path("results_dir")

    print("fitting the density-weighted hybrid on validation")
    model, weights = run_density(config, items, train, validation)
    weights.to_csv(results_dir / WEIGHTS_OUTPUT, index=False)
    print(weights.round(4).to_string(index=False))
    print("")
    print("fitted curve: w(n) = sigmoid("
          + format(model.intercept, ".4f") + " + "
          + format(model.slope, ".4f") + " * log(1 + n))")
    print("wrote " + str(results_dir / WEIGHTS_OUTPUT))

    print("")
    print("sweeping fixed weights on validation")
    frontier = run_frontier(config, items, train, validation, run_id)
    frontier.to_csv(results_dir / FRONTIER_OUTPUT, index=False)
    print("wrote " + str(results_dir / FRONTIER_OUTPUT))

    chosen = choose_frontier_weight(frontier)
    print("")
    print("frontier weight within " + format(RMSE_TOLERANCE, ".2f")
          + " RMSE of pure CF: w=" + format(chosen, ".1f"))
    print("")
    print("components fingerprint: " + component_fingerprint(config))
    print("Write the constants and the fingerprint into configs/default.yaml by hand;")
    print("this script does not edit it. The fingerprint makes a stale curve fail loudly.")


if __name__ == "__main__":
    main()

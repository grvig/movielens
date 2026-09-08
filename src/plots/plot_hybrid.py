"""Figures for the two hybrids.

    python -m src.plots.plot_hybrid

Reads ``results/hybrid_weights.csv`` and ``results/hybrid_frontier.csv`` and writes
``figures/hybrid.pdf``: the fitted density curve beside the accuracy-coverage frontier.

Both panels exist to show the same thing from different angles - that blending two models
which are each individually worse than the alternative still beats either one, because
their errors are decorrelated.
"""

import argparse

import pandas as pd
import matplotlib.pyplot as plt

from src.config import load_config
from src.models.hybrid import weight_for_history
from src.plots.style import apply_style
from src.plots.style import save

WEIGHTS_INPUT = "hybrid_weights.csv"
FRONTIER_INPUT = "hybrid_frontier.csv"
OUTPUT_NAME = "hybrid.pdf"

CONTENT_COLOUR = "#2e8b57"
CF_COLOUR = "#8c3025"
HYBRID_COLOUR = "#6a3d9a"


def draw_weight_curve(axis, weights):
    intercept = float(weights["fitted_intercept"].iloc[0])
    slope = float(weights["fitted_slope"].iloc[0])
    histories = [1, 2, 5, 10, 20, 50, 100, 200, 400, 700]
    fitted = []
    for history in histories:
        fitted.append(weight_for_history(history, intercept, slope))

    axis.plot(histories, fitted, color=HYBRID_COLOUR,
              label="fitted w(n) = sigmoid(a + b log(1+n))")
    axis.scatter(weights["median_history"], weights["best_weight"],
                 color=HYBRID_COLOUR, zorder=3, label="best weight per history bin")
    axis.axhline(0.5, color="#999999", linewidth=0.8, linestyle=":")
    axis.set_xscale("log")
    axis.set_xlabel("training ratings the user has")
    axis.set_ylabel("weight given to the content model")
    axis.set_ylim(0.0, 1.0)
    axis.set_title("Learned weighting: content-heavy when sparse")
    axis.legend(loc="upper right")


def draw_frontier(axis, frontier):
    axis.plot(frontier["coverage_at_10"], frontier["precision_at_10"],
              color=HYBRID_COLOUR, marker="o")
    for _, row in frontier.iterrows():
        if row["weight"] in [0.0, 0.4, 1.0]:
            axis.annotate(
                "w=" + format(row["weight"], ".1f"),
                (row["coverage_at_10"], row["precision_at_10"]),
                textcoords="offset points", xytext=(6, 4), fontsize=8,
            )
    pure_cf = frontier[frontier["weight"] == 0.0]
    pure_content = frontier[frontier["weight"] == 1.0]
    axis.scatter(pure_cf["coverage_at_10"], pure_cf["precision_at_10"],
                 color=CF_COLOUR, zorder=4, label="pure matrix factorisation")
    axis.scatter(pure_content["coverage_at_10"], pure_content["precision_at_10"],
                 color=CONTENT_COLOUR, zorder=4, label="pure content-based")
    axis.set_xlabel("catalogue coverage@10")
    axis.set_ylabel("precision@10")
    axis.set_title("The blend beats both of its endpoints")
    axis.legend(loc="lower right")


def draw(weights, frontier):
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    draw_weight_curve(axes[0], weights)
    draw_frontier(axes[1], frontier)
    figure.suptitle("Hybrid weighting, fitted on validation", y=1.03, fontsize=11)
    return figure


def main():
    parser = argparse.ArgumentParser(description="Plot the hybrid weighting results.")
    parser.add_argument("--config", default=None, help="path to a config file")
    args = parser.parse_args()

    config = load_config(args.config)
    results_dir = config.path("results_dir")
    for name in [WEIGHTS_INPUT, FRONTIER_INPUT]:
        if not (results_dir / name).exists():
            raise FileNotFoundError(
                "missing " + str(results_dir / name) +
                ". Run python -m src.experiments.run_hybrid first."
            )
    weights = pd.read_csv(results_dir / WEIGHTS_INPUT)
    frontier = pd.read_csv(results_dir / FRONTIER_INPUT)

    apply_style()
    figure = draw(weights, frontier)
    save(figure, config.path("figures_dir") / OUTPUT_NAME)


if __name__ == "__main__":
    main()

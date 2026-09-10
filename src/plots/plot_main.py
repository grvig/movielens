"""The main comparison figures.

    python -m src.plots.plot_main

Reads ``results/main_results.csv`` and writes two figures:

``figures/accuracy.pdf``
    RMSE and MAE per model, as bars. The plain accuracy view.

``figures/tradeoff.pdf``
    The figure the report leads with. Left panel plots accuracy against catalogue coverage;
    right panel plots precision@10 against popularity bias. Together they say the thing the
    whole project is about: the models do not order the same way on the two metric
    families, and precision@10 tracks popularity rather than accuracy.

Both read the committed CSV and fit nothing.
"""

import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import load_config
from src.experiments.registry import MODEL_ORDER
from src.plots.style import apply_style
from src.plots.style import colour_for
from src.plots.style import label_for
from src.plots.style import save

INPUT_NAME = "main_results.csv"
ACCURACY_OUTPUT = "accuracy.pdf"
TRADEOFF_OUTPUT = "tradeoff.pdf"
REPORT_K = 10


def metric_series(frame, metric, k=None):
    selected = frame[frame["metric"] == metric]
    if k is not None:
        selected = selected[selected["k"] == k]
    return selected.set_index("model")["value"].reindex(MODEL_ORDER)


def draw_accuracy(frame):
    rmse = metric_series(frame, "rmse")
    mae = metric_series(frame, "mae")
    positions = np.arange(len(MODEL_ORDER))
    colours = []
    for name in MODEL_ORDER:
        colours.append(colour_for(name))

    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for axis, values, title in [(axes[0], rmse, "Test RMSE"), (axes[1], mae, "Test MAE")]:
        axis.barh(positions, values.to_numpy(), color=colours, height=0.65)
        axis.set_yticks(positions)
        axis.set_yticklabels([label_for(name) for name in MODEL_ORDER])
        axis.invert_yaxis()
        axis.set_xlim(0.9, float(values.max()) * 1.03)
        axis.set_xlabel(title + "  (lower is better)")
        axis.set_title(title)
        for position, value in zip(positions, values.to_numpy()):
            axis.text(value + 0.004, position, format(value, ".3f"),
                      va="center", fontsize=7.5)
    figure.suptitle("Rating accuracy, all models, identical protocol", y=1.02, fontsize=11)
    return figure


def annotate_points(axis, x_values, y_values):
    for name in MODEL_ORDER:
        axis.scatter(x_values[name], y_values[name], color=colour_for(name),
                     s=46, zorder=3)
        axis.annotate(label_for(name), (x_values[name], y_values[name]),
                      textcoords="offset points", xytext=(6, 3), fontsize=7.5)


def draw_tradeoff(frame):
    rmse = metric_series(frame, "rmse")
    coverage = metric_series(frame, "coverage", REPORT_K)
    precision = metric_series(frame, "precision_at_k", REPORT_K)
    popularity = metric_series(frame, "popularity_percentile", REPORT_K)

    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))

    annotate_points(axes[0], coverage, rmse)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("catalogue coverage@10  (higher is wider)")
    axes[0].set_ylabel("test RMSE  (lower is better, axis inverted)")
    axes[0].set_title("Accuracy against reach: top-right is better on both")

    annotate_points(axes[1], popularity, precision)
    correlation = float(np.corrcoef(popularity.to_numpy(), precision.to_numpy())[0, 1])
    axes[1].set_xlabel("mean popularity percentile of recommendations@10")
    axes[1].set_ylabel("precision@10")
    axes[1].set_title(
        "precision@10 tracks popularity  (r = " + format(correlation, ".2f") + ")"
    )

    figure.suptitle(
        "The two metric families do not order the models the same way", y=1.02, fontsize=11
    )
    return figure


def main():
    parser = argparse.ArgumentParser(description="Plot the main comparison.")
    parser.add_argument("--config", default=None, help="path to a config file")
    args = parser.parse_args()

    config = load_config(args.config)
    input_path = config.path("results_dir") / INPUT_NAME
    if not input_path.exists():
        raise FileNotFoundError(
            "missing " + str(input_path) +
            ". Run python -m src.experiments.run_main first."
        )
    frame = pd.read_csv(input_path)

    apply_style()
    figures_dir = config.path("figures_dir")
    save(draw_accuracy(frame), figures_dir / ACCURACY_OUTPUT)
    save(draw_tradeoff(frame), figures_dir / TRADEOFF_OUTPUT)


if __name__ == "__main__":
    main()

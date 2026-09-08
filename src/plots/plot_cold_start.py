"""Figures for the cold-start experiment.

    python -m src.plots.plot_cold_start

Reads ``results/cold_start.csv`` and writes ``figures/cold_start.pdf``: how each model's
accuracy and ranking change as a user's training history shrinks from 20 ratings to 1.

The x axis is log-scaled. The interesting behaviour is between 1 and 5 ratings, where the
collaborative models have almost nothing to work with; on a linear axis that whole region
is squeezed into the left eighth of the plot.
"""

import argparse

import pandas as pd
import matplotlib.pyplot as plt

from src.config import load_config
from src.experiments.registry import MODEL_ORDER
from src.plots.style import apply_style
from src.plots.style import colour_for
from src.plots.style import label_for
from src.plots.style import linestyle_for
from src.plots.style import save

INPUT_NAME = "cold_start.csv"
OUTPUT_NAME = "cold_start.pdf"
REPORT_K = 10

PANELS = [
    ("rmse", None, "Test RMSE", "lower is better"),
    ("precision_at_k", REPORT_K, "precision@10", "higher is better"),
    ("coverage", REPORT_K, "Catalogue coverage@10", "higher is wider"),
]


def series_for(frame, model, metric, k):
    selected = frame[(frame["model"] == model) & (frame["metric"] == metric)]
    if k is not None:
        selected = selected[selected["k"] == k]
    ordered = selected.sort_values("history")
    return ordered["history"].to_numpy(), ordered["value"].to_numpy()


def draw(frame, models):
    figure, axes = plt.subplots(1, len(PANELS), figsize=(12.5, 3.8))
    for axis, (metric, k, title, note) in zip(axes, PANELS):
        for model in models:
            history, values = series_for(frame, model, metric, k)
            if len(history) == 0:
                continue
            axis.plot(
                history,
                values,
                marker="o",
                color=colour_for(model),
                linestyle=linestyle_for(model),
                label=label_for(model),
            )
        axis.set_xscale("log")
        axis.set_xticks(sorted(frame["history"].unique()))
        axis.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        axis.set_xlabel("training ratings kept per user")
        axis.set_title(title + "  (" + note + ")")
    axes[0].set_ylabel("value")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.08))
    figure.suptitle(
        "How each model degrades as a user's history shrinks", y=1.02, fontsize=11
    )
    return figure


def main():
    parser = argparse.ArgumentParser(description="Plot the cold-start experiment.")
    parser.add_argument("--config", default=None, help="path to a config file")
    args = parser.parse_args()

    config = load_config(args.config)
    input_path = config.path("results_dir") / INPUT_NAME
    if not input_path.exists():
        raise FileNotFoundError(
            "missing " + str(input_path) +
            ". Run python -m src.experiments.run_cold_start first."
        )
    frame = pd.read_csv(input_path)

    apply_style()
    models = []
    for name in MODEL_ORDER:
        if name in set(frame["model"]):
            models.append(name)
    figure = draw(frame, models)
    save(figure, config.path("figures_dir") / OUTPUT_NAME)


if __name__ == "__main__":
    main()

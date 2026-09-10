"""Figures for the hyperparameter sweeps.

    python -m src.plots.plot_tuning

Reads ``results/tuning_knn.csv`` and ``results/tuning_mf.csv`` and writes
``figures/tuning.pdf``.

Each panel plots one sweep's configurations as points: validation RMSE against validation
precision@10. If the two criteria agreed, the cloud would slope down-right — better RMSE
with better precision. It slopes the other way in both sweeps, which is the project's
sharpest single result: *within one algorithm*, selecting on accuracy costs ranking.

The configuration each sweep would pick under either criterion is marked, so the distance
between the two markers is the cost of the choice.
"""

import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import load_config
from src.plots.style import apply_style
from src.plots.style import save

OUTPUT_NAME = "tuning.pdf"
REPORT_K = 10
SWEEPS = [
    ("tuning_knn.csv", "Item-kNN, 48 configurations", "#26418f"),
    ("tuning_mf.csv", "Matrix factorisation, 32 configurations", "#8c3025"),
]


def sweep_table(frame):
    rmse = frame[frame["metric"] == "rmse"][["variant", "value"]]
    rmse = rmse.rename(columns={"value": "rmse"}).set_index("variant")
    precision = frame[(frame["metric"] == "precision_at_k") & (frame["k"] == REPORT_K)]
    precision = precision[["variant", "value"]]
    precision = precision.rename(columns={"value": "precision"}).set_index("variant")
    coverage = frame[(frame["metric"] == "coverage") & (frame["k"] == REPORT_K)]
    coverage = coverage[["variant", "value"]]
    coverage = coverage.rename(columns={"value": "coverage"}).set_index("variant")
    return rmse.join(precision).join(coverage).dropna()


def draw_panel(axis, table, title, colour):
    axis.scatter(table["rmse"], table["precision"], color=colour, alpha=0.55, s=32)

    best_rmse = table["rmse"].idxmin()
    best_precision = table["precision"].idxmax()
    axis.scatter(table.loc[best_rmse, "rmse"], table.loc[best_rmse, "precision"],
                 facecolors="none", edgecolors="#111111", s=140, linewidths=1.4,
                 zorder=4, label="picked on RMSE")
    axis.scatter(table.loc[best_precision, "rmse"], table.loc[best_precision, "precision"],
                 marker="s", facecolors="none", edgecolors="#111111", s=140,
                 linewidths=1.4, zorder=4, label="picked on precision@10")

    correlation = float(np.corrcoef(table["rmse"].to_numpy(),
                                    table["precision"].to_numpy())[0, 1])
    axis.set_xlabel("validation RMSE  (lower is better)")
    axis.set_ylabel("validation precision@10  (higher is better)")
    axis.set_title(title + "\nr = " + format(correlation, ".2f")
                   + "; positive means the criteria disagree")
    axis.legend(loc="upper left")


def draw(tables):
    figure, axes = plt.subplots(1, len(tables), figsize=(11.5, 4.4))
    for axis, (table, title, colour) in zip(axes, tables):
        draw_panel(axis, table, title, colour)
    figure.suptitle(
        "Selecting on accuracy costs ranking, within a single algorithm",
        y=1.04, fontsize=11,
    )
    return figure


def main():
    parser = argparse.ArgumentParser(description="Plot the hyperparameter sweeps.")
    parser.add_argument("--config", default=None, help="path to a config file")
    args = parser.parse_args()

    config = load_config(args.config)
    results_dir = config.path("results_dir")
    tables = []
    for name, title, colour in SWEEPS:
        path = results_dir / name
        if not path.exists():
            raise FileNotFoundError(
                "missing " + str(path) +
                ". Run python -m src.experiments.run_tuning first."
            )
        tables.append((sweep_table(pd.read_csv(path)), title, colour))

    apply_style()
    save(draw(tables), config.path("figures_dir") / OUTPUT_NAME)


if __name__ == "__main__":
    main()

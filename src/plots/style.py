"""Shared figure style.

Every plotting script imports from here so figures in the report look like one set rather
than seven. Nothing in this module reads results or fits anything - plotting scripts take
their numbers from the CSVs in ``results/``, which is what keeps a change of axis label
from turning into a refit.

Figures are written as PDF. The report is typeset, so vector output stays sharp at any
size, and a PDF diff at least tells you the figure changed.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

FIGURE_DPI = 150
DEFAULT_SIZE = (7.0, 4.2)

# One colour per model, held constant across every figure so a reader can follow a model
# from one plot to the next. The two ranking-tuned variants share the hue of their
# accuracy-tuned twin and are drawn dashed, because they are the same algorithm.
MODEL_COLOURS = {
    "global_mean": "#9aa0a6",
    "user_mean": "#7f8c8d",
    "item_mean": "#5d6d7e",
    "most_popular": "#b4741f",
    "content": "#2e8b57",
    "item_knn": "#26418f",
    "item_knn_ranking": "#26418f",
    "mf": "#8c3025",
    "mf_ranking": "#8c3025",
    "hybrid": "#6a3d9a",
    "hybrid_frontier": "#6a3d9a",
}

DASHED_MODELS = ["item_knn_ranking", "mf_ranking", "hybrid_frontier"]

MODEL_LABELS = {
    "global_mean": "Global mean",
    "user_mean": "User mean",
    "item_mean": "Item mean",
    "most_popular": "Most popular",
    "content": "Content-based",
    "item_knn": "Item-kNN",
    "item_knn_ranking": "Item-kNN (ranking)",
    "mf": "MF",
    "mf_ranking": "MF (ranking)",
    "hybrid": "Hybrid (density)",
    "hybrid_frontier": "Hybrid (w=0.4)",
}


def apply_style():
    plt.rcParams.update({
        "figure.figsize": DEFAULT_SIZE,
        "figure.dpi": FIGURE_DPI,
        "savefig.dpi": FIGURE_DPI,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "lines.linewidth": 1.6,
        "lines.markersize": 4,
    })


def colour_for(model):
    return MODEL_COLOURS.get(model, "#333333")


def linestyle_for(model):
    if model in DASHED_MODELS:
        return "--"
    return "-"


def label_for(model):
    return MODEL_LABELS.get(model, model)


def save(figure, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)
    print("wrote " + str(path))
    return path

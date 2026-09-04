"""Regenerate the results table in README.md from the committed CSV.

    python -m src.experiments.make_readme_table

Rewrites whatever sits between the RESULTS_TABLE_START and RESULTS_TABLE_END markers. The
CSV is the source of truth; the README is a view of it. Hand-editing that block means the
report can quote a number no run ever produced, which is the failure this script exists to
prevent.
"""

import argparse

import pandas as pd

from src.config import load_config
from src.experiments.registry import MODEL_ORDER

START_MARKER = "<!-- RESULTS_TABLE_START -->"
END_MARKER = "<!-- RESULTS_TABLE_END -->"
RESULTS_NAME = "main_results.csv"

COLUMNS = [
    ("rmse", None, "RMSE"),
    ("mae", None, "MAE"),
    ("precision_at_k", 10, "P@10"),
    ("recall_at_k", 10, "R@10"),
    ("coverage", 10, "Cov@10"),
    ("popularity_percentile", 10, "PopPct@10"),
    ("gini", 10, "Gini@10"),
]

MODEL_LABELS = {
    "global_mean": "Global mean",
    "user_mean": "User mean",
    "item_mean": "Item mean",
    "most_popular": "Most popular",
    "content": "Content-based",
    "item_knn": "Item-kNN",
    "mf": "Matrix factorisation",
}


def build_table(frame):
    columns = {}
    for metric, k, label in COLUMNS:
        selected = frame[frame["metric"] == metric]
        if k is not None:
            selected = selected[selected["k"] == k]
        columns[label] = selected.set_index("model")["value"]
    table = pd.DataFrame(columns).reindex(MODEL_ORDER)

    lines = []
    lines.append("| Model | " + " | ".join(label for _, _, label in COLUMNS) + " |")
    lines.append("|---" * (len(COLUMNS) + 1) + "|")
    for model in MODEL_ORDER:
        cells = [MODEL_LABELS.get(model, model)]
        for _, _, label in COLUMNS:
            value = table.loc[model, label]
            cells.append(format(float(value), ".4f"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_block(frame, seed):
    run_ids = sorted(set(frame["run_id"].astype(str)))
    n_users = int(frame[frame["metric"] == "precision_at_k"]["n_users"].max())
    n_ratings = int(frame[frame["metric"] == "rmse"]["n_ratings"].max())

    lines = []
    lines.append(build_table(frame))
    lines.append("")
    lines.append("Test split: " + str(n_ratings) + " held-out ratings; ranking metrics "
                 "averaged over the " + str(n_users) + " users with at least one relevant "
                 "held-out item. Seed " + str(seed) + ", run " + run_ids[0] + ".")
    lines.append("")
    lines.append("Lower is better for RMSE, MAE and the two popularity columns. Higher is "
                 "better for precision, recall and coverage.")
    return "\n".join(lines)


def rewrite_readme(readme_path, block):
    text = readme_path.read_text(encoding="utf-8")
    start = text.find(START_MARKER)
    end = text.find(END_MARKER)
    if start < 0 or end < 0:
        raise ValueError(
            "could not find the results markers in " + str(readme_path) +
            "; expected " + START_MARKER + " and " + END_MARKER
        )
    if end < start:
        raise ValueError("results markers are in the wrong order in " + str(readme_path))
    updated = (
        text[: start + len(START_MARKER)]
        + "\n"
        + block
        + "\n"
        + text[end:]
    )
    readme_path.write_text(updated, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Regenerate the README results table.")
    parser.add_argument("--config", default=None, help="path to a config file")
    args = parser.parse_args()

    config = load_config(args.config)
    results_path = config.path("results_dir") / RESULTS_NAME
    if not results_path.exists():
        raise FileNotFoundError(
            "missing " + str(results_path) +
            ". Run python -m src.experiments.run_main first."
        )
    frame = pd.read_csv(results_path)
    readme_path = config.project_root / "README.md"
    rewrite_readme(readme_path, build_block(frame, config.seed))
    print("updated the results table in " + str(readme_path))


if __name__ == "__main__":
    main()

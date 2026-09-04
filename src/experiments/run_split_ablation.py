"""Quantify what a random split would have bought us.

    python -m src.experiments.run_split_ablation

Runs the identical pipeline twice, changing one thing: how the data is divided. Once with
the per-user temporal split the project uses, and once with a random row split of the same
proportions. Everything else - models, hyperparameters, metrics, seed - is held fixed.

The random split leaks. A user's future ratings land in training, so the model gets to see
someone rate a sequel before predicting they liked the original, and every metric improves
for a reason that has nothing to do with the model being better. The gap between the two
columns is the size of that illusion on this dataset.

This is worth its own experiment rather than a paragraph, for two reasons. It justifies the
protocol with a number instead of an appeal to the literature, and the gap is usually
larger for the stronger models, which means a random split does not merely inflate results
uniformly - it distorts the ranking between the models being compared.
"""

import argparse

import numpy as np
import pandas as pd

from src.config import load_config
from src.data.splitting import check_fractions
from src.data.splitting import temporal_split
from src.evaluation.harness import evaluate_model
from src.evaluation.harness import make_run_id
from src.evaluation.harness import results_frame
from src.evaluation.harness import write_results
from src.experiments.registry import MODEL_ORDER
from src.experiments.registry import build_model
from src.experiments.run_main import load_items

OUTPUT_NAME = "split_ablation.csv"
RATINGS_NAME = "ratings.parquet"
COMPARISON_K = 10


def random_split(ratings, config):
    """Split rows at random, ignoring time entirely.

    Deliberately the naive thing: shuffle the rows and cut. This is what a project that
    had not thought about the problem would do, which is exactly what the ablation needs
    to measure.
    """
    settings = config.section("split")
    train_frac = float(settings["train_frac"])
    val_frac = float(settings["val_frac"])
    test_frac = float(settings["test_frac"])
    check_fractions(train_frac, val_frac, test_frac)

    rng = config.fresh_rng()
    order = rng.permutation(len(ratings))
    shuffled = ratings.iloc[order].reset_index(drop=True)
    train_end = int(round(len(shuffled) * train_frac))
    val_end = train_end + int(round(len(shuffled) * val_frac))
    train = shuffled.iloc[:train_end].reset_index(drop=True)
    validation = shuffled.iloc[train_end:val_end].reset_index(drop=True)
    test = shuffled.iloc[val_end:].reset_index(drop=True)
    return train, validation, test


def evaluate_under_split(config, items, split_name, train, validation, test, run_id):
    rows = []
    for name in MODEL_ORDER:
        model = build_model(name, config, items, validation=validation)
        model.fit(train)
        rows.extend(
            evaluate_model(model, train, test, config, split_name, run_id=run_id)
        )
        print("  " + split_name.ljust(9) + " " + name)
    return rows


def leakage_report(frame, metric, k=None):
    """Temporal against random for one metric, with the inflation attributed per model."""
    selected = frame[frame["metric"] == metric]
    if k is not None:
        selected = selected[selected["k"] == k]
    pivoted = selected.pivot_table(index="model", columns="split", values="value")
    pivoted = pivoted.reindex(MODEL_ORDER)
    if "temporal" not in pivoted.columns or "random" not in pivoted.columns:
        return pivoted
    pivoted["difference"] = pivoted["random"] - pivoted["temporal"]
    denominator = pivoted["temporal"].replace(0.0, np.nan)
    pivoted["percent"] = 100.0 * pivoted["difference"] / denominator
    return pivoted


def print_report(frame):
    print("")
    print("RMSE (lower is better; random should look deceptively good)")
    print("-" * 72)
    print(leakage_report(frame, "rmse").round(4).to_string())
    print("")
    print("precision@" + str(COMPARISON_K))
    print("-" * 72)
    print(leakage_report(frame, "precision_at_k", COMPARISON_K).round(4).to_string())
    print("")


def main():
    parser = argparse.ArgumentParser(
        description="Compare the temporal split against a random row split."
    )
    parser.add_argument("--config", default=None, help="path to a config file")
    args = parser.parse_args()

    config = load_config(args.config)
    config.ensure_output_dirs()

    ratings_path = config.path("processed_dir") / RATINGS_NAME
    if not ratings_path.exists():
        raise FileNotFoundError(
            "missing " + str(ratings_path) + ". Run python scripts/preprocess.py first."
        )
    ratings = pd.read_parquet(ratings_path)
    items = load_items(config)
    run_id = make_run_id()

    rows = []
    temporal = temporal_split(ratings, config)
    rows.extend(
        evaluate_under_split(config, items, "temporal", temporal[0], temporal[1],
                             temporal[2], run_id)
    )
    random = random_split(ratings, config)
    rows.extend(
        evaluate_under_split(config, items, "random", random[0], random[1],
                             random[2], run_id)
    )

    frame = results_frame(rows)
    output_path = config.path("results_dir") / OUTPUT_NAME
    write_results(rows, output_path)
    print("wrote " + str(output_path) + " (" + str(len(frame)) + " rows)")
    print_report(frame)


if __name__ == "__main__":
    main()

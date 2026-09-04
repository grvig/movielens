"""Run every model through the protocol and write the main results table.

    python -m src.experiments.run_main

Fits each model on train, evaluates on test, and writes long-format rows to
``results/main_results.csv``. Validation is passed to the models that use it - content for
its cosine calibration, matrix factorisation for early stopping - and is never touched by
the evaluation itself.

This script does not plot. Figures are made by ``src/plots/`` reading the CSV this writes,
so changing an axis label never means refitting a model.
"""

import argparse
import time

import pandas as pd

from src.config import load_config
from src.data.splitting import load_splits
from src.evaluation.harness import evaluate_model
from src.evaluation.harness import make_run_id
from src.evaluation.harness import results_frame
from src.evaluation.harness import write_results
from src.experiments.registry import MODEL_ORDER
from src.experiments.registry import build_model

OUTPUT_NAME = "main_results.csv"
ITEMS_NAME = "items.parquet"

HEADLINE_METRICS = ["rmse", "mae"]
HEADLINE_K = 10


def load_items(config):
    path = config.path("processed_dir") / ITEMS_NAME
    if not path.exists():
        raise FileNotFoundError(
            "missing " + str(path) + ". Run python scripts/preprocess.py first."
        )
    return pd.read_parquet(path)


def run(config, cache):
    train, validation, test = load_splits(config.path("processed_dir"))
    items = load_items(config)
    run_id = make_run_id()

    print("train " + str(len(train)) + " ratings, "
          + "validation " + str(len(validation)) + ", test " + str(len(test)))
    print("")

    rows = []
    for name in MODEL_ORDER:
        started = time.time()
        model = build_model(name, config, items, validation=validation, cache=cache)
        model.fit(train)
        fit_seconds = time.time() - started

        started = time.time()
        rows.extend(evaluate_model(model, train, test, config, "test", run_id=run_id))
        evaluate_seconds = time.time() - started

        print(name.ljust(14)
              + " fit " + format(fit_seconds, ".1f") + "s"
              + "  eval " + format(evaluate_seconds, ".1f") + "s")

    return rows


def print_summary(frame):
    """The comparison as the report wants to read it: accuracy beside catalogue behaviour."""
    columns = {}
    for metric in HEADLINE_METRICS:
        selected = frame[frame["metric"] == metric].set_index("model")["value"]
        columns[metric] = selected
    for metric in ["precision_at_k", "recall_at_k", "coverage", "popularity_percentile"]:
        selected = frame[(frame["metric"] == metric) & (frame["k"] == HEADLINE_K)]
        columns[metric + "@" + str(HEADLINE_K)] = selected.set_index("model")["value"]

    summary = pd.DataFrame(columns)
    summary = summary.reindex(MODEL_ORDER)
    print("")
    print("results at k=" + str(HEADLINE_K))
    print("-" * 78)
    print(summary.round(4).to_string())
    print("")


def main():
    parser = argparse.ArgumentParser(description="Run all models through the protocol.")
    parser.add_argument("--config", default=None, help="path to a config file")
    parser.add_argument("--cache", action="store_true",
                        help="reuse cached matrix factorisation fits")
    args = parser.parse_args()

    config = load_config(args.config)
    config.ensure_output_dirs()

    rows = run(config, args.cache)
    frame = results_frame(rows)
    output_path = config.path("results_dir") / OUTPUT_NAME
    write_results(rows, output_path)
    print("wrote " + str(output_path) + " (" + str(len(frame)) + " rows)")
    print_summary(frame)


if __name__ == "__main__":
    main()

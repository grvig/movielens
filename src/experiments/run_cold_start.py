"""How each model degrades as a user's history shrinks.

    python -m src.experiments.run_cold_start

Picks a sample of users, truncates *only their* training histories to 1, 3, 5, 10 and 20
ratings, refits every model at each level, and evaluates on those users alone. Writes
``results/cold_start.csv`` with a ``history`` column.

This is the experiment that tests the project's remaining untested prediction: that
content-based scoring holds up better than collaborative filtering when a user has almost
no history, because it can score an item from its features without needing co-raters.

The design decision that matters is that only the sampled users are truncated. Truncating
everybody degrades the item similarities and latent factors as well, and the figure would
then show how a model copes with a small dataset rather than how it copes with a new user.
The other users stay intact so the model's view of the catalogue is normal.

Runtime note: this refits every model once per history level, so it is the most expensive
script here. Matrix factorisation fits dominate; --cache reuses them across reruns.
"""

import argparse

import pandas as pd

from src.config import load_config
from src.data.splitting import history_lengths
from src.data.splitting import load_splits
from src.data.splitting import select_cold_users
from src.data.splitting import truncate_user_histories
from src.evaluation.harness import evaluate_model
from src.evaluation.harness import make_run_id
from src.evaluation.harness import results_frame
from src.evaluation.harness import write_results
from src.experiments.registry import MODEL_ORDER
from src.experiments.registry import build_model
from src.experiments.run_main import load_items

OUTPUT_NAME = "cold_start.csv"
HISTORY_LEVELS = [1, 3, 5, 10, 20]
DEFAULT_COLD_USERS = 200
REPORT_K = 10


def evaluate_at_level(config, items, train, validation, test, cold_users, level, run_id,
                      cache):
    """Truncate, refit everything, evaluate on the truncated users only."""
    truncated = truncate_user_histories(train, cold_users, level)
    cold_test = test[test["user_id"].isin(cold_users)]

    rows = []
    for name in MODEL_ORDER:
        model = build_model(name, config, items, validation=validation, cache=cache)
        model.fit(truncated)
        rows.extend(
            evaluate_model(model, truncated, cold_test, config, "test",
                           variant="history=" + str(level), run_id=run_id)
        )
    lengths = history_lengths(truncated, cold_users)
    mean_history = sum(lengths.values()) / max(1, len(lengths))
    print("  history " + str(level).rjust(3)
          + "  mean kept " + format(mean_history, ".1f")
          + "  test ratings " + str(len(cold_test)))
    return rows


def add_history_column(rows, level):
    for row in rows:
        row["history"] = level
    return rows


def print_report(frame, metric, k=None):
    selected = frame[frame["metric"] == metric]
    if k is not None:
        selected = selected[selected["k"] == k]
    table = selected.pivot_table(index="model", columns="history", values="value")
    table = table.reindex(MODEL_ORDER)
    label = metric
    if k is not None:
        label = metric + "@" + str(k)
    print("")
    print(label + " by training history length")
    print("-" * 74)
    print(table.round(4).to_string())
    print("")


def main():
    parser = argparse.ArgumentParser(
        description="Measure how models degrade as user history shrinks."
    )
    parser.add_argument("--config", default=None, help="path to a config file")
    parser.add_argument("--users", type=int, default=DEFAULT_COLD_USERS)
    parser.add_argument("--cache", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    config.ensure_output_dirs()
    train, validation, test = load_splits(config.path("processed_dir"))
    items = load_items(config)
    run_id = make_run_id()

    cold_users = select_cold_users(train, args.users, config.fresh_rng(), min_history=20)
    print("truncating " + str(len(cold_users)) + " of "
          + str(train["user_id"].nunique()) + " users")

    rows = []
    for level in HISTORY_LEVELS:
        level_rows = evaluate_at_level(
            config, items, train, validation, test, cold_users, level, run_id, args.cache
        )
        rows.extend(add_history_column(level_rows, level))

    frame = pd.DataFrame(rows)
    output_path = config.path("results_dir") / OUTPUT_NAME
    write_results_with_history(frame, output_path)
    print("")
    print("wrote " + str(output_path) + " (" + str(len(frame)) + " rows)")

    print_report(frame, "rmse")
    print_report(frame, "precision_at_k", REPORT_K)
    print_report(frame, "coverage", REPORT_K)


def write_results_with_history(frame, path):
    """Same schema as every other results file, plus the history column."""
    ordered = results_frame(frame.to_dict("records"))
    ordered["history"] = frame["history"].to_numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered.to_csv(path, index=False)
    return path


if __name__ == "__main__":
    main()

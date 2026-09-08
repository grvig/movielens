"""Tests for the cold-start experiment runner."""

import numpy as np
import pandas as pd
import pytest

from src.data.splitting import load_splits
from src.data.splitting import temporal_split
from src.data.splitting import write_splits
from src.experiments.registry import MODEL_ORDER
from src.experiments.run_cold_start import HISTORY_LEVELS
from src.experiments.run_cold_start import add_history_column
from src.experiments.run_cold_start import evaluate_at_level
from src.experiments.run_cold_start import write_results_with_history
from src.experiments.run_main import load_items


@pytest.fixture
def prepared_config(config, tmp_path, synthetic_ratings, synthetic_items):
    processed = tmp_path / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    config.values["paths"]["processed_dir"] = str(processed)
    config.values["paths"]["results_dir"] = str(tmp_path / "results")
    config.values["paths"]["fits_dir"] = str(tmp_path / "fits")
    config.values["models"]["mf"]["n_epochs"] = 3
    config.values["models"]["mf_ranking"]["n_epochs"] = 3
    train, validation, test = temporal_split(synthetic_ratings, config)
    write_splits(train, validation, test, processed)
    synthetic_items.to_parquet(processed / "items.parquet", index=False)
    return config


def test_history_levels_are_increasing():
    assert HISTORY_LEVELS == sorted(HISTORY_LEVELS)
    assert HISTORY_LEVELS[0] == 1


def test_every_model_is_evaluated_at_a_level(prepared_config):
    train, validation, test = load_splits(prepared_config.path("processed_dir"))
    items = load_items(prepared_config)
    cold = np.array([1, 2, 3])
    rows = evaluate_at_level(
        prepared_config, items, train, validation, test, cold, 3, "run", False
    )
    frame = pd.DataFrame(rows)
    assert set(frame["model"]) == set(MODEL_ORDER)


def test_the_variant_records_the_history_level(prepared_config):
    train, validation, test = load_splits(prepared_config.path("processed_dir"))
    items = load_items(prepared_config)
    rows = evaluate_at_level(
        prepared_config, items, train, validation, test, np.array([1, 2]), 5, "run", False
    )
    assert set(pd.DataFrame(rows)["variant"]) == {"history=5"}


def test_evaluation_is_restricted_to_the_truncated_users(prepared_config):
    """Scoring untouched users too would dilute the effect being measured."""
    train, validation, test = load_splits(prepared_config.path("processed_dir"))
    items = load_items(prepared_config)
    cold = np.array([1, 2])
    rows = evaluate_at_level(
        prepared_config, items, train, validation, test, cold, 3, "run", False
    )
    frame = pd.DataFrame(rows)
    cold_test = test[test["user_id"].isin(cold)]
    reported = frame[frame["metric"] == "rmse"]["n_ratings"].max()
    assert reported == len(cold_test)


def test_a_shorter_history_changes_the_predictions(prepared_config):
    train, validation, test = load_splits(prepared_config.path("processed_dir"))
    items = load_items(prepared_config)
    cold = np.array([1, 2, 3, 4])
    long_rows = evaluate_at_level(
        prepared_config, items, train, validation, test, cold, 20, "run", False
    )
    short_rows = evaluate_at_level(
        prepared_config, items, train, validation, test, cold, 1, "run", False
    )
    long_frame = pd.DataFrame(long_rows)
    short_frame = pd.DataFrame(short_rows)
    long_rmse = long_frame[long_frame["model"] == "item_knn"]["value"].iloc[0]
    short_rmse = short_frame[short_frame["model"] == "item_knn"]["value"].iloc[0]
    assert long_rmse != short_rmse


def test_add_history_column_tags_every_row():
    rows = [{"model": "mf"}, {"model": "content"}]
    tagged = add_history_column(rows, 5)
    for row in tagged:
        assert row["history"] == 5


def test_written_file_keeps_the_standard_schema_plus_history(prepared_config, tmp_path):
    train, validation, test = load_splits(prepared_config.path("processed_dir"))
    items = load_items(prepared_config)
    rows = add_history_column(
        evaluate_at_level(
            prepared_config, items, train, validation, test, np.array([1]), 3, "run", False
        ),
        3,
    )
    path = tmp_path / "cold_start.csv"
    write_results_with_history(pd.DataFrame(rows), path)
    written = pd.read_csv(path)
    assert "history" in written.columns
    assert "model" in written.columns
    assert "value" in written.columns
    assert set(written["history"]) == {3}


def test_history_column_stays_aligned_with_its_rows(prepared_config, tmp_path):
    """The column is reattached after a reindex, so misalignment is the risk."""
    train, validation, test = load_splits(prepared_config.path("processed_dir"))
    items = load_items(prepared_config)
    rows = []
    for level in [1, 5]:
        rows.extend(add_history_column(
            evaluate_at_level(
                prepared_config, items, train, validation, test, np.array([1]), level,
                "run", False
            ),
            level,
        ))
    path = tmp_path / "cold_start.csv"
    write_results_with_history(pd.DataFrame(rows), path)
    written = pd.read_csv(path)
    source = pd.DataFrame(rows)
    assert written["history"].tolist() == source["history"].tolist()
    assert written["model"].tolist() == source["model"].tolist()

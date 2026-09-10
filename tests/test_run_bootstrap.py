"""Tests for the bootstrap experiment runner."""

import numpy as np
import pytest

from src.data.splitting import load_splits
from src.data.splitting import temporal_split
from src.data.splitting import write_splits
from src.evaluation.ranking import relevant_items_by_user
from src.experiments.registry import build_model
from src.experiments.run_bootstrap import build_precision_table
from src.experiments.run_bootstrap import collect
from src.experiments.run_bootstrap import difference_rows
from src.experiments.run_bootstrap import per_user_precision
from src.experiments.run_bootstrap import summarise
from src.experiments.run_main import load_items
from tests.conftest import refresh_hybrid_fingerprint


@pytest.fixture
def prepared_config(config, tmp_path, synthetic_ratings, synthetic_items):
    processed = tmp_path / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    config.values["paths"]["processed_dir"] = str(processed)
    config.values["paths"]["results_dir"] = str(tmp_path / "results")
    config.values["paths"]["fits_dir"] = str(tmp_path / "fits")
    config.values["models"]["mf"]["n_epochs"] = 3
    refresh_hybrid_fingerprint(config)
    train, validation, test = temporal_split(synthetic_ratings, config)
    write_splits(train, validation, test, processed)
    synthetic_items.to_parquet(processed / "items.parquet", index=False)
    return config


def test_collect_returns_one_table_per_model(prepared_config):
    train, validation, test = load_splits(prepared_config.path("processed_dir"))
    accuracy, models, relevant = collect(prepared_config, train, validation, test, False)
    assert set(accuracy.keys()) == set(models.keys())
    for table in accuracy.values():
        assert "sum_squared_error" in table.columns


def test_per_user_precision_is_nan_for_users_with_nothing_relevant(prepared_config):
    train, validation, test = load_splits(prepared_config.path("processed_dir"))
    items = load_items(prepared_config)
    model = build_model("most_popular", prepared_config, items).fit(train)
    relevant = relevant_items_by_user(test, 4.0)

    users = sorted(train["user_id"].unique().tolist())
    values = per_user_precision(model, relevant, users, 5)
    for position, user_id in enumerate(users):
        if user_id in relevant and len(relevant[user_id]) > 0:
            assert not np.isnan(values[position])
        else:
            assert np.isnan(values[position])


def test_precision_values_are_bounded(prepared_config):
    train, validation, test = load_splits(prepared_config.path("processed_dir"))
    _, models, relevant = collect(prepared_config, train, validation, test, False)
    users = sorted(train["user_id"].unique().tolist())
    vectors = build_precision_table(models, relevant, users, 5)
    for values in vectors.values():
        usable = values[~np.isnan(values)]
        assert np.all(usable >= 0.0)
        assert np.all(usable <= 1.0)


def test_difference_rows_cover_every_pair():
    rng = np.random.default_rng(3)
    samples = {"a": rng.normal(1.0, 0.1, 200), "b": rng.normal(1.2, 0.1, 200)}
    rows = difference_rows("rmse", samples, 0.95)
    assert len(rows) == 1
    assert rows[0]["kind"] == "difference"
    assert rows[0]["metric"] == "rmse"


def test_summarise_produces_a_point_row():
    samples = np.random.default_rng(2).normal(1.0, 0.05, 500)
    row = summarise("mf", "rmse", samples, 0.95)
    assert row["kind"] == "point"
    assert row["model_b"] == ""
    assert row["lower"] < row["difference"] < row["upper"]


def test_the_bootstrap_is_reproducible(prepared_config):
    """Two runs from the same seed must give identical intervals."""
    from src.evaluation.bootstrap import accuracy_samples
    from src.evaluation.bootstrap import bootstrap_user_indices

    train, validation, test = load_splits(prepared_config.path("processed_dir"))
    accuracy, _, _ = collect(prepared_config, train, validation, test, False)
    table = accuracy["item_mean"]

    first = accuracy_samples(
        table, bootstrap_user_indices(len(table), 50, prepared_config.fresh_rng()), "rmse"
    )
    second = accuracy_samples(
        table, bootstrap_user_indices(len(table), 50, prepared_config.fresh_rng()), "rmse"
    )
    assert np.array_equal(first, second)

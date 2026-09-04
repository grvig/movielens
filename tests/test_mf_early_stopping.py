"""Tests for matrix factorisation early stopping and fit caching."""

import numpy as np
import pandas as pd
import pytest

from src.models.mf import MatrixFactorization


@pytest.fixture
def noisy_split():
    """Training data an over-parameterised model will memorise, plus held-out ratings.

    The validation ratings deliberately do not follow the training noise, so validation
    error has to turn upwards while training error keeps falling.
    """
    rng = np.random.default_rng(5)
    train_rows = []
    val_rows = []
    for user_id in range(1, 21):
        for item_id in range(1, 11):
            rating = float(rng.integers(1, 6))
            train_rows.append((user_id, item_id, rating, 100 + item_id))
        for item_id in range(1, 4):
            val_rows.append((user_id, item_id, 3.0, 200 + item_id))
    columns = ["user_id", "item_id", "rating", "timestamp"]
    return pd.DataFrame(train_rows, columns=columns), pd.DataFrame(val_rows, columns=columns)


@pytest.fixture
def cache_config(config, tmp_path):
    config.values["paths"]["fits_dir"] = str(tmp_path / "fits")
    return config


def test_without_validation_it_runs_every_epoch(config, noisy_split):
    train, _ = noisy_split
    model = MatrixFactorization(config, n_epochs=8).fit(train)
    assert len(model.train_history) == 8
    assert model.val_history == []
    assert model.best_epoch is None


def test_validation_history_is_recorded(config, noisy_split):
    train, validation = noisy_split
    model = MatrixFactorization(config, n_epochs=8, validation=validation).fit(train)
    assert len(model.val_history) > 0
    assert len(model.val_history) == len(model.train_history)


def test_it_stops_early_when_validation_stops_improving(config, noisy_split):
    train, validation = noisy_split
    model = MatrixFactorization(
        config, n_epochs=200, patience=3, validation=validation
    ).fit(train)
    assert len(model.train_history) < 200


def test_it_keeps_the_best_epoch_not_the_last(config, noisy_split):
    """Detecting overfitting and then returning the overfitted model would be pointless."""
    train, validation = noisy_split
    model = MatrixFactorization(
        config, n_epochs=200, patience=3, validation=validation
    ).fit(train)
    assert model.best_epoch is not None
    assert model.val_history[model.best_epoch] == pytest.approx(min(model.val_history))
    assert model.best_epoch <= len(model.val_history) - 1


def test_restored_parameters_match_the_best_validation_error(config, noisy_split):
    train, validation = noisy_split
    model = MatrixFactorization(
        config, n_epochs=200, patience=3, validation=validation
    ).fit(train)
    arrays = model.validation_arrays()
    assert model.training_rmse(*arrays) == pytest.approx(min(model.val_history), abs=1e-9)


def test_validation_rows_for_unknown_items_are_dropped(config, noisy_split):
    train, _ = noisy_split
    validation = pd.DataFrame(
        [(1, 9999, 4.0, 300), (1, 1, 4.0, 301)],
        columns=["user_id", "item_id", "rating", "timestamp"],
    )
    model = MatrixFactorization(config, n_epochs=3, validation=validation).fit(train)
    users, items, values = model.validation_arrays()
    assert len(values) == 1


def test_validation_with_nothing_usable_disables_early_stopping(config, noisy_split):
    train, _ = noisy_split
    validation = pd.DataFrame(
        [(999, 9999, 4.0, 300)], columns=["user_id", "item_id", "rating", "timestamp"]
    )
    model = MatrixFactorization(config, n_epochs=4, validation=validation).fit(train)
    assert model.validation_arrays() is None
    assert len(model.train_history) == 4


def test_cache_round_trips_the_parameters(cache_config, noisy_split):
    train, _ = noisy_split
    first = MatrixFactorization(cache_config, n_epochs=5, cache=True).fit(train)
    assert first.loaded_from_cache is False

    second = MatrixFactorization(cache_config, n_epochs=5, cache=True).fit(train)
    assert second.loaded_from_cache is True
    assert np.array_equal(first.user_factors, second.user_factors)
    assert np.array_equal(first.item_bias, second.item_bias)
    assert first.train_history == pytest.approx(second.train_history)


def test_cached_and_fresh_models_predict_identically(cache_config, noisy_split):
    train, _ = noisy_split
    first = MatrixFactorization(cache_config, n_epochs=5, cache=True).fit(train)
    second = MatrixFactorization(cache_config, n_epochs=5, cache=True).fit(train)
    candidates = np.array([1, 2, 3, 4, 5])
    assert np.allclose(first.predict(1, candidates), second.predict(1, candidates))


def test_different_hyperparameters_do_not_share_a_cache_entry(cache_config, noisy_split):
    train, _ = noisy_split
    MatrixFactorization(cache_config, n_epochs=5, n_factors=4, cache=True).fit(train)
    other = MatrixFactorization(cache_config, n_epochs=5, n_factors=8, cache=True).fit(train)
    assert other.loaded_from_cache is False


def test_different_training_data_does_not_share_a_cache_entry(cache_config, noisy_split):
    """A truncated history must not resolve to the full-history fit.

    This is the cold-start experiment's failure mode: same model, same hyperparameters,
    deliberately less data, and a cache keyed on shape alone would return the wrong model.
    """
    train, _ = noisy_split
    MatrixFactorization(cache_config, n_epochs=5, cache=True).fit(train)
    truncated = train[train["item_id"] <= 5]
    other = MatrixFactorization(cache_config, n_epochs=5, cache=True).fit(truncated)
    assert other.loaded_from_cache is False


def test_caching_is_off_by_default(config, noisy_split):
    train, _ = noisy_split
    model = MatrixFactorization(config, n_epochs=3).fit(train)
    assert model.cache is False
    assert model.loaded_from_cache is False

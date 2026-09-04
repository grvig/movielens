"""Tests for the biased matrix factorisation model."""

import numpy as np
import pandas as pd
import pytest

from src.models.baselines import GlobalMean
from src.models.mf import MatrixFactorization


@pytest.fixture
def structured_ratings():
    """Two user groups with opposite tastes over two item groups.

    A rank-1 factorisation should be able to recover this almost exactly, so it is a
    dataset where a working implementation must clearly beat the global mean.
    """
    rows = []
    for user_id in range(1, 7):
        if user_id <= 3:
            liked = [1, 2, 3]
            disliked = [4, 5, 6]
        else:
            liked = [4, 5, 6]
            disliked = [1, 2, 3]
        timestamp = 100
        for item_id in liked:
            rows.append((user_id, item_id, 5.0, timestamp))
            timestamp = timestamp + 1
        for item_id in disliked:
            rows.append((user_id, item_id, 1.0, timestamp))
            timestamp = timestamp + 1
    return pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "timestamp"])


def training_rmse(model, ratings):
    total = 0.0
    count = 0
    for user_id, group in ratings.groupby("user_id"):
        items = group["item_id"].to_numpy(dtype=np.int64)
        predicted = model.predict(user_id, items)
        actual = group["rating"].to_numpy(dtype=np.float64)
        total = total + float(np.sum((actual - predicted) ** 2))
        count = count + len(actual)
    return float(np.sqrt(total / count))


def test_predictions_are_finite_and_in_range(config, structured_ratings):
    model = MatrixFactorization(config).fit(structured_ratings)
    predictions = model.predict(1, np.array([1, 2, 3, 4, 5, 6]))
    assert np.all(np.isfinite(predictions))
    assert np.all(predictions >= model.rating_min)
    assert np.all(predictions <= model.rating_max)


def test_it_learns_the_structure_better_than_the_global_mean(config, structured_ratings):
    model = MatrixFactorization(config).fit(structured_ratings)
    baseline = GlobalMean(config).fit(structured_ratings)
    assert training_rmse(model, structured_ratings) < training_rmse(
        baseline, structured_ratings
    )


def test_it_separates_the_two_taste_groups(config, structured_ratings):
    model = MatrixFactorization(config).fit(structured_ratings)
    group_one = model.predict(1, np.array([1, 4]))
    group_two = model.predict(4, np.array([1, 4]))
    assert group_one[0] > group_one[1]
    assert group_two[1] > group_two[0]


def test_training_error_falls_over_epochs(config, structured_ratings):
    model = MatrixFactorization(config).fit(structured_ratings)
    assert len(model.train_history) == model.n_epochs
    assert model.train_history[-1] < model.train_history[0]


def test_fitting_twice_is_bit_identical(config, structured_ratings):
    first = MatrixFactorization(config).fit(structured_ratings)
    second = MatrixFactorization(config).fit(structured_ratings)
    assert np.array_equal(first.user_factors, second.user_factors)
    assert np.array_equal(first.item_factors, second.item_factors)
    assert first.train_history == second.train_history


def test_item_bias_reflects_how_well_an_item_is_liked(config):
    rows = []
    for user_id in range(1, 9):
        rows.append((user_id, 1, 5.0, 100))
        rows.append((user_id, 2, 1.0, 101))
    ratings = pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "timestamp"])
    model = MatrixFactorization(config).fit(ratings)
    assert model.item_bias[0] > model.item_bias[1]


def test_unknown_user_uses_the_item_bias_only(config, structured_ratings):
    model = MatrixFactorization(config).fit(structured_ratings)
    predictions = model.predict(999, np.array([1, 4]))
    expected = model.clip(model.global_mean + model.item_bias[[0, 3]])
    assert np.allclose(predictions, expected)


def test_unknown_item_falls_back_to_the_global_mean(config, structured_ratings):
    model = MatrixFactorization(config).fit(structured_ratings)
    assert model.predict(1, np.array([9999]))[0] == pytest.approx(model.global_mean)


def test_all_unknown_items_are_handled(config, structured_ratings):
    model = MatrixFactorization(config).fit(structured_ratings)
    predictions = model.predict(1, np.array([9998, 9999]))
    assert np.allclose(predictions, model.global_mean)


def test_stronger_regularisation_shrinks_the_factors(config, structured_ratings):
    light = MatrixFactorization(config, regularisation=0.001).fit(structured_ratings)
    heavy = MatrixFactorization(config, regularisation=5.0).fit(structured_ratings)
    assert np.linalg.norm(heavy.user_factors) < np.linalg.norm(light.user_factors)


def test_hyperparameters_override_the_config(config):
    model = MatrixFactorization(config, n_factors=4, learning_rate=0.02,
                                regularisation=0.01, n_epochs=3)
    assert model.n_factors == 4
    assert model.learning_rate == pytest.approx(0.02)
    assert model.regularisation == pytest.approx(0.01)
    assert model.n_epochs == 3


def test_recommendations_exclude_training_items(config, structured_ratings):
    model = MatrixFactorization(config).fit(structured_ratings)
    recommendations = model.recommend(1, 3)
    assert set(recommendations.tolist()).isdisjoint({1, 2, 3, 4, 5, 6})


def test_mf_runs_on_the_shared_fixture(config, synthetic_ratings):
    model = MatrixFactorization(config, n_epochs=5).fit(synthetic_ratings)
    predictions = model.predict(1, np.array([1, 2, 3]))
    assert np.all(np.isfinite(predictions))
    assert len(model.recommend(1, 5)) == 5

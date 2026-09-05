"""Tests for the paired user-level bootstrap.

These check the statistical properties the report will lean on: that intervals cover the
truth, that pairing detects differences that separate intervals would miss, and that
heavy raters are weighted correctly.
"""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.bootstrap import accuracy_samples
from src.evaluation.bootstrap import align_per_user
from src.evaluation.bootstrap import bootstrap_user_indices
from src.evaluation.bootstrap import compare_models
from src.evaluation.bootstrap import difference_interval
from src.evaluation.bootstrap import percentile_interval
from src.evaluation.bootstrap import ranking_samples
from src.evaluation.metrics import aggregate_user_accuracy
from src.evaluation.metrics import per_user_accuracy


def make_per_user(n_users, squared_error, n_ratings):
    return pd.DataFrame(
        {
            "user_id": np.arange(1, n_users + 1),
            "n_ratings": np.full(n_users, n_ratings, dtype=np.int64),
            "sum_squared_error": np.full(n_users, squared_error, dtype=np.float64),
            "sum_absolute_error": np.full(n_users, squared_error, dtype=np.float64),
        }
    )


def test_indices_have_one_row_per_iteration():
    rng = np.random.default_rng(1)
    indices = bootstrap_user_indices(50, 200, rng)
    assert indices.shape == (200, 50)
    assert indices.min() >= 0
    assert indices.max() < 50


def test_resampling_is_with_replacement():
    rng = np.random.default_rng(1)
    indices = bootstrap_user_indices(50, 200, rng)
    has_duplicate = False
    for row in indices:
        if len(set(row.tolist())) < len(row):
            has_duplicate = True
            break
    assert has_duplicate


def test_bootstrap_is_reproducible_under_the_seed():
    first = bootstrap_user_indices(30, 50, np.random.default_rng(7))
    second = bootstrap_user_indices(30, 50, np.random.default_rng(7))
    assert np.array_equal(first, second)


def test_accuracy_samples_centre_on_the_observed_metric():
    per_user = make_per_user(200, squared_error=4.0, n_ratings=4)
    observed = aggregate_user_accuracy(per_user)["rmse"]
    rng = np.random.default_rng(3)
    indices = bootstrap_user_indices(len(per_user), 400, rng)
    samples = accuracy_samples(per_user, indices, "rmse")
    assert float(np.mean(samples)) == pytest.approx(observed, abs=1e-6)


def test_accuracy_samples_reject_an_unknown_metric():
    per_user = make_per_user(10, 1.0, 1)
    indices = bootstrap_user_indices(10, 5, np.random.default_rng(1))
    with pytest.raises(ValueError, match="rmse or mae"):
        accuracy_samples(per_user, indices, "precision")


def test_heavy_raters_dominate_the_accuracy_bootstrap():
    """A user with 100 ratings must not carry the same weight as one with 1."""
    per_user = pd.DataFrame(
        {
            "user_id": [1, 2],
            "n_ratings": [1, 100],
            "sum_squared_error": [100.0, 100.0],
            "sum_absolute_error": [10.0, 100.0],
        }
    )
    indices = np.array([[0, 1]])
    sample = accuracy_samples(per_user, indices, "rmse")[0]
    # 200 squared error over 101 ratings, not the average of the two per-user rates
    assert sample == pytest.approx(np.sqrt(200.0 / 101.0))


def test_ranking_samples_ignore_users_with_no_defined_value():
    values = np.array([1.0, np.nan, 1.0, 1.0])
    indices = np.array([[0, 1, 2, 3]])
    assert ranking_samples(values, indices)[0] == pytest.approx(1.0)


def test_interval_brackets_the_centre():
    samples = np.random.default_rng(5).normal(0.5, 0.1, 5000)
    lower, upper = percentile_interval(samples, 0.95)
    assert lower < 0.5 < upper
    assert upper - lower < 0.5


def test_a_wider_confidence_level_gives_a_wider_interval():
    samples = np.random.default_rng(5).normal(0.0, 1.0, 5000)
    narrow = percentile_interval(samples, 0.80)
    wide = percentile_interval(samples, 0.99)
    assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])


def test_empty_samples_give_nan_rather_than_raising():
    lower, upper = percentile_interval(np.array([]))
    assert np.isnan(lower)
    assert np.isnan(upper)


def test_identical_models_show_no_difference():
    samples = np.random.default_rng(2).normal(1.0, 0.1, 2000)
    outcome = difference_interval(samples, samples)
    assert outcome["difference"] == pytest.approx(0.0)
    assert outcome["excludes_zero"] is False


def test_a_clear_gap_excludes_zero():
    rng = np.random.default_rng(4)
    better = rng.normal(0.90, 0.01, 2000)
    worse = rng.normal(1.10, 0.01, 2000)
    outcome = difference_interval(better, worse)
    assert outcome["difference"] < 0.0
    assert outcome["upper"] < 0.0
    assert outcome["excludes_zero"] is True


def test_pairing_detects_a_gap_that_separate_intervals_would_miss():
    """The reason the resample is paired rather than run per model.

    Both models vary a lot across users, but one is consistently a little better. Their
    individual intervals overlap heavily; the paired difference does not touch zero.
    """
    rng = np.random.default_rng(11)
    shared_variation = rng.normal(1.0, 0.30, 4000)
    first = shared_variation
    second = shared_variation + 0.05

    first_interval = percentile_interval(first)
    second_interval = percentile_interval(second)
    assert first_interval[1] > second_interval[0]

    outcome = difference_interval(first, second)
    assert outcome["excludes_zero"] is True


def test_align_restricts_to_users_every_model_scored():
    frames = {
        "a": pd.DataFrame({"user_id": [1, 2, 3], "value": [1.0, 2.0, 3.0]}),
        "b": pd.DataFrame({"user_id": [2, 3, 4], "value": [2.0, 3.0, 4.0]}),
    }
    aligned, shared = align_per_user(frames)
    assert shared == [2, 3]
    for frame in aligned.values():
        assert list(frame["user_id"]) == [2, 3]


def test_compare_models_covers_every_pair():
    rng = np.random.default_rng(6)
    samples = {
        "a": rng.normal(1.0, 0.1, 500),
        "b": rng.normal(1.1, 0.1, 500),
        "c": rng.normal(1.2, 0.1, 500),
    }
    frame = compare_models(samples)
    assert len(frame) == 3
    pairs = set(zip(frame["model_a"], frame["model_b"]))
    assert pairs == {("a", "b"), ("a", "c"), ("b", "c")}


def test_bootstrap_agrees_with_the_metrics_module():
    """The bootstrap must resample the same quantity the results table reports."""
    predictions = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 3],
            "rating": [5.0, 3.0, 4.0, 2.0, 1.0],
            "predicted": [4.0, 1.0, 4.0, 3.0, 3.0],
        }
    )
    per_user = per_user_accuracy(predictions)
    observed = aggregate_user_accuracy(per_user)["rmse"]
    indices = np.array([[0, 1, 2]])
    assert accuracy_samples(per_user, indices, "rmse")[0] == pytest.approx(observed)

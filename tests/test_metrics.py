"""Metric tests against values worked out by hand.

Checking a metric against a second implementation of the same metric proves nothing, so
the expected numbers here are computed on paper and written in as literals.
"""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import accuracy_metrics
from src.evaluation.metrics import aggregate_user_accuracy
from src.evaluation.metrics import mae
from src.evaluation.metrics import per_user_accuracy
from src.evaluation.metrics import rmse


def test_perfect_predictions_score_zero():
    actual = [3.0, 4.0, 5.0]
    assert rmse(actual, actual) == 0.0
    assert mae(actual, actual) == 0.0


def test_rmse_matches_a_hand_computed_case():
    # errors are 1 and 2, so the mean squared error is (1 + 4) / 2 = 2.5
    actual = [1.0, 2.0]
    predicted = [2.0, 4.0]
    assert rmse(actual, predicted) == pytest.approx(np.sqrt(2.5))


def test_mae_matches_a_hand_computed_case():
    # errors are 1 and 2, so the mean absolute error is 1.5
    assert mae([1.0, 2.0], [2.0, 4.0]) == pytest.approx(1.5)


def test_rmse_punishes_large_errors_harder_than_mae():
    # errors of 0 and 4: mae is 2.0 but rmse is sqrt(8) = 2.83
    actual = [2.0, 2.0]
    lumpy = [2.0, 6.0]
    assert mae(actual, lumpy) == pytest.approx(2.0)
    assert rmse(actual, lumpy) == pytest.approx(np.sqrt(8.0))
    assert rmse(actual, lumpy) > mae(actual, lumpy)


def test_empty_input_gives_nan_rather_than_raising():
    assert np.isnan(rmse([], []))
    assert np.isnan(mae([], []))


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        rmse([1.0, 2.0], [1.0])


def test_accuracy_metrics_reports_the_rating_count():
    result = accuracy_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert result["n_ratings"] == 3
    assert result["rmse"] == 0.0


def test_per_user_accuracy_sums_by_user():
    predictions = pd.DataFrame(
        {
            "user_id": [1, 1, 2],
            "rating": [5.0, 3.0, 4.0],
            "predicted": [4.0, 1.0, 4.0],
        }
    )
    per_user = per_user_accuracy(predictions).set_index("user_id")
    # user 1 has errors of 1 and 2, user 2 has an error of 0
    assert per_user.loc[1, "n_ratings"] == 2
    assert per_user.loc[1, "sum_squared_error"] == pytest.approx(5.0)
    assert per_user.loc[1, "sum_absolute_error"] == pytest.approx(3.0)
    assert per_user.loc[2, "sum_squared_error"] == pytest.approx(0.0)


def test_aggregating_per_user_sums_recovers_the_global_metric():
    predictions = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 2],
            "rating": [5.0, 3.0, 4.0, 2.0, 1.0],
            "predicted": [4.0, 1.0, 4.0, 3.0, 3.0],
        }
    )
    direct = accuracy_metrics(predictions["rating"], predictions["predicted"])
    recovered = aggregate_user_accuracy(per_user_accuracy(predictions))
    assert recovered["rmse"] == pytest.approx(direct["rmse"])
    assert recovered["mae"] == pytest.approx(direct["mae"])
    assert recovered["n_ratings"] == direct["n_ratings"]


def test_unequal_rating_counts_are_weighted_correctly():
    """The heavy user must dominate, which per-user means would hide."""
    predictions = pd.DataFrame(
        {
            "user_id": [1, 2, 2, 2, 2],
            "rating": [5.0, 3.0, 3.0, 3.0, 3.0],
            "predicted": [1.0, 3.0, 3.0, 3.0, 3.0],
        }
    )
    recovered = aggregate_user_accuracy(per_user_accuracy(predictions))
    # one error of 4 across five ratings: mean squared error is 16 / 5 = 3.2
    assert recovered["rmse"] == pytest.approx(np.sqrt(3.2))


def test_per_user_accuracy_requires_its_columns():
    with pytest.raises(ValueError):
        per_user_accuracy(pd.DataFrame({"user_id": [1], "rating": [3.0]}))

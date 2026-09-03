"""Tests for the ranking and catalogue metrics, against hand-computed values."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.ranking import catalogue_coverage
from src.evaluation.ranking import item_popularity
from src.evaluation.ranking import mean_popularity_percentile
from src.evaluation.ranking import mean_recommended_popularity
from src.evaluation.ranking import popularity_percentiles
from src.evaluation.ranking import precision_at_k
from src.evaluation.ranking import recall_at_k
from src.evaluation.ranking import recommendation_gini
from src.evaluation.ranking import relevant_items_by_user


def test_precision_counts_hits_over_k():
    # two of the top five are relevant
    recommended = [10, 20, 30, 40, 50]
    relevant = {20, 40, 99}
    assert precision_at_k(recommended, relevant, 5) == pytest.approx(2.0 / 5.0)


def test_recall_counts_hits_over_the_relevant_set():
    # two of the three relevant items are retrieved
    recommended = [10, 20, 30, 40, 50]
    relevant = {20, 40, 99}
    assert recall_at_k(recommended, relevant, 5) == pytest.approx(2.0 / 3.0)


def test_only_the_top_k_counts():
    recommended = [10, 20, 30, 40, 50]
    relevant = {50}
    assert precision_at_k(recommended, relevant, 3) == 0.0
    assert precision_at_k(recommended, relevant, 5) == pytest.approx(1.0 / 5.0)


def test_a_short_list_is_penalised_not_rescaled():
    """Returning two items when k is five scores 1/5, not 1/2."""
    assert precision_at_k([10, 20], {10}, 5) == pytest.approx(1.0 / 5.0)


def test_perfect_and_empty_rankings():
    assert precision_at_k([1, 2, 3], {1, 2, 3}, 3) == 1.0
    assert recall_at_k([1, 2, 3], {1, 2, 3}, 3) == 1.0
    assert precision_at_k([7, 8, 9], {1, 2, 3}, 3) == 0.0


def test_users_with_no_relevant_items_give_nan():
    assert np.isnan(precision_at_k([1, 2], set(), 2))
    assert np.isnan(recall_at_k([1, 2], set(), 2))


def test_non_positive_k_raises():
    with pytest.raises(ValueError):
        precision_at_k([1], {1}, 0)
    with pytest.raises(ValueError):
        recall_at_k([1], {1}, -1)


def test_relevant_items_uses_the_threshold():
    holdout = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 2],
            "item_id": [10, 11, 12, 20],
            "rating": [5.0, 3.0, 4.0, 2.0],
        }
    )
    relevant = relevant_items_by_user(holdout, 4.0)
    assert relevant[1] == {10, 12}
    assert 2 not in relevant


def test_coverage_counts_distinct_items():
    recommendations = {1: [10, 20], 2: [20, 30], 3: [10, 20]}
    # three distinct items out of a catalogue of ten
    assert catalogue_coverage(recommendations, 10) == pytest.approx(0.3)


def test_coverage_is_one_when_everything_is_shown():
    recommendations = {1: [1, 2], 2: [3, 4]}
    assert catalogue_coverage(recommendations, 4) == 1.0


def test_mean_recommended_popularity_averages_training_counts():
    train = pd.DataFrame(
        {
            "user_id": [1, 2, 3, 1, 2, 1],
            "item_id": [10, 10, 10, 20, 20, 30],
            "rating": [4.0, 4.0, 4.0, 3.0, 3.0, 5.0],
        }
    )
    popularity = item_popularity(train)
    assert popularity == {10: 3, 20: 2, 30: 1}
    recommendations = {1: [10, 20]}
    # counts of 3 and 2 average to 2.5
    assert mean_recommended_popularity(recommendations, popularity) == pytest.approx(2.5)


def test_popularity_percentiles_run_from_zero_to_one():
    percentiles = popularity_percentiles({10: 3, 20: 2, 30: 1})
    assert percentiles[30] == pytest.approx(0.0)
    assert percentiles[20] == pytest.approx(0.5)
    assert percentiles[10] == pytest.approx(1.0)


def test_recommending_only_blockbusters_scores_near_one():
    percentiles = popularity_percentiles({10: 3, 20: 2, 30: 1})
    blockbusters = {1: [10], 2: [10]}
    long_tail = {1: [30], 2: [30]}
    assert mean_popularity_percentile(blockbusters, percentiles) == pytest.approx(1.0)
    assert mean_popularity_percentile(long_tail, percentiles) == pytest.approx(0.0)


def test_gini_is_zero_when_exposure_is_even():
    catalogue = [1, 2, 3, 4]
    recommendations = {1: [1, 2], 2: [3, 4]}
    assert recommendation_gini(recommendations, catalogue) == pytest.approx(0.0)


def test_gini_rises_when_one_item_takes_everything():
    catalogue = [1, 2, 3, 4]
    even = {1: [1, 2], 2: [3, 4]}
    lopsided = {1: [1], 2: [1], 3: [1]}
    assert recommendation_gini(lopsided, catalogue) > recommendation_gini(even, catalogue)


def test_coverage_and_gini_disagree_by_design():
    """Same coverage, very different exposure: this is why both are reported."""
    catalogue = [1, 2, 3, 4]
    even = {1: [1], 2: [2], 3: [3], 4: [4]}
    lopsided = {1: [1], 2: [1], 3: [1], 4: [1, 2, 3, 4]}
    assert catalogue_coverage(even, 4) == catalogue_coverage(lopsided, 4)
    assert recommendation_gini(lopsided, catalogue) > recommendation_gini(even, catalogue)

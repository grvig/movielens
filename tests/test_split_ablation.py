"""Tests for the random-split ablation.

These check the ablation is a fair comparison - same proportions, same data, one variable
changed - and that the random split really does leak, since an ablation that failed to
leak would be measuring nothing.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.splitting import assert_no_leakage
from src.data.splitting import temporal_split
from src.evaluation.harness import results_frame
from src.experiments.run_split_ablation import evaluate_under_split
from src.experiments.run_split_ablation import leakage_report
from src.experiments.run_split_ablation import random_split


def test_random_split_partitions_the_data(config, synthetic_ratings):
    train, validation, test = random_split(synthetic_ratings, config)
    total = len(train) + len(validation) + len(test)
    assert total == len(synthetic_ratings)

    original = set(zip(synthetic_ratings["user_id"], synthetic_ratings["item_id"]))
    recovered = set()
    for frame in [train, validation, test]:
        recovered.update(zip(frame["user_id"], frame["item_id"]))
    assert recovered == original


def test_random_split_matches_the_temporal_split_proportions(config, synthetic_ratings):
    """One variable changes between the two arms, and it is not the split sizes."""
    temporal = temporal_split(synthetic_ratings, config)
    random = random_split(synthetic_ratings, config)
    for temporal_frame, random_frame in zip(temporal, random):
        assert abs(len(temporal_frame) - len(random_frame)) <= 2


def test_random_split_actually_leaks(config, synthetic_ratings):
    """If this passed the leakage check, the ablation would be comparing nothing."""
    train, validation, test = random_split(synthetic_ratings, config)
    with pytest.raises(AssertionError):
        assert_no_leakage(train, validation, test)


def test_temporal_split_does_not_leak(config, synthetic_ratings):
    train, validation, test = temporal_split(synthetic_ratings, config)
    assert_no_leakage(train, validation, test)


def test_random_split_is_deterministic(config, synthetic_ratings):
    first = random_split(synthetic_ratings, config)
    second = random_split(synthetic_ratings, config)
    for frame_one, frame_two in zip(first, second):
        pd.testing.assert_frame_equal(frame_one, frame_two)


def test_random_split_is_not_the_temporal_split(config, synthetic_ratings):
    temporal_train, _, _ = temporal_split(synthetic_ratings, config)
    random_train, _, _ = random_split(synthetic_ratings, config)
    temporal_pairs = set(zip(temporal_train["user_id"], temporal_train["item_id"]))
    random_pairs = set(zip(random_train["user_id"], random_train["item_id"]))
    assert temporal_pairs != random_pairs


def test_leakage_report_computes_the_difference():
    frame = pd.DataFrame(
        {
            "model": ["item_knn", "item_knn"],
            "split": ["temporal", "random"],
            "metric": ["rmse", "rmse"],
            "k": [None, None],
            "value": [1.0, 0.8],
        }
    )
    report = leakage_report(frame, "rmse")
    assert report.loc["item_knn", "difference"] == pytest.approx(-0.2)
    assert report.loc["item_knn", "percent"] == pytest.approx(-20.0)


def test_leakage_report_survives_a_zero_baseline():
    """Precision of zero under the temporal split must not raise on the percentage."""
    frame = pd.DataFrame(
        {
            "model": ["global_mean", "global_mean"],
            "split": ["temporal", "random"],
            "metric": ["precision_at_k", "precision_at_k"],
            "k": [10, 10],
            "value": [0.0, 0.0],
        }
    )
    report = leakage_report(frame, "precision_at_k", 10)
    assert np.isnan(report.loc["global_mean", "percent"])


def test_both_arms_produce_comparable_rows(config, synthetic_ratings, synthetic_items):
    temporal = temporal_split(synthetic_ratings, config)
    random = random_split(synthetic_ratings, config)
    rows = evaluate_under_split(
        config, synthetic_items, "temporal", temporal[0], temporal[1], temporal[2], "run"
    )
    rows.extend(
        evaluate_under_split(
            config, synthetic_items, "random", random[0], random[1], random[2], "run"
        )
    )
    frame = results_frame(rows)
    assert set(frame["split"]) == {"temporal", "random"}
    counts = frame.groupby("split").size()
    assert counts["temporal"] == counts["random"]

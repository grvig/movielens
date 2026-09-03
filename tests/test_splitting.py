"""Tests for the per-user temporal split.

If any of these fail, every number in results/ is wrong.  They are the most important
tests in the project.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.splitting import assert_no_leakage
from src.data.splitting import split_summary
from src.data.splitting import temporal_split
from src.data.splitting import user_split_sizes


def test_split_partitions_the_data(config, synthetic_ratings):
    train, val, test = temporal_split(synthetic_ratings, config)
    total = len(train) + len(val) + len(test)
    assert total == len(synthetic_ratings)

    original = set(zip(synthetic_ratings["user_id"], synthetic_ratings["item_id"]))
    recovered = set()
    for frame in [train, val, test]:
        recovered.update(zip(frame["user_id"], frame["item_id"]))
    assert recovered == original


def test_splits_do_not_overlap(config, synthetic_ratings):
    train, val, test = temporal_split(synthetic_ratings, config)
    train_pairs = set(zip(train["user_id"], train["item_id"]))
    val_pairs = set(zip(val["user_id"], val["item_id"]))
    test_pairs = set(zip(test["user_id"], test["item_id"]))
    assert train_pairs.isdisjoint(val_pairs)
    assert train_pairs.isdisjoint(test_pairs)
    assert val_pairs.isdisjoint(test_pairs)


def test_no_temporal_leakage(config, synthetic_ratings):
    train, val, test = temporal_split(synthetic_ratings, config)
    assert_no_leakage(train, val, test)


def test_leakage_assertion_actually_catches_leakage(config, synthetic_ratings):
    """A random split must trip the guard, otherwise the guard proves nothing."""
    shuffled = synthetic_ratings.sample(frac=1.0, random_state=3).reset_index(drop=True)
    cut_one = int(len(shuffled) * 0.7)
    cut_two = int(len(shuffled) * 0.8)
    train = shuffled.iloc[:cut_one]
    val = shuffled.iloc[cut_one:cut_two]
    test = shuffled.iloc[cut_two:]
    with pytest.raises(AssertionError):
        assert_no_leakage(train, val, test)


def test_every_user_is_ordered_in_time(config, synthetic_ratings):
    train, val, test = temporal_split(synthetic_ratings, config)
    for user_id in synthetic_ratings["user_id"].unique():
        user_train = train[train["user_id"] == user_id]
        user_val = val[val["user_id"] == user_id]
        user_test = test[test["user_id"] == user_id]
        if len(user_val) > 0:
            assert user_train["timestamp"].max() <= user_val["timestamp"].min()
        if len(user_test) > 0:
            assert user_train["timestamp"].max() <= user_test["timestamp"].min()
        if len(user_val) > 0 and len(user_test) > 0:
            assert user_val["timestamp"].max() <= user_test["timestamp"].min()


def test_every_user_keeps_training_ratings(config, synthetic_ratings):
    train, _, _ = temporal_split(synthetic_ratings, config)
    users_in = set(synthetic_ratings["user_id"].unique())
    users_out = set(train["user_id"].unique())
    assert users_in == users_out


def test_split_is_deterministic(config, synthetic_ratings):
    first = temporal_split(synthetic_ratings, config)
    second = temporal_split(synthetic_ratings, config)
    for frame_one, frame_two in zip(first, second):
        pd.testing.assert_frame_equal(frame_one, frame_two)


def test_ties_on_timestamp_break_on_item_id(config):
    """All ratings share a timestamp, so only the item-id tie-break can order them."""
    rows = []
    for item_id in [5, 3, 1, 4, 2, 6, 9, 7, 8, 10]:
        rows.append((1, item_id, 3.0, 900000000))
    tied = pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "timestamp"])
    train, val, test = temporal_split(tied, config)
    ordered_items = list(train["item_id"]) + list(val["item_id"]) + list(test["item_id"])
    assert ordered_items == sorted(ordered_items)


def test_user_split_sizes_respect_the_minimum():
    n_train, n_val = user_split_sizes(6, 0.1, 0.2, 5)
    assert n_train == 6
    assert n_val == 0

    n_train, n_val = user_split_sizes(20, 0.1, 0.2, 5)
    assert n_train == 14
    assert n_val == 2


def test_split_sizes_are_roughly_the_configured_fractions(config, synthetic_ratings):
    train, val, test = temporal_split(synthetic_ratings, config)
    total = len(synthetic_ratings)
    assert 0.60 < len(train) / total < 0.80
    assert 0.15 < len(test) / total < 0.30


def test_split_summary_reports_all_three(config, synthetic_ratings):
    train, val, test = temporal_split(synthetic_ratings, config)
    summary = split_summary(train, val, test)
    assert list(summary["split"]) == ["train", "val", "test"]
    assert summary["ratings"].sum() == len(synthetic_ratings)

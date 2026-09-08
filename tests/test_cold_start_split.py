"""Tests for the cold-start history truncation.

The design decision these pin down is that truncation is *selective*. Truncating every
user's history would measure how a model copes with a small dataset; truncating a subset
measures how it copes with a new user, which is the question the experiment asks.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.splitting import history_lengths
from src.data.splitting import select_cold_users
from src.data.splitting import truncate_user_histories


@pytest.fixture
def train():
    rows = []
    for user_id in range(1, 6):
        for position in range(20):
            rows.append((user_id, position + 1, 4.0, 1000 + position))
    return pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "timestamp"])


def test_only_the_named_users_are_truncated(train):
    truncated = truncate_user_histories(train, [1, 2], 5)
    counts = truncated.groupby("user_id").size()
    assert counts[1] == 5
    assert counts[2] == 5
    assert counts[3] == 20
    assert counts[4] == 20
    assert counts[5] == 20


def test_the_most_recent_ratings_are_kept(train):
    """The split is temporal, so recent history is closest to the evaluation period."""
    truncated = truncate_user_histories(train, [1], 3)
    kept = truncated[truncated["user_id"] == 1]
    assert sorted(kept["timestamp"].tolist()) == [1017, 1018, 1019]


def test_truncating_to_one_leaves_exactly_one_rating(train):
    truncated = truncate_user_histories(train, [1, 2, 3], 1)
    counts = truncated.groupby("user_id").size()
    for user_id in [1, 2, 3]:
        assert counts[user_id] == 1


def test_a_larger_limit_than_the_history_is_a_no_op(train):
    truncated = truncate_user_histories(train, [1], 500)
    assert len(truncated) == len(train)


def test_truncation_only_removes_rows(train):
    truncated = truncate_user_histories(train, [1, 2], 5)
    original = set(zip(train["user_id"], train["item_id"]))
    remaining = set(zip(truncated["user_id"], truncated["item_id"]))
    assert remaining.issubset(original)
    assert len(truncated) == len(train) - 30


def test_columns_and_dtypes_survive(train):
    truncated = truncate_user_histories(train, [1], 5)
    assert list(truncated.columns) == list(train.columns)
    assert truncated["user_id"].dtype == train["user_id"].dtype


def test_truncation_is_deterministic(train):
    first = truncate_user_histories(train, [1, 3], 4)
    second = truncate_user_histories(train, [1, 3], 4)
    pd.testing.assert_frame_equal(first, second)


def test_nested_truncation_is_monotone(train):
    """A shorter history must be a subset of a longer one for the same user."""
    long_history = truncate_user_histories(train, [1], 10)
    short_history = truncate_user_histories(train, [1], 3)
    long_items = set(long_history[long_history["user_id"] == 1]["item_id"])
    short_items = set(short_history[short_history["user_id"] == 1]["item_id"])
    assert short_items.issubset(long_items)


def test_cold_user_selection_is_reproducible(train):
    first = select_cold_users(train, 3, np.random.default_rng(7))
    second = select_cold_users(train, 3, np.random.default_rng(7))
    assert np.array_equal(first, second)


def test_cold_user_selection_returns_everyone_when_asked_for_too_many(train):
    chosen = select_cold_users(train, 500, np.random.default_rng(1))
    assert len(chosen) == 5


def test_cold_user_selection_respects_a_minimum_history(train):
    short = pd.DataFrame(
        [(9, 1, 4.0, 100)], columns=["user_id", "item_id", "rating", "timestamp"]
    )
    combined = pd.concat([train, short], ignore_index=True)
    chosen = select_cold_users(combined, 100, np.random.default_rng(1), min_history=20)
    assert 9 not in set(chosen.tolist())


def test_history_lengths_reports_zero_for_a_missing_user(train):
    lengths = history_lengths(train, [1, 999])
    assert lengths[1] == 20
    assert lengths[999] == 0

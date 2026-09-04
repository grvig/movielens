"""Tests for the cosine-to-rating calibration.

The calibration exists because a cosine similarity is not a rating. These tests check that
the fitted scale recovers a known value, that it is fitted on validation rather than
anything else, and that it degrades sensibly when there is nothing to fit on.
"""

import numpy as np
import pandas as pd
import pytest

from src.models.content import ContentBased
from src.models.content import clamp_scale


@pytest.fixture
def items():
    rows = []
    for item_id in [1, 2, 3]:
        rows.append({"item_id": item_id, "title": "Horror " + str(item_id),
                     "release_year": 1990, "genre_Horror": 1, "genre_Romance": 0})
    for item_id in [4, 5, 6]:
        rows.append({"item_id": item_id, "title": "Romance " + str(item_id),
                     "release_year": 1990, "genre_Horror": 0, "genre_Romance": 1})
    return pd.DataFrame(rows)


@pytest.fixture
def train():
    rows = [
        (1, 1, 5.0, 1000), (1, 2, 5.0, 1001), (1, 4, 1.0, 1002),
        (2, 4, 5.0, 1000), (2, 5, 5.0, 1001), (2, 1, 1.0, 1002),
        (3, 3, 3.0, 1000), (3, 6, 3.0, 1001),
    ]
    return pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "timestamp"])


@pytest.fixture
def wide_config(config):
    config.values["features"]["min_df"] = 1
    return config


def test_without_validation_the_config_value_is_used(wide_config, items, train):
    model = ContentBased(wide_config, items).fit(train)
    assert model.calibration_fitted is False
    assert model.calibration_scale == pytest.approx(model.default_scale)


def test_calibration_recovers_a_planted_scale(wide_config, items, train):
    """Build validation ratings that are exactly user_mean + 2.0 * cosine.

    The least-squares fit has to come back with 2.0, or the regression is wrong.
    """
    uncalibrated = ContentBased(wide_config, items).fit(train)
    planted_scale = 2.0
    rows = []
    for user_id in [1, 2]:
        candidates = uncalibrated.candidate_items(user_id)
        similarities = uncalibrated.rank_scores(user_id, candidates)
        centre = uncalibrated.user_mean[user_id]
        for item_id, similarity in zip(candidates, similarities):
            rows.append((user_id, int(item_id), centre + planted_scale * similarity, 2000))
    validation = pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "timestamp"])

    model = ContentBased(wide_config, items, validation=validation).fit(train)
    assert model.calibration_fitted is True
    assert model.calibration_scale == pytest.approx(planted_scale, abs=1e-6)


def test_calibration_improves_predictions_on_the_data_it_was_fitted_to(
    wide_config, items, train
):
    uncalibrated = ContentBased(wide_config, items).fit(train)
    rows = []
    for user_id in [1, 2]:
        candidates = uncalibrated.candidate_items(user_id)
        similarities = uncalibrated.rank_scores(user_id, candidates)
        centre = uncalibrated.user_mean[user_id]
        for item_id, similarity in zip(candidates, similarities):
            rows.append((user_id, int(item_id), centre + 2.0 * similarity, 2000))
    validation = pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "timestamp"])

    calibrated = ContentBased(wide_config, items, validation=validation).fit(train)
    before = squared_error(uncalibrated, validation)
    after = squared_error(calibrated, validation)
    assert after < before


def test_empty_validation_falls_back_to_the_default(wide_config, items, train):
    empty = pd.DataFrame(columns=["user_id", "item_id", "rating", "timestamp"])
    model = ContentBased(wide_config, items, validation=empty).fit(train)
    assert model.calibration_scale == pytest.approx(model.default_scale)


def test_validation_for_unknown_users_falls_back_to_the_default(wide_config, items, train):
    """Users with no profile contribute nothing, so there is nothing to fit."""
    validation = pd.DataFrame(
        [(999, 3, 4.0, 2000)], columns=["user_id", "item_id", "rating", "timestamp"]
    )
    model = ContentBased(wide_config, items, validation=validation).fit(train)
    assert model.calibration_scale == pytest.approx(model.default_scale)


def test_calibrated_predictions_stay_in_the_rating_range(wide_config, items, train):
    validation = pd.DataFrame(
        [(1, 3, 5.0, 2000), (1, 5, 1.0, 2001), (2, 6, 5.0, 2000)],
        columns=["user_id", "item_id", "rating", "timestamp"],
    )
    model = ContentBased(wide_config, items, validation=validation).fit(train)
    predictions = model.predict(1, np.array([3, 5, 6]))
    assert np.all(predictions >= model.rating_min)
    assert np.all(predictions <= model.rating_max)


def test_clamp_rejects_a_negative_scale():
    assert clamp_scale(-3.0, 4.0) == 0.0


def test_clamp_caps_a_runaway_scale():
    assert clamp_scale(500.0, 4.0) == 4.0


def test_clamp_leaves_a_sane_scale_alone():
    assert clamp_scale(1.75, 4.0) == pytest.approx(1.75)


def squared_error(model, frame):
    total = 0.0
    for user_id, group in frame.groupby("user_id"):
        items_seen = group["item_id"].to_numpy(dtype=np.int64)
        predicted = model.predict(user_id, items_seen)
        actual = group["rating"].to_numpy(dtype=np.float64)
        total = total + float(np.sum((actual - predicted) ** 2))
    return total

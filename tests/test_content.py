"""Tests for the content-based model.

The behaviour worth pinning down is the mean-centred weighting: a disliked item has to
push the profile away from its features, not towards them.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.features import build_item_features
from src.models.content import ContentBased
from src.models.content import weighted_profile


@pytest.fixture
def polarised_items():
    """Two clean genre groups with nothing in common, so cosine signs are unambiguous."""
    rows = []
    for item_id in [1, 2, 3]:
        rows.append({"item_id": item_id, "title": "Horror Film " + str(item_id),
                     "release_year": 1990, "genre_Horror": 1, "genre_Romance": 0})
    for item_id in [4, 5, 6]:
        rows.append({"item_id": item_id, "title": "Romance Film " + str(item_id),
                     "release_year": 1990, "genre_Horror": 0, "genre_Romance": 1})
    return pd.DataFrame(rows)


@pytest.fixture
def polarised_ratings():
    """User 1 loves horror and hates romance. User 2 is the mirror image.

    User 3 exists only so that items 3 and 6 appear somewhere in training. The candidate
    catalogue is the set of items seen during training, so an item nobody rated is not
    recommendable by anyone and the mirrored-ranking test would have nothing to choose
    between.
    """
    rows = [
        (1, 1, 5.0, 1000), (1, 2, 5.0, 1001), (1, 4, 1.0, 1002),
        (2, 4, 5.0, 1000), (2, 5, 5.0, 1001), (2, 1, 1.0, 1002),
        (3, 3, 3.0, 1000), (3, 6, 3.0, 1001),
    ]
    return pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "timestamp"])


@pytest.fixture
def polarised_config(config):
    config.values["features"]["min_df"] = 1
    return config


def test_profile_prefers_the_genre_the_user_rated_highly(
    polarised_config, polarised_items, polarised_ratings
):
    model = ContentBased(polarised_config, polarised_items).fit(polarised_ratings)
    horror_score = model.rank_scores(1, np.array([3]))[0]
    romance_score = model.rank_scores(1, np.array([6]))[0]
    assert horror_score > romance_score


def test_a_disliked_item_pushes_the_profile_away(
    polarised_config, polarised_items, polarised_ratings
):
    """User 1 rated a romance film 1 out of 5, so unseen romance must score negative.

    Weighting by raw rating instead of the centred rating would make this positive, which
    is the bug this test exists to catch.
    """
    model = ContentBased(polarised_config, polarised_items).fit(polarised_ratings)
    assert model.rank_scores(1, np.array([6]))[0] < 0.0


def test_mirrored_users_get_mirrored_rankings(
    polarised_config, polarised_items, polarised_ratings
):
    model = ContentBased(polarised_config, polarised_items).fit(polarised_ratings)
    assert model.recommend(1, 1)[0] == 3
    assert model.recommend(2, 1)[0] == 6


def test_candidates_are_training_items_only(
    polarised_config, polarised_items, polarised_ratings
):
    """An item with no training ratings is not recommendable, for any model.

    Content-based scoring could in principle rank an item nobody has rated, since it only
    needs the item's features. Allowing that would give it a candidate set the
    collaborative models cannot have, and the comparison would stop being like for like.
    """
    unrated = polarised_ratings[polarised_ratings["item_id"] != 6]
    model = ContentBased(polarised_config, polarised_items).fit(unrated)
    assert 6 not in set(model.candidate_items(1).tolist())


def test_cosine_scores_stay_in_range(polarised_config, polarised_items, polarised_ratings):
    model = ContentBased(polarised_config, polarised_items).fit(polarised_ratings)
    scores = model.rank_scores(1, np.array([1, 2, 3, 4, 5, 6]))
    assert np.all(scores >= -1.0 - 1e-9)
    assert np.all(scores <= 1.0 + 1e-9)


def test_predictions_are_centred_on_the_user_mean(
    polarised_config, polarised_items, polarised_ratings
):
    model = ContentBased(polarised_config, polarised_items).fit(polarised_ratings)
    predictions = model.predict(1, np.array([3, 6]))
    assert np.all(predictions >= model.rating_min)
    assert np.all(predictions <= model.rating_max)
    # the liked genre must predict higher than the disliked one
    assert predictions[0] > predictions[1]


def test_unknown_user_scores_zero_everywhere(
    polarised_config, polarised_items, polarised_ratings
):
    model = ContentBased(polarised_config, polarised_items).fit(polarised_ratings)
    assert np.allclose(model.rank_scores(999, np.array([1, 2, 3])), 0.0)


def test_unknown_item_is_skipped_not_crashed(
    polarised_config, polarised_items, polarised_ratings
):
    model = ContentBased(polarised_config, polarised_items).fit(polarised_ratings)
    scores = model.rank_scores(1, np.array([1, 9999]))
    assert scores[1] == 0.0


def test_profiles_are_unit_length(polarised_config, polarised_items, polarised_ratings):
    model = ContentBased(polarised_config, polarised_items).fit(polarised_ratings)
    for user_id in [1, 2]:
        norm = np.linalg.norm(model.profile_for(user_id))
        assert norm == pytest.approx(1.0)


def test_flat_rater_falls_back_to_an_unweighted_mean(polarised_config, polarised_items):
    """Every weight is zero when a user rates everything the same, so the average is
    undefined and the model must not return NaN."""
    flat = pd.DataFrame(
        [(1, 1, 3.0, 1000), (1, 2, 3.0, 1001)],
        columns=["user_id", "item_id", "rating", "timestamp"],
    )
    model = ContentBased(polarised_config, polarised_items).fit(flat)
    profile = model.profile_for(1)
    assert np.all(np.isfinite(profile))
    assert np.linalg.norm(profile) == pytest.approx(1.0)


def test_weighted_profile_handles_zero_weights(polarised_config, polarised_items):
    matrix, _, _ = build_item_features(polarised_items, polarised_config)
    profile = weighted_profile(matrix, [0, 1], np.array([0.0, 0.0]))
    assert np.all(np.isfinite(profile))


def test_content_works_on_the_shared_fixture(config, synthetic_ratings, synthetic_items):
    model = ContentBased(config, synthetic_items).fit(synthetic_ratings)
    recommendations = model.recommend(1, 5)
    assert len(recommendations) == 5
    predictions = model.predict(1, np.array([1, 2, 3]))
    assert np.all(np.isfinite(predictions))

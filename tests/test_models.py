"""Conformance tests for the shared recommender interface.

Every model registered in ``MODEL_CLASSES`` is put through the same checks.  The point of
the project is that the harness treats all models identically, so a model that needs an
exception here is a model that would quietly invalidate the comparison.

New model families get added to MODEL_CLASSES and must pass unchanged.
"""

import numpy as np
import pytest

from src.models.baselines import GlobalMean
from src.models.baselines import ItemMean
from src.models.baselines import MostPopular
from src.models.baselines import UserMean

MODEL_CLASSES = [GlobalMean, UserMean, ItemMean, MostPopular]

UNKNOWN_USER = 9999
UNKNOWN_ITEM = 9999


@pytest.fixture(params=MODEL_CLASSES, ids=lambda cls: cls.name)
def fitted_model(request, config, synthetic_ratings):
    model = request.param(config)
    model.fit(synthetic_ratings)
    return model


def test_fit_returns_self(config, synthetic_ratings):
    for model_class in MODEL_CLASSES:
        model = model_class(config)
        assert model.fit(synthetic_ratings) is model


def test_predict_before_fit_raises(config):
    for model_class in MODEL_CLASSES:
        model = model_class(config)
        with pytest.raises(RuntimeError):
            model.predict(1, np.array([1, 2, 3]))


def test_predict_returns_one_score_per_candidate(fitted_model):
    candidates = np.array([1, 2, 3, 4], dtype=np.int64)
    scores = fitted_model.predict(1, candidates)
    assert len(scores) == len(candidates)
    assert np.asarray(scores).dtype.kind == "f"


def test_predictions_stay_inside_the_rating_range(fitted_model, synthetic_ratings):
    items = np.sort(synthetic_ratings["item_id"].unique())
    for user_id in synthetic_ratings["user_id"].unique():
        scores = fitted_model.predict(user_id, items)
        assert np.all(scores >= fitted_model.rating_min)
        assert np.all(scores <= fitted_model.rating_max)


def test_unknown_user_falls_back_to_something_sane(fitted_model):
    candidates = np.array([1, 2, 3], dtype=np.int64)
    scores = fitted_model.predict(UNKNOWN_USER, candidates)
    assert len(scores) == 3
    assert np.all(np.isfinite(scores))


def test_unknown_item_does_not_crash(fitted_model):
    candidates = np.array([1, UNKNOWN_ITEM], dtype=np.int64)
    scores = fitted_model.predict(1, candidates)
    assert len(scores) == 2
    assert np.all(np.isfinite(scores))


def test_recommend_returns_k_items(fitted_model):
    recommendations = fitted_model.recommend(1, 5)
    assert len(recommendations) == 5
    assert len(set(recommendations.tolist())) == 5


def test_recommend_excludes_training_items(fitted_model, synthetic_ratings):
    for user_id in synthetic_ratings["user_id"].unique():
        seen = set(synthetic_ratings[synthetic_ratings["user_id"] == user_id]["item_id"])
        recommendations = fitted_model.recommend(user_id, 5)
        assert set(recommendations.tolist()).isdisjoint(seen)


def test_recommend_caps_at_the_candidate_count(fitted_model):
    recommendations = fitted_model.recommend(1, 10000)
    assert len(recommendations) == len(fitted_model.candidate_items(1))


def test_recommend_is_deterministic(fitted_model):
    first = fitted_model.recommend(3, 8)
    second = fitted_model.recommend(3, 8)
    assert np.array_equal(first, second)


def test_recommendation_ties_break_on_item_id(config, synthetic_ratings):
    """GlobalMean scores everything identically, so only the tie-break orders the list."""
    model = GlobalMean(config).fit(synthetic_ratings)
    recommendations = model.recommend(1, 5)
    assert list(recommendations) == sorted(recommendations)


def test_most_popular_ranks_by_training_count(config, synthetic_ratings):
    model = MostPopular(config).fit(synthetic_ratings)
    counts = synthetic_ratings.groupby("item_id").size()
    recommendations = model.recommend(1, 4)
    ranked_counts = [counts.get(int(item), 0) for item in recommendations]
    assert ranked_counts == sorted(ranked_counts, reverse=True)


def test_most_popular_matches_global_mean_on_rating_prediction(config, synthetic_ratings):
    popular = MostPopular(config).fit(synthetic_ratings)
    globals_ = GlobalMean(config).fit(synthetic_ratings)
    candidates = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    assert np.allclose(popular.predict(1, candidates), globals_.predict(1, candidates))


def test_shrinkage_pulls_sparse_biases_towards_zero(config, synthetic_ratings):
    model = UserMean(config).fit(synthetic_ratings)
    for user_id, bias in model.user_bias.items():
        raw = synthetic_ratings[synthetic_ratings["user_id"] == user_id]["rating"].mean()
        raw_bias = raw - model.global_mean
        assert abs(bias) <= abs(raw_bias) + 1e-9

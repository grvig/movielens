"""Conformance tests for the shared recommender interface.

Every model in ``MODEL_FACTORIES`` is put through the same checks. The point of the project
is that the harness treats all models identically, so a model that needs an exception here
is a model that would quietly invalidate the comparison.

Models take different constructor arguments - the content model needs the item table, the
collaborative models do not - so each entry is a factory taking ``(config, items)`` rather
than a bare class. The constructor is where model-specific setup is allowed to live; the
``fit`` / ``predict`` / ``recommend`` surface is where it is not.
"""

import numpy as np
import pytest

from src.models.baselines import GlobalMean
from src.models.baselines import ItemMean
from src.models.baselines import MostPopular
from src.models.baselines import UserMean
from src.models.content import ContentBased
from src.models.item_knn import ItemKNN
from src.models.mf import MatrixFactorization

UNKNOWN_USER = 9999
UNKNOWN_ITEM = 9999

MODEL_FACTORIES = {
    "global_mean": lambda config, items: GlobalMean(config),
    "user_mean": lambda config, items: UserMean(config),
    "item_mean": lambda config, items: ItemMean(config),
    "most_popular": lambda config, items: MostPopular(config),
    "content": lambda config, items: ContentBased(config, items),
    "item_knn": lambda config, items: ItemKNN(config),
    "item_knn_ranking": lambda config, items: ItemKNN(
        config, name="item_knn_ranking", **config.model_params("item_knn_ranking")
    ),
    "mf": lambda config, items: MatrixFactorization(config, n_epochs=5),
    "mf_ranking": lambda config, items: MatrixFactorization(
        config, name="mf_ranking", n_epochs=5
    ),
}


@pytest.fixture(params=sorted(MODEL_FACTORIES.keys()))
def model_factory(request, config, synthetic_items):
    def build():
        return MODEL_FACTORIES[request.param](config, synthetic_items)
    return build


@pytest.fixture
def fitted_model(model_factory, synthetic_ratings):
    return model_factory().fit(synthetic_ratings)


def test_fit_returns_self(model_factory, synthetic_ratings):
    model = model_factory()
    assert model.fit(synthetic_ratings) is model


def test_model_declares_a_name(fitted_model):
    assert isinstance(fitted_model.name, str)
    assert fitted_model.name != "recommender"


def test_predict_before_fit_raises(model_factory):
    model = model_factory()
    with pytest.raises(RuntimeError):
        model.predict(1, np.array([1, 2, 3]))


def test_predict_returns_one_float_score_per_candidate(fitted_model):
    candidates = np.array([1, 2, 3, 4], dtype=np.int64)
    scores = np.asarray(fitted_model.predict(1, candidates))
    assert scores.shape == (4,)
    assert scores.dtype.kind == "f"


def test_predictions_stay_inside_the_rating_range(fitted_model, synthetic_ratings):
    items = np.sort(synthetic_ratings["item_id"].unique())
    for user_id in synthetic_ratings["user_id"].unique():
        scores = fitted_model.predict(user_id, items)
        assert np.all(scores >= fitted_model.rating_min)
        assert np.all(scores <= fitted_model.rating_max)
        assert np.all(np.isfinite(scores))


def test_predict_accepts_an_empty_candidate_list(fitted_model):
    scores = np.asarray(fitted_model.predict(1, np.array([], dtype=np.int64)))
    assert len(scores) == 0


def test_unknown_user_is_handled(fitted_model):
    candidates = np.array([1, 2, 3], dtype=np.int64)
    scores = fitted_model.predict(UNKNOWN_USER, candidates)
    assert len(scores) == 3
    assert np.all(np.isfinite(scores))


def test_unknown_item_is_handled(fitted_model):
    candidates = np.array([1, UNKNOWN_ITEM], dtype=np.int64)
    scores = fitted_model.predict(1, candidates)
    assert len(scores) == 2
    assert np.all(np.isfinite(scores))


def test_rank_scores_returns_one_score_per_candidate(fitted_model):
    candidates = np.array([1, 2, 3, 4], dtype=np.int64)
    scores = np.asarray(fitted_model.rank_scores(1, candidates))
    assert scores.shape == (4,)
    assert np.all(np.isfinite(scores))


def test_recommend_returns_k_distinct_items(fitted_model):
    recommendations = fitted_model.recommend(1, 5)
    assert len(recommendations) == 5
    assert len(set(recommendations.tolist())) == 5


def test_recommend_excludes_training_items(fitted_model, synthetic_ratings):
    for user_id in synthetic_ratings["user_id"].unique():
        seen = set(synthetic_ratings[synthetic_ratings["user_id"] == user_id]["item_id"])
        recommendations = fitted_model.recommend(user_id, 5)
        assert set(recommendations.tolist()).isdisjoint(seen)


def test_recommend_only_returns_known_items(fitted_model):
    recommendations = fitted_model.recommend(1, 10)
    assert set(recommendations.tolist()).issubset(fitted_model.known_items)


def test_recommend_caps_at_the_candidate_count(fitted_model):
    recommendations = fitted_model.recommend(1, 10000)
    assert len(recommendations) == len(fitted_model.candidate_items(1))


def test_recommend_for_an_unknown_user_still_returns_items(fitted_model):
    recommendations = fitted_model.recommend(UNKNOWN_USER, 5)
    assert len(recommendations) == 5


def test_recommend_is_deterministic(fitted_model):
    assert np.array_equal(fitted_model.recommend(3, 8), fitted_model.recommend(3, 8))


def test_refitting_reproduces_the_same_recommendations(model_factory, synthetic_ratings):
    """Two independent fits on the same data must agree, or the seed is not doing its job."""
    first = model_factory().fit(synthetic_ratings)
    second = model_factory().fit(synthetic_ratings)
    assert np.array_equal(first.recommend(2, 10), second.recommend(2, 10))
    candidates = first.candidate_items(2)
    assert np.allclose(first.predict(2, candidates), second.predict(2, candidates))


def test_global_mean_ties_break_on_item_id(config, synthetic_ratings):
    """GlobalMean scores everything identically, so only the tie-break orders the list."""
    model = GlobalMean(config).fit(synthetic_ratings)
    recommendations = model.recommend(1, 5)
    assert list(recommendations) == sorted(recommendations)


def test_most_popular_ranks_by_training_count(config, synthetic_ratings):
    model = MostPopular(config).fit(synthetic_ratings)
    counts = synthetic_ratings.groupby("item_id").size()
    recommendations = model.recommend(1, 4)
    ranked = [counts.get(int(item), 0) for item in recommendations]
    assert ranked == sorted(ranked, reverse=True)


def test_most_popular_matches_global_mean_on_rating_prediction(config, synthetic_ratings):
    popular = MostPopular(config).fit(synthetic_ratings)
    baseline = GlobalMean(config).fit(synthetic_ratings)
    candidates = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    assert np.allclose(popular.predict(1, candidates), baseline.predict(1, candidates))


def test_shrinkage_pulls_sparse_biases_towards_zero(config, synthetic_ratings):
    model = UserMean(config).fit(synthetic_ratings)
    for user_id, bias in model.user_bias.items():
        raw = synthetic_ratings[synthetic_ratings["user_id"] == user_id]["rating"].mean()
        assert abs(bias) <= abs(raw - model.global_mean) + 1e-9


def test_every_model_family_is_covered():
    """A new model family that is not registered here is not actually being checked."""
    assert len(MODEL_FACTORIES) == 9
    assert set(MODEL_FACTORIES.keys()) == {
        "global_mean", "user_mean", "item_mean", "most_popular", "content",
        "item_knn", "item_knn_ranking", "mf", "mf_ranking",
    }

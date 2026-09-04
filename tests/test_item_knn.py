"""Tests for the item-based kNN model."""

import numpy as np
import pandas as pd
import pytest

from src.models.item_knn import ItemKNN
from src.models.item_knn import weighted_deviation


@pytest.fixture
def agreeing_ratings():
    """Items 1 and 2 move together; item 3 moves against them.

    User 4 has rated item 1 highly and item 3 poorly, so a neighbourhood model should
    predict a high rating for item 2.
    """
    rows = [
        (1, 1, 5.0, 100), (1, 2, 5.0, 101), (1, 3, 1.0, 102),
        (2, 1, 4.0, 100), (2, 2, 4.0, 101), (2, 3, 2.0, 102),
        (3, 1, 2.0, 100), (3, 2, 2.0, 101), (3, 3, 5.0, 102),
        (4, 1, 5.0, 100), (4, 3, 1.0, 102),
    ]
    return pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "timestamp"])


def test_predicts_high_for_an_item_similar_to_one_the_user_liked(config, agreeing_ratings):
    model = ItemKNN(config, neighbourhood_size=5, shrinkage=0.0).fit(agreeing_ratings)
    prediction = model.predict(4, np.array([2]))[0]
    assert prediction > model.item_means[1]


def test_a_user_who_dislikes_the_cluster_gets_a_low_prediction(config):
    rows = [
        (1, 1, 5.0, 100), (1, 2, 5.0, 101),
        (2, 1, 4.0, 100), (2, 2, 4.0, 101),
        (3, 1, 1.0, 100), (3, 2, 1.0, 101),
        (4, 1, 1.0, 100),
    ]
    ratings = pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "timestamp"])
    model = ItemKNN(config, neighbourhood_size=5, shrinkage=0.0).fit(ratings)
    assert model.predict(4, np.array([2]))[0] < model.item_means[1]


def test_predictions_stay_in_the_rating_range(config, agreeing_ratings):
    model = ItemKNN(config, neighbourhood_size=5, shrinkage=0.0).fit(agreeing_ratings)
    predictions = model.predict(4, np.array([1, 2, 3]))
    assert np.all(predictions >= model.rating_min)
    assert np.all(predictions <= model.rating_max)


def test_unknown_user_falls_back_to_item_means(config, agreeing_ratings):
    model = ItemKNN(config, neighbourhood_size=5, shrinkage=0.0).fit(agreeing_ratings)
    predictions = model.predict(999, np.array([1, 2, 3]))
    assert np.allclose(predictions, model.item_means[[0, 1, 2]])


def test_unknown_item_falls_back_to_the_global_mean(config, agreeing_ratings):
    model = ItemKNN(config, neighbourhood_size=5, shrinkage=0.0).fit(agreeing_ratings)
    assert model.predict(4, np.array([9999]))[0] == pytest.approx(model.global_mean)


def test_both_centring_modes_run_and_agree_on_direction(config, agreeing_ratings):
    item_centred = ItemKNN(config, neighbourhood_size=5, shrinkage=0.0,
                           centering="item").fit(agreeing_ratings)
    user_centred = ItemKNN(config, neighbourhood_size=5, shrinkage=0.0,
                           centering="user").fit(agreeing_ratings)
    assert item_centred.predict(4, np.array([2]))[0] > item_centred.item_means[1]
    assert user_centred.predict(4, np.array([2]))[0] > user_centred.item_means[1]


def test_deviations_use_the_centring_means(config, agreeing_ratings):
    """Item centring subtracts a per-item mean; user centring subtracts a per-user one."""
    item_centred = ItemKNN(config, centering="item").fit(agreeing_ratings)
    positions, deviations = item_centred.deviations_for(1)
    expected = np.array([5.0, 5.0, 1.0]) - item_centred.centre_means[positions]
    assert np.allclose(np.sort(deviations), np.sort(expected))

    user_centred = ItemKNN(config, centering="user").fit(agreeing_ratings)
    _, user_deviations = user_centred.deviations_for(1)
    # user 1 rated 5, 5, 1 so the mean is 11 / 3 and deviations must sum to zero
    assert float(np.sum(user_deviations)) == pytest.approx(0.0, abs=1e-9)


def test_uniform_shrinkage_cancels_out(config, agreeing_ratings):
    """When every pair has the same co-raters, shrinkage changes nothing.

    The prediction divides by the sum of absolute similarities, so scaling every
    similarity by the same factor leaves the ratio untouched. Shrinkage is a
    re-weighting between neighbours, not a pull towards the item mean, and expecting the
    latter is an easy way to mis-read what this hyperparameter does.
    """
    unshrunk = ItemKNN(config, neighbourhood_size=5, shrinkage=0.0).fit(agreeing_ratings)
    shrunk = ItemKNN(config, neighbourhood_size=5, shrinkage=1000.0).fit(agreeing_ratings)
    assert shrunk.predict(4, np.array([2]))[0] == pytest.approx(
        unshrunk.predict(4, np.array([2]))[0]
    )


def test_shrinkage_favours_the_better_supported_neighbour(config):
    """Items 1 and 2 share six co-raters; items 2 and 3 share only two.

    User 7 liked item 1 and disliked item 3, so the two neighbours pull the prediction for
    item 2 in opposite directions. Shrinkage discounts the thinly-supported pair harder, so
    the prediction has to move towards what the well-supported neighbour implies.
    """
    rows = []
    paired = [(1, 5.0), (2, 4.0), (3, 2.0), (4, 1.0), (5, 5.0), (6, 1.0)]
    for user_id, rating in paired:
        rows.append((user_id, 1, rating, 100))
        rows.append((user_id, 2, rating, 101))
    rows.append((1, 3, 5.0, 102))
    rows.append((2, 3, 4.0, 102))
    rows.append((7, 1, 5.0, 100))
    rows.append((7, 3, 1.0, 102))
    ratings = pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "timestamp"])

    unshrunk = ItemKNN(config, neighbourhood_size=5, shrinkage=0.0).fit(ratings)
    shrunk = ItemKNN(config, neighbourhood_size=5, shrinkage=10.0).fit(ratings)
    assert shrunk.predict(7, np.array([2]))[0] > unshrunk.predict(7, np.array([2]))[0]


def test_neighbourhood_size_is_respected(config, agreeing_ratings):
    """A neighbourhood of one may only use the single most similar rated item."""
    model = ItemKNN(config, neighbourhood_size=1, shrinkage=0.0).fit(agreeing_ratings)
    prediction = model.predict(4, np.array([2]))[0]
    assert np.isfinite(prediction)


def test_recommendations_exclude_training_items(config, agreeing_ratings):
    model = ItemKNN(config, neighbourhood_size=5, shrinkage=0.0).fit(agreeing_ratings)
    recommendations = model.recommend(4, 5)
    assert set(recommendations.tolist()).isdisjoint({1, 3})


def test_ranking_matches_prediction_order(config, agreeing_ratings):
    model = ItemKNN(config, neighbourhood_size=5, shrinkage=0.0).fit(agreeing_ratings)
    candidates = model.candidate_items(4)
    assert np.allclose(
        model.rank_scores(4, candidates), model.predict(4, candidates)
    )


def test_fitting_twice_gives_identical_predictions(config, agreeing_ratings):
    first = ItemKNN(config).fit(agreeing_ratings).predict(4, np.array([2]))
    second = ItemKNN(config).fit(agreeing_ratings).predict(4, np.array([2]))
    assert np.allclose(first, second)


def test_weighted_deviation_is_a_similarity_weighted_mean():
    # (0.5 * 2.0 + 0.5 * -1.0) / (0.5 + 0.5) = 0.5
    deviations = {0: 2.0, 1: -1.0}
    result = weighted_deviation(np.array([0, 1]), np.array([0.5, 0.5]), deviations)
    assert result == pytest.approx(0.5)


def test_weighted_deviation_with_no_neighbours_is_zero():
    assert weighted_deviation(np.array([]), np.array([]), {}) == 0.0


def test_a_single_five_star_rating_does_not_win_the_ranking(config):
    """The failure found on the first full run against the real dataset.

    An item rated once at 5.0 has a raw mean of exactly 5.0 and outranks a genuinely
    well-liked item with hundreds of ratings. Eight of the top ten for user 1 were items
    with a single rating. The base term has to be shrunk, exactly as ItemMean shrinks.
    """
    rows = []
    for user_id in range(1, 31):
        # A genuinely well-liked item, and a mass of poorly-rated ones so that the overall
        # mean sits well below it. Without that contrast, shrinking towards the overall
        # mean has nothing to pull against and the test proves nothing.
        rows.append((user_id, 1, 4.5, 100))
        rows.append((user_id, 4, 2.0, 101))
    rows.append((1, 2, 5.0, 102))
    rows.append((31, 4, 2.0, 103))
    ratings = pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "timestamp"])

    unshrunk = ItemKNN(config, mean_shrinkage=0.0).fit(ratings)
    assert unshrunk.recommend(31, 1)[0] == 2

    shrunk = ItemKNN(config, mean_shrinkage=10.0).fit(ratings)
    assert shrunk.recommend(31, 1)[0] == 1


def test_mean_shrinkage_matches_the_item_mean_baseline(config, agreeing_ratings):
    """Both models estimate an item's level, so both must regularise it the same way."""
    from src.models.baselines import ItemMean

    knn = ItemKNN(config).fit(agreeing_ratings)
    baseline = ItemMean(config).fit(agreeing_ratings)
    assert knn.mean_shrinkage == pytest.approx(baseline.shrinkage)


def test_knn_runs_on_the_shared_fixture(config, synthetic_ratings):
    model = ItemKNN(config).fit(synthetic_ratings)
    predictions = model.predict(1, np.array([1, 2, 3]))
    assert np.all(np.isfinite(predictions))
    assert len(model.recommend(1, 5)) == 5

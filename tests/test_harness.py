"""Tests for the evaluation harness.

The harness has to treat every model identically, so these tests check the shape and the
protocol rather than any particular model's numbers.
"""

import numpy as np
import pytest

from src.data.splitting import temporal_split
from src.evaluation.harness import RESULT_COLUMNS
from src.evaluation.harness import evaluate_model
from src.evaluation.harness import predict_holdout
from src.evaluation.harness import results_frame
from src.evaluation.harness import safe_mean
from src.models.baselines import GlobalMean
from src.models.baselines import ItemMean
from src.models.baselines import MostPopular
from src.models.baselines import UserMean

MODEL_CLASSES = [GlobalMean, UserMean, ItemMean, MostPopular]


@pytest.fixture(scope="module")
def split(config, synthetic_ratings):
    return temporal_split(synthetic_ratings, config)


@pytest.fixture(params=MODEL_CLASSES, ids=lambda cls: cls.name)
def evaluated(request, config, split):
    train, _, test = split
    model = request.param(config).fit(train)
    rows = evaluate_model(model, train, test, config, "test")
    return model, rows


def test_every_model_produces_the_same_row_schema(evaluated):
    _, rows = evaluated
    frame = results_frame(rows)
    assert list(frame.columns) == RESULT_COLUMNS
    assert len(frame) > 0


def test_all_metrics_are_reported_at_all_k(evaluated, config):
    _, rows = evaluated
    frame = results_frame(rows)
    k_values = list(config.section("evaluation")["k_values"])
    ranking_metrics = ["precision_at_k", "recall_at_k", "coverage", "gini"]
    for metric in ranking_metrics:
        reported = sorted(frame[frame["metric"] == metric]["k"].tolist())
        assert reported == sorted(k_values)
    for metric in ["rmse", "mae"]:
        assert len(frame[frame["metric"] == metric]) == 1


def test_metric_values_are_in_range(evaluated):
    _, rows = evaluated
    frame = results_frame(rows)
    bounded = ["precision_at_k", "recall_at_k", "coverage", "popularity_percentile"]
    for metric in bounded:
        values = frame[frame["metric"] == metric]["value"].to_numpy(dtype=np.float64)
        usable = values[~np.isnan(values)]
        assert np.all(usable >= 0.0)
        assert np.all(usable <= 1.0)


def test_rmse_is_positive_and_finite(evaluated):
    _, rows = evaluated
    frame = results_frame(rows)
    value = frame[frame["metric"] == "rmse"]["value"].iloc[0]
    assert np.isfinite(value)
    assert value > 0.0


def test_evaluation_is_deterministic(config, split):
    train, _, test = split
    first = results_frame(evaluate_model(ItemMean(config).fit(train), train, test, config, "test"))
    second = results_frame(evaluate_model(ItemMean(config).fit(train), train, test, config, "test"))
    assert np.allclose(
        first["value"].to_numpy(dtype=np.float64),
        second["value"].to_numpy(dtype=np.float64),
        equal_nan=True,
    )


def test_context_columns_are_carried_through(config, split):
    train, _, test = split
    model = UserMean(config).fit(train)
    rows = evaluate_model(model, train, test, config, "test", variant="tuned", run_id="abc")
    frame = results_frame(rows)
    assert set(frame["model"]) == {"user_mean"}
    assert set(frame["variant"]) == {"tuned"}
    assert set(frame["split"]) == {"test"}
    assert set(frame["run_id"]) == {"abc"}
    assert set(frame["seed"]) == {config.seed}


def test_unfitted_model_is_rejected(config, split):
    train, _, test = split
    with pytest.raises(RuntimeError):
        evaluate_model(GlobalMean(config), train, test, config, "test")


def test_predict_holdout_covers_every_held_out_rating(config, split):
    train, _, test = split
    model = ItemMean(config).fit(train)
    predictions = predict_holdout(model, test)
    assert len(predictions) == len(test)
    assert np.all(np.isfinite(predictions["predicted"].to_numpy(dtype=np.float64)))


def test_recommendations_never_include_training_items(config, split):
    train, _, test = split
    model = MostPopular(config).fit(train)
    for user_id in test["user_id"].unique():
        seen = set(train[train["user_id"] == user_id]["item_id"])
        assert set(model.recommend(user_id, 10).tolist()).isdisjoint(seen)


def test_safe_mean_ignores_users_with_no_relevant_items():
    assert safe_mean([1.0, float("nan"), 3.0]) == pytest.approx(2.0)
    assert np.isnan(safe_mean([float("nan")]))

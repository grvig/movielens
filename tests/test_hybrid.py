"""Tests for the weighted hybrids.

The property that matters most is normalisation. The two components score on completely
different scales, so a blend of raw scores would be dominated by whichever has the larger
spread — and it would look perfectly plausible while being wrong.
"""

import numpy as np
import pandas as pd
import pytest

from src.models.content import ContentBased
from src.models.hybrid import WeightedHybrid
from src.models.hybrid import fit_weight_curve
from src.models.hybrid import normalise_scores
from src.models.hybrid import weight_for_history
from src.models.mf import MatrixFactorization


def build(config, items, **kwargs):
    return WeightedHybrid(
        config,
        ContentBased(config, items),
        MatrixFactorization(config, n_epochs=5),
        **kwargs,
    )


def test_normalisation_gives_zero_mean_and_unit_spread():
    normalised = normalise_scores([1.0, 2.0, 3.0, 4.0])
    assert float(np.mean(normalised)) == pytest.approx(0.0, abs=1e-9)
    assert float(np.std(normalised)) == pytest.approx(1.0)


def test_normalisation_puts_different_scales_on_one_axis():
    """Cosines in [-1, 1] and ratings in [1, 5] must become comparable."""
    cosines = normalise_scores([-0.5, 0.0, 0.5])
    ratings = normalise_scores([2.0, 3.0, 4.0])
    assert np.allclose(cosines, ratings)


def test_constant_scores_normalise_to_zero_not_nan():
    """A model with nothing to say about a user must contribute nothing, not NaN."""
    normalised = normalise_scores([3.0, 3.0, 3.0])
    assert np.all(normalised == 0.0)
    assert np.all(np.isfinite(normalised))


def test_empty_scores_are_handled():
    assert len(normalise_scores([])) == 0


def test_weight_curve_recovers_a_planted_relationship():
    lengths = np.array([5, 20, 60, 150, 400], dtype=np.float64)
    intercept, slope = 2.0, -0.5
    weights = 1.0 / (1.0 + np.exp(-(intercept + slope * np.log1p(lengths))))
    fitted_intercept, fitted_slope = fit_weight_curve(lengths, weights)
    assert fitted_intercept == pytest.approx(intercept, abs=0.05)
    assert fitted_slope == pytest.approx(slope, abs=0.05)


def test_weight_curve_handles_extreme_bin_optima():
    """Optima of exactly 0 or 1 have infinite logits and must be clipped, not crash."""
    intercept, slope = fit_weight_curve([5, 50, 500], [0.0, 0.0, 0.0])
    assert np.isfinite(intercept)
    assert np.isfinite(slope)


def test_weight_curve_with_one_point_is_flat():
    assert fit_weight_curve([10], [0.5]) == (0.0, 0.0)


def test_weight_for_history_is_bounded():
    for history in [0, 1, 10, 1000]:
        weight = weight_for_history(history, 3.0, -1.0)
        assert 0.0 <= weight <= 1.0


def test_a_negative_slope_means_less_content_as_history_grows():
    sparse = weight_for_history(1, 2.0, -0.8)
    dense = weight_for_history(500, 2.0, -0.8)
    assert sparse > dense


def test_fixed_mode_uses_the_configured_weight(config, synthetic_ratings, synthetic_items):
    model = build(config, synthetic_items, weight_mode="fixed", fixed_weight=0.3)
    model.fit(synthetic_ratings)
    assert model.weight_for(1) == pytest.approx(0.3)
    assert model.weight_for(2) == pytest.approx(0.3)


def test_weight_zero_reproduces_the_collaborative_component(
    config, synthetic_ratings, synthetic_items
):
    model = build(config, synthetic_items, weight_mode="fixed", fixed_weight=0.0)
    model.fit(synthetic_ratings)
    candidates = model.candidate_items(1)
    assert np.allclose(
        model.predict(1, candidates), model.cf_model.predict(1, candidates)
    )


def test_weight_one_reproduces_the_content_component(
    config, synthetic_ratings, synthetic_items
):
    model = build(config, synthetic_items, weight_mode="fixed", fixed_weight=1.0)
    model.fit(synthetic_ratings)
    candidates = model.candidate_items(1)
    assert np.allclose(
        model.predict(1, candidates), model.content_model.predict(1, candidates)
    )


def test_blended_ranking_sits_between_its_components(
    config, synthetic_ratings, synthetic_items
):
    model = build(config, synthetic_items, weight_mode="fixed", fixed_weight=0.5)
    model.fit(synthetic_ratings)
    candidates = model.candidate_items(1)
    content, collaborative = model.component_rank_scores(1, candidates)
    blended = model.rank_scores(1, candidates)
    lower = np.minimum(content, collaborative)
    upper = np.maximum(content, collaborative)
    assert np.all(blended >= lower - 1e-9)
    assert np.all(blended <= upper + 1e-9)


def test_predictions_stay_in_the_rating_range(config, synthetic_ratings, synthetic_items):
    model = build(config, synthetic_items, weight_mode="fixed", fixed_weight=0.5)
    model.fit(synthetic_ratings)
    predictions = model.predict(1, model.candidate_items(1))
    assert np.all(predictions >= model.rating_min)
    assert np.all(predictions <= model.rating_max)


def test_an_unknown_weight_mode_is_rejected(config, synthetic_items):
    with pytest.raises(ValueError, match="weight_mode"):
        build(config, synthetic_items, weight_mode="sideways")


def test_density_curve_records_a_point_per_bin(config, synthetic_ratings, synthetic_items):
    validation = synthetic_ratings.copy()
    model = build(config, synthetic_items, weight_mode="density", validation=validation)
    model.fit(synthetic_ratings)
    assert len(model.curve_points) >= 1
    for point in model.curve_points:
        assert 0.0 <= point["best_weight"] <= 1.0
        assert point["users"] > 0


def test_density_mode_without_validation_stays_neutral(
    config, synthetic_ratings, synthetic_items
):
    model = build(config, synthetic_items, weight_mode="density")
    model.fit(synthetic_ratings)
    assert model.intercept == 0.0
    assert model.slope == 0.0
    assert model.weight_for(1) == pytest.approx(0.5)


def test_hybrid_recommends_and_excludes_training_items(
    config, synthetic_ratings, synthetic_items
):
    model = build(config, synthetic_items, weight_mode="fixed", fixed_weight=0.5)
    model.fit(synthetic_ratings)
    seen = set(synthetic_ratings[synthetic_ratings["user_id"] == 1]["item_id"])
    recommendations = model.recommend(1, 5)
    assert len(recommendations) == 5
    assert set(recommendations.tolist()).isdisjoint(seen)


def test_fingerprint_is_stable_for_unchanged_config(config):
    from src.models.hybrid import component_fingerprint

    assert component_fingerprint(config) == component_fingerprint(config)


def test_fingerprint_changes_when_a_component_is_retuned(config):
    from src.models.hybrid import component_fingerprint

    before = component_fingerprint(config)
    config.values["models"]["mf"]["regularisation"] = 0.42
    assert component_fingerprint(config) != before


def test_fingerprint_changes_when_the_features_change(config):
    from src.models.hybrid import component_fingerprint

    before = component_fingerprint(config)
    config.values["features"]["min_df"] = 99
    assert component_fingerprint(config) != before


def test_matching_fingerprint_passes(config):
    from src.models.hybrid import check_fingerprint
    from src.models.hybrid import component_fingerprint

    check_fingerprint(config, component_fingerprint(config))


def test_a_stale_fingerprint_fails_loudly(config):
    from src.models.hybrid import check_fingerprint

    with pytest.raises(ValueError, match="run_hybrid"):
        check_fingerprint(config, "deadbeefdeadbeef")


def test_an_absent_fingerprint_is_not_enforced(config):
    """Older configs without the field must keep working."""
    from src.models.hybrid import check_fingerprint

    check_fingerprint(config, None)
    check_fingerprint(config, "")


def test_the_committed_config_fingerprint_is_current(config):
    """The recorded constants must match the components actually configured."""
    from src.models.hybrid import component_fingerprint

    recorded = config.model_params("hybrid")["components"]
    assert recorded == component_fingerprint(config)

"""Tests for the hybrid fitting and frontier experiments."""

import numpy as np
import pandas as pd
import pytest

from src.data.splitting import load_splits
from src.data.splitting import temporal_split
from src.data.splitting import write_splits
from src.experiments.run_hybrid import build_hybrid
from src.experiments.run_hybrid import choose_frontier_weight
from src.experiments.run_hybrid import run_density
from src.experiments.run_hybrid import value_of
from src.models.hybrid import WEIGHT_GRID


@pytest.fixture
def prepared_config(config, tmp_path, synthetic_ratings, synthetic_items):
    processed = tmp_path / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    config.values["paths"]["processed_dir"] = str(processed)
    config.values["paths"]["results_dir"] = str(tmp_path / "results")
    config.values["paths"]["fits_dir"] = str(tmp_path / "fits")
    config.values["models"]["mf"]["n_epochs"] = 3
    train, validation, test = temporal_split(synthetic_ratings, config)
    write_splits(train, validation, test, processed)
    synthetic_items.to_parquet(processed / "items.parquet", index=False)
    return config


def test_weight_grid_spans_both_endpoints():
    assert WEIGHT_GRID[0] == 0.0
    assert WEIGHT_GRID[-1] == 1.0
    assert WEIGHT_GRID == sorted(WEIGHT_GRID)


def test_density_fit_returns_a_point_per_populated_bin(prepared_config, synthetic_items):
    train, validation, _ = load_splits(prepared_config.path("processed_dir"))
    model, frame = run_density(prepared_config, synthetic_items, train, validation)
    assert len(frame) >= 1
    for column in ["best_weight", "median_history", "fitted_weight", "fitted_slope"]:
        assert column in frame.columns
    assert np.all(frame["best_weight"] >= 0.0)
    assert np.all(frame["best_weight"] <= 1.0)


def test_the_fitted_curve_is_recorded_on_every_row(prepared_config, synthetic_items):
    train, validation, _ = load_splits(prepared_config.path("processed_dir"))
    model, frame = run_density(prepared_config, synthetic_items, train, validation)
    assert frame["fitted_intercept"].nunique() == 1
    assert frame["fitted_slope"].nunique() == 1
    assert float(frame["fitted_intercept"].iloc[0]) == pytest.approx(model.intercept)


def test_frontier_weight_choice_respects_the_tolerance():
    frontier = pd.DataFrame({
        "weight": [0.0, 0.2, 0.4, 0.6],
        "rmse": [0.95, 0.96, 0.97, 1.10],
    })
    # 0.95 + 0.02 = 0.97, so 0.4 qualifies and 0.6 does not
    assert choose_frontier_weight(frontier, tolerance=0.02) == pytest.approx(0.4)


def test_a_tight_tolerance_falls_back_to_pure_collaborative():
    frontier = pd.DataFrame({"weight": [0.0, 0.2], "rmse": [0.95, 1.50]})
    assert choose_frontier_weight(frontier, tolerance=0.001) == 0.0


def test_frontier_choice_never_exceeds_the_grid():
    frontier = pd.DataFrame({"weight": WEIGHT_GRID, "rmse": [0.95] * len(WEIGHT_GRID)})
    assert choose_frontier_weight(frontier) == 1.0


def test_value_of_selects_metric_and_k():
    frame = pd.DataFrame({
        "metric": ["rmse", "precision_at_k", "precision_at_k"],
        "k": [None, 5, 10],
        "value": [0.9, 0.01, 0.02],
    })
    assert value_of(frame, "rmse") == pytest.approx(0.9)
    assert value_of(frame, "precision_at_k", 10) == pytest.approx(0.02)


def test_fixed_endpoints_reduce_to_the_components(prepared_config, synthetic_items):
    """w=0 must equal pure CF and w=1 pure content, or the blend is miswired."""
    train, validation, _ = load_splits(prepared_config.path("processed_dir"))
    pure_cf = build_hybrid(prepared_config, synthetic_items, "fixed", 0.0, validation)
    pure_cf.fit(train)
    pure_content = build_hybrid(prepared_config, synthetic_items, "fixed", 1.0, validation)
    pure_content.fit(train)

    candidates = pure_cf.candidate_items(1)
    assert np.allclose(
        pure_cf.predict(1, candidates), pure_cf.cf_model.predict(1, candidates)
    )
    assert np.allclose(
        pure_content.predict(1, candidates),
        pure_content.content_model.predict(1, candidates),
    )


def test_density_fitting_never_reads_the_test_split(prepared_config, synthetic_items):
    """Corrupting test must not move the fitted curve."""
    train, validation, _ = load_splits(prepared_config.path("processed_dir"))
    before, _ = run_density(prepared_config, synthetic_items, train, validation)

    processed = prepared_config.path("processed_dir")
    test = pd.read_parquet(processed / "test.parquet")
    test["rating"] = 1.0
    test.to_parquet(processed / "test.parquet", index=False)

    after, _ = run_density(prepared_config, synthetic_items, train, validation)
    assert after.intercept == pytest.approx(before.intercept)
    assert after.slope == pytest.approx(before.slope)

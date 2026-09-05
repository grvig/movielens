"""Tests for the hyperparameter sweeps.

The property that matters most here is that selection happens on validation and never on
test. A sweep that picks its winner on test data reports a number that cannot be trusted,
and nothing about the output would look wrong.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.splitting import load_splits
from src.data.splitting import temporal_split
from src.data.splitting import write_splits
from src.evaluation.harness import results_frame
from src.experiments.run_tuning import best_configuration
from src.experiments.run_tuning import build_knn
from src.experiments.run_tuning import build_mf
from src.experiments.run_tuning import knn_configurations
from src.experiments.run_tuning import mf_configurations
from src.experiments.run_tuning import sweep
from src.experiments.run_tuning import variant_label


@pytest.fixture
def prepared_config(config, tmp_path, synthetic_ratings):
    processed = tmp_path / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    config.values["paths"]["processed_dir"] = str(processed)
    config.values["paths"]["results_dir"] = str(tmp_path / "results")
    config.values["paths"]["fits_dir"] = str(tmp_path / "fits")
    config.values["models"]["mf"]["n_epochs"] = 3
    train, validation, test = temporal_split(synthetic_ratings, config)
    write_splits(train, validation, test, processed)
    return config


def test_knn_grid_covers_every_combination():
    configurations = knn_configurations()
    assert len(configurations) == 6 * 4 * 2
    sizes = set()
    for parameters in configurations:
        sizes.add(parameters["neighbourhood_size"])
    assert sizes == {5, 10, 20, 40, 80, 160}


def test_mf_grid_covers_every_combination():
    assert len(mf_configurations()) == 4 * 4 * 2


def test_variant_labels_are_stable_and_unique():
    """Labels are the join key between the CSV and the report, so they must not collide."""
    labels = set()
    for parameters in knn_configurations():
        labels.add(variant_label(parameters))
    assert len(labels) == len(knn_configurations())
    assert variant_label({"b": 2, "a": 1}) == variant_label({"a": 1, "b": 2})


def test_sweep_records_one_variant_per_configuration(prepared_config):
    train, validation, _ = load_splits(prepared_config.path("processed_dir"))
    configurations = knn_configurations()[:3]
    rows = sweep(prepared_config, configurations, build_knn, train, validation, "run")
    frame = results_frame(rows)
    assert frame["variant"].nunique() == 3


def test_sweep_evaluates_on_validation_not_test(prepared_config):
    train, validation, _ = load_splits(prepared_config.path("processed_dir"))
    rows = sweep(prepared_config, knn_configurations()[:2], build_knn, train, validation,
                 "run")
    frame = results_frame(rows)
    assert set(frame["split"]) == {"val"}


def test_corrupting_test_does_not_change_the_sweep(prepared_config):
    """Direct evidence that selection never reads the test split."""
    train, validation, _ = load_splits(prepared_config.path("processed_dir"))
    configurations = knn_configurations()[:3]
    before = results_frame(
        sweep(prepared_config, configurations, build_knn, train, validation, "run")
    )

    processed = prepared_config.path("processed_dir")
    test = pd.read_parquet(processed / "test.parquet")
    test["rating"] = 1.0
    test.to_parquet(processed / "test.parquet", index=False)

    after = results_frame(
        sweep(prepared_config, configurations, build_knn, train, validation, "run")
    )
    assert np.allclose(
        before["value"].to_numpy(dtype=np.float64),
        after["value"].to_numpy(dtype=np.float64),
        equal_nan=True,
    )


def test_best_configuration_picks_the_lowest_rmse():
    frame = pd.DataFrame(
        {
            "variant": ["a", "b", "c"],
            "metric": ["rmse", "rmse", "rmse"],
            "k": [None, None, None],
            "value": [1.2, 0.9, 1.0],
        }
    )
    variant, value = best_configuration(frame)
    assert variant == "b"
    assert value == pytest.approx(0.9)


def test_different_knn_settings_give_different_results(prepared_config):
    train, validation, _ = load_splits(prepared_config.path("processed_dir"))
    configurations = [
        {"neighbourhood_size": 1, "shrinkage": 0.0, "centering": "item"},
        {"neighbourhood_size": 160, "shrinkage": 100.0, "centering": "user"},
    ]
    frame = results_frame(
        sweep(prepared_config, configurations, build_knn, train, validation, "run")
    )
    values = frame[frame["metric"] == "rmse"]["value"].tolist()
    assert values[0] != values[1]


def test_mf_builder_passes_validation_through_for_early_stopping(prepared_config):
    train, validation, _ = load_splits(prepared_config.path("processed_dir"))
    parameters = {"n_factors": 4, "regularisation": 0.05, "learning_rate": 0.01}
    model = build_mf(prepared_config, parameters, validation)
    assert model.validation is not None
    assert model.n_factors == 4
    assert model.cache is True

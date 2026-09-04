"""End-to-end test of the main experiment runner.

The real dataset is not committed, so this builds a miniature processed directory from the
synthetic fixture and drives the whole pipeline through it: split, fit every model,
evaluate, write the CSV. It is the only test that exercises the runner itself, and it
catches the class of bug where a model is fine in isolation but the runner wires it up
wrongly.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.splitting import temporal_split
from src.data.splitting import write_splits
from src.evaluation.harness import RESULT_COLUMNS
from src.evaluation.harness import results_frame
from src.experiments.registry import MODEL_ORDER
from src.experiments.registry import build_all
from src.experiments.registry import build_model
from src.experiments.run_main import load_items
from src.experiments.run_main import run


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


def test_runner_produces_rows_for_every_model(prepared_config):
    rows = run(prepared_config, cache=False)
    frame = results_frame(rows)
    assert set(frame["model"]) == set(MODEL_ORDER)
    assert list(frame.columns) == RESULT_COLUMNS


def test_all_models_share_one_run_id(prepared_config):
    """A results file mixing run ids would make the comparison unreproducible."""
    frame = results_frame(run(prepared_config, cache=False))
    assert frame["run_id"].nunique() == 1
    assert frame["seed"].nunique() == 1


def test_every_model_reports_the_same_metrics(prepared_config):
    frame = results_frame(run(prepared_config, cache=False))
    per_model = frame.groupby("model")["metric"].apply(lambda values: sorted(set(values)))
    reference = per_model.iloc[0]
    for metrics in per_model:
        assert metrics == reference


def test_metric_values_are_finite_and_bounded(prepared_config):
    frame = results_frame(run(prepared_config, cache=False))
    bounded = ["precision_at_k", "recall_at_k", "coverage", "popularity_percentile"]
    for metric in bounded:
        values = frame[frame["metric"] == metric]["value"].to_numpy(dtype=np.float64)
        usable = values[~np.isnan(values)]
        assert np.all(usable >= 0.0)
        assert np.all(usable <= 1.0)
    rmse = frame[frame["metric"] == "rmse"]["value"].to_numpy(dtype=np.float64)
    assert np.all(np.isfinite(rmse))
    assert np.all(rmse > 0.0)


def test_the_run_is_reproducible(prepared_config):
    first = results_frame(run(prepared_config, cache=False))
    second = results_frame(run(prepared_config, cache=False))
    assert np.allclose(
        first["value"].to_numpy(dtype=np.float64),
        second["value"].to_numpy(dtype=np.float64),
        equal_nan=True,
    )


def test_evaluation_never_reads_the_validation_split(prepared_config, synthetic_ratings):
    """Corrupting validation must not move a test-split number.

    Validation legitimately reaches two models, for calibration and early stopping. This
    checks it does not leak into the evaluation itself, where it would inflate results.
    """
    baseline = results_frame(run(prepared_config, cache=False))
    baseline = baseline[baseline["model"].isin(["item_mean", "item_knn", "most_popular"])]

    processed = prepared_config.path("processed_dir")
    validation = pd.read_parquet(processed / "val.parquet")
    corrupted = validation.copy()
    corrupted["rating"] = 1.0
    corrupted.to_parquet(processed / "val.parquet", index=False)

    after = results_frame(run(prepared_config, cache=False))
    after = after[after["model"].isin(["item_mean", "item_knn", "most_popular"])]
    assert np.allclose(
        baseline["value"].to_numpy(dtype=np.float64),
        after["value"].to_numpy(dtype=np.float64),
        equal_nan=True,
    )


def test_missing_items_file_gives_a_useful_error(config, tmp_path):
    config.values["paths"]["processed_dir"] = str(tmp_path / "empty")
    with pytest.raises(FileNotFoundError, match="preprocess"):
        load_items(config)


def test_registry_builds_every_model(config, synthetic_items):
    models = build_all(config, synthetic_items)
    assert len(models) == len(MODEL_ORDER)
    names = []
    for model in models:
        names.append(model.name)
    assert names == MODEL_ORDER


def test_registry_rejects_an_unknown_model(config, synthetic_items):
    with pytest.raises(ValueError, match="unknown model"):
        build_model("random_forest", config, synthetic_items)

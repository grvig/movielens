"""Tests for the main comparison and tuning figures."""

import numpy as np
import pandas as pd
import pytest

from src.experiments.registry import MODEL_ORDER
from src.plots.plot_main import draw_accuracy
from src.plots.plot_main import draw_tradeoff
from src.plots.plot_main import main as main_plot
from src.plots.plot_main import metric_series
from src.plots.plot_tuning import draw as draw_tuning
from src.plots.plot_tuning import sweep_table
from src.plots.style import apply_style
from src.plots.style import save


@pytest.fixture
def results_frame():
    rows = []
    for position, model in enumerate(MODEL_ORDER):
        rows.append({"model": model, "metric": "rmse", "k": None,
                     "value": 1.0 + position * 0.01})
        rows.append({"model": model, "metric": "mae", "k": None,
                     "value": 0.8 + position * 0.01})
        for metric, base in [("precision_at_k", 0.02), ("coverage", 0.2),
                             ("popularity_percentile", 0.6)]:
            rows.append({"model": model, "metric": metric, "k": 10,
                         "value": base + position * 0.01})
    return pd.DataFrame(rows)


@pytest.fixture
def sweep_frame():
    rows = []
    for index in range(12):
        variant = "config=" + str(index)
        rows.append({"variant": variant, "metric": "rmse", "k": None,
                     "value": 0.95 + index * 0.005})
        rows.append({"variant": variant, "metric": "precision_at_k", "k": 10,
                     "value": 0.015 + index * 0.001})
        rows.append({"variant": variant, "metric": "coverage", "k": 10,
                     "value": 0.14 + index * 0.002})
    return pd.DataFrame(rows)


def test_metric_series_covers_every_model(results_frame):
    series = metric_series(results_frame, "rmse")
    assert list(series.index) == MODEL_ORDER
    assert series.notna().all()


def test_metric_series_filters_on_k(results_frame):
    series = metric_series(results_frame, "precision_at_k", 10)
    assert len(series) == len(MODEL_ORDER)


def test_accuracy_figure_has_two_panels(results_frame):
    apply_style()
    figure = draw_accuracy(results_frame)
    assert len(figure.axes) == 2
    for axis in figure.axes:
        assert len(axis.get_yticklabels()) == len(MODEL_ORDER)


def test_accuracy_axes_are_inverted_so_best_is_top(results_frame):
    apply_style()
    figure = draw_accuracy(results_frame)
    for axis in figure.axes:
        bottom, top = axis.get_ylim()
        assert bottom > top


def test_tradeoff_figure_has_two_panels(results_frame):
    apply_style()
    figure = draw_tradeoff(results_frame)
    assert len(figure.axes) == 2


def test_tradeoff_rmse_axis_is_inverted(results_frame):
    """RMSE is better when lower, so the axis flips to keep "up" meaning "better"."""
    apply_style()
    figure = draw_tradeoff(results_frame)
    bottom, top = figure.axes[0].get_ylim()
    assert bottom > top


def test_tradeoff_reports_a_correlation_in_its_title(results_frame):
    apply_style()
    figure = draw_tradeoff(results_frame)
    assert "r = " in figure.axes[1].get_title()


def test_sweep_table_joins_the_three_metrics(sweep_frame):
    table = sweep_table(sweep_frame)
    assert len(table) == 12
    for column in ["rmse", "precision", "coverage"]:
        assert column in table.columns
    assert table.notna().all().all()


def test_tuning_panel_marks_both_selection_criteria(sweep_frame):
    apply_style()
    table = sweep_table(sweep_frame)
    figure = draw_tuning([(table, "sweep", "#26418f"), (table, "sweep", "#8c3025")])
    assert len(figure.axes) == 2
    for axis in figure.axes:
        labels = [text.get_text() for text in axis.get_legend().get_texts()]
        assert "picked on RMSE" in labels
        assert "picked on precision@10" in labels


def test_figures_write_non_empty_files(results_frame, tmp_path):
    apply_style()
    for name, figure in [("accuracy.pdf", draw_accuracy(results_frame)),
                         ("tradeoff.pdf", draw_tradeoff(results_frame))]:
        path = save(figure, tmp_path / name)
        assert path.stat().st_size > 0


def test_missing_results_file_gives_a_useful_error(config, tmp_path, monkeypatch):
    config.values["paths"]["results_dir"] = str(tmp_path / "nothing")
    monkeypatch.setattr("src.plots.plot_main.load_config", lambda path: config)
    monkeypatch.setattr("sys.argv", ["plot_main"])
    with pytest.raises(FileNotFoundError, match="run_main"):
        main_plot()

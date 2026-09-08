"""Tests for the plotting layer.

Figures are checked for the properties the report depends on - that a model keeps the same
colour across every figure, that plotting reads CSVs rather than refitting anything, and
that the script fails loudly when the results file is missing - rather than for pixels.
"""

import pandas as pd
import pytest

from src.experiments.registry import MODEL_ORDER
from src.plots.plot_cold_start import draw
from src.plots.plot_cold_start import main as cold_start_main
from src.plots.plot_cold_start import series_for
from src.plots.style import DASHED_MODELS
from src.plots.style import MODEL_COLOURS
from src.plots.style import apply_style
from src.plots.style import colour_for
from src.plots.style import label_for
from src.plots.style import linestyle_for
from src.plots.style import save


@pytest.fixture
def cold_start_frame():
    rows = []
    for model in MODEL_ORDER:
        for history in [1, 3, 5, 10, 20]:
            rows.append({"model": model, "metric": "rmse", "k": None,
                         "value": 1.2 - history * 0.005, "history": history})
            rows.append({"model": model, "metric": "precision_at_k", "k": 10,
                         "value": 0.01 + history * 0.001, "history": history})
            rows.append({"model": model, "metric": "coverage", "k": 10,
                         "value": 0.2 + history * 0.002, "history": history})
    return pd.DataFrame(rows)


def test_every_registered_model_has_a_colour():
    for name in MODEL_ORDER:
        assert name in MODEL_COLOURS


def test_ranking_variants_share_their_twins_colour():
    """Same algorithm, same hue; the dash is what separates them."""
    assert colour_for("mf_ranking") == colour_for("mf")
    assert colour_for("item_knn_ranking") == colour_for("item_knn")
    for name in DASHED_MODELS:
        assert linestyle_for(name) == "--"
    assert linestyle_for("mf") == "-"


def test_unknown_model_still_gets_a_colour_and_label():
    assert colour_for("something_new").startswith("#")
    assert label_for("something_new") == "something_new"


def test_every_registered_model_has_a_label():
    for name in MODEL_ORDER:
        assert label_for(name) != ""


def test_series_are_ordered_by_history(cold_start_frame):
    history, values = series_for(cold_start_frame, "mf", "rmse", None)
    assert list(history) == sorted(history)
    assert len(values) == len(history)


def test_series_filters_on_k(cold_start_frame):
    history, values = series_for(cold_start_frame, "mf", "precision_at_k", 10)
    assert len(history) == 5


def test_series_for_a_missing_model_is_empty(cold_start_frame):
    history, values = series_for(cold_start_frame, "nonexistent", "rmse", None)
    assert len(history) == 0


def test_draw_produces_one_panel_per_metric(cold_start_frame):
    apply_style()
    figure = draw(cold_start_frame, MODEL_ORDER)
    assert len(figure.axes) == 3
    for axis in figure.axes:
        assert axis.get_xscale() == "log"


def test_saving_writes_a_file(cold_start_frame, tmp_path):
    apply_style()
    figure = draw(cold_start_frame, MODEL_ORDER)
    path = save(figure, tmp_path / "figures" / "cold_start.pdf")
    assert path.exists()
    assert path.stat().st_size > 0


def test_missing_results_file_gives_a_useful_error(config, tmp_path, monkeypatch):
    config.values["paths"]["results_dir"] = str(tmp_path / "nothing")
    monkeypatch.setattr("src.plots.plot_cold_start.load_config", lambda path: config)
    monkeypatch.setattr("sys.argv", ["plot_cold_start"])
    with pytest.raises(FileNotFoundError, match="run_cold_start"):
        cold_start_main()

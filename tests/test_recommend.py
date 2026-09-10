"""Tests for the recommendation demo CLI."""

import pandas as pd
import pytest

from src.data.splitting import temporal_split
from src.data.splitting import write_splits
from src.experiments.recommend import build_item_lookup
from src.experiments.recommend import describe
from src.experiments.recommend import genre_list
from src.experiments.recommend import genre_text
from src.experiments.recommend import run
from tests.conftest import refresh_hybrid_fingerprint


@pytest.fixture
def prepared_config(config, tmp_path, synthetic_ratings, synthetic_items):
    processed = tmp_path / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    config.values["paths"]["processed_dir"] = str(processed)
    config.values["paths"]["results_dir"] = str(tmp_path / "results")
    config.values["paths"]["fits_dir"] = str(tmp_path / "fits")
    config.values["models"]["mf"]["n_epochs"] = 3
    refresh_hybrid_fingerprint(config)

    train, validation, test = temporal_split(synthetic_ratings, config)
    write_splits(train, validation, test, processed)
    synthetic_items.to_parquet(processed / "items.parquet", index=False)
    return config


def test_title_year_is_not_duplicated():
    """ML-100K release dates disagree with title years, which produced "Fargo (1996) (1997)".

    The title already carries a year, so nothing should be appended even though the
    release_year column says something different.
    """
    items = pd.DataFrame(
        [{"item_id": 1, "title": "Fargo (1996)", "release_year": 1997, "genre_Drama": 1}]
    )
    lookup = build_item_lookup(items)
    assert lookup[1]["label"] == "Fargo (1996)"


def test_year_is_appended_when_the_title_has_none():
    items = pd.DataFrame(
        [{"item_id": 1, "title": "Some Film", "release_year": 1994, "genre_Drama": 1}]
    )
    assert build_item_lookup(items)[1]["label"] == "Some Film (1994)"


def test_missing_year_is_left_off():
    items = pd.DataFrame(
        [{"item_id": 1, "title": "Unknown Film", "release_year": 0, "genre_Drama": 1}]
    )
    assert build_item_lookup(items)[1]["label"] == "Unknown Film"


def test_genres_are_read_from_the_flag_columns():
    row = pd.Series({"item_id": 1, "title": "x", "genre_Drama": 1,
                     "genre_Comedy": 0, "genre_War": 1})
    assert genre_list(row) == ["Drama", "War"]


def test_the_unknown_genre_is_not_displayed():
    """ML-100K has an "unknown" genre flag, which is noise on a slide."""
    row = pd.Series({"item_id": 1, "title": "x", "genre_unknown": 1, "genre_Drama": 1})
    assert genre_list(row) == ["Drama"]


def test_long_titles_are_truncated_to_the_column_width():
    items = pd.DataFrame(
        [{"item_id": 1, "title": "A" * 80, "release_year": 1990, "genre_Drama": 1}]
    )
    lookup = build_item_lookup(items)
    rendered = describe(lookup, 1, width=44)
    assert len(rendered) == 44
    assert rendered.endswith("...")


def test_short_titles_are_padded_to_the_column_width():
    items = pd.DataFrame(
        [{"item_id": 1, "title": "Ran (1985)", "release_year": 1985, "genre_Drama": 1}]
    )
    assert len(describe(build_item_lookup(items), 1, width=44)) == 44


def test_unknown_item_renders_without_crashing():
    assert "9999" in describe({}, 9999)
    assert genre_text({}, 9999) == ""


def test_genre_text_is_capped(monkeypatch):
    items = pd.DataFrame(
        [{"item_id": 1, "title": "x (1990)", "release_year": 1990,
          "genre_Drama": 1, "genre_War": 1, "genre_Romance": 1, "genre_Comedy": 1}]
    )
    lookup = build_item_lookup(items)
    assert genre_text(lookup, 1, limit=3).count("|") == 2


def test_run_reports_hit_counts_for_every_model(prepared_config, capsys):
    summary = run(prepared_config, user_id=1, k=5, model_names=["most_popular", "content"],
                  cache=False)
    assert set(summary.keys()) == {"most_popular", "content"}
    for hits in summary.values():
        assert 0 <= hits <= 5


def test_run_prints_titles_not_just_ids(prepared_config, capsys):
    run(prepared_config, user_id=1, k=5, model_names=["most_popular"], cache=False)
    output = capsys.readouterr().out
    assert "Movie" in output
    assert "training ratings" in output
    assert "held out" in output


def test_unknown_user_is_rejected(prepared_config):
    with pytest.raises(ValueError, match="not in the training split"):
        run(prepared_config, user_id=99999, k=5, model_names=["most_popular"], cache=False)


def test_hit_count_matches_the_relevant_set(prepared_config, capsys):
    """The reported hits must equal the overlap the metrics would compute."""
    from src.data.splitting import load_splits
    from src.evaluation.ranking import relevant_items_by_user
    from src.experiments.registry import build_model
    from src.experiments.run_main import load_items

    train, validation, test = load_splits(prepared_config.path("processed_dir"))
    items = load_items(prepared_config)
    threshold = float(prepared_config.section("evaluation")["relevance_threshold"])
    relevant = relevant_items_by_user(test, threshold)

    summary = run(prepared_config, user_id=1, k=5, model_names=["most_popular"], cache=False)
    model = build_model("most_popular", prepared_config, items).fit(train)
    expected = len(set(model.recommend(1, 5).tolist()) & relevant.get(1, set()))
    assert summary["most_popular"] == expected

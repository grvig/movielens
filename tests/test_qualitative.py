"""Tests for the synthetic-user qualitative check.

Title matching is the part that decides whether this is useful or useless: a bad join looks
exactly like a bad recommender.
"""

import pandas as pd
import pytest

from src.experiments.run_qualitative import SYNTHETIC_USER_ID
from src.experiments.run_qualitative import build_title_index
from src.experiments.run_qualitative import match_history
from src.experiments.run_qualitative import normalise_title
from src.experiments.run_qualitative import pick_column
from src.experiments.run_qualitative import synthetic_ratings


@pytest.fixture
def items():
    return pd.DataFrame([
        {"item_id": 1, "title": "Godfather, The (1972)", "release_year": 1972,
         "genre_Drama": 1},
        {"item_id": 2, "title": "Star Wars (1977)", "release_year": 1977, "genre_Action": 1},
        {"item_id": 3, "title": "Fargo (1996)", "release_year": 1997, "genre_Crime": 1},
    ])


def test_trailing_article_is_rotated_back():
    """MovieLens writes "Godfather, The"; an export writes "The Godfather"."""
    assert normalise_title("Godfather, The (1972)") == normalise_title("The Godfather")


def test_year_and_punctuation_are_stripped():
    assert normalise_title("Star Wars (1977)") == "star wars"
    assert normalise_title("Se7en: Director's Cut") == normalise_title("Se7en Directors Cut")


def test_case_is_ignored():
    assert normalise_title("FARGO") == normalise_title("fargo")


def test_title_index_covers_both_keys(items):
    by_title, by_title_year = build_title_index(items)
    assert by_title[normalise_title("The Godfather")] == 1
    assert by_title_year[(normalise_title("Star Wars"), 1977)] == 2


def test_year_disambiguates_when_the_title_is_ambiguous(items):
    """Fargo's title year is 1996 but its release_date year is 1997; both must resolve."""
    _, by_title_year = build_title_index(items)
    assert by_title_year[(normalise_title("Fargo"), 1996)] == 3


def test_letterboxd_column_names_are_recognised():
    frame = pd.DataFrame({"Name": ["x"], "Year": [1977], "Rating": [4.5]})
    assert pick_column(frame, ["Name", "name"]) == "Name"
    assert pick_column(frame, ["Rating", "rating"]) == "Rating"
    assert pick_column(frame, ["nothing"]) is None


def test_matching_reports_what_it_could_not_find(items):
    history = pd.DataFrame({
        "Name": ["The Godfather", "Star Wars", "Some Film From 2019"],
        "Rating": [5.0, 4.0, 3.0],
    })
    matched, unmatched = match_history(history, items, 1.0)
    assert len(matched) == 2
    assert unmatched == ["Some Film From 2019"]


def test_a_missing_title_column_is_rejected(items):
    with pytest.raises(ValueError, match="title column"):
        match_history(pd.DataFrame({"foo": [1]}), items, 1.0)


def test_a_history_without_ratings_still_matches(items):
    history = pd.DataFrame({"Name": ["Star Wars"]})
    matched, unmatched = match_history(history, items, 1.0)
    assert len(matched) == 1
    assert matched["rating"].iloc[0] is None


def test_rating_scale_is_applied(items):
    history = pd.DataFrame({"Name": ["Star Wars"], "Rating": [5.0]})
    matched, _ = match_history(history, items, 0.5)
    assert matched["rating"].iloc[0] == pytest.approx(2.5)


def test_synthetic_rows_use_the_injected_user_and_clip(items):
    matched = pd.DataFrame([
        {"item_id": 1, "rating": 9.0, "title": "a"},
        {"item_id": 2, "rating": None, "title": "b"},
    ])
    rows = synthetic_ratings(matched, default_rating=4.0, base_timestamp=1000)
    assert set(rows["user_id"]) == {SYNTHETIC_USER_ID}
    assert rows["rating"].max() <= 5.0
    assert rows["rating"].iloc[1] == pytest.approx(4.0)
    assert list(rows["timestamp"]) == [1000, 1001]


def test_synthetic_timestamps_are_after_training(items):
    """The injected user must look like the most recent thing in the data."""
    matched = pd.DataFrame([{"item_id": 1, "rating": 4.0, "title": "a"}])
    rows = synthetic_ratings(matched, 4.0, base_timestamp=999999)
    assert rows["timestamp"].min() == 999999

"""Recommend for a real person, from an exported watch history.

    python -m src.experiments.run_qualitative --history my_letterboxd_export.csv

Takes a CSV of films and ratings, matches the titles against the MovieLens catalogue,
injects the result as a synthetic user, and prints what each model recommends. Writes
``results/qualitative.csv``.

This is the one check in the project that a person can judge without knowing what
precision@10 means. Numbers say most-popular wins on ranking; this says whether the films
it puts up are ones you would actually watch, and it exposes failure modes an aggregate
cannot - a model that only suggests films from one decade, or one that has clearly latched
onto a single item in your history.

Expected input: a CSV with a title column and optionally a rating column. Letterboxd
exports ``Name``, ``Year`` and ``Rating`` (out of 5); those names are recognised, as are
``title`` and ``rating``. Anything unmatched is reported rather than silently dropped,
because a 40% match rate would otherwise look like a bad recommender rather than a bad
join.
"""

import argparse
import re

import numpy as np
import pandas as pd

from src.config import load_config
from src.data.splitting import load_splits
from src.evaluation.ranking import item_popularity
from src.experiments.recommend import build_item_lookup
from src.experiments.recommend import describe
from src.experiments.recommend import genre_text
from src.experiments.registry import MODEL_ORDER
from src.experiments.registry import build_model
from src.experiments.run_main import load_items

OUTPUT_NAME = "qualitative.csv"
SYNTHETIC_USER_ID = 944
DEFAULT_K = 10
TITLE_COLUMNS = ["Name", "name", "Title", "title", "film", "Film"]
RATING_COLUMNS = ["Rating", "rating", "Score", "score"]
YEAR_COLUMNS = ["Year", "year"]
ARTICLE_SUFFIX = re.compile(r",\s*(the|a|an)$", re.IGNORECASE)
YEAR_IN_TITLE = re.compile(r"\(\s*(\d{4})\s*\)")
NON_WORD = re.compile(r"[^a-z0-9]+")
# Apostrophes are dropped rather than treated as separators, so "Director's Cut"
# matches "Directors Cut". Different sources punctuate the same title differently.
APOSTROPHES = re.compile(r"['‘’`]")


def pick_column(frame, candidates):
    for name in candidates:
        if name in frame.columns:
            return name
    return None


def normalise_title(title):
    """A comparable key for a film title.

    MovieLens writes "Godfather, The (1972)" where an export writes "The Godfather", so the
    trailing-article form is rotated back before punctuation and case are stripped.
    """
    text = str(title)
    text = YEAR_IN_TITLE.sub(" ", text)
    text = text.strip()
    match = ARTICLE_SUFFIX.search(text)
    if match:
        text = match.group(1) + " " + text[: match.start()]
    text = APOSTROPHES.sub("", text.lower())
    return NON_WORD.sub(" ", text).strip()


def build_title_index(items):
    """Map normalised title to item id, and (title, year) to item id for tie-breaking."""
    by_title = {}
    by_title_year = {}
    for _, row in items.iterrows():
        key = normalise_title(row["title"])
        item_id = int(row["item_id"])
        by_title.setdefault(key, item_id)
        year = YEAR_IN_TITLE.search(str(row["title"]))
        if year:
            by_title_year[(key, int(year.group(1)))] = item_id
        elif int(row["release_year"]) > 0:
            by_title_year[(key, int(row["release_year"]))] = item_id
    return by_title, by_title_year


def match_history(history, items, rating_scale):
    """Join an exported history onto MovieLens item ids."""
    title_column = pick_column(history, TITLE_COLUMNS)
    if title_column is None:
        raise ValueError(
            "no title column found; expected one of " + ", ".join(TITLE_COLUMNS)
        )
    rating_column = pick_column(history, RATING_COLUMNS)
    year_column = pick_column(history, YEAR_COLUMNS)
    by_title, by_title_year = build_title_index(items)

    matched = []
    unmatched = []
    for position, row in history.iterrows():
        key = normalise_title(row[title_column])
        item_id = None
        if year_column is not None and not pd.isna(row[year_column]):
            item_id = by_title_year.get((key, int(row[year_column])))
        if item_id is None:
            item_id = by_title.get(key)
        if item_id is None:
            unmatched.append(str(row[title_column]))
            continue
        if rating_column is None or pd.isna(row[rating_column]):
            rating = None
        else:
            rating = float(row[rating_column]) * rating_scale
        matched.append({"item_id": item_id, "rating": rating,
                        "title": str(row[title_column])})
    return pd.DataFrame(matched), unmatched


def synthetic_ratings(matched, default_rating, base_timestamp):
    """Turn matched films into ratings rows for the injected user."""
    rows = []
    for position, row in matched.iterrows():
        rating = row["rating"]
        if rating is None or pd.isna(rating):
            rating = default_rating
        rating = float(np.clip(rating, 1.0, 5.0))
        rows.append((SYNTHETIC_USER_ID, int(row["item_id"]), rating,
                     base_timestamp + position))
    return pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "timestamp"])


def recommend_for_synthetic(config, items, train, validation, injected, k):
    combined = pd.concat([train, injected], ignore_index=True)
    lookup = build_item_lookup(items)
    popularity = item_popularity(combined)
    rows = []
    for name in MODEL_ORDER:
        model = build_model(name, config, items, validation=validation, cache=False)
        model.fit(combined)
        recommendations = model.recommend(SYNTHETIC_USER_ID, k)
        scores = model.predict(SYNTHETIC_USER_ID, recommendations)
        print("")
        print(name)
        for position, item_id in enumerate(recommendations):
            item_id = int(item_id)
            print("  " + str(position + 1).ljust(4)
                  + format(float(scores[position]), ".2f").ljust(7)
                  + str(popularity.get(item_id, 0)).ljust(6)
                  + describe(lookup, item_id) + "  " + genre_text(lookup, item_id))
            rows.append({
                "model": name,
                "rank": position + 1,
                "item_id": item_id,
                "title": lookup.get(item_id, {}).get("label", ""),
                "genres": genre_text(lookup, item_id),
                "predicted": float(scores[position]),
                "train_ratings": int(popularity.get(item_id, 0)),
            })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Recommend for a synthetic user built from an exported history."
    )
    parser.add_argument("--config", default=None, help="path to a config file")
    parser.add_argument("--history", required=True, help="CSV of watched films")
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--rating-scale", type=float, default=1.0,
                        help="multiplier onto the 1-5 scale; Letterboxd is already 1-5")
    parser.add_argument("--default-rating", type=float, default=4.0,
                        help="rating to assume for films with no score in the export")
    args = parser.parse_args()

    config = load_config(args.config)
    config.ensure_output_dirs()
    train, validation, test = load_splits(config.path("processed_dir"))
    items = load_items(config)

    history = pd.read_csv(args.history)
    matched, unmatched = match_history(history, items, args.rating_scale)
    print("matched " + str(len(matched)) + " of " + str(len(history))
          + " films onto the MovieLens catalogue")
    if len(unmatched) > 0:
        print("unmatched (" + str(len(unmatched)) + "), first 10:")
        for title in unmatched[:10]:
            print("  " + title)
    if len(matched) == 0:
        raise SystemExit(
            "nothing matched. MovieLens 100K only contains films released up to 1998, so a "
            "modern watch history will mostly miss."
        )

    injected = synthetic_ratings(matched, args.default_rating,
                                 int(train["timestamp"].max()) + 1)
    frame = recommend_for_synthetic(config, items, train, validation, injected, args.k)

    output_path = config.path("results_dir") / OUTPUT_NAME
    frame.to_csv(output_path, index=False)
    print("")
    print("wrote " + str(output_path))


if __name__ == "__main__":
    main()

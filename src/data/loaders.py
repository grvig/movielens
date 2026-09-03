"""Readers for the raw MovieLens 100K files.

MovieLens 100K ships four files this project uses, all plain text:

``u.data``
    Tab separated, no header: user id, item id, rating, unix timestamp.
``u.item``
    Pipe separated, latin-1 encoded: movie id, title, release date, video release date,
    IMDb URL, then 19 binary genre flags.
``u.genre``
    Pipe separated genre name and index, which fixes the order of those 19 flags.
``u.user``
    Pipe separated demographics.  Loaded for the data card only; no model uses it.

Note for the report: this dataset has **no tag file**.  The 19 genre flags plus the title
and release year are the entire content signal available, which is why the content-based
model builds its documents from genres, title tokens and release decade.
"""

import re

import numpy as np
import pandas as pd

RATINGS_FILE = "u.data"
ITEMS_FILE = "u.item"
GENRE_FILE = "u.genre"
USERS_FILE = "u.user"

ITEM_BASE_COLUMNS = ["item_id", "title", "release_date", "video_release_date", "imdb_url"]
YEAR_PATTERN = re.compile(r"\((\d{4})\)")


def load_ratings(raw_dir):
    path = raw_dir / RATINGS_FILE
    frame = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["user_id", "item_id", "rating", "timestamp"],
        engine="python",
    )
    frame["user_id"] = frame["user_id"].astype(np.int64)
    frame["item_id"] = frame["item_id"].astype(np.int64)
    frame["rating"] = frame["rating"].astype(np.float64)
    frame["timestamp"] = frame["timestamp"].astype(np.int64)
    return frame


def load_genre_names(raw_dir):
    """Genre names in flag order, taken from u.genre rather than hardcoded."""
    path = raw_dir / GENRE_FILE
    names = {}
    with open(path, "r", encoding="latin-1") as handle:
        for line in handle:
            line = line.strip()
            if line == "":
                continue
            parts = line.split("|")
            if len(parts) != 2:
                continue
            names[int(parts[1])] = parts[0]
    ordered = []
    for index in sorted(names.keys()):
        ordered.append(names[index])
    return ordered


def load_items(raw_dir):
    """Item table with one ``genre_<Name>`` column per flag."""
    genre_names = load_genre_names(raw_dir)
    genre_columns = []
    for name in genre_names:
        genre_columns.append("genre_" + name)
    path = raw_dir / ITEMS_FILE
    frame = pd.read_csv(
        path,
        sep="|",
        header=None,
        names=ITEM_BASE_COLUMNS + genre_columns,
        encoding="latin-1",
        engine="python",
    )
    frame["item_id"] = frame["item_id"].astype(np.int64)
    frame["title"] = frame["title"].fillna("").astype(str)
    frame["release_year"] = extract_release_years(frame)
    for column in genre_columns:
        frame[column] = frame[column].fillna(0).astype(np.int64)
    keep = ["item_id", "title", "release_year"] + genre_columns
    return frame[keep]


def extract_release_years(frame):
    """Prefer the release_date column, fall back to a year in the title.

    A handful of ML-100K rows have an empty release date, and one has no year anywhere;
    those become 0 rather than NaN so the column stays integer-typed.
    """
    years = []
    for _, row in frame.iterrows():
        year = year_from_release_date(row["release_date"])
        if year == 0:
            year = year_from_title(row["title"])
        years.append(year)
    return np.array(years, dtype=np.int64)


def year_from_release_date(value):
    if not isinstance(value, str):
        return 0
    parts = value.strip().split("-")
    if len(parts) != 3:
        return 0
    if not parts[2].isdigit():
        return 0
    return int(parts[2])


def year_from_title(title):
    if not isinstance(title, str):
        return 0
    matches = YEAR_PATTERN.findall(title)
    if len(matches) == 0:
        return 0
    return int(matches[-1])


def load_users(raw_dir):
    path = raw_dir / USERS_FILE
    frame = pd.read_csv(
        path,
        sep="|",
        header=None,
        names=["user_id", "age", "gender", "occupation", "zip_code"],
        encoding="latin-1",
        engine="python",
    )
    frame["user_id"] = frame["user_id"].astype(np.int64)
    return frame


def genre_columns(items):
    columns = []
    for column in items.columns:
        if column.startswith("genre_"):
            columns.append(column)
    return columns

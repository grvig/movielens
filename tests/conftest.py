"""Shared test fixtures.

The synthetic dataset here exists so models and metrics can be developed and tested
before the real MovieLens files are downloaded, and so the test suite runs in under a
second without touching data/.  It has the same column names and dtypes as the real
processed ratings table.
"""

import numpy as np
import pandas as pd
import pytest

from src.config import load_config

N_USERS = 12
N_ITEMS = 30
MIN_RATINGS = 8
MAX_RATINGS = 12
BASE_TIMESTAMP = 880000000


@pytest.fixture(scope="session")
def config():
    return load_config()


@pytest.fixture(scope="session")
def synthetic_ratings():
    """A small ratings frame with strictly increasing timestamps per user."""
    rng = np.random.default_rng(7)
    rows = []
    for user_id in range(1, N_USERS + 1):
        n_ratings = int(rng.integers(MIN_RATINGS, MAX_RATINGS + 1))
        items = rng.choice(np.arange(1, N_ITEMS + 1), size=n_ratings, replace=False)
        for position, item_id in enumerate(items):
            rating = float(rng.integers(1, 6))
            timestamp = BASE_TIMESTAMP + user_id * 10000 + position * 60
            rows.append((user_id, int(item_id), rating, timestamp))
    frame = pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "timestamp"])
    frame["user_id"] = frame["user_id"].astype(np.int64)
    frame["item_id"] = frame["item_id"].astype(np.int64)
    frame["rating"] = frame["rating"].astype(np.float64)
    frame["timestamp"] = frame["timestamp"].astype(np.int64)
    return frame


@pytest.fixture(scope="session")
def synthetic_items():
    """An item table with the same shape as the real one: id, title, year, genres."""
    rng = np.random.default_rng(11)
    genre_names = ["Action", "Comedy", "Drama", "Horror", "Romance", "Sci-Fi"]
    rows = []
    for item_id in range(1, N_ITEMS + 1):
        year = int(rng.integers(1950, 1999))
        flags = rng.integers(0, 2, size=len(genre_names))
        if flags.sum() == 0:
            flags[0] = 1
        row = {"item_id": item_id, "title": "Movie " + str(item_id), "release_year": year}
        for name, flag in zip(genre_names, flags):
            row["genre_" + name] = int(flag)
        rows.append(row)
    return pd.DataFrame(rows)

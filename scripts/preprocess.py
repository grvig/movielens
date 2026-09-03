"""Turn the raw MovieLens files into the processed tables every experiment reads.

Run after the download::

    python scripts/preprocess.py

Writes to ``data/processed/``:

``ratings.parquet``
    user_id, item_id, rating, timestamp for all 100,000 ratings.
``items.parquet``
    item_id, title, release_year and one column per genre flag.
``train.parquet``, ``val.parquet``, ``test.parquet``
    The per-user temporal split, written once so nothing recomputes it at runtime.

It also prints a data card.  Those numbers go straight into the dataset section of the
report, so they are computed here once rather than retyped from memory.
"""

import argparse
import sys
from pathlib import Path

# Running a file inside scripts/ puts scripts/ on the import path, not the repository
# root, so "import src" would fail on a fresh clone unless the package had been installed
# first. Putting the root on the path here means the documented commands work straight
# after a git clone, with pip install -e . left as an optional convenience.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.config import load_config
from src.data.loaders import genre_columns
from src.data.loaders import load_items
from src.data.loaders import load_ratings
from src.data.splitting import assert_no_leakage
from src.data.splitting import split_summary
from src.data.splitting import temporal_split
from src.data.splitting import write_splits

RATINGS_OUTPUT = "ratings.parquet"
ITEMS_OUTPUT = "items.parquet"


def print_data_card(ratings, items):
    n_ratings = len(ratings)
    n_users = ratings["user_id"].nunique()
    n_items = ratings["item_id"].nunique()
    density = 100.0 * n_ratings / (n_users * n_items)
    per_user = ratings.groupby("user_id").size()
    per_item = ratings.groupby("item_id").size()
    first = ratings["timestamp"].min()
    last = ratings["timestamp"].max()
    span_days = (last - first) / 86400.0

    print("")
    print("data card")
    print("---------")
    print("  ratings                 " + str(n_ratings))
    print("  users                   " + str(n_users))
    print("  items rated             " + str(n_items))
    print("  items in catalogue      " + str(len(items)))
    print("  density                 " + format(density, ".2f") + "%")
    print("  ratings per user        min " + str(int(per_user.min())) +
          ", median " + str(int(per_user.median())) +
          ", max " + str(int(per_user.max())))
    print("  ratings per item        min " + str(int(per_item.min())) +
          ", median " + str(int(per_item.median())) +
          ", max " + str(int(per_item.max())))
    print("  rating scale            " + str(ratings["rating"].min()) +
          " to " + str(ratings["rating"].max()))
    print("  timestamp span          " + format(span_days, ".0f") + " days")
    print("  genre flags             " + str(len(genre_columns(items))))
    print("")


def main():
    parser = argparse.ArgumentParser(description="Build the processed MovieLens tables.")
    parser.add_argument("--config", default=None, help="path to a config file")
    args = parser.parse_args()

    config = load_config(args.config)
    config.ensure_output_dirs()
    raw_dir = config.path("raw_dir") / config.section("download")["extract_subdir"]
    processed_dir = config.path("processed_dir")

    if not raw_dir.exists():
        raise FileNotFoundError(
            "raw data not found at " + str(raw_dir) +
            ". Run python scripts/download_data.py first."
        )

    print("reading raw files from " + str(raw_dir))
    ratings = load_ratings(raw_dir)
    items = load_items(raw_dir)

    ratings_path = processed_dir / RATINGS_OUTPUT
    items_path = processed_dir / ITEMS_OUTPUT
    ratings.to_parquet(ratings_path, index=False)
    items.to_parquet(items_path, index=False)
    print("wrote " + str(ratings_path))
    print("wrote " + str(items_path))

    print_data_card(ratings, items)

    train, val, test = temporal_split(ratings, config)
    assert_no_leakage(train, val, test)
    paths = write_splits(train, val, test, processed_dir)
    for name in ["train", "val", "test"]:
        print("wrote " + str(paths[name]))

    print("")
    print("split summary")
    print("-------------")
    print(split_summary(train, val, test).to_string(index=False))
    print("")


if __name__ == "__main__":
    main()

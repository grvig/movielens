"""Per-user temporal splitting.

Each user's ratings are ordered in time and cut into train / validation / test, so the
model only ever sees a user's past when predicting their future.

The alternative — splitting rows at random — leaks future interactions into training and
inflates every metric, because the model gets to see a user rating a sequel before it
predicts they liked the original.  ``src/experiments/run_split_ablation.py`` measures
exactly how much that is worth on this dataset; the short version is that a random split
makes every model look better than it is, and the fancier the model the more it gains.

Ties on timestamp are broken by ascending item id.  Without a tie-break the split would
depend on pandas' sort stability and stop being reproducible across versions.
"""

import numpy as np
import pandas as pd

SPLIT_NAMES = ["train", "val", "test"]
SORT_COLUMNS = ["user_id", "timestamp", "item_id"]


def temporal_split(ratings, config):
    """Split ``ratings`` into three frames, holding out each user's most recent ratings.

    Users with too few ratings to leave ``min_train_ratings`` behind keep everything in
    train and contribute to no held-out metric.  MovieLens 100K guarantees at least 20
    ratings per user, so in practice this never triggers, but the guard means the code
    also works on the truncated histories used by the cold-start experiment.
    """
    settings = config.section("split")
    train_frac = float(settings["train_frac"])
    val_frac = float(settings["val_frac"])
    test_frac = float(settings["test_frac"])
    min_train = int(settings["min_train_ratings"])
    check_fractions(train_frac, val_frac, test_frac)

    ordered = ratings.sort_values(SORT_COLUMNS, kind="mergesort").reset_index(drop=True)
    labels = np.empty(len(ordered), dtype=object)
    positions_by_user = ordered.groupby("user_id").indices

    for user_id in sorted(positions_by_user.keys()):
        positions = positions_by_user[user_id]
        n_train, n_val = user_split_sizes(len(positions), val_frac, test_frac, min_train)
        labels[positions[:n_train]] = "train"
        labels[positions[n_train:n_train + n_val]] = "val"
        labels[positions[n_train + n_val:]] = "test"

    ordered["split"] = labels
    train = ordered[ordered["split"] == "train"].drop(columns=["split"])
    val = ordered[ordered["split"] == "val"].drop(columns=["split"])
    test = ordered[ordered["split"] == "test"].drop(columns=["split"])
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def user_split_sizes(n_ratings, val_frac, test_frac, min_train):
    """Sizes of the train and validation blocks for one user; the rest is test."""
    n_test = int(round(n_ratings * test_frac))
    n_val = int(round(n_ratings * val_frac))
    if n_test < 1:
        n_test = 1
    if n_val < 1:
        n_val = 1
    n_train = n_ratings - n_val - n_test
    if n_train < min_train:
        return n_ratings, 0
    return n_train, n_val


def check_fractions(train_frac, val_frac, test_frac):
    total = train_frac + val_frac + test_frac
    if abs(total - 1.0) > 1e-9:
        raise ValueError("split fractions must sum to 1.0, got " + format(total, ".4f"))


def assert_no_leakage(train, val, test):
    """Raise unless every user's train precedes their val, which precedes their test.

    This is the guard rail for the whole project.  If it ever fails, every number in
    results/ is wrong, so it runs as a test and again inside preprocessing.
    """
    train_last = train.groupby("user_id")["timestamp"].max()
    check_ordering(train_last, val, "train", "val")
    check_ordering(train_last, test, "train", "test")
    if len(val) > 0:
        val_last = val.groupby("user_id")["timestamp"].max()
        check_ordering(val_last, test, "val", "test")


def check_ordering(earlier_last, later, earlier_name, later_name):
    if len(later) == 0:
        return
    later_first = later.groupby("user_id")["timestamp"].min()
    shared = earlier_last.index.intersection(later_first.index)
    if len(shared) == 0:
        return
    offenders = shared[earlier_last.loc[shared].values > later_first.loc[shared].values]
    if len(offenders) > 0:
        raise AssertionError(
            "temporal leakage: " + str(len(offenders)) + " users have " + earlier_name +
            " ratings after their first " + later_name + " rating; first offenders: " +
            str(list(offenders[:5]))
        )


def split_summary(train, val, test):
    """One row per split, for the console and for the report's protocol section."""
    rows = []
    frames = [train, val, test]
    for name, frame in zip(SPLIT_NAMES, frames):
        rows.append({
            "split": name,
            "ratings": len(frame),
            "users": frame["user_id"].nunique(),
            "items": frame["item_id"].nunique(),
        })
    return pd.DataFrame(rows)


def write_splits(train, val, test, processed_dir):
    paths = {}
    frames = [train, val, test]
    for name, frame in zip(SPLIT_NAMES, frames):
        path = processed_dir / (name + ".parquet")
        frame.to_parquet(path, index=False)
        paths[name] = path
    return paths


def load_splits(processed_dir):
    frames = {}
    for name in SPLIT_NAMES:
        path = processed_dir / (name + ".parquet")
        if not path.exists():
            raise FileNotFoundError(
                "missing " + str(path) + ". Run python scripts/preprocess.py first."
            )
        frames[name] = pd.read_parquet(path)
    return frames["train"], frames["val"], frames["test"]


def select_cold_users(train, n_users, rng, min_history=1):
    """Choose the users whose history the cold-start experiment will truncate."""
    counts = train.groupby("user_id").size()
    eligible = np.sort(counts[counts >= min_history].index.to_numpy())
    if len(eligible) <= n_users:
        return eligible
    chosen = rng.choice(eligible, size=n_users, replace=False)
    return np.sort(chosen)


def truncate_user_histories(train, user_ids, max_ratings):
    """Keep only each named user's most recent ``max_ratings`` training ratings.

    Two decisions here shape what the cold-start experiment actually measures.

    **Only the named users are truncated.** Truncating everybody would starve the whole
    model - the item similarities and latent factors would all degrade together - and the
    figure would show how a model copes with a small dataset rather than how it copes with
    a new user. Leaving the other 743 users intact keeps the model's view of the catalogue
    normal and isolates the variable.

    **The most recent ratings are kept, not the earliest.** The split is already temporal,
    so a user's training ratings all precede their test ratings. Keeping the most recent n
    minimises the time gap to the evaluation period, so the figure measures how much
    history a model needs rather than how stale that history is.
    """
    cold = set(int(value) for value in user_ids)
    ordered = train.sort_values(SORT_COLUMNS, kind="mergesort").reset_index(drop=True)
    keep = []
    for user_id, group in ordered.groupby("user_id"):
        positions = group.index.to_numpy()
        if int(user_id) not in cold:
            keep.extend(positions.tolist())
            continue
        keep.extend(positions[-max_ratings:].tolist())
    keep.sort()
    return ordered.loc[keep].reset_index(drop=True)


def history_lengths(train, user_ids):
    """Training history length for each named user, zero when they have none left."""
    counts = train.groupby("user_id").size()
    lengths = {}
    for user_id in user_ids:
        lengths[int(user_id)] = int(counts.get(int(user_id), 0))
    return lengths

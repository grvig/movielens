"""Paired bootstrap confidence intervals over users.

A results table gives point estimates. It does not say whether matrix factorisation beating
item-kNN by 0.02 RMSE is a real difference or the luck of which 943 users happened to be in
this dataset. That question has to be answered before the report claims a gap, and this is
what answers it.

The resampling unit is the **user**, not the rating. Users are the independent observations
here; individual ratings from the same person are not, and resampling ratings would treat
one person's 700 ratings as 700 independent pieces of evidence and produce intervals far
too narrow.

Two details make the comparison meaningful:

**The resample is paired.** Both models are scored on the *same* bootstrap sample of users
on every iteration, so the interval is around the difference rather than around each model
separately. Two overlapping individual intervals can still correspond to a difference that
is consistently positive; comparing separate intervals is the standard way to under-detect
a real effect.

**Per-user metrics are re-weighted, not re-averaged.** A user with 700 ratings and a user
with 4 must not carry equal weight in an RMSE, so the accuracy bootstrap works from the
per-user error sums produced by :mod:`src.evaluation.metrics`.
"""

import numpy as np
import pandas as pd

DEFAULT_ITERATIONS = 1000
DEFAULT_CONFIDENCE = 0.95


def bootstrap_user_indices(n_users, iterations, rng):
    """One matrix of resampled user positions, shared by every model being compared."""
    return rng.integers(0, n_users, size=(iterations, n_users))


def accuracy_samples(per_user, indices, metric):
    """Bootstrap distribution of RMSE or MAE from per-user error sums."""
    counts = per_user["n_ratings"].to_numpy(dtype=np.float64)
    if metric == "rmse":
        totals = per_user["sum_squared_error"].to_numpy(dtype=np.float64)
    elif metric == "mae":
        totals = per_user["sum_absolute_error"].to_numpy(dtype=np.float64)
    else:
        raise ValueError("accuracy metric must be rmse or mae, got " + str(metric))

    resampled_totals = totals[indices].sum(axis=1)
    resampled_counts = counts[indices].sum(axis=1)
    resampled_counts[resampled_counts == 0] = np.nan
    means = resampled_totals / resampled_counts
    if metric == "rmse":
        return np.sqrt(means)
    return means


def ranking_samples(per_user_values, indices):
    """Bootstrap distribution of a per-user ranking metric, ignoring undefined users."""
    values = np.asarray(per_user_values, dtype=np.float64)
    resampled = values[indices]
    return np.nanmean(resampled, axis=1)


def percentile_interval(samples, confidence=DEFAULT_CONFIDENCE):
    usable = np.asarray(samples, dtype=np.float64)
    usable = usable[~np.isnan(usable)]
    if usable.size == 0:
        return float("nan"), float("nan")
    tail = (1.0 - confidence) / 2.0
    lower = float(np.percentile(usable, 100.0 * tail))
    upper = float(np.percentile(usable, 100.0 * (1.0 - tail)))
    return lower, upper


def difference_interval(samples_a, samples_b, confidence=DEFAULT_CONFIDENCE):
    """Interval around the paired difference a - b, plus whether it excludes zero."""
    differences = np.asarray(samples_a, dtype=np.float64) - np.asarray(
        samples_b, dtype=np.float64
    )
    lower, upper = percentile_interval(differences, confidence)
    usable = differences[~np.isnan(differences)]
    if usable.size == 0:
        point = float("nan")
    else:
        point = float(np.mean(usable))
    significant = False
    if not np.isnan(lower) and not np.isnan(upper):
        if lower > 0.0 or upper < 0.0:
            significant = True
    return {
        "difference": point,
        "lower": lower,
        "upper": upper,
        "excludes_zero": significant,
    }


def align_per_user(frames, key="user_id"):
    """Restrict every model's per-user table to the users all of them scored.

    Comparing models over different user sets would make the paired resample meaningless,
    since a given bootstrap draw would not refer to the same people in both arms.
    """
    shared = None
    for frame in frames.values():
        users = set(frame[key].tolist())
        if shared is None:
            shared = users
        else:
            shared = shared & users
    aligned = {}
    for name, frame in frames.items():
        selected = frame[frame[key].isin(shared)]
        aligned[name] = selected.sort_values(key).reset_index(drop=True)
    return aligned, sorted(shared)


def compare_models(samples_by_model, confidence=DEFAULT_CONFIDENCE):
    """Every pairwise difference, as rows ready for a results CSV."""
    rows = []
    names = sorted(samples_by_model.keys())
    for index, first in enumerate(names):
        for second in names[index + 1:]:
            outcome = difference_interval(
                samples_by_model[first], samples_by_model[second], confidence
            )
            outcome["model_a"] = first
            outcome["model_b"] = second
            rows.append(outcome)
    return pd.DataFrame(rows)

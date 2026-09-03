"""Evaluation metrics.

The report puts two families of metric side by side, and they answer different questions:

*   **Accuracy** - RMSE and MAE on held-out ratings. How close is the predicted rating to
    the real one?
*   **Ranking and catalogue behaviour** - precision@k, recall@k, coverage and popularity
    bias. Of the things it puts in front of a user, how many are good, and how much of the
    catalogue does it ever bother to use?

A model can win one family and lose the other badly, which is the tension the results
section of the report is built around. This module holds the accuracy half.

Every metric also has a per-user form. The aggregate is what goes in the results table,
but the bootstrap needs per-user values to resample over, so they are computed once here
and reused rather than recomputed later from a second code path.
"""

import numpy as np
import pandas as pd


def rmse(actual, predicted):
    """Root mean squared error. Returns NaN on empty input rather than raising."""
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    check_same_length(actual, predicted)
    if actual.size == 0:
        return float("nan")
    errors = actual - predicted
    return float(np.sqrt(np.mean(errors * errors)))


def mae(actual, predicted):
    """Mean absolute error. Returns NaN on empty input rather than raising."""
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    check_same_length(actual, predicted)
    if actual.size == 0:
        return float("nan")
    return float(np.mean(np.abs(actual - predicted)))


def check_same_length(actual, predicted):
    if actual.shape[0] != predicted.shape[0]:
        raise ValueError(
            "actual and predicted differ in length: "
            + str(actual.shape[0])
            + " vs "
            + str(predicted.shape[0])
        )


def accuracy_metrics(actual, predicted):
    """The accuracy family as one dict, ready to become results rows."""
    return {
        "rmse": rmse(actual, predicted),
        "mae": mae(actual, predicted),
        "n_ratings": int(np.asarray(actual).shape[0]),
    }


def per_user_accuracy(predictions):
    """Per-user error sums, the unit the bootstrap resamples over.

    The frame needs columns user_id, rating and predicted.

    Sums are returned rather than means because a bootstrap over users has to re-weight by
    each user rating count. Averaging per user and then averaging those averages would
    silently give a 4-rating user the same weight as a 200-rating one.
    """
    require_columns(predictions, ["user_id", "rating", "predicted"])
    actual = predictions["rating"].to_numpy(dtype=np.float64)
    predicted = predictions["predicted"].to_numpy(dtype=np.float64)
    errors = actual - predicted
    frame = pd.DataFrame(
        {
            "user_id": predictions["user_id"].to_numpy(),
            "squared_error": errors * errors,
            "absolute_error": np.abs(errors),
        }
    )
    grouped = frame.groupby("user_id")
    summary = grouped.agg(
        n_ratings=("squared_error", "size"),
        sum_squared_error=("squared_error", "sum"),
        sum_absolute_error=("absolute_error", "sum"),
    )
    return summary.reset_index()


def aggregate_user_accuracy(per_user):
    """Recover RMSE and MAE from per-user sums, weighted by rating count."""
    total_ratings = float(per_user["n_ratings"].sum())
    if total_ratings == 0:
        return {"rmse": float("nan"), "mae": float("nan"), "n_ratings": 0}
    mean_squared = float(per_user["sum_squared_error"].sum()) / total_ratings
    mean_absolute = float(per_user["sum_absolute_error"].sum()) / total_ratings
    return {
        "rmse": float(np.sqrt(mean_squared)),
        "mae": mean_absolute,
        "n_ratings": int(total_ratings),
    }


def require_columns(frame, columns):
    missing = []
    for column in columns:
        if column not in frame.columns:
            missing.append(column)
    if len(missing) > 0:
        raise ValueError("frame is missing columns: " + ", ".join(missing))

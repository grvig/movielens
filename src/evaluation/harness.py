"""The shared evaluation harness.

One function, :func:`evaluate_model`, runs any model through the whole protocol and
returns result rows. It never branches on model type. Every model is fitted, asked for
rating predictions on the held-out ratings, and asked for a top-k list per user, using the
same candidate rule and the same relevance threshold. That uniformity is the entire reason
the comparison between four model families means anything.

Results come out in long format, one row per measurement::

    model, variant, split, metric, k, value, n_users, n_ratings, seed, run_id, timestamp

Long format rather than wide because a wide table has to be redesigned every time a metric
is added, and the plotting scripts filter on metric anyway. The README table is generated
from these rows; it is not the source of truth.
"""

import time

import numpy as np
import pandas as pd

from src.evaluation.metrics import accuracy_metrics
from src.evaluation.metrics import per_user_accuracy
from src.evaluation.ranking import catalogue_coverage
from src.evaluation.ranking import item_popularity
from src.evaluation.ranking import mean_popularity_percentile
from src.evaluation.ranking import mean_recommended_popularity
from src.evaluation.ranking import popularity_percentiles
from src.evaluation.ranking import precision_at_k
from src.evaluation.ranking import recall_at_k
from src.evaluation.ranking import recommendation_gini
from src.evaluation.ranking import relevant_items_by_user

RESULT_COLUMNS = [
    "model",
    "variant",
    "split",
    "metric",
    "k",
    "value",
    "n_users",
    "n_ratings",
    "seed",
    "run_id",
    "timestamp",
]


def evaluate_model(model, train, holdout, config, split_name, variant="default", run_id=None):
    """Fit nothing, evaluate everything. The model must already be fitted.

    Fitting is left to the caller so that an experiment sweeping hyperparameters can fit
    once and evaluate against several holdouts without refitting.
    """
    model.check_fitted()
    if run_id is None:
        run_id = make_run_id()

    predictions = predict_holdout(model, holdout)
    accuracy = accuracy_metrics(predictions["rating"], predictions["predicted"])
    per_user = per_user_accuracy(predictions)

    evaluation = config.section("evaluation")
    k_values = list(evaluation["k_values"])
    threshold = float(evaluation["relevance_threshold"])
    max_k = max(k_values)

    relevant = relevant_items_by_user(holdout, threshold)
    scored_users = sorted(relevant.keys())
    recommendations = recommend_for_users(model, scored_users, max_k)

    popularity = item_popularity(train)
    percentiles = popularity_percentiles(popularity)
    catalogue = np.sort(train["item_id"].unique())

    context = {
        "model": model.name,
        "variant": variant,
        "split": split_name,
        "seed": config.seed,
        "run_id": run_id,
        "timestamp": int(time.time()),
    }

    rows = []
    rows.append(make_row(context, "rmse", None, accuracy["rmse"],
                         len(per_user), accuracy["n_ratings"]))
    rows.append(make_row(context, "mae", None, accuracy["mae"],
                         len(per_user), accuracy["n_ratings"]))

    for k in k_values:
        precisions = []
        recalls = []
        top_k_by_user = {}
        for user_id in scored_users:
            top_k = recommendations[user_id][:k]
            top_k_by_user[user_id] = top_k
            precisions.append(precision_at_k(top_k, relevant[user_id], k))
            recalls.append(recall_at_k(top_k, relevant[user_id], k))

        n_users = len(scored_users)
        rows.append(make_row(context, "precision_at_k", k, safe_mean(precisions), n_users, 0))
        rows.append(make_row(context, "recall_at_k", k, safe_mean(recalls), n_users, 0))
        rows.append(make_row(context, "coverage", k,
                             catalogue_coverage(top_k_by_user, len(catalogue)), n_users, 0))
        rows.append(make_row(context, "mean_popularity", k,
                             mean_recommended_popularity(top_k_by_user, popularity),
                             n_users, 0))
        rows.append(make_row(context, "popularity_percentile", k,
                             mean_popularity_percentile(top_k_by_user, percentiles),
                             n_users, 0))
        rows.append(make_row(context, "gini", k,
                             recommendation_gini(top_k_by_user, catalogue), n_users, 0))

    return rows


def predict_holdout(model, holdout):
    """Ask the model for one prediction per held-out rating, user by user."""
    frames = []
    for user_id, group in holdout.groupby("user_id"):
        items = group["item_id"].to_numpy(dtype=np.int64)
        predicted = np.asarray(model.predict(user_id, items), dtype=np.float64)
        if len(predicted) != len(items):
            raise ValueError(
                model.name + " returned " + str(len(predicted)) +
                " predictions for " + str(len(items)) + " candidates"
            )
        frame = group[["user_id", "item_id", "rating"]].copy()
        frame["predicted"] = predicted
        frames.append(frame)
    if len(frames) == 0:
        return pd.DataFrame(columns=["user_id", "item_id", "rating", "predicted"])
    return pd.concat(frames, ignore_index=True)


def recommend_for_users(model, user_ids, k):
    """Top-k once per user at the largest k; smaller k values are prefixes of it."""
    recommendations = {}
    for user_id in user_ids:
        recommendations[user_id] = list(model.recommend(user_id, k))
    return recommendations


def safe_mean(values):
    """Mean over the users that have a defined value, ignoring NaN."""
    array = np.asarray(values, dtype=np.float64)
    usable = array[~np.isnan(array)]
    if usable.size == 0:
        return float("nan")
    return float(np.mean(usable))


def make_row(context, metric, k, value, n_users, n_ratings):
    row = dict(context)
    row["metric"] = metric
    row["k"] = k
    row["value"] = value
    row["n_users"] = n_users
    row["n_ratings"] = n_ratings
    return row


def make_run_id():
    return time.strftime("%Y%m%dT%H%M%S")


def results_frame(rows):
    frame = pd.DataFrame(rows)
    if len(frame) == 0:
        return pd.DataFrame(columns=RESULT_COLUMNS)
    return frame[RESULT_COLUMNS]


def write_results(rows, path):
    frame = results_frame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def pivot_for_report(frame, metric, k=None):
    """One column per model for a single metric, which is the shape the README wants."""
    selected = frame[frame["metric"] == metric]
    if k is not None:
        selected = selected[selected["k"] == k]
    return selected.pivot_table(index="model", values="value", aggfunc="mean")

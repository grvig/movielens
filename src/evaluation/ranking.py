"""Ranking and catalogue-behaviour metrics.

The second half of the metric suite. Where :mod:`src.evaluation.metrics` asks how close a
predicted rating is, this module asks what a model actually puts in front of people and
how much of the catalogue it is willing to use.

Protocol decisions baked in here, all of them locked in configs/default.yaml:

*   Candidates are every training item the user has not already rated. No sampled
    negatives - with 1682 items the full scan is cheap, and sampling would add a second
    protocol decision to defend in the report.
*   An item is relevant if its held-out rating is at or above the relevance threshold
    (4.0). Everything else, including a 3, counts as not relevant.
*   Users with no relevant held-out item are excluded from precision and recall. Recall
    would divide by zero and precision would be pinned at zero through no fault of the
    model. Every results row carries its own n_users so this is never guesswork later.

Coverage and popularity bias are catalogue-level, not per-user, so they are computed over
the union of everyone's recommendations.
"""

import numpy as np

EPSILON = 1e-12


def precision_at_k(recommended, relevant, k):
    """Fraction of the top k that is relevant.

    The denominator is k, not len(recommended). A model that returns fewer than k items is
    being penalised for the shortfall, which is the honest reading.
    """
    if k <= 0:
        raise ValueError("k must be positive, got " + str(k))
    top = list(recommended)[:k]
    if len(relevant) == 0:
        return float("nan")
    hits = count_hits(top, relevant)
    return float(hits) / float(k)


def recall_at_k(recommended, relevant, k):
    """Fraction of the relevant items that made it into the top k."""
    if k <= 0:
        raise ValueError("k must be positive, got " + str(k))
    if len(relevant) == 0:
        return float("nan")
    top = list(recommended)[:k]
    hits = count_hits(top, relevant)
    return float(hits) / float(len(relevant))


def count_hits(top, relevant):
    relevant_set = set(int(item) for item in relevant)
    hits = 0
    for item in top:
        if int(item) in relevant_set:
            hits = hits + 1
    return hits


def relevant_items_by_user(holdout, threshold):
    """Map user id to the set of held-out items they rated at or above the threshold."""
    liked = holdout[holdout["rating"] >= threshold]
    grouped = liked.groupby("user_id")["item_id"]
    relevant = {}
    for user_id, items in grouped:
        relevant[int(user_id)] = set(int(item) for item in items)
    return relevant


def catalogue_coverage(recommendations_by_user, catalogue_size):
    """Fraction of the catalogue that appears in at least one user top-k list.

    This is the number that separates a model recommending the same twenty blockbusters to
    everyone from one that spreads across the catalogue, and it is half the story the
    results section tells.
    """
    if catalogue_size <= 0:
        return float("nan")
    distinct = set()
    for items in recommendations_by_user.values():
        for item in items:
            distinct.add(int(item))
    return float(len(distinct)) / float(catalogue_size)


def item_popularity(train_ratings):
    """Training rating count per item, the popularity measure used throughout."""
    counts = train_ratings.groupby("item_id").size()
    popularity = {}
    for item_id, count in counts.items():
        popularity[int(item_id)] = int(count)
    return popularity


def mean_recommended_popularity(recommendations_by_user, popularity):
    """Mean training rating count of everything recommended, across all users.

    Raw counts are readable but scale with the dataset, so
    :func:`mean_popularity_percentile` is the version to compare across k and across
    experiments. Both are reported.
    """
    values = []
    for items in recommendations_by_user.values():
        for item in items:
            values.append(float(popularity.get(int(item), 0)))
    if len(values) == 0:
        return float("nan")
    return float(np.mean(values))


def popularity_percentiles(popularity):
    """Map item id to its popularity rank as a fraction, 0 rarest and 1 most rated."""
    if len(popularity) == 0:
        return {}
    item_ids = sorted(popularity.keys())
    counts = np.array([popularity[item_id] for item_id in item_ids], dtype=np.float64)
    order = np.argsort(counts, kind="mergesort")
    ranks = np.empty(len(counts), dtype=np.float64)
    ranks[order] = np.arange(len(counts), dtype=np.float64)
    if len(counts) > 1:
        ranks = ranks / float(len(counts) - 1)
    percentiles = {}
    for position, item_id in enumerate(item_ids):
        percentiles[item_id] = float(ranks[position])
    return percentiles


def mean_popularity_percentile(recommendations_by_user, percentiles):
    """Mean popularity percentile of recommended items, comparable across datasets."""
    values = []
    for items in recommendations_by_user.values():
        for item in items:
            values.append(percentiles.get(int(item), 0.0))
    if len(values) == 0:
        return float("nan")
    return float(np.mean(values))


def recommendation_gini(recommendations_by_user, catalogue_items):
    """Gini coefficient over how often each catalogue item is recommended.

    0 means every item is recommended equally often, 1 means one item takes everything.
    Coverage only asks whether an item was ever shown; Gini asks how lopsided the exposure
    is, which is the sharper version of the same question.
    """
    if len(catalogue_items) == 0:
        return float("nan")
    counts = {}
    for item_id in catalogue_items:
        counts[int(item_id)] = 0
    for items in recommendations_by_user.values():
        for item in items:
            key = int(item)
            if key in counts:
                counts[key] = counts[key] + 1
    values = np.sort(np.array(list(counts.values()), dtype=np.float64))
    total = values.sum()
    if total <= EPSILON:
        return float("nan")
    n = len(values)
    index = np.arange(1, n + 1, dtype=np.float64)
    weighted = np.sum((2.0 * index - n - 1.0) * values)
    return float(weighted / (n * total))

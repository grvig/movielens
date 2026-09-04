"""Content-based recommendation over TF-IDF item documents.

An item is a sparse TF-IDF vector over genre tokens, title tokens and a release-decade
token (see :mod:`src.data.features`). A user is the rating-weighted average of the items
they rated, and candidates are scored by cosine similarity to that profile.

The weighting is **mean-centred**, which is the part that matters::

    profile = sum((rating - user_mean) * item_vector) / sum(abs(rating - user_mean))

Weighting by the raw rating instead is the classic content-based bug. Every weight is then
positive, so a user who rated a horror film 1 out of 5 gets pushed *towards* horror, and
every profile converges on "the genres I happen to watch" rather than "the genres I like".
Centring on the user mean makes a below-average rating push away from those features,
which is the whole point of having ratings rather than a watch list.

This model is expected to lose on RMSE and win on catalogue coverage. It has no way to
learn that a film is popular, so it recommends across the long tail, which is exactly the
trade-off the report is built around.
"""

import numpy as np

from src.data.features import build_item_features
from src.data.features import build_item_index
from src.models.base import Recommender

EPSILON = 1e-12


class ContentBased(Recommender):
    """Cosine similarity between a user profile and item TF-IDF vectors.

    The item table arrives through the constructor rather than through ``fit``, so the
    ``fit(train_ratings)`` signature stays identical to every other model and the harness
    keeps treating them all the same way.
    """

    name = "content"

    def __init__(self, config, items):
        super().__init__(config)
        params = config.model_params("content")
        self.calibration_scale = float(params["calibration_scale"])
        self.items = items
        self.feature_matrix = None
        self.item_index = {}
        self.user_index = {}
        self.profile_matrix = None
        self.user_mean = {}

    def fit(self, train_ratings):
        self.record_training_data(train_ratings)
        self.feature_matrix, feature_item_ids, _ = build_item_features(self.items, self.config)
        self.item_index = build_item_index(feature_item_ids)
        self.user_mean = user_means(train_ratings)
        self.build_profiles(train_ratings)
        return self

    def build_profiles(self, train_ratings):
        """One unit-length profile vector per user, stored as a dense matrix."""
        user_ids = np.sort(train_ratings["user_id"].unique())
        self.user_index = {}
        for position, user_id in enumerate(user_ids):
            self.user_index[int(user_id)] = position

        n_features = self.feature_matrix.shape[1]
        profiles = np.zeros((len(user_ids), n_features), dtype=np.float64)

        for user_id, group in train_ratings.groupby("user_id"):
            item_ids = group["item_id"].to_numpy(dtype=np.int64)
            ratings = group["rating"].to_numpy(dtype=np.float64)
            centre = self.user_mean.get(int(user_id), self.global_mean)
            rows, weights = self.aligned_rows_and_weights(item_ids, ratings - centre)
            if len(rows) == 0:
                continue
            profile = weighted_profile(self.feature_matrix, rows, weights)
            profiles[self.user_index[int(user_id)]] = profile

        self.profile_matrix = profiles

    def aligned_rows_and_weights(self, item_ids, weights):
        """Row positions and their weights, dropping items with no feature vector.

        Rows and weights are collected together rather than filtered separately. Filtering
        the rows and then truncating the weights would silently pair each weight with the
        wrong item as soon as one item in the middle was missing.
        """
        rows = []
        kept_weights = []
        for item_id, weight in zip(item_ids, weights):
            position = self.item_index.get(int(item_id))
            if position is None:
                continue
            rows.append(position)
            kept_weights.append(weight)
        return rows, np.array(kept_weights, dtype=np.float64)

    def profile_for(self, user_id):
        position = self.user_index.get(int(user_id))
        if position is None:
            return None
        return self.profile_matrix[position]

    def rank_scores(self, user_id, candidate_items):
        """Cosine similarity in [-1, 1]. Unknown users score zero everywhere."""
        self.check_fitted()
        profile = self.profile_for(user_id)
        scores = np.zeros(len(candidate_items), dtype=np.float64)
        if profile is None:
            return scores
        positions = []
        rows = []
        for position, item_id in enumerate(candidate_items):
            row = self.item_index.get(int(item_id))
            if row is None:
                continue
            positions.append(position)
            rows.append(row)
        if len(rows) == 0:
            return scores
        # One sparse-dense product for every candidate at once. Scoring row by row in a
        # Python loop costs 1682 sparse slices per user and dominates a full evaluation.
        scores[positions] = np.asarray(self.feature_matrix[rows].dot(profile)).ravel()
        return scores

    def predict(self, user_id, candidate_items):
        """Map cosine similarity onto the rating scale.

        A cosine is not a rating, so it has to be shifted onto the user's own scale and
        stretched by some factor. The factor is a fixed config value here; the next commit
        fits it on the validation split instead, which is the honest version.
        """
        self.check_fitted()
        similarities = self.rank_scores(user_id, candidate_items)
        centre = self.user_mean.get(int(user_id), self.global_mean)
        return self.clip(centre + self.calibration_scale * similarities)


def user_means(train_ratings):
    means = {}
    grouped = train_ratings.groupby("user_id")["rating"].mean()
    for user_id, value in grouped.items():
        means[int(user_id)] = float(value)
    return means


def weighted_profile(feature_matrix, rows, weights):
    """Mean-centred rating-weighted average of item vectors, normalised to unit length.

    Falls back to an unweighted mean when a user rated everything identically, which makes
    every weight zero and the weighted average undefined.
    """
    selected = feature_matrix[rows]
    total_weight = float(np.sum(np.abs(weights)))
    if total_weight < EPSILON:
        profile = np.asarray(selected.mean(axis=0)).ravel()
    else:
        # selected.T.dot(weights) rather than weights @ selected: a dense array on the
        # left of a scipy sparse matrix does not reliably dispatch to sparse matmul, and
        # gets the wrong answer without raising.
        profile = np.asarray(selected.T.dot(weights)).ravel() / total_weight
    norm = float(np.linalg.norm(profile))
    if norm < EPSILON:
        return profile
    return profile / norm

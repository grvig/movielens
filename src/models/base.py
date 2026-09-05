"""The shared recommender interface.

Every model in :mod:`src.models` implements this contract and nothing else.  The
evaluation harness never branches on model type, which is the entire reason the
comparison between four model families is fair.  If a model needs special handling, the
special handling goes inside the model.

Three methods make up the contract:

``fit(train_ratings)``
    Learn from a ratings frame with columns ``user_id``, ``item_id``, ``rating``,
    ``timestamp``.

``predict(user_id, candidate_items)``
    Return one predicted rating per candidate item, clipped into the rating range.

``recommend(user_id, k)``
    Return the top ``k`` item ids for a user, excluding items already rated in training.
"""

from abc import ABC
from abc import abstractmethod

import numpy as np

RATING_COLUMNS = ["user_id", "item_id", "rating", "timestamp"]


class Recommender(ABC):
    """Base class holding the bookkeeping every model needs."""

    name = "recommender"

    def __init__(self, config, name=None):
        # A model class can back more than one registered model when the same algorithm is
        # tuned two ways. The name is what the results table keys on, so it has to differ
        # even though the class does not.
        if name is not None:
            self.name = name
        self.config = config
        evaluation = config.section("evaluation")
        self.rating_min = float(evaluation["rating_min"])
        self.rating_max = float(evaluation["rating_max"])
        self.global_mean = 0.0
        self.all_items = np.array([], dtype=np.int64)
        self.known_users = set()
        self.known_items = set()
        self.train_items_by_user = {}
        self.is_fitted = False

    # -- contract ---------------------------------------------------------------

    @abstractmethod
    def fit(self, train_ratings):
        """Learn model parameters from the training ratings."""

    @abstractmethod
    def predict(self, user_id, candidate_items):
        """Return predicted ratings for ``candidate_items`` as a float array."""

    def recommend(self, user_id, k):
        """Return the top ``k`` item ids for ``user_id``.

        Ties are broken by ascending item id so that two runs of the same model produce
        byte-identical recommendation lists.
        """
        candidates = self.candidate_items(user_id)
        if candidates.size == 0:
            return np.array([], dtype=np.int64)
        scores = self.rank_scores(user_id, candidates)
        order = np.lexsort((candidates, -np.asarray(scores, dtype=np.float64)))
        top = order[:k]
        return candidates[top]

    def rank_scores(self, user_id, candidate_items):
        """Scores used for ranking.

        Defaults to the predicted rating.  Models whose ranking signal is not a rating
        (most-popular counts, content cosine similarity) override this and leave
        ``predict`` alone, so the two metric families stay independent.
        """
        return self.predict(user_id, candidate_items)

    # -- shared bookkeeping -----------------------------------------------------

    def record_training_data(self, train_ratings):
        """Store the state every model needs.  Call this first in every ``fit``."""
        self.global_mean = float(train_ratings["rating"].mean())
        self.all_items = np.sort(train_ratings["item_id"].unique()).astype(np.int64)
        self.known_users = set(train_ratings["user_id"].unique().tolist())
        self.known_items = set(self.all_items.tolist())
        grouped = train_ratings.groupby("user_id")["item_id"]
        self.train_items_by_user = {}
        for user_id, items in grouped:
            self.train_items_by_user[int(user_id)] = set(int(i) for i in items)
        self.is_fitted = True

    def candidate_items(self, user_id):
        """All training items the user has not already rated."""
        seen = self.train_items_by_user.get(int(user_id), set())
        if len(seen) == 0:
            return self.all_items.copy()
        mask = np.array([item not in seen for item in self.all_items], dtype=bool)
        return self.all_items[mask]

    def clip(self, values):
        return np.clip(np.asarray(values, dtype=np.float64), self.rating_min, self.rating_max)

    def check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError(self.name + " has not been fitted")

    def fallback_scores(self, candidate_items):
        """Global mean for every candidate, used for unknown users and items."""
        return np.full(len(candidate_items), self.global_mean, dtype=np.float64)

    def __repr__(self):
        return self.__class__.__name__ + "(name=" + repr(self.name) + ")"

"""Non-personalised and near-non-personalised baselines.

Half the value of this project's report is showing how much — or how little — the fancy
models beat these.  There are two distinct floors here and they measure different things:

*   ``GlobalMean``, ``UserMean`` and ``ItemMean`` are the **rating-prediction** floor.
    Global-mean and user-mean have no per-item signal at all, so their recommendation
    lists are effectively arbitrary and their precision@k will sit near zero.  That is
    correct behaviour, not a bug.
*   ``MostPopular`` is the **ranking** floor, and it is the one that is hard to beat.  Its
    rating predictions are just the global mean, so its RMSE matches GlobalMean exactly;
    only its ranking numbers are interesting.

Both mean models shrink their biases towards the global mean.  A raw user mean over a
20-rating user is a noisy estimate, and shrinkage is one line that measurably helps.
"""

import numpy as np

from src.models.base import Recommender


class GlobalMean(Recommender):
    """Predict the training global mean for everything."""

    name = "global_mean"

    def fit(self, train_ratings):
        self.record_training_data(train_ratings)
        return self

    def predict(self, user_id, candidate_items):
        self.check_fitted()
        return self.clip(self.fallback_scores(candidate_items))


class UserMean(Recommender):
    """Predict the global mean plus a shrunk per-user bias."""

    name = "user_mean"

    def __init__(self, config):
        super().__init__(config)
        params = config.model_params("user_mean")
        self.shrinkage = float(params["shrinkage"])
        self.user_bias = {}

    def fit(self, train_ratings):
        self.record_training_data(train_ratings)
        self.user_bias = shrunk_bias(train_ratings, "user_id", self.global_mean, self.shrinkage)
        return self

    def predict(self, user_id, candidate_items):
        self.check_fitted()
        bias = self.user_bias.get(int(user_id), 0.0)
        scores = np.full(len(candidate_items), self.global_mean + bias, dtype=np.float64)
        return self.clip(scores)


class ItemMean(Recommender):
    """Predict the global mean plus a shrunk per-item bias."""

    name = "item_mean"

    def __init__(self, config):
        super().__init__(config)
        params = config.model_params("item_mean")
        self.shrinkage = float(params["shrinkage"])
        self.item_bias = {}

    def fit(self, train_ratings):
        self.record_training_data(train_ratings)
        self.item_bias = shrunk_bias(train_ratings, "item_id", self.global_mean, self.shrinkage)
        return self

    def predict(self, user_id, candidate_items):
        self.check_fitted()
        scores = np.empty(len(candidate_items), dtype=np.float64)
        for position, item_id in enumerate(candidate_items):
            bias = self.item_bias.get(int(item_id), 0.0)
            scores[position] = self.global_mean + bias
        return self.clip(scores)


class MostPopular(Recommender):
    """Rank by training rating count; predict the global mean.

    Keeping the two apart is deliberate.  Popularity is a ranking signal and not a rating
    estimate, and bending it into one would make this baseline look like something it is
    not.  Its RMSE is expected to equal GlobalMean's.
    """

    name = "most_popular"

    def __init__(self, config):
        super().__init__(config)
        self.item_counts = {}

    def fit(self, train_ratings):
        self.record_training_data(train_ratings)
        counts = train_ratings.groupby("item_id").size()
        self.item_counts = {}
        for item_id, count in counts.items():
            self.item_counts[int(item_id)] = int(count)
        return self

    def predict(self, user_id, candidate_items):
        self.check_fitted()
        return self.clip(self.fallback_scores(candidate_items))

    def rank_scores(self, user_id, candidate_items):
        self.check_fitted()
        scores = np.empty(len(candidate_items), dtype=np.float64)
        for position, item_id in enumerate(candidate_items):
            scores[position] = float(self.item_counts.get(int(item_id), 0))
        return scores


def shrunk_bias(train_ratings, key_column, global_mean, shrinkage):
    """Bias per group, pulled towards zero in proportion to how little data backs it.

    bias = sum(rating - global_mean) / (count + shrinkage)
    """
    deviations = train_ratings["rating"] - global_mean
    grouped = deviations.groupby(train_ratings[key_column])
    totals = grouped.sum()
    counts = grouped.size()
    bias = {}
    for key in totals.index:
        bias[int(key)] = float(totals.loc[key] / (counts.loc[key] + shrinkage))
    return bias

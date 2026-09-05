"""Item-based k-nearest-neighbour collaborative filtering.

A rating is predicted from how the user rated the most similar items they have already
seen::

    r(u, i) = mean(i) + sum_j s(i, j) * (r(u, j) - mean(j)) / sum_j abs(s(i, j))

where j runs over the ``neighbourhood_size`` items most similar to i that the user has
rated. The similarity itself comes from :mod:`src.models.similarity`, which handles the
centring convention and the significance shrinkage.

The deviations are taken against the same means the similarity was centred on. Mixing the
two - centring the similarity on user means and then adding back item means - is an easy
mistake that produces plausible-looking predictions that are quietly wrong, so the means
used here are the ones returned by the centring step rather than recomputed.

One property of this formulation is worth knowing before tuning it. Because the weighted
average divides by the sum of absolute similarities, **shrinkage does not pull predictions
towards the item mean**. Scaling every similarity by the same factor leaves the ratio
unchanged. Shrinkage only bites when co-occurrence counts differ between neighbours, where
it re-weights the thinly-supported ones downwards, and when it changes which items make it
into the top-k neighbourhood at all. Expecting a regularisation-style pull towards the mean
would make the shrinkage sweep look broken when it is behaving correctly.

Expected behaviour: better RMSE than any baseline, worse than matrix factorisation, and
noticeably wider catalogue coverage than MF because a neighbourhood model can only
recommend items that share co-raters with something the user already liked.
"""

import numpy as np

from src.data.matrix import build_rating_matrix
from src.models.base import Recommender
from src.models.similarity import EPSILON
from src.models.similarity import apply_shrinkage
from src.models.similarity import centre_matrix
from src.models.similarity import cooccurrence_counts
from src.models.similarity import cosine_similarity
from src.models.similarity import top_neighbours


class ItemKNN(Recommender):
    """Neighbourhood model over mean-centred item similarities."""

    name = "item_knn"

    def __init__(self, config, neighbourhood_size=None, shrinkage=None, centering=None,
                 mean_shrinkage=None, name=None):
        super().__init__(config, name=name)
        params = config.model_params("item_knn")
        if neighbourhood_size is None:
            neighbourhood_size = params["neighbourhood_size"]
        if shrinkage is None:
            shrinkage = params["shrinkage"]
        if centering is None:
            centering = params["centering"]
        if mean_shrinkage is None:
            mean_shrinkage = params["mean_shrinkage"]
        self.neighbourhood_size = int(neighbourhood_size)
        self.shrinkage = float(shrinkage)
        self.centering = str(centering)
        self.mean_shrinkage = float(mean_shrinkage)
        self.rating_matrix = None
        self.similarity = None
        self.centre_means = None
        self.item_means = None

    def fit(self, train_ratings):
        self.record_training_data(train_ratings)
        self.rating_matrix = build_rating_matrix(train_ratings)
        centred, means = centre_matrix(self.rating_matrix.matrix, self.centering)
        self.centre_means = means
        self.similarity = cosine_similarity(centred)
        counts = cooccurrence_counts(self.rating_matrix.matrix)
        self.similarity = apply_shrinkage(self.similarity, counts, self.shrinkage)
        self.item_means = item_mean_vector(self.rating_matrix, self.mean_shrinkage)
        return self

    def deviations_for(self, user_id):
        """The user's rated item positions and their deviations from the centring means.

        Item centring stores one mean per item, so the deviation for a rating is looked up
        by column. User centring stores one mean per user, so every deviation for that user
        uses the same value.
        """
        positions, values = self.rating_matrix.user_row(user_id)
        if len(positions) == 0:
            return positions, values
        if self.centering == "item":
            return positions, values - self.centre_means[positions]
        user_position = self.rating_matrix.user_position.get(int(user_id))
        return positions, values - self.centre_means[user_position]

    def predict(self, user_id, candidate_items):
        self.check_fitted()
        scores = np.full(len(candidate_items), self.global_mean, dtype=np.float64)
        rated_positions, deviations = self.deviations_for(user_id)
        item_positions = self.rating_matrix.item_positions(candidate_items)
        deviation_by_position = dict(zip(rated_positions.tolist(), deviations.tolist()))

        for index, position in enumerate(item_positions):
            if position < 0:
                continue
            base = self.item_means[position]
            if len(rated_positions) == 0:
                scores[index] = base
                continue
            neighbours, similarities = top_neighbours(
                self.similarity[position], rated_positions, self.neighbourhood_size
            )
            scores[index] = base + weighted_deviation(
                neighbours, similarities, deviation_by_position
            )
        return self.clip(scores)

    def rank_scores(self, user_id, candidate_items):
        """Rank by predicted rating.

        A neighbourhood model's prediction already carries its confidence in the item, so
        unlike popularity or cosine there is no separate ranking signal to keep apart.
        """
        return self.predict(user_id, candidate_items)


def weighted_deviation(neighbours, similarities, deviation_by_position):
    """Similarity-weighted mean of the user's deviations on the neighbouring items."""
    if len(neighbours) == 0:
        return 0.0
    numerator = 0.0
    denominator = 0.0
    for position, similarity in zip(neighbours, similarities):
        deviation = deviation_by_position.get(int(position))
        if deviation is None:
            continue
        numerator = numerator + similarity * deviation
        denominator = denominator + abs(similarity)
    if denominator < EPSILON:
        return 0.0
    return numerator / denominator


def item_mean_vector(rating_matrix, shrinkage):
    """Per-item mean rating, shrunk towards the overall mean.

        mean(i) = overall + sum(rating - overall) / (count + shrinkage)

    The shrinkage is not optional dressing. This is the base term the prediction is built
    on, and MovieLens 100K has twelve items whose entire training history is one or two
    ratings that happen to be 5.0. With a raw mean those items score a perfect 5 and take
    over every top-k list, which is what happened on the first full run: eight of the ten
    recommendations for user 1 were items with a single rating.

    The ItemMean baseline already shrinks with the same parameter, so leaving it out here
    also made the two models incomparable, in a project whose premise is that every model
    gets identical treatment.

    Note that this deliberately does not change the deviation term. The similarity is
    centred on raw means, which is the standard arrangement: the base term is an estimate
    of an item's rating level and needs regularising, while the centring is about the
    relative structure the similarity is measured over.
    """
    as_csc = rating_matrix.matrix.tocsc()
    sums = np.asarray(as_csc.sum(axis=0)).ravel()
    counts = np.diff(as_csc.indptr).astype(np.float64)
    overall = float(sums.sum() / max(1.0, counts.sum()))
    deviations = sums - overall * counts
    return overall + deviations / (counts + shrinkage)

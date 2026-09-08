"""Weighted blends of the content-based and matrix factorisation models.

Two hybrids share this code, differing only in where the blend weight comes from.

``density``
    The hybrid the project plan specified: the weight is a function of how many ratings the
    user has, ``w(n) = sigmoid(a + b * log(1 + n))``, with ``a`` and ``b`` fitted on
    validation. The plan expected content-heavy blends for sparse profiles and CF-heavy
    ones for dense profiles, with a crossover in between.

``fixed``
    A constant weight, used to trace the accuracy-versus-coverage frontier. Content-based
    reaches 46% of the catalogue against matrix factorisation's 15%, so sliding the weight
    answers what a coverage target costs in accuracy.

Normalisation is the part that has to be right. The two models' ranking signals are not on
the same scale at all - content scores are cosine similarities in [-1, 1], matrix
factorisation scores are predicted ratings in [1, 5] - so a weight applied to raw scores
would be meaningless and would silently make every hybrid result wrong. Scores are z-scored
**per user** across that user's candidate set before blending.

Rating predictions are blended directly rather than z-scored, because both models already
return a rating on the same [1, 5] scale there and the blend stays interpretable.
"""

import numpy as np

from src.models.base import Recommender

EPSILON = 1e-12
WEIGHT_MODES = ["density", "fixed"]
DEFAULT_BINS = [(1, 20), (21, 40), (41, 100), (101, 250), (251, 100000)]
WEIGHT_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
LOGIT_CLIP = 0.05


def normalise_scores(scores):
    """Z-score a user's candidate scores so two models can be compared on one axis.

    A constant score vector - which happens for a model with nothing to say about a user -
    has zero spread, so it is returned as all zeros rather than dividing by zero. That
    makes the component contribute nothing to the blend, which is the right behaviour.
    """
    values = np.asarray(scores, dtype=np.float64)
    if values.size == 0:
        return values
    spread = float(np.std(values))
    if spread < EPSILON:
        return np.zeros_like(values)
    return (values - float(np.mean(values))) / spread


def sigmoid(value):
    return 1.0 / (1.0 + np.exp(-value))


def weight_for_history(n_ratings, intercept, slope):
    return float(sigmoid(intercept + slope * np.log1p(max(0, n_ratings))))


def fit_weight_curve(history_lengths, best_weights):
    """Least-squares fit of logit(w) against log(1 + history).

    Bin optima of exactly 0 or 1 have infinite logits, so they are clipped inwards first.
    That biases the fitted curve slightly towards the middle, which is the conservative
    direction: it understates rather than overstates how extreme the weighting should be.
    """
    lengths = np.asarray(history_lengths, dtype=np.float64)
    weights = np.clip(np.asarray(best_weights, dtype=np.float64), LOGIT_CLIP, 1.0 - LOGIT_CLIP)
    if len(lengths) < 2:
        return 0.0, 0.0
    x = np.log1p(lengths)
    y = np.log(weights / (1.0 - weights))
    slope, intercept = np.polyfit(x, y, 1)
    return float(intercept), float(slope)


class WeightedHybrid(Recommender):
    """Blend of a content model and a matrix factorisation model."""

    name = "hybrid"

    def __init__(self, config, content_model, cf_model, weight_mode="density",
                 fixed_weight=0.5, validation=None, name=None):
        super().__init__(config, name=name)
        if weight_mode not in WEIGHT_MODES:
            raise ValueError("weight_mode must be one of " + ", ".join(WEIGHT_MODES))
        self.content_model = content_model
        self.cf_model = cf_model
        self.weight_mode = weight_mode
        self.fixed_weight = float(fixed_weight)
        self.validation = validation
        self.intercept = 0.0
        self.slope = 0.0
        self.curve_points = []
        self.train_history = {}

    def fit(self, train_ratings):
        self.record_training_data(train_ratings)
        self.content_model.fit(train_ratings)
        self.cf_model.fit(train_ratings)
        counts = train_ratings.groupby("user_id").size()
        self.train_history = {}
        for user_id, count in counts.items():
            self.train_history[int(user_id)] = int(count)
        if self.weight_mode == "density" and self.validation is not None:
            self.fit_density_curve(self.validation)
        return self

    def weight_for(self, user_id):
        """How much of the blend goes to the content model, in [0, 1]."""
        if self.weight_mode == "fixed":
            return self.fixed_weight
        history = self.train_history.get(int(user_id), 0)
        return weight_for_history(history, self.intercept, self.slope)

    def component_rank_scores(self, user_id, candidate_items):
        content = normalise_scores(self.content_model.rank_scores(user_id, candidate_items))
        collaborative = normalise_scores(self.cf_model.rank_scores(user_id, candidate_items))
        return content, collaborative

    def rank_scores(self, user_id, candidate_items):
        self.check_fitted()
        content, collaborative = self.component_rank_scores(user_id, candidate_items)
        weight = self.weight_for(user_id)
        return weight * content + (1.0 - weight) * collaborative

    def predict(self, user_id, candidate_items):
        """Blend the two rating predictions directly.

        Both components return a rating on the same scale here, so z-scoring would throw
        away the very thing RMSE is measuring.
        """
        self.check_fitted()
        content = np.asarray(
            self.content_model.predict(user_id, candidate_items), dtype=np.float64
        )
        collaborative = np.asarray(
            self.cf_model.predict(user_id, candidate_items), dtype=np.float64
        )
        weight = self.weight_for(user_id)
        return self.clip(weight * content + (1.0 - weight) * collaborative)

    def fit_density_curve(self, validation, bins=None, k=10):
        """Choose the best content weight per history bin, then fit a curve through them.

        Fitted on validation and never on test. For each user the two component score
        vectors are computed once and then blended at every candidate weight, so the search
        costs one scoring pass rather than one per weight.
        """
        if bins is None:
            bins = DEFAULT_BINS
        threshold = float(self.config.section("evaluation")["relevance_threshold"])
        liked = validation[validation["rating"] >= threshold]
        relevant = {}
        for user_id, group in liked.groupby("user_id"):
            relevant[int(user_id)] = set(int(item) for item in group["item_id"])

        self.curve_points = []
        centres = []
        best_weights = []
        for low, high in bins:
            users = self.users_in_bin(relevant.keys(), low, high)
            if len(users) == 0:
                continue
            scores = self.weight_grid_scores(users, relevant, k)
            best_index = int(np.argmax(scores))
            centre = float(np.median([self.train_history.get(u, 0) for u in users]))
            centres.append(centre)
            best_weights.append(WEIGHT_GRID[best_index])
            self.curve_points.append({
                "bin_low": low,
                "bin_high": high,
                "users": len(users),
                "median_history": centre,
                "best_weight": WEIGHT_GRID[best_index],
                "best_precision": float(scores[best_index]),
                "precision_at_weight_zero": float(scores[0]),
                "precision_at_weight_one": float(scores[-1]),
            })
        self.intercept, self.slope = fit_weight_curve(centres, best_weights)
        return self.curve_points

    def users_in_bin(self, user_ids, low, high):
        chosen = []
        for user_id in user_ids:
            history = self.train_history.get(int(user_id), 0)
            if low <= history <= high:
                chosen.append(int(user_id))
        return sorted(chosen)

    def weight_grid_scores(self, users, relevant, k):
        """Mean precision@k over these users at each weight on the grid."""
        totals = np.zeros(len(WEIGHT_GRID), dtype=np.float64)
        counted = 0
        for user_id in users:
            candidates = self.candidate_items(user_id)
            if candidates.size == 0:
                continue
            content, collaborative = self.component_rank_scores(user_id, candidates)
            liked = relevant.get(int(user_id), set())
            if len(liked) == 0:
                continue
            counted = counted + 1
            for position, weight in enumerate(WEIGHT_GRID):
                blended = weight * content + (1.0 - weight) * collaborative
                order = np.lexsort((candidates, -blended))
                top = candidates[order[:k]]
                hits = 0
                for item in top:
                    if int(item) in liked:
                        hits = hits + 1
                totals[position] = totals[position] + float(hits) / float(k)
        if counted == 0:
            return totals
        return totals / counted

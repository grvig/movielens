"""Matrix factorisation trained by SGD on the observed entries.

The standard biased model::

    r(u, i) = mu + b_u + b_i + p_u . q_i

fitted by stochastic gradient descent over observed ratings only, with L2 regularisation
on all four parameter groups. Training on observed entries alone is the point: treating the
94% of the matrix that is missing as zeros would teach the model that most films are
terrible rather than that most films are unseen.

The biases matter more than they look. Without them the latent factors spend their capacity
learning that some users rate generously and some films are widely liked, which is exactly
the structure the two bias vectors capture in far fewer parameters.

Implementation note: the inner loop runs over pre-extracted numpy integer arrays rather
than a DataFrame. Iterating rows of a DataFrame for 70,000 ratings times 60 epochs is the
one part of this project that can turn a sweep into an afternoon.
"""

import hashlib

import numpy as np

from src.data.matrix import build_rating_matrix
from src.models.base import Recommender

IMPROVEMENT_TOLERANCE = 1e-5


class MatrixFactorization(Recommender):
    """Biased matrix factorisation with L2-regularised SGD."""

    name = "mf"

    def __init__(self, config, n_factors=None, learning_rate=None, regularisation=None,
                 n_epochs=None, validation=None, patience=None, cache=False):
        super().__init__(config)
        params = config.model_params("mf")
        if patience is None:
            patience = params["patience"]
        self.patience = int(patience)
        self.validation = validation
        self.cache = cache
        self.val_history = []
        self.best_epoch = None
        self.loaded_from_cache = False
        if n_factors is None:
            n_factors = params["n_factors"]
        if learning_rate is None:
            learning_rate = params["learning_rate"]
        if regularisation is None:
            regularisation = params["regularisation"]
        if n_epochs is None:
            n_epochs = params["n_epochs"]
        self.n_factors = int(n_factors)
        self.learning_rate = float(learning_rate)
        self.regularisation = float(regularisation)
        self.n_epochs = int(n_epochs)
        self.init_scale = float(params["init_scale"])
        self.rating_matrix = None
        self.user_bias = None
        self.item_bias = None
        self.user_factors = None
        self.item_factors = None
        self.train_history = []

    def fit(self, train_ratings):
        self.record_training_data(train_ratings)
        self.rating_matrix = build_rating_matrix(train_ratings)
        users, items, ratings = self.training_arrays(train_ratings)

        if self.cache and self.load_from_cache(users, items, ratings):
            return self

        # One generator for the whole fit, taken from the shared seed. Initialisation and
        # shuffling draw from the same stream, so a refit reproduces bit for bit.
        rng = self.config.fresh_rng()
        self.initialise_parameters(
            self.rating_matrix.n_users, self.rating_matrix.n_items, rng
        )
        self.train_history = []
        self.val_history = []
        validation_arrays = self.validation_arrays()

        best_rmse = float("inf")
        best_parameters = None
        epochs_without_improvement = 0

        for epoch in range(self.n_epochs):
            order = rng.permutation(len(ratings))
            self.run_epoch(users, items, ratings, order)
            self.train_history.append(self.training_rmse(users, items, ratings))

            if validation_arrays is None:
                continue

            validation_rmse = self.training_rmse(*validation_arrays)
            self.val_history.append(validation_rmse)
            if validation_rmse < best_rmse - IMPROVEMENT_TOLERANCE:
                best_rmse = validation_rmse
                best_parameters = self.snapshot()
                self.best_epoch = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement = epochs_without_improvement + 1
                if epochs_without_improvement >= self.patience:
                    break

        # Restore the best epoch rather than keeping the last one. Without this, early
        # stopping would detect the overfitting and then hand back the overfitted model.
        if best_parameters is not None:
            self.restore(best_parameters)

        if self.cache:
            self.save_to_cache(users, items, ratings)
        return self

    def validation_arrays(self):
        """Validation ratings as matrix positions, dropping users and items not in train.

        A validation rating for an item the model has never seen tells us nothing about
        whether it is overfitting, so those rows are excluded rather than scored against
        the global-mean fallback.
        """
        if self.validation is None or len(self.validation) == 0:
            return None
        users = []
        items = []
        values = []
        for user_id, item_id, rating in zip(
            self.validation["user_id"],
            self.validation["item_id"],
            self.validation["rating"],
        ):
            user_position = self.rating_matrix.user_position.get(int(user_id))
            item_position = self.rating_matrix.item_position.get(int(item_id))
            if user_position is None or item_position is None:
                continue
            users.append(user_position)
            items.append(item_position)
            values.append(float(rating))
        if len(values) == 0:
            return None
        return (
            np.array(users, dtype=np.int64),
            np.array(items, dtype=np.int64),
            np.array(values, dtype=np.float64),
        )

    def snapshot(self):
        return (
            self.user_bias.copy(),
            self.item_bias.copy(),
            self.user_factors.copy(),
            self.item_factors.copy(),
        )

    def restore(self, parameters):
        self.user_bias = parameters[0]
        self.item_bias = parameters[1]
        self.user_factors = parameters[2]
        self.item_factors = parameters[3]

    def training_arrays(self, train_ratings):
        """Ratings as three aligned numpy arrays of matrix positions and values."""
        user_positions = np.array(
            [self.rating_matrix.user_position[int(value)] for value in train_ratings["user_id"]],
            dtype=np.int64,
        )
        item_positions = np.array(
            [self.rating_matrix.item_position[int(value)] for value in train_ratings["item_id"]],
            dtype=np.int64,
        )
        values = train_ratings["rating"].to_numpy(dtype=np.float64)
        return user_positions, item_positions, values

    def initialise_parameters(self, n_users, n_items, rng):
        self.user_bias = np.zeros(n_users, dtype=np.float64)
        self.item_bias = np.zeros(n_items, dtype=np.float64)
        self.user_factors = rng.normal(0.0, self.init_scale, (n_users, self.n_factors))
        self.item_factors = rng.normal(0.0, self.init_scale, (n_items, self.n_factors))

    def run_epoch(self, users, items, ratings, order):
        """One SGD pass. Updates are applied in place, one observed rating at a time."""
        learning_rate = self.learning_rate
        regularisation = self.regularisation
        for index in order:
            user = users[index]
            item = items[index]
            user_vector = self.user_factors[user]
            item_vector = self.item_factors[item]
            prediction = (
                self.global_mean
                + self.user_bias[user]
                + self.item_bias[item]
                + float(np.dot(user_vector, item_vector))
            )
            error = ratings[index] - prediction

            self.user_bias[user] = self.user_bias[user] + learning_rate * (
                error - regularisation * self.user_bias[user]
            )
            self.item_bias[item] = self.item_bias[item] + learning_rate * (
                error - regularisation * self.item_bias[item]
            )
            # The item vector is updated from the pre-update user vector, so a copy is
            # needed. Updating in sequence would feed the new user vector into the item
            # gradient and quietly change the algorithm.
            previous_user = user_vector.copy()
            self.user_factors[user] = user_vector + learning_rate * (
                error * item_vector - regularisation * user_vector
            )
            self.item_factors[item] = item_vector + learning_rate * (
                error * previous_user - regularisation * item_vector
            )

    def cache_key(self, users, items, ratings):
        """Identify a fit by its hyperparameters, its seed and the exact training data.

        The data goes into the key as a hash of the raw arrays rather than as a row count.
        A cache keyed on shape alone would happily return a model trained on the full
        history when asked for one trained on a truncated cold-start history, which is
        precisely the experiment where that mistake would be hardest to notice.
        """
        digest = hashlib.md5()
        digest.update(users.tobytes())
        digest.update(items.tobytes())
        digest.update(ratings.tobytes())
        parts = [
            self.name,
            str(self.n_factors),
            str(self.learning_rate),
            str(self.regularisation),
            str(self.n_epochs),
            str(self.patience),
            str(self.init_scale),
            str(self.config.seed),
            str(self.validation is not None),
            digest.hexdigest(),
        ]
        return hashlib.md5("|".join(parts).encode("utf-8")).hexdigest()

    def cache_path(self, users, items, ratings):
        directory = self.config.path("fits_dir")
        return directory / (self.name + "_" + self.cache_key(users, items, ratings) + ".npz")

    def load_from_cache(self, users, items, ratings):
        path = self.cache_path(users, items, ratings)
        if not path.exists():
            return False
        stored = np.load(path, allow_pickle=False)
        self.user_bias = stored["user_bias"]
        self.item_bias = stored["item_bias"]
        self.user_factors = stored["user_factors"]
        self.item_factors = stored["item_factors"]
        self.train_history = stored["train_history"].tolist()
        self.val_history = stored["val_history"].tolist()
        best_epoch = int(stored["best_epoch"])
        if best_epoch < 0:
            self.best_epoch = None
        else:
            self.best_epoch = best_epoch
        self.loaded_from_cache = True
        return True

    def save_to_cache(self, users, items, ratings):
        path = self.cache_path(users, items, ratings)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.best_epoch is None:
            best_epoch = -1
        else:
            best_epoch = self.best_epoch
        np.savez(
            path,
            user_bias=self.user_bias,
            item_bias=self.item_bias,
            user_factors=self.user_factors,
            item_factors=self.item_factors,
            train_history=np.array(self.train_history, dtype=np.float64),
            val_history=np.array(self.val_history, dtype=np.float64),
            best_epoch=np.array(best_epoch, dtype=np.int64),
        )
        return path

    def raw_predictions(self, users, items):
        """Unclipped predictions for aligned position arrays."""
        dot = np.sum(self.user_factors[users] * self.item_factors[items], axis=1)
        return self.global_mean + self.user_bias[users] + self.item_bias[items] + dot

    def training_rmse(self, users, items, ratings):
        predictions = np.clip(
            self.raw_predictions(users, items), self.rating_min, self.rating_max
        )
        errors = ratings - predictions
        return float(np.sqrt(np.mean(errors * errors)))

    def predict(self, user_id, candidate_items):
        self.check_fitted()
        item_positions = self.rating_matrix.item_positions(candidate_items)
        known = item_positions >= 0
        scores = np.full(len(candidate_items), self.global_mean, dtype=np.float64)
        if not np.any(known):
            return self.clip(scores)

        user_position = self.rating_matrix.user_position.get(int(user_id))
        columns = item_positions[known]
        if user_position is None:
            # An unseen user has no bias and no factors, so the best available estimate is
            # the global mean plus what is known about the item itself.
            scores[known] = self.global_mean + self.item_bias[columns]
            return self.clip(scores)

        rows = np.full(len(columns), user_position, dtype=np.int64)
        scores[known] = self.raw_predictions(rows, columns)
        return self.clip(scores)

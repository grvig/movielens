"""Sparse user-by-item rating matrix.

Both collaborative models work off the same matrix, so it is built in one place with one
set of index maps. Positions in the matrix are dense 0..n-1 indices, which are not the
same thing as MovieLens user and item ids; keeping the two clearly separated is the
easiest way to avoid an off-by-one that silently shifts every prediction by one item.
"""

import numpy as np
from scipy.sparse import csr_matrix


class RatingMatrix:
    """A CSR rating matrix plus the maps between ids and matrix positions."""

    def __init__(self, matrix, user_ids, item_ids):
        self.matrix = matrix
        self.user_ids = user_ids
        self.item_ids = item_ids
        self.user_position = build_position_map(user_ids)
        self.item_position = build_position_map(item_ids)

    @property
    def n_users(self):
        return self.matrix.shape[0]

    @property
    def n_items(self):
        return self.matrix.shape[1]

    def user_row(self, user_id):
        """The user's rated item positions and ratings, or empty arrays if unknown."""
        position = self.user_position.get(int(user_id))
        if position is None:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float64)
        start = self.matrix.indptr[position]
        end = self.matrix.indptr[position + 1]
        return self.matrix.indices[start:end], self.matrix.data[start:end]

    def item_positions(self, item_ids):
        """Matrix columns for the given item ids; -1 marks an item not in training."""
        positions = np.full(len(item_ids), -1, dtype=np.int64)
        for index, item_id in enumerate(item_ids):
            position = self.item_position.get(int(item_id))
            if position is not None:
                positions[index] = position
        return positions


def build_rating_matrix(train_ratings):
    """Build the CSR matrix from a ratings frame.

    Rows are users and columns are items, both sorted by id so the layout is deterministic
    across runs. Duplicate user-item pairs would be summed by the CSR constructor rather
    than rejected, so they are checked for explicitly.
    """
    user_ids = np.sort(train_ratings["user_id"].unique()).astype(np.int64)
    item_ids = np.sort(train_ratings["item_id"].unique()).astype(np.int64)
    user_position = build_position_map(user_ids)
    item_position = build_position_map(item_ids)

    rows = np.array(
        [user_position[int(value)] for value in train_ratings["user_id"]], dtype=np.int64
    )
    columns = np.array(
        [item_position[int(value)] for value in train_ratings["item_id"]], dtype=np.int64
    )
    values = train_ratings["rating"].to_numpy(dtype=np.float64)

    check_no_duplicates(rows, columns)
    matrix = csr_matrix(
        (values, (rows, columns)), shape=(len(user_ids), len(item_ids))
    )
    return RatingMatrix(matrix, user_ids, item_ids)


def check_no_duplicates(rows, columns):
    pairs = set(zip(rows.tolist(), columns.tolist()))
    if len(pairs) != len(rows):
        raise ValueError(
            "training data contains duplicate user-item pairs; the sparse constructor "
            "would sum them into ratings above the rating scale"
        )


def build_position_map(ids):
    positions = {}
    for position, value in enumerate(ids):
        positions[int(value)] = position
    return positions


def binary_pattern(matrix):
    """Copy of the matrix with every stored value set to 1.

    Used for co-occurrence counts. This has to come from the original matrix rather than a
    centred one, because centring can turn an observed rating into an exact zero and the
    count would then quietly drop it.
    """
    pattern = matrix.copy()
    pattern.data = np.ones_like(pattern.data)
    return pattern

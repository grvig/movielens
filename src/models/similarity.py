"""Item-item similarity for the neighbourhood model.

Three steps, each of which changes the answer materially:

**Mean centring.** Raw cosine between rating vectors mostly measures who rated what, not
who agreed. Subtracting a mean first makes the similarity about agreement. Two conventions
exist and the config picks between them: ``item`` centring subtracts each item's mean and
``user`` centring subtracts each user's mean, the latter being what the literature calls
adjusted cosine. User centring removes the effect of a user's personal scale, which is why
it usually wins on this kind of data, but both are implemented so the report can show the
difference rather than assert it.

**Cosine similarity** between the centred item columns.

**Significance shrinkage.** Two items rated by the same two people can reach a similarity
of 1.0 on no evidence at all. Multiplying by ``n_ij / (n_ij + lambda)`` pulls similarities
towards zero in proportion to how little co-rating supports them, and without it those
accidental pairs dominate every neighbourhood.
"""

import numpy as np

from src.data.matrix import binary_pattern

EPSILON = 1e-12
CENTERING_MODES = ["item", "user"]


def centre_matrix(matrix, mode):
    """Subtract item or user means from the observed entries only.

    Only stored entries are touched. A sparse zero means "not rated" and must stay that
    way; subtracting a mean from the whole dense grid would invent a rating for every
    user-item pair that was never observed.
    """
    if mode not in CENTERING_MODES:
        raise ValueError("centering must be one of " + ", ".join(CENTERING_MODES))
    centred = matrix.tocsr(copy=True).astype(np.float64)

    if mode == "item":
        means = column_means(centred)
        centred.data = centred.data - means[centred.indices]
    else:
        means = row_means(centred)
        row_of_entry = np.repeat(np.arange(centred.shape[0]), np.diff(centred.indptr))
        centred.data = centred.data - means[row_of_entry]

    return centred, means


def column_means(matrix):
    """Mean of the observed entries in each column, zero for an empty column."""
    as_csc = matrix.tocsc()
    sums = np.asarray(as_csc.sum(axis=0)).ravel()
    counts = np.diff(as_csc.indptr).astype(np.float64)
    means = np.zeros(matrix.shape[1], dtype=np.float64)
    nonzero = counts > 0
    means[nonzero] = sums[nonzero] / counts[nonzero]
    return means


def row_means(matrix):
    """Mean of the observed entries in each row, zero for an empty row."""
    sums = np.asarray(matrix.sum(axis=1)).ravel()
    counts = np.diff(matrix.indptr).astype(np.float64)
    means = np.zeros(matrix.shape[0], dtype=np.float64)
    nonzero = counts > 0
    means[nonzero] = sums[nonzero] / counts[nonzero]
    return means


def cosine_similarity(centred):
    """Dense item-by-item cosine similarity over the centred columns.

    1682 items gives a 22 MB dense matrix, which is small enough to hold and much faster to
    index than a sparse one during scoring.
    """
    columns = centred.tocsc()
    norms = np.sqrt(np.asarray(columns.multiply(columns).sum(axis=0)).ravel())
    norms[norms < EPSILON] = 1.0
    scale = 1.0 / norms
    normalised = columns.multiply(scale[np.newaxis, :]).tocsc()
    similarity = np.asarray((normalised.T @ normalised).todense(), dtype=np.float64)
    np.fill_diagonal(similarity, 0.0)
    return similarity


def cooccurrence_counts(matrix):
    """How many users rated both items, for every item pair."""
    pattern = binary_pattern(matrix).tocsc()
    counts = np.asarray((pattern.T @ pattern).todense(), dtype=np.float64)
    np.fill_diagonal(counts, 0.0)
    return counts


def apply_shrinkage(similarity, counts, shrinkage):
    """Pull similarities towards zero in proportion to how little evidence backs them."""
    if shrinkage <= 0.0:
        return similarity
    factor = counts / (counts + shrinkage)
    return similarity * factor


def build_item_similarity(rating_matrix, centering, shrinkage):
    """Centre, take cosine similarity, then shrink. Returns a dense item-by-item matrix."""
    centred, _ = centre_matrix(rating_matrix.matrix, centering)
    similarity = cosine_similarity(centred)
    counts = cooccurrence_counts(rating_matrix.matrix)
    return apply_shrinkage(similarity, counts, shrinkage)


def top_neighbours(similarity_row, rated_positions, neighbourhood_size):
    """The k most similar items among those the user actually rated.

    Only positive similarities are kept. A negative neighbour would contribute a term
    pushed in the opposite direction, which is defensible in principle but makes the
    weighted average unstable when the denominator uses absolute values.
    """
    if len(rated_positions) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64)
    similarities = similarity_row[rated_positions]
    positive = similarities > 0.0
    if not np.any(positive):
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64)
    candidate_positions = np.asarray(rated_positions)[positive]
    candidate_similarities = similarities[positive]
    if len(candidate_similarities) > neighbourhood_size:
        keep = np.argpartition(candidate_similarities, -neighbourhood_size)
        keep = keep[-neighbourhood_size:]
        candidate_positions = candidate_positions[keep]
        candidate_similarities = candidate_similarities[keep]
    return candidate_positions, candidate_similarities

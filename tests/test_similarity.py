"""Tests for the rating matrix and the item-item similarity pipeline."""

import numpy as np
import pandas as pd
import pytest

from src.data.matrix import binary_pattern
from src.data.matrix import build_rating_matrix
from src.models.similarity import apply_shrinkage
from src.models.similarity import centre_matrix
from src.models.similarity import cooccurrence_counts
from src.models.similarity import cosine_similarity
from src.models.similarity import top_neighbours


@pytest.fixture
def small_ratings():
    """Items 1 and 2 always agree; item 3 always disagrees with them."""
    rows = [
        (1, 1, 5.0, 100), (1, 2, 5.0, 101), (1, 3, 1.0, 102),
        (2, 1, 4.0, 100), (2, 2, 4.0, 101), (2, 3, 2.0, 102),
        (3, 1, 1.0, 100), (3, 2, 1.0, 101), (3, 3, 5.0, 102),
    ]
    return pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "timestamp"])


def test_matrix_shape_and_ids(small_ratings):
    rating_matrix = build_rating_matrix(small_ratings)
    assert rating_matrix.n_users == 3
    assert rating_matrix.n_items == 3
    assert list(rating_matrix.user_ids) == [1, 2, 3]
    assert list(rating_matrix.item_ids) == [1, 2, 3]


def test_user_row_returns_the_users_ratings(small_ratings):
    rating_matrix = build_rating_matrix(small_ratings)
    positions, values = rating_matrix.user_row(1)
    assert sorted(positions.tolist()) == [0, 1, 2]
    assert sorted(values.tolist()) == [1.0, 5.0, 5.0]


def test_unknown_user_row_is_empty(small_ratings):
    rating_matrix = build_rating_matrix(small_ratings)
    positions, values = rating_matrix.user_row(999)
    assert len(positions) == 0
    assert len(values) == 0


def test_item_positions_mark_unknown_items_with_minus_one(small_ratings):
    rating_matrix = build_rating_matrix(small_ratings)
    positions = rating_matrix.item_positions(np.array([1, 999, 3]))
    assert positions[0] == 0
    assert positions[1] == -1
    assert positions[2] == 2


def test_duplicate_pairs_are_rejected():
    """The CSR constructor would sum them into a rating of 10 rather than complain."""
    duplicated = pd.DataFrame(
        [(1, 1, 5.0, 100), (1, 1, 5.0, 101)],
        columns=["user_id", "item_id", "rating", "timestamp"],
    )
    with pytest.raises(ValueError):
        build_rating_matrix(duplicated)


def test_item_centring_only_touches_observed_entries(small_ratings):
    rating_matrix = build_rating_matrix(small_ratings)
    centred, means = centre_matrix(rating_matrix.matrix, "item")
    assert centred.nnz == rating_matrix.matrix.nnz
    # item 1 has ratings 5, 4, 1 so its mean is 10 / 3
    assert means[0] == pytest.approx(10.0 / 3.0)


def test_user_centring_uses_row_means(small_ratings):
    rating_matrix = build_rating_matrix(small_ratings)
    _, means = centre_matrix(rating_matrix.matrix, "user")
    # user 1 rated 5, 5, 1 so their mean is 11 / 3
    assert means[0] == pytest.approx(11.0 / 3.0)


def test_centring_rejects_an_unknown_mode(small_ratings):
    rating_matrix = build_rating_matrix(small_ratings)
    with pytest.raises(ValueError):
        centre_matrix(rating_matrix.matrix, "sideways")


def test_agreeing_items_are_similar_and_disagreeing_items_are_not(small_ratings):
    rating_matrix = build_rating_matrix(small_ratings)
    centred, _ = centre_matrix(rating_matrix.matrix, "user")
    similarity = cosine_similarity(centred)
    assert similarity[0, 1] > 0.9
    assert similarity[0, 2] < 0.0


def test_similarity_diagonal_is_zeroed(small_ratings):
    rating_matrix = build_rating_matrix(small_ratings)
    centred, _ = centre_matrix(rating_matrix.matrix, "item")
    similarity = cosine_similarity(centred)
    assert np.allclose(np.diag(similarity), 0.0)


def test_similarity_is_symmetric_and_bounded(small_ratings):
    rating_matrix = build_rating_matrix(small_ratings)
    centred, _ = centre_matrix(rating_matrix.matrix, "item")
    similarity = cosine_similarity(centred)
    assert np.allclose(similarity, similarity.T)
    assert np.all(similarity >= -1.0 - 1e-9)
    assert np.all(similarity <= 1.0 + 1e-9)


def test_cooccurrence_counts_co_raters(small_ratings):
    rating_matrix = build_rating_matrix(small_ratings)
    counts = cooccurrence_counts(rating_matrix.matrix)
    # all three users rated both item 1 and item 2
    assert counts[0, 1] == 3.0
    assert counts[0, 0] == 0.0


def test_binary_pattern_survives_a_zero_valued_entry():
    """A centred rating can be exactly zero; the co-occurrence count must still see it."""
    ratings = pd.DataFrame(
        [(1, 1, 3.0, 100), (2, 1, 3.0, 101), (1, 2, 4.0, 102), (2, 2, 4.0, 103)],
        columns=["user_id", "item_id", "rating", "timestamp"],
    )
    rating_matrix = build_rating_matrix(ratings)
    centred, _ = centre_matrix(rating_matrix.matrix, "item")
    assert np.allclose(centred.data, 0.0)
    counts = cooccurrence_counts(rating_matrix.matrix)
    assert counts[0, 1] == 2.0
    assert binary_pattern(rating_matrix.matrix).sum() == 4.0


def test_shrinkage_pulls_towards_zero():
    similarity = np.array([[0.0, 1.0], [1.0, 0.0]])
    counts = np.array([[0.0, 2.0], [2.0, 0.0]])
    # 2 co-raters with lambda 50 keeps only 2 / 52 of the similarity
    shrunk = apply_shrinkage(similarity, counts, 50.0)
    assert shrunk[0, 1] == pytest.approx(2.0 / 52.0)


def test_shrinkage_barely_touches_well_supported_pairs():
    similarity = np.array([[0.0, 1.0], [1.0, 0.0]])
    counts = np.array([[0.0, 500.0], [500.0, 0.0]])
    shrunk = apply_shrinkage(similarity, counts, 50.0)
    assert shrunk[0, 1] > 0.9


def test_zero_shrinkage_is_a_no_op():
    similarity = np.array([[0.0, 0.7], [0.7, 0.0]])
    counts = np.array([[0.0, 3.0], [3.0, 0.0]])
    assert np.allclose(apply_shrinkage(similarity, counts, 0.0), similarity)


def test_top_neighbours_keeps_the_most_similar():
    similarity_row = np.array([0.9, 0.1, 0.5, 0.7])
    positions, values = top_neighbours(similarity_row, [0, 1, 2, 3], 2)
    assert sorted(values.tolist()) == pytest.approx([0.7, 0.9])
    assert sorted(positions.tolist()) == [0, 3]


def test_top_neighbours_drops_negative_similarities():
    similarity_row = np.array([-0.9, 0.2])
    positions, values = top_neighbours(similarity_row, [0, 1], 5)
    assert positions.tolist() == [1]
    assert values.tolist() == pytest.approx([0.2])


def test_top_neighbours_handles_no_usable_neighbours():
    positions, values = top_neighbours(np.array([-0.5, -0.2]), [0, 1], 3)
    assert len(positions) == 0
    positions, values = top_neighbours(np.array([0.5]), [], 3)
    assert len(positions) == 0

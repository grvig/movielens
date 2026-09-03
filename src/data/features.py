"""Item content features.

MovieLens 100K ships no tag file.  The 19 binary genre flags, the title and the release
year are the entire content signal available, and TF-IDF over 19 binary flags alone gives
a vocabulary of 19 with almost no discriminating power.

So an item's document is assembled from three parts, listed in ``features.document_parts``
in the config:

``genres``
    One token per set flag, e.g. ``genre_film-noir``.
``title_tokens``
    The title with its release year stripped, lowercased and split into word tokens.
    Sequels, franchises and shared proper nouns end up sharing terms, which is real
    signal — "star", "trek", "wars", "batman" all discriminate.
``decade``
    A single ``decade_1970`` style token, which separates the 1930s catalogue from the
    1990s one without giving the model a continuous year to overfit.

The part list is config-driven so switching to a dataset that does have tags (for example
ml-latest-small) means adding a builder and a config entry, not rewriting this module.
"""

import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

YEAR_IN_PARENS = re.compile(r"\(\s*\d{4}\s*\)")
NON_WORD = re.compile(r"[^a-z0-9]+")

TITLE_STOPWORDS = {
    "a", "an", "and", "the", "of", "in", "on", "at", "to", "for", "il", "la", "le",
    "les", "el", "un", "une", "der", "die", "das",
}


def build_item_documents(items, parts):
    """Return a list of whitespace-joined token strings, one per row of ``items``."""
    documents = []
    for _, row in items.iterrows():
        tokens = []
        if "genres" in parts:
            tokens.extend(genre_tokens(row))
        if "title_tokens" in parts:
            tokens.extend(title_tokens(row["title"]))
        if "decade" in parts:
            tokens.extend(decade_tokens(row["release_year"]))
        if len(tokens) == 0:
            tokens.append("unknown_content")
        documents.append(" ".join(tokens))
    return documents


def genre_tokens(row):
    tokens = []
    for column in row.index:
        if not column.startswith("genre_"):
            continue
        if int(row[column]) != 1:
            continue
        name = column[len("genre_"):].lower().replace(" ", "-")
        tokens.append("genre_" + name)
    return tokens


def title_tokens(title):
    if not isinstance(title, str):
        return []
    stripped = YEAR_IN_PARENS.sub(" ", title).lower()
    pieces = NON_WORD.split(stripped)
    tokens = []
    for piece in pieces:
        if piece == "":
            continue
        if piece in TITLE_STOPWORDS:
            continue
        if len(piece) < 2:
            continue
        tokens.append("title_" + piece)
    return tokens


def decade_tokens(release_year):
    year = int(release_year)
    if year <= 0:
        return []
    decade = (year // 10) * 10
    return ["decade_" + str(decade)]


def build_item_features(items, config):
    """Build the TF-IDF matrix over item documents.

    Returns the sparse matrix (rows aligned with ``item_ids``), the item id array, and
    the vocabulary as a list of terms.  Rows are L2 normalised by the vectoriser, so a
    dot product between two rows is already a cosine similarity.
    """
    settings = config.section("features")
    parts = list(settings["document_parts"])
    min_df = int(settings["min_df"])
    sublinear_tf = bool(settings["sublinear_tf"])

    documents = build_item_documents(items, parts)
    vectoriser = TfidfVectorizer(
        analyzer="word",
        tokenizer=str.split,
        preprocessor=None,
        lowercase=False,
        token_pattern=None,
        min_df=min_df,
        sublinear_tf=sublinear_tf,
        norm="l2",
    )
    matrix = vectoriser.fit_transform(documents)
    item_ids = items["item_id"].to_numpy(dtype=np.int64)
    vocabulary = list(vectoriser.get_feature_names_out())
    return matrix, item_ids, vocabulary


def build_item_index(item_ids):
    """Map item id to row position in the feature matrix."""
    index = {}
    for position, item_id in enumerate(item_ids):
        index[int(item_id)] = position
    return index


def describe_features(matrix, vocabulary):
    """Summary numbers for the report's feature-engineering paragraph."""
    n_items = matrix.shape[0]
    n_terms = matrix.shape[1]
    nonzero = matrix.getnnz()
    if n_items > 0:
        terms_per_item = nonzero / n_items
    else:
        terms_per_item = 0.0
    return {
        "items": n_items,
        "vocabulary": n_terms,
        "nonzero_entries": nonzero,
        "mean_terms_per_item": round(terms_per_item, 2),
        "density_percent": round(100.0 * nonzero / max(1, n_items * n_terms), 3),
        "example_terms": vocabulary[:8],
    }

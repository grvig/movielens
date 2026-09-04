"""One place that knows how to construct every model.

The experiment scripts, and later the tuning sweeps, all need the same list of models built
the same way. Keeping that list here rather than repeating it per script means an
experiment cannot quietly evaluate a differently-configured version of a model than the one
in the results table.

Models differ in what they need at construction time - the content model takes the item
table, matrix factorisation takes a validation split for early stopping - so each entry is
a builder function with a uniform signature. Everything after construction is the shared
interface.
"""

from src.models.baselines import GlobalMean
from src.models.baselines import ItemMean
from src.models.baselines import MostPopular
from src.models.baselines import UserMean
from src.models.content import ContentBased
from src.models.item_knn import ItemKNN
from src.models.mf import MatrixFactorization

MODEL_ORDER = [
    "global_mean",
    "user_mean",
    "item_mean",
    "most_popular",
    "content",
    "item_knn",
    "mf",
]


def build_model(name, config, items, validation=None, cache=False):
    """Construct one model by name, wired to whatever extra data it needs."""
    if name == "global_mean":
        return GlobalMean(config)
    if name == "user_mean":
        return UserMean(config)
    if name == "item_mean":
        return ItemMean(config)
    if name == "most_popular":
        return MostPopular(config)
    if name == "content":
        return ContentBased(config, items, validation=validation)
    if name == "item_knn":
        return ItemKNN(config)
    if name == "mf":
        return MatrixFactorization(config, validation=validation, cache=cache)
    raise ValueError("unknown model: " + name)


def build_all(config, items, validation=None, cache=False):
    """Every model in report order, unfitted."""
    models = []
    for name in MODEL_ORDER:
        models.append(build_model(name, config, items, validation=validation, cache=cache))
    return models

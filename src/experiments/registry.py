"""One place that knows how to construct every model.

The experiment scripts, and later the tuning sweeps, all need the same list of models built
the same way. Keeping that list here rather than repeating it per script means an
experiment cannot quietly evaluate a differently-configured version of a model than the one
in the results table.

Models differ in what they need at construction time - the content model takes the item
table, matrix factorisation takes a validation split for early stopping - so each entry is
a builder function with a uniform signature. Everything after construction is the shared
interface.

Two models appear twice under different names. Both sweeps in results/tuning_*.csv found
that validation RMSE and validation precision@10 pick different configurations of the same
algorithm, and pick them strongly: item-kNN loses 46 percent of its precision when selected
on RMSE, and even changes centring convention. Registering the accuracy-tuned and
ranking-tuned settings as separate models puts that in the results table as a row rather
than leaving it in the report as an assertion.
"""

from src.models.baselines import GlobalMean
from src.models.baselines import ItemMean
from src.models.baselines import MostPopular
from src.models.baselines import UserMean
from src.models.content import ContentBased
from src.models.hybrid import WeightedHybrid
from src.models.item_knn import ItemKNN
from src.models.mf import MatrixFactorization

MODEL_ORDER = [
    "global_mean",
    "user_mean",
    "item_mean",
    "most_popular",
    "content",
    "item_knn",
    "item_knn_ranking",
    "mf",
    "mf_ranking",
    "hybrid",
    "hybrid_frontier",
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
    if name == "item_knn_ranking":
        return ItemKNN(config, name=name, **config.model_params(name))
    if name == "mf":
        return MatrixFactorization(config, validation=validation, cache=cache)
    if name == "mf_ranking":
        params = config.model_params(name)
        return MatrixFactorization(
            config,
            name=name,
            n_factors=params["n_factors"],
            learning_rate=params["learning_rate"],
            regularisation=params["regularisation"],
            n_epochs=params["n_epochs"],
            patience=params["patience"],
            validation=validation,
            cache=cache,
        )
    if name == "hybrid":
        params = config.model_params("hybrid")
        model = WeightedHybrid(
            config,
            ContentBased(config, items, validation=validation),
            MatrixFactorization(config, validation=validation, cache=cache),
            weight_mode="density",
            validation=None,
            name=name,
        )
        # The curve was fitted on validation by run_hybrid and recorded in the config, so
        # fitting the model does not refit it. Refitting per run would make the model's
        # behaviour depend on which experiment happened to construct it.
        model.intercept = float(params["curve_intercept"])
        model.slope = float(params["curve_slope"])
        return model
    if name == "hybrid_frontier":
        params = config.model_params("hybrid_frontier")
        return WeightedHybrid(
            config,
            ContentBased(config, items, validation=validation),
            MatrixFactorization(config, validation=validation, cache=cache),
            weight_mode="fixed",
            fixed_weight=float(params["fixed_weight"]),
            name=name,
        )
    raise ValueError("unknown model: " + name)


def build_all(config, items, validation=None, cache=False):
    """Every model in report order, unfitted."""
    models = []
    for name in MODEL_ORDER:
        models.append(build_model(name, config, items, validation=validation, cache=cache))
    return models

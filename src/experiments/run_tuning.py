"""Hyperparameter sweeps for item-kNN and matrix factorisation.

    python -m src.experiments.run_tuning --model knn
    python -m src.experiments.run_tuning --model mf

Both sweeps select on the **validation** split and never touch test. The winning
configuration is printed so it can be written into configs/default.yaml by hand; this
script deliberately does not edit the config itself, because a sweep that silently rewrites
the protocol makes every previously committed result unattributable.

Selection is on validation RMSE. That is a choice worth stating rather than assuming: these
models are also being judged on ranking, and the configuration with the best RMSE is not
always the one with the best precision@10. Both are recorded per configuration so the
report can show where they disagree.
"""

import argparse
import itertools
import time

from src.config import load_config
from src.data.splitting import load_splits
from src.evaluation.harness import evaluate_model
from src.evaluation.harness import make_run_id
from src.evaluation.harness import results_frame
from src.evaluation.harness import write_results
from src.models.item_knn import ItemKNN
from src.models.mf import MatrixFactorization

KNN_OUTPUT = "tuning_knn.csv"
MF_OUTPUT = "tuning_mf.csv"
SELECTION_METRIC = "rmse"
REPORT_K = 10

KNN_NEIGHBOURHOODS = [5, 10, 20, 40, 80, 160]
KNN_SHRINKAGE = [0.0, 10.0, 50.0, 100.0]
KNN_CENTERING = ["item", "user"]

MF_FACTORS = [8, 16, 32, 64]
MF_REGULARISATION = [0.005, 0.02, 0.05, 0.1]
MF_LEARNING_RATES = [0.005, 0.01]


def variant_label(parameters):
    parts = []
    for key in sorted(parameters.keys()):
        parts.append(key + "=" + str(parameters[key]))
    return ",".join(parts)


def knn_configurations():
    configurations = []
    for size, shrinkage, centering in itertools.product(
        KNN_NEIGHBOURHOODS, KNN_SHRINKAGE, KNN_CENTERING
    ):
        configurations.append({
            "neighbourhood_size": size,
            "shrinkage": shrinkage,
            "centering": centering,
        })
    return configurations


def mf_configurations():
    configurations = []
    for factors, regularisation, learning_rate in itertools.product(
        MF_FACTORS, MF_REGULARISATION, MF_LEARNING_RATES
    ):
        configurations.append({
            "n_factors": factors,
            "regularisation": regularisation,
            "learning_rate": learning_rate,
        })
    return configurations


def build_knn(config, parameters, validation):
    return ItemKNN(
        config,
        neighbourhood_size=parameters["neighbourhood_size"],
        shrinkage=parameters["shrinkage"],
        centering=parameters["centering"],
    )


def build_mf(config, parameters, validation):
    return MatrixFactorization(
        config,
        n_factors=parameters["n_factors"],
        regularisation=parameters["regularisation"],
        learning_rate=parameters["learning_rate"],
        validation=validation,
        cache=True,
    )


def sweep(config, configurations, builder, train, validation, run_id):
    rows = []
    total = len(configurations)
    for index, parameters in enumerate(configurations):
        started = time.time()
        model = builder(config, parameters, validation)
        model.fit(train)
        label = variant_label(parameters)
        rows.extend(
            evaluate_model(model, train, validation, config, "val",
                           variant=label, run_id=run_id)
        )
        elapsed = time.time() - started
        print("  " + str(index + 1).rjust(3) + "/" + str(total) + "  "
              + label.ljust(58) + format(elapsed, ".1f") + "s")
    return rows


def best_configuration(frame, metric=SELECTION_METRIC):
    selected = frame[frame["metric"] == metric]
    ordered = selected.sort_values("value")
    return ordered.iloc[0]["variant"], float(ordered.iloc[0]["value"])


def print_leaderboard(frame, limit=8):
    accuracy = frame[frame["metric"] == SELECTION_METRIC][["variant", "value"]]
    accuracy = accuracy.rename(columns={"value": "rmse"}).set_index("variant")
    precision = frame[(frame["metric"] == "precision_at_k") & (frame["k"] == REPORT_K)]
    precision = precision[["variant", "value"]].rename(
        columns={"value": "p@" + str(REPORT_K)}
    ).set_index("variant")
    coverage = frame[(frame["metric"] == "coverage") & (frame["k"] == REPORT_K)]
    coverage = coverage[["variant", "value"]].rename(
        columns={"value": "cov@" + str(REPORT_K)}
    ).set_index("variant")

    table = accuracy.join(precision).join(coverage).sort_values("rmse")
    print("")
    print("best " + str(limit) + " by validation RMSE")
    print("-" * 92)
    print(table.head(limit).round(4).to_string())

    by_precision = table.sort_values("p@" + str(REPORT_K), ascending=False)
    print("")
    print("best 3 by validation precision@" + str(REPORT_K)
          + " (not always the same configuration)")
    print("-" * 92)
    print(by_precision.head(3).round(4).to_string())
    print("")


def main():
    parser = argparse.ArgumentParser(description="Sweep hyperparameters on validation.")
    parser.add_argument("--config", default=None, help="path to a config file")
    parser.add_argument("--model", required=True, choices=["knn", "mf"],
                        help="which sweep to run")
    args = parser.parse_args()

    config = load_config(args.config)
    config.ensure_output_dirs()
    train, validation, test = load_splits(config.path("processed_dir"))
    run_id = make_run_id()

    if args.model == "knn":
        configurations = knn_configurations()
        builder = build_knn
        output_name = KNN_OUTPUT
    else:
        configurations = mf_configurations()
        builder = build_mf
        output_name = MF_OUTPUT

    print("sweeping " + str(len(configurations)) + " configurations for " + args.model)
    rows = sweep(config, configurations, builder, train, validation, run_id)

    frame = results_frame(rows)
    output_path = config.path("results_dir") / output_name
    write_results(rows, output_path)
    print("")
    print("wrote " + str(output_path) + " (" + str(len(frame)) + " rows)")

    print_leaderboard(frame)
    variant, value = best_configuration(frame)
    print("best by validation RMSE: " + variant + "  (" + format(value, ".4f") + ")")
    print("Write it into configs/default.yaml by hand; this script does not edit it.")


if __name__ == "__main__":
    main()

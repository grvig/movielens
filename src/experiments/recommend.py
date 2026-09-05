"""Print what each model actually recommends, as films rather than as decimals.

    python -m src.experiments.recommend --user 1

Every other script in this project reports aggregate metrics over 943 users. This one shows
a single user: what they rated highly in training, what each model puts in front of them,
and which of those recommendations they went on to rate 4 or more in the held-out period.

That last part is what makes it diagnostic rather than decorative. A precision@10 of 0.04
is hard to argue with and hard to learn from; ten titles with the hits marked shows you
whether a model is failing because it is wrong or because the metric is strict.

It writes nothing to results/ and fits nothing differently from run_main, so it cannot
affect a reported number.
"""

import argparse
import re

from src.config import load_config
from src.data.splitting import load_splits
from src.evaluation.ranking import item_popularity
from src.experiments.registry import MODEL_ORDER
from src.experiments.registry import build_model
from src.experiments.run_main import load_items

YEAR_IN_TITLE = re.compile(r"\(\s*\d{4}\s*\)")

DEFAULT_K = 10
PROFILE_SIZE = 8
HIT_MARKER = "*"
MISS_MARKER = " "


def genre_list(row):
    genres = []
    for column in row.index:
        if not column.startswith("genre_"):
            continue
        if int(row[column]) != 1:
            continue
        name = column[len("genre_"):]
        if name.lower() == "unknown":
            continue
        genres.append(name)
    return genres


def build_item_lookup(items):
    """Map item id to a short display string and its genre list.

    The year is only appended when the title carries none of its own. Testing for this
    item's particular year is not enough: ML-100K's release_date is a distribution date and
    sometimes disagrees with the year printed in the title, so Fargo has a title year of
    1996 and a release_date year of 1997 and would render as "Fargo (1996) (1997)".
    """
    lookup = {}
    for _, row in items.iterrows():
        year = int(row["release_year"])
        title = str(row["title"])
        if YEAR_IN_TITLE.search(title):
            label = title
        elif year > 0:
            label = title + " (" + str(year) + ")"
        else:
            label = title
        lookup[int(row["item_id"])] = {
            "label": label,
            "genres": genre_list(row),
        }
    return lookup


def describe(lookup, item_id, width=44):
    entry = lookup.get(int(item_id))
    if entry is None:
        return "item " + str(item_id)
    label = entry["label"]
    if len(label) > width:
        label = label[: width - 3] + "..."
    return label.ljust(width)


def genre_text(lookup, item_id, limit=3):
    entry = lookup.get(int(item_id))
    if entry is None:
        return ""
    return "|".join(entry["genres"][:limit])


def print_user_profile(train, test, lookup, user_id, threshold):
    history = train[train["user_id"] == user_id]
    held_out = test[test["user_id"] == user_id]
    liked = held_out[held_out["rating"] >= threshold]

    print("")
    print("user " + str(user_id) + ": " + str(len(history)) + " training ratings, "
          + str(len(held_out)) + " held out, " + str(len(liked)) + " of them rated "
          + format(threshold, ".0f") + " or more")
    print("  mean training rating " + format(float(history["rating"].mean()), ".2f"))
    print("")
    print("  top rated in training")
    favourites = history.sort_values(["rating", "item_id"], ascending=[False, True])
    for _, row in favourites.head(PROFILE_SIZE).iterrows():
        item_id = int(row["item_id"])
        print("    " + format(float(row["rating"]), ".0f") + "  "
              + describe(lookup, item_id) + "  " + genre_text(lookup, item_id))
    print("")
    print("  held out and liked (what a good recommendation would have found)")
    if len(liked) == 0:
        print("    none")
    for _, row in liked.sort_values("item_id").head(PROFILE_SIZE).iterrows():
        item_id = int(row["item_id"])
        print("    " + format(float(row["rating"]), ".0f") + "  "
              + describe(lookup, item_id) + "  " + genre_text(lookup, item_id))
    return set(int(value) for value in liked["item_id"])


def print_recommendations(model, user_id, k, lookup, popularity, relevant):
    recommendations = model.recommend(user_id, k)
    scores = model.predict(user_id, recommendations)
    hits = 0
    print("")
    print(model.name)
    print("  " + "rank".ljust(5) + "pred".ljust(7) + "n".ljust(6)
          + "title".ljust(46) + "genres")
    for position, item_id in enumerate(recommendations):
        item_id = int(item_id)
        if item_id in relevant:
            marker = HIT_MARKER
            hits = hits + 1
        else:
            marker = MISS_MARKER
        print("  " + marker + str(position + 1).ljust(4)
              + format(float(scores[position]), ".2f").ljust(7)
              + str(popularity.get(item_id, 0)).ljust(6)
              + describe(lookup, item_id) + "  " + genre_text(lookup, item_id))
    print("  " + str(hits) + " of " + str(k) + " were held-out likes")
    return hits


def run(config, user_id, k, model_names, cache):
    train, validation, test = load_splits(config.path("processed_dir"))
    items = load_items(config)
    lookup = build_item_lookup(items)
    popularity = item_popularity(train)
    threshold = float(config.section("evaluation")["relevance_threshold"])

    if user_id not in set(train["user_id"].unique()):
        raise ValueError("user " + str(user_id) + " is not in the training split")

    relevant = print_user_profile(train, test, lookup, user_id, threshold)

    summary = {}
    for name in model_names:
        model = build_model(name, config, items, validation=validation, cache=cache)
        model.fit(train)
        summary[name] = print_recommendations(
            model, user_id, k, lookup, popularity, relevant
        )
    return summary


def print_summary(summary, k):
    print("")
    print("hits in the top " + str(k))
    print("-" * 34)
    for name, hits in summary.items():
        print("  " + name.ljust(16) + str(hits))
    print("")


def main():
    parser = argparse.ArgumentParser(
        description="Show each model's recommendations for one user, as films."
    )
    parser.add_argument("--config", default=None, help="path to a config file")
    parser.add_argument("--user", type=int, default=1, help="user id to explain")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="how many to show")
    parser.add_argument("--models", default=None,
                        help="comma separated model names, default all")
    parser.add_argument("--no-cache", action="store_true",
                        help="refit matrix factorisation instead of loading a cached fit")
    args = parser.parse_args()

    config = load_config(args.config)
    config.ensure_output_dirs()

    if args.models is None:
        model_names = list(MODEL_ORDER)
    else:
        model_names = []
        for name in args.models.split(","):
            cleaned = name.strip()
            if cleaned != "":
                model_names.append(cleaned)

    summary = run(config, args.user, args.k, model_names, not args.no_cache)
    print_summary(summary, args.k)


if __name__ == "__main__":
    main()

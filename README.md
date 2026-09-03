# MovieLens 100K — four recommenders, one protocol

A controlled comparison of four recommender families on MovieLens 100K. The deliverable is
not a recommender; it is the comparison. Four approaches are pushed through an identical
split, an identical candidate set and an identical metric suite, so we can say precisely
where each one wins and what it pays for the win.

The claim the repository exists to support:

> Under one protocol, matrix factorisation wins on rating accuracy while recommending a
> narrow, popular slice of the catalogue. Content-based scoring loses on accuracy but
> spreads across the catalogue and survives cold users. A hybrid weighted by profile
> density beats both around the crossover.

## Quick start

```bash
pip install -r requirements.txt
pip install -e .
python scripts/download_data.py
python scripts/preprocess.py
```

`download_data.py` fetches `ml-100k.zip` from GroupLens, verifies its MD5 against
`configs/default.yaml` and extracts to `data/raw/`. `preprocess.py` writes the processed
tables and the train/validation/test split into `data/processed/`, prints a data card, and
asserts that the split has no temporal leakage.

Nothing under `data/` is committed. A fresh clone reproduces it from those two commands.

## Running experiments

```bash
python -m src.experiments.run_main
```

Experiments write CSVs into `results/`. Plotting scripts read those CSVs and write into
`figures/`. Training code never plots — otherwise every axis-label change in the report
means re-running a matrix factorisation sweep.

## Protocol

These decisions are locked. Changing any of them invalidates every CSV in `results/`.

| Decision | Value | Why |
|---|---|---|
| Split | Per-user temporal, 70 / 10 / 20 | A random row split leaks a user's future into their training history and inflates every metric |
| Ordering | `(timestamp, item_id)` | Many ML-100K users rated in one session with identical timestamps; the item-id tie-break keeps the split reproducible |
| Seed | `20260903`, set once in `configs/default.yaml` | No module seeds anything on its own |
| Candidates | All training items the user has not rated | No sampled negatives; with 1682 items the full scan is cheap and sampling would be a second decision to defend |
| Relevance | Held-out rating ≥ 4.0 | A 3 counts as not relevant |
| Scored users | Those with ≥ 1 relevant held-out item | Recall would divide by zero otherwise; `n_users` travels with every results row |
| k | 5, 10, 20 | |
| Predictions | Clipped to [1, 5], unknown user or item falls back to the global mean | Enforced by a shared conformance test, not by convention |

## The model interface

Every model implements the same three methods and the harness never branches on model
type. That is what makes the comparison fair, and it is also how three people work in
parallel without anyone's model special-casing the harness.

```python
model.fit(train_ratings)                    # learn from user_id, item_id, rating, timestamp
model.predict(user_id, candidate_items)     # one predicted rating per candidate, clipped
model.recommend(user_id, k)                 # top-k item ids, excluding training items
```

`rank_scores()` is overridable for models whose ranking signal is not a rating — popularity
counts, content cosine similarity — so the accuracy and ranking families stay independent.

## Results

<!-- RESULTS_TABLE_START -->
_Not yet populated. Generated from `results/main_results.csv` by
`python -m src.experiments.make_readme_table` once the main experiment runner lands._
<!-- RESULTS_TABLE_END -->

Metrics reported side by side, per model:

- **Accuracy** — RMSE, MAE on held-out ratings.
- **Ranking** — precision@k, recall@k at k = 5, 10, 20.
- **Catalogue** — coverage@k, mean recommended popularity, popularity percentile, Gini.

## Layout

```
configs/default.yaml     seed, paths, split ratios, protocol, hyperparameters
scripts/                 download_data.py, preprocess.py
src/config.py            config loading and global seeding
src/data/                loaders, item features, temporal splitting
src/models/              base.py is the contract; one module per model family
src/evaluation/          metrics.py (accuracy), ranking.py, harness.py
src/experiments/         scripts that run a config and dump CSVs
src/plots/               scripts that read CSVs and write figures
tests/                   pytest, run locally; no CI
results/                 committed CSVs, the report's source of truth
figures/                 committed figures
```

## Development

```bash
python -m pytest
```

Tests run locally; there is no CI. The suite uses a synthetic fixture rather than the real
dataset, so it runs in under a second and works on a clone with no `data/` directory.

`tests/test_splitting.py` is the important one — it asserts that no user's training ratings
postdate their held-out ratings. If it fails, every number in `results/` is wrong.

`tests/test_models.py` is parameterised over every model class. A model that needs an
exception there is a model that would quietly invalidate the comparison.

## Dataset notes

MovieLens 100K: 943 users, 1682 movies, 100,000 ratings on a 1–5 integer scale, collected
September 1997 to April 1998, with at least 20 ratings per user.

Two things worth stating up front, both of which shape the design:

**There is no tag file.** `u.item` carries the title, release date, IMDb URL and 19 binary
genre flags — that is the entire content signal. Tag data starts at `ml-latest-small` and
`ml-25m`. TF-IDF over 19 binary flags has a vocabulary of 19 and barely discriminates, so
item documents are built from genre tokens plus title tokens plus a release-decade token.
`features.document_parts` in the config is a list, so adding tags later means adding a
builder, not a rewrite.

**Timestamps cluster into sessions.** Many users rated everything in one or two sittings,
so for those users the held-out "future" is minutes away rather than months. The per-user
temporal split is still the right call, but the median train-to-test gap is reported as a
diagnostic and written up as a limitation.

## Team

| Lane | Owner | Scope |
|---|---|---|
| A | data & harness | config, loaders, splitting, metrics, harness, cold-start |
| B | baselines & content | four baselines, TF-IDF features, content model, qualitative check |
| C | collaborative filtering | item-kNN, matrix factorisation, hyperparameter sweeps |

The hybrid is joint work, since it needs both a tuned MF and a working content model.

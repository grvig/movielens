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
python scripts/download_data.py
python scripts/preprocess.py
```

That is the whole setup. Everything runs from a bare clone at the repository root — no
`pip install -e .` needed, and no `PYTHONPATH` to set.

`download_data.py` fetches `ml-100k.zip` from GroupLens, verifies its MD5 against
`configs/default.yaml` and extracts to `data/raw/`. `preprocess.py` writes the processed
tables and the train/validation/test split into `data/processed/`, prints a data card, and
asserts that the split has no temporal leakage.

Nothing under `data/` is committed. A fresh clone reproduces it from those two commands.

## Running experiments

```bash
python -m src.experiments.run_main
```

Fits all seven models on train, evaluates on test, writes `results/main_results.csv` and
prints the comparison table. Add `--cache` to reuse stored matrix factorisation fits.

```bash
python -m src.experiments.run_split_ablation
```

Runs the identical pipeline under the temporal split and under a naive random row split,
and reports the gap. This is the experiment that justifies the protocol with a number.

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

Models differ in what they need at construction time, and that is the only place they are
allowed to differ. `src/experiments/registry.py` is the single place that knows how to
build each one.

## The models

| Model | Ranking signal | Notes |
|---|---|---|
| `global_mean` | none (tie-break only) | Rating-prediction floor. Precision near zero is correct, not a bug |
| `user_mean` | none (tie-break only) | Shrunk user bias, β = 10 |
| `item_mean` | predicted rating | Shrunk item bias |
| `most_popular` | training rating count | The **ranking** floor, and the hard one to beat. RMSE equals `global_mean` by construction |
| `content` | cosine to user profile | Mean-centred TF-IDF profile; cosine calibrated onto the rating scale on validation |
| `item_knn` | predicted rating | Mean-centred cosine with significance shrinkage |
| `mf` | predicted rating | Biases + L2, SGD, early stopping on validation |

Two behaviours that read as bugs and are not:

**Shrinkage does not pull kNN predictions towards the item mean.** The weighted average
divides by the sum of absolute similarities, so scaling every similarity by the same factor
cancels exactly. Shrinkage only bites when co-occurrence counts differ between neighbours,
or when it changes which items reach the top-k neighbourhood.

**`global_mean` and `user_mean` produce near-zero precision@k.** They have no per-item
signal, so their ranking is arbitrary up to the item-id tie-break. They are the rating
floor; `most_popular` is the ranking floor.

## Where validation is used

Validation reaches exactly two models — `content` for its cosine calibration, `mf` for
early stopping — and never reaches evaluation. A test overwrites every validation rating
and asserts the test-split numbers of the models that ignore validation do not move.

## Results

<!-- RESULTS_TABLE_START -->
_Not yet populated._ The pipeline runs end to end and is tested against a synthetic
fixture, but no numbers here are real yet: the dataset has not been downloaded, because
`files.grouplens.org` has been serving an expired TLS certificate since 28 Aug 2026.
Certificate verification was deliberately **not** disabled — the pinned MD5 has never been
confirmed against GroupLens, so trusting an unauthenticated download because it matches an
unverified hash would not be verification.

Once the download succeeds, `python -m src.experiments.run_main` populates this block.
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
src/data/matrix.py       sparse rating matrix shared by both collaborative models
src/models/              base.py is the contract; one module per model family
src/models/similarity.py centring, cosine and shrinkage for item-kNN
src/evaluation/          metrics.py (accuracy), ranking.py, harness.py
src/experiments/         registry.py builds models; run_*.py run a config and dump CSVs
src/plots/               scripts that read CSVs and write figures
tests/                   pytest, run locally; no CI
results/                 committed CSVs, the report's source of truth
figures/                 committed figures
```

## Development

```bash
python -m pytest
```

Run experiments and tests from the repository root, so that `src` resolves:

```bash
python -m src.experiments.run_main
```

Tests run locally; there is no CI. The suite uses a synthetic fixture rather than the real
dataset, so it runs in under a second and works on a clone with no `data/` directory.

`tests/test_splitting.py` is the important one — it asserts that no user's training ratings
postdate their held-out ratings. If it fails, every number in `results/` is wrong.

`tests/test_models.py` is parameterised over every model family — all seven pass the same
contract with no exceptions. A model that needs an exception there is a model that would
quietly invalidate the comparison.

`tests/test_run_main.py` drives the whole pipeline against a miniature processed directory
built from the fixture, so the runner wiring is verified without the real dataset.

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

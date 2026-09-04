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

`download_data.py` fetches `ml-100k.zip` from GroupLens, checks it against both GroupLens's
published `.md5` and the one pinned in `configs/default.yaml`, and extracts to `data/raw/`.

> **GroupLens's certificate expired on 28 Aug 2026.** Until they renew, add
> `--allow-expired-cert`. That does *not* disable verification — it pins the certificate,
> requiring the server to present exactly the one whose SHA-256 is in the config and
> ignoring only the expiry date. It trusts no certificate authority, so it is stricter than
> the default path. Delete the flag and the pin once they renew. `preprocess.py` writes the processed
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
| `item_knn` | predicted rating | Mean-centred cosine with significance shrinkage; base term shrunk like `item_mean` |
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
| Model | RMSE | MAE | P@10 | R@10 | Cov@10 | PopPct@10 | Gini@10 |
|---|---|---|---|---|---|---|---|
| Global mean | 1.2149 | 1.0065 | 0.0233 | 0.0236 | 0.0190 | 0.8343 | 0.9923 |
| User mean | 1.1570 | 0.9271 | 0.0233 | 0.0236 | 0.0190 | 0.8343 | 0.9923 |
| Item mean | 1.0913 | 0.8774 | 0.0434 | 0.0401 | 0.0324 | 0.9135 | 0.9912 |
| Most popular | 1.2149 | 1.0065 | 0.0657 | 0.0682 | 0.0419 | 0.9935 | 0.9870 |
| Content-based | 1.1422 | 0.9093 | 0.0201 | 0.0193 | 0.4552 | 0.6168 | 0.8359 |
| Item-kNN | 1.0171 | 0.7993 | 0.0362 | 0.0262 | 0.3410 | 0.6709 | 0.9115 |
| Matrix factorisation | 0.9962 | 0.7874 | 0.0422 | 0.0378 | 0.1727 | 0.8219 | 0.9633 |

Test split: 20000 held-out ratings; ranking metrics averaged over the 906 users with at least one relevant held-out item. Seed 20260903, run 20260904T065729.

Lower is better for RMSE, MAE and the two popularity columns. Higher is better for precision, recall and coverage.
<!-- RESULTS_TABLE_END -->

Regenerate with `python -m src.experiments.make_readme_table` after a run. The CSV is the
source of truth; this table is a view of it. These are **untuned** numbers — the sweeps
have not run yet.

Metrics reported side by side, per model:

- **Accuracy** — RMSE, MAE on held-out ratings.
- **Ranking** — precision@k, recall@k at k = 5, 10, 20.
- **Catalogue** — coverage@k, mean recommended popularity, popularity percentile, Gini.

### What the first run shows

**The accuracy/catalogue tension is real and large.** Matrix factorisation wins RMSE
(0.9962) while showing 17% of the catalogue at a popularity percentile of 0.82.
Content-based is 15% worse on RMSE (1.1422) but reaches 46% of the catalogue at 0.62. They
are not competing on the same axis, which is the backbone of the results section.

**No personalised model beats the popularity baseline on precision@10.** Most-popular
scores 0.0657; the best personalised model manages 0.0422. It does that by showing everyone
essentially the same 4% of the catalogue at the 99th popularity percentile. This is the
honest headline, and the argument for reporting catalogue metrics at all — on precision
alone the correct conclusion would be "don't personalise".

**Item-kNN is the balanced one.** Second-best RMSE, twice MF's coverage, a markedly lower
popularity percentile. Worth saying plainly, because the narrative that MF simply wins does
not survive looking at the other columns.

### Why not a random split

`results/split_ablation.csv` runs the identical pipeline under both splits. Every model
looks better under a random row split — RMSE by 5.5–10.3%, precision@10 by 27–87%.

The distortion is not uniform, and it lands on the number the report leads with:

| Metric | Temporal winner | Random winner |
|---|---|---|
| RMSE | Matrix factorisation (0.9962) | **Item-kNN** (0.9289) |

Under the temporal split MF beats item-kNN by 0.0209 RMSE. Under a random split the order
reverses and item-kNN wins by 0.0035. **The choice of split changes which model is reported
as the most accurate.** The random-split margin is small, so calling that a firm reversal
needs the bootstrap intervals rather than point estimates — but it is more than enough to
justify the protocol.

One expectation this experiment overturned: the inflation is *larger for the weaker
models*, not the stronger ones. Content-based gains most on RMSE (−10.3%) and MF least
(−6.4%); most-popular gains most on precision (+87%) and content least (+27%). In hindsight
that is the more sensible prediction — MF already extracts most of the available signal, so
leaked information has less headroom to help it.

Two artifacts to be aware of before quoting anything:

- `global_mean` scores a non-zero precision@10 of 0.0233 despite having no ranking signal
  at all. Its tie-break is ascending item id, and low item ids in ML-100K are the early,
  popular films. That number is an artifact of the dataset's id ordering, not a signal.
- `most_popular` and `global_mean` have identical RMSE by construction — most-popular
  predicts the global mean and only its ranking differs.

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

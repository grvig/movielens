# MovieLens 100K — four recommenders, one protocol

A controlled comparison of four recommender families on MovieLens 100K: baselines,
content-based, collaborative filtering (item-kNN and matrix factorisation) and a hybrid.
Every method goes through the same split, the same candidate set and the same metrics, so
the differences are the methods and nothing else. A demo app on top serves the same trained
models to real people, including people who aren't in the dataset.

Item-kNN and matrix factorisation are each registered twice, tuned once for accuracy and
once for ranking, and the hybrid is registered twice as well (density-weighted and at a
fixed weight). Eleven models in total.

## Findings

- **No personalised model beats a popularity baseline on precision@10**, and every one of
  those gaps survives a bootstrap over users. The reason is that precision@10 on this data
  mostly measures popularity: across the nine models with a real ranking signal it
  correlates with popularity bias at r = 0.93 and with RMSE at r = −0.03.
- **Matrix factorisation wins on rating accuracy while recommending a narrow, popular slice
  of the catalogue** (RMSE 0.9996, 13% catalogue coverage, popularity percentile 0.92).
  Content-based is 14% worse on RMSE and reaches 3.5 times as much of the catalogue.
- **Tuning for accuracy costs ranking for item-kNN, but not for matrix factorisation.**
  Across 48 kNN settings validation RMSE and precision@10 disagree (r = +0.80), mostly
  through the centring choice. Across 32 MF settings they agree (r = −0.42).
- **Content-based does not cope better with cold start.** With one rating it is the least
  accurate personalised method; matrix factorisation's bias terms give it a floor.
- **The hybrid does not significantly beat plain matrix factorisation on precision.** Its
  real gain is catalogue coverage, and the density-dependent crossover the plan predicted is
  not there once matrix factorisation is regularised properly.
- **A random train/test split flatters every method** — RMSE by 3–10%, precision@10 by
  27–87% — which is why the protocol hides each user's most recent ratings instead.

<details>
<summary>Findings that changed during the project, and why</summary>

Four reported results changed when a defect in matrix factorisation was fixed on
22 September 2026 (see [The matrix factorisation fix](#the-matrix-factorisation-fix)):

- The original thesis said MF recommends "a narrow, popular slice". Before the fix MF had
  the *lowest* popularity percentile of any personalised model (0.58), so the thesis was
  revised to "narrow and obscure". That was the defect talking: MF was filling 43% of its
  lists with films rated only a handful of times. After the fix MF is narrow and popular
  (0.92), and the original thesis stands.
- The density-weighted hybrid appeared to confirm the plan's crossover (content-heavy for
  sparse users, CF-heavy for dense ones) and to beat both components by a wide margin. With
  MF fixed the fitted weight is small everywhere, there is no crossover, and the hybrid's
  precision gain over MF is not significant. Content had been compensating for MF's defect.
- MF's own sweep appeared to show accuracy and ranking pulling apart (r = +0.67). After the
  fix they agree (r = −0.42). The item-kNN version of that finding was unaffected.
- Under a random split item-kNN had briefly overtaken MF on RMSE, which suggested the choice
  of split changes the reported winner. After the fix MF wins under both splits, so that
  claim is withdrawn; the inflation itself still holds.

</details>

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

`preprocess.py` writes the processed tables and the train/validation/test split into
`data/processed/`, prints a data card, and asserts that the split has no temporal leakage.

Nothing under `data/` is committed. A fresh clone reproduces it from those two commands.

## Demo app

```bash
pip install -r requirements.txt
streamlit run app/Home.py
```

Opens at http://localhost:8501 and needs the processed data from the quick start. The first
load fits all eleven models, which takes about a minute; after that everything is cached.

- **Recommend** — rate a few films, or upload your Letterboxd `ratings.csv`, and see each
  method's picks as a row of posters. You aren't in the training data: each trained model
  is forked and you're folded into the fork, so nothing is retrained and nothing is stored.
- **Browse users** — any of the 943 users, with the films they went on to rate highly
  outlined, so you can see which methods guessed right.
- **Results** and **Experiments** — the committed CSVs as interactive charts. The sentences
  describing each experiment are generated from the CSVs, so a re-run can't leave them stale.

Posters are optional. Get a free API key from themoviedb.org and put it in
`.streamlit/secrets.toml`, which is gitignored:

```toml
TMDB_API_KEY = "your key"
```

Without a key the app shows title cards instead. The app serves the same trained models the
experiments evaluate, but it never produces a reported number.

## Running experiments

```bash
python -m src.experiments.run_main
```

Fits all eleven models on train, evaluates on test, writes `results/main_results.csv` and
prints the comparison table. Add `--cache` to reuse stored matrix factorisation fits.

```bash
python -m src.experiments.run_split_ablation
```

Runs the identical pipeline under the temporal split and under a naive random row split,
and reports the gap. This is the experiment that justifies the protocol with a number.

```bash
python -m src.experiments.run_tuning --model knn
python -m src.experiments.run_tuning --model mf
```

Sweeps hyperparameters on validation and prints the winner. It deliberately does not edit
`configs/default.yaml` — a sweep that silently rewrites the protocol makes every previously
committed result unattributable.

```bash
python -m src.experiments.run_bootstrap
```

Resamples users 1000 times to put confidence intervals on every pairwise difference, so
the report can say a gap is real rather than assert it.

```bash
python -m src.experiments.run_hybrid
python -m src.plots.plot_hybrid
```

Fits the density weighting curve and sweeps the fixed-weight frontier, both on validation.
Like the tuning sweeps it prints the constants rather than editing the config.

```bash
python -m src.experiments.run_cold_start
python -m src.plots.plot_cold_start
```

Truncates 200 users' histories to 1, 3, 5, 10 and 20 ratings, refits everything at each
level and evaluates on those users alone. The most expensive script here — 45 fits.

```bash
python -m src.experiments.recommend --user 1
```

Prints each model's top-10 for one user as actual film titles, with the held-out likes
marked. The only script here that shows recommendations rather than metrics.

```bash
python -m src.experiments.run_qualitative --history my_export.csv
```

Matches an exported watch history onto the catalogue, injects it as user 944, and prints
what each model recommends for a real person. Accepts Letterboxd's `Name,Year,Rating`
columns or plain `title,rating`. Unmatched titles are reported rather than dropped — a poor
join otherwise looks exactly like a poor recommender. Note that ML-100K stops at 1998, so a
modern history will mostly miss.

```bash
python -m src.plots.plot_main
python -m src.plots.plot_tuning
python -m src.plots.plot_hybrid
python -m src.plots.plot_cold_start
```

Regenerate every figure from the committed CSVs.

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
| `item_knn` | predicted rating | Mean-centred cosine with significance shrinkage; base term shrunk like `item_mean`. Tuned on validation RMSE |
| `item_knn_ranking` | predicted rating | Same algorithm, tuned on validation precision@10. Lands on the other centring convention |
| `mf` | predicted rating | Biases + L2, SGD, early stopping on validation. Biases shrunk like `item_mean` (see [the fix](#the-matrix-factorisation-fix)). Tuned on validation RMSE |
| `mf_ranking` | predicted rating | Same algorithm, tuned on validation precision@10. Not significantly different from `mf` on precision since the fix |
| `hybrid` | blended, z-scored per user | Content + MF, weight a fitted function of the user's history length |
| `hybrid_frontier` | blended, z-scored per user | Content + MF at a fixed weight, chosen on the validation frontier |

Two behaviours that read as bugs and are not:

**Shrinkage does not pull kNN predictions towards the item mean.** The weighted average
divides by the sum of absolute similarities, so scaling every similarity by the same factor
cancels exactly. Shrinkage only bites when co-occurrence counts differ between neighbours,
or when it changes which items reach the top-k neighbourhood.

**`global_mean` and `user_mean` produce near-zero precision@k.** They have no per-item
signal, so their ranking is arbitrary up to the item-id tie-break. They are the rating
floor; `most_popular` is the ranking floor.

## Where validation is used

Validation reaches the models in three places — `content` for its cosine calibration, `mf`
for early stopping, and `run_hybrid` for fitting the hybrid's weights — and never reaches
evaluation. A test overwrites every validation rating
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
| Item-kNN (RMSE-tuned) | 1.0162 | 0.7992 | 0.0357 | 0.0246 | 0.3378 | 0.6708 | 0.9130 |
| Item-kNN (ranking-tuned) | 1.1153 | 0.8754 | 0.0455 | 0.0367 | 0.3740 | 0.8133 | 0.9097 |
| Matrix factorisation (RMSE-tuned) | 0.9996 | 0.7908 | 0.0523 | 0.0464 | 0.1308 | 0.9162 | 0.9773 |
| Matrix factorisation (ranking-tuned) | 1.0108 | 0.8007 | 0.0519 | 0.0446 | 0.1498 | 0.9191 | 0.9705 |
| Hybrid (density-weighted) | 1.0084 | 0.7991 | 0.0542 | 0.0484 | 0.1333 | 0.9166 | 0.9745 |
| Hybrid (fixed weight) | 1.0227 | 0.8116 | 0.0545 | 0.0490 | 0.1886 | 0.9026 | 0.9577 |

Test split: 20000 held-out ratings; ranking metrics averaged over the 906 users with at least one relevant held-out item. Seed 20260903, run 20260922T172819.

Lower is better for RMSE, MAE and the two popularity columns. Higher is better for precision, recall and coverage.
<!-- RESULTS_TABLE_END -->

Regenerate with `python -m src.experiments.make_readme_table` after a run. The CSV is the
source of truth; this table is a view of it. Hyperparameters come from the sweeps in
`results/tuning_knn.csv` and `results/tuning_mf.csv`, selected on validation.

Metrics reported side by side, per model:

- **Accuracy** — RMSE, MAE on held-out ratings.
- **Ranking** — precision@k, recall@k at k = 5, 10, 20.
- **Catalogue** — coverage@k, mean recommended popularity, popularity percentile, Gini.

### What the results show

**precision@10 measures popularity, not accuracy.** Across the nine models with a real
ranking signal (leaving out global and user mean, whose rankings are arbitrary),
precision@10 correlates with popularity bias at r = 0.93 and with RMSE at r = −0.03. Over
all eleven the figures are 0.76 and −0.38. Nine models is a small sample and several are
variants of one another, so treat these as a description of this table rather than an
estimate of anything wider.

**No personalised model beats most-popular on precision@10.** Most-popular scores 0.0657;
the best personalised model, the fixed-weight hybrid, scores 0.0545. Most-popular gets
there by showing everyone the same 4% of the catalogue at the 99th popularity percentile.

**Accuracy and reach pull apart.** Matrix factorisation has the best RMSE (0.9996) and
recommends from 13% of the catalogue; content-based is 14% worse on RMSE and reaches 46%.

**For item-kNN, tuning for accuracy costs ranking.** Validation RMSE and precision@10
disagree across its 48 settings (r = +0.80): item centring wins on accuracy, user centring
(adjusted cosine) on ranking, and the precision-selected setting scores 46% higher on
precision than the RMSE-selected one. That is why item-kNN is registered twice. For matrix
factorisation the two criteria agree (r = −0.42 across 32 settings), and on the test set
the ranking-tuned variant is not significantly different from the accuracy-tuned one on
precision.

### Are these gaps real?

`results/bootstrap_ci.csv` resamples the 906 scored users 1000 times, paired across models,
and gives a 95% interval for every pairwise difference. Users are resampled rather than
ratings, because one person's ratings are not independent of each other.

| Comparison | Difference | 95% interval | Real |
|---|---|---|---|
| MF − most-popular, P@10 | −0.0134 | [−0.0203, −0.0067] | yes |
| Hybrid (fixed) − most-popular, P@10 | −0.0111 | [−0.0183, −0.0040] | yes |
| Item-kNN − most-popular, P@10 | −0.0300 | [−0.0366, −0.0234] | yes |
| Item-kNN − MF, RMSE | +0.0163 | [+0.0071, +0.0288] | yes |
| Hybrid (density) − MF, P@10 | +0.0019 | [−0.0004, +0.0041] | no |
| MF − MF (ranking), P@10 | +0.0003 | [−0.0035, +0.0041] | no |
| Item-kNN − Item-kNN (ranking), RMSE | −0.0995 | [−0.1217, −0.0778] | yes |

Most-popular's lead on precision is real against every personalised model, including the
closest. MF's RMSE lead over item-kNN is real. The hybrid's precision edge over MF is not.

### The matrix factorisation fix

The first version of matrix factorisation used plain L2 on its biases. Summed over an item's
n ratings, per-rating L2 minimises Σ err² + nλb², which solves to b = mean deviation / (1 + λ)
— a constant shrink factor, whatever n is. A film rated once kept almost its whole deviation.
Every other model here shrinks by n / (n + β) (item mean, user mean and the item-kNN base
term all use β = 10), so MF treated thin evidence differently from everything it was being
compared against.

The effect was large. 43% of MF's top-10 slots went to films with fewer than ten ratings,
and those were liked 0.08% of the time. A bias penalty of β/n per rating instead solves to
exactly Σ dev / (n + β); with it that share drops to 0.1%, precision@10 roughly doubles
(0.0267 → 0.0523 on test) and RMSE rises slightly (0.9877 → 0.9996). The MF sweep, the
hybrid and every experiment were re-run afterwards. Four earlier findings changed; they are
listed in the collapsed section at the top.

It showed up through the demo app rather than the metrics: MF's top pick for user 1 was a
film with seven training ratings.

### The hybrid

`results/hybrid_weights.csv`, `results/hybrid_frontier.csv`, `figures/hybrid.pdf`. Both
hybrids blend content and MF scores, z-scored per user first, because cosine similarities
and predicted ratings are not on the same scale.

The density-weighted hybrid learns a content weight as a function of how many ratings a user
has, w(n) = sigmoid(a + b·log(1 + n)), fitted on validation. The plan expected a crossover:
content-heavy for sparse users, CF-heavy for dense ones. The fitted curve is
sigmoid(−2.90 + 0.23·log(1 + n)) — a content weight of about 0.10 for sparse users and 0.17
for dense ones, so no crossover — and the best weight per history group (0.1, 0.0, 0.3,
0.2, 0.1) shows no pattern in history length.

On validation the best fixed blend (0.2 content) beats pure MF on precision@10 by 14%, and
precision peaks in the middle of the weight sweep rather than at either end. On test the
hybrid's precision edge over MF is not significant. What does hold up is reach: the
fixed-weight hybrid covers 19% of the catalogue against MF's 13% at similar precision.

Before the MF fix this section reported a clear crossover and a large gain. Content was
compensating for MF's thin-item problem, most of all for sparse users; once MF was fixed
there was little left for it to compensate for.

### Cold start

`results/cold_start.csv`, `figures/cold_start.pdf`. 200 users have their training history
cut to 1, 3, 5, 10 and 20 ratings. Everyone else is left intact, every model is retrained at
each level, and only the truncated users are scored.

The plan predicted content-based would cope best with almost no history, since it doesn't
need co-raters. It copes worst:

| Test RMSE | 1 rating | 20 ratings |
|---|---|---|
| Hybrid (density) | **1.0845** | 1.0289 |
| Matrix factorisation | 1.0873 | 1.0269 |
| Item-kNN | 1.2536 | 1.0400 |
| Content-based | **1.4761** | 1.1262 |
| Global mean | 1.2074 | 1.2086 |

With one rating content-based is worse than predicting the global mean, and its precision@10
is the lowest of the personalised models (0.0147). Matrix factorisation's bias terms give it
a floor: one rating says nothing about taste, but μ + b_u + b_i is still a reasonable
prediction. Content-based does keep its reach — 43% catalogue coverage at one rating.

### Why not a random split

`results/split_ablation.csv` runs the same pipeline under the temporal split and under a
random row split of the same proportions. The random split lets a user's later ratings into
training and makes every model look better: RMSE improves by 3–10% and precision@10 by
27–87%. The inflation is uneven — most-popular gains most on precision (+87%) and
content-based least (+27%) — so a random split distorts the comparison, not just the
absolute numbers.

Things to know before quoting anything:

- `global_mean` scores a non-zero precision@10 of 0.0233 despite having no ranking signal.
  Its tie-break is ascending item id, and low item ids in ML-100K are the early, popular
  films.
- `most_popular` and `global_mean` have identical RMSE by construction — most-popular
  predicts the global mean and only its ranking differs.
- The `*_ranking` models and `hybrid_frontier` are the same code as their twins with
  different settings, not different algorithms.

## Layout

```
configs/default.yaml     seed, paths, split ratios, protocol, hyperparameters
scripts/                 download_data.py, preprocess.py
src/config.py            config loading and global seeding
src/data/                loaders, item features, temporal splitting
src/data/matrix.py       sparse rating matrix shared by both collaborative models
src/models/              base.py is the contract; one module per model family
src/models/similarity.py centring, cosine and shrinkage for item-kNN
src/evaluation/          metrics.py (accuracy), ranking.py, harness.py, bootstrap.py
src/experiments/         registry.py builds models; run_*.py run a config and dump CSVs
src/experiments/recommend.py  the demo CLI, films rather than decimals
src/plots/               style.py plus scripts that read CSVs and write figures
src/engine/              the recommendation engine (fold-in, forking) and poster lookup
app/                     the Streamlit demo; Home.py routes to app/views/
tests/                   pytest, run locally; no CI
results/                 committed CSVs, the report's source of truth
figures/                 committed figures
```

## Figures

| File | Shows |
|---|---|
| `figures/accuracy.pdf` | RMSE and MAE per model |
| `figures/tradeoff.pdf` | **The headline.** Accuracy against reach, and precision against popularity bias |
| `figures/tuning.pdf` | Both sweeps: accuracy and ranking disagree for item-kNN, agree for MF |
| `figures/hybrid.pdf` | The fitted weighting curve, and precision against coverage across blend weights |
| `figures/cold_start.pdf` | How each model degrades as history shrinks |

## Development

```bash
python -m pytest
```

A local gate is available, since there is no CI by design:

```bash
pip install pre-commit
pre-commit install
```

That runs ruff and the test suite before each commit, and checks the commit message is a
single line with no attribution trailer. `tests/test_commit_message.py` asserts every commit
already in this repository would pass that gate.

Run experiments and tests from the repository root, so that `src` resolves:

```bash
python -m src.experiments.run_main
```

Tests run locally; there is no CI. Almost all of the suite uses a synthetic fixture rather
than the real dataset, so it works on a clone with no `data/` directory. The exception is
`tests/test_app.py`, which drives every page of the demo app headlessly; it needs the
processed data and skips itself without it.

`tests/test_fold_in.py` checks that adding a new user to a trained model gives exactly the
same recommendations as training on that user, for every model where that has a closed
form. The app depends on it: it is what makes the app serve the same models the report
evaluates.

`tests/test_splitting.py` is the important one — it asserts that no user's training ratings
postdate their held-out ratings. If it fails, every number in `results/` is wrong.

`tests/test_models.py` is parameterised over every model family — all eleven pass the
same contract with no exceptions. A model that needs an exception there is a model that would
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

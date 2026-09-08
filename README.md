# MovieLens 100K — four recommenders, one protocol

A controlled comparison of four recommender families on MovieLens 100K. The deliverable is
not a recommender; it is the comparison. Four approaches are pushed through an identical
split, an identical candidate set and an identical metric suite, so we can say precisely
where each one wins and what it pays for the win.

Two of the four are registered twice, tuned once for accuracy and once for ranking, because
the two criteria select different hyperparameters for the same algorithm. Nine models in
total.

The claim the repository exists to support, as the results now stand:

> Under one protocol, **no personalised model beats a popularity baseline on precision@10**
> — and the reason is that precision@10 largely measures popularity rather than accuracy
> (r = 0.91 against popularity bias, 0.49 against RMSE). Tuning an algorithm on ranking
> instead of accuracy moves it *towards* the popularity baseline rather than past it.
> Matrix factorisation wins on rating accuracy while concentrating on a narrow slice of the
> catalogue; content-based scoring is 15% worse on RMSE and spreads three times wider.

One part of this is still a prediction rather than a finding: that a hybrid weighted by
profile density beats both at the crossover. The other prediction — that content-based
scoring survives cold users better than collaborative filtering — has now been tested and
**is false on this dataset**. See [Cold start](#cold-start).

<details>
<summary>What this replaced, and why</summary>

The original thesis read: *"matrix factorisation wins on rating accuracy while recommending
a narrow, popular slice of the catalogue."* The first half held. The second did not.

Accuracy-tuned matrix factorisation has the **lowest** popularity percentile of any
personalised model here (0.5816, against 0.6168 for content-based and 0.6708 for item-kNN).
It is narrow — 15% coverage, Gini 0.97 — but narrow and *obscure*, not narrow and popular.
The narrow-and-popular description belongs to the ranking-tuned variant (0.8976), which is
a different set of hyperparameters for the same algorithm.

Coverage and popularity bias are separate axes, and this dataset separates them: a model
can concentrate hard on a small set of items without those being the popular ones. Keeping
the original wording would have meant reporting a result the numbers contradict.

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

> **GroupLens's certificate expired on 28 Aug 2026.** Until they renew, add
> `--allow-expired-cert`. That does *not* disable verification — it pins the certificate,
> requiring the server to present exactly the one whose SHA-256 is in the config and
> ignoring only the expiry date. It trusts no certificate authority, so it is stricter than
> the default path. Delete the flag and the pin once they renew.

`preprocess.py` writes the processed tables and the train/validation/test split into
`data/processed/`, prints a data card, and asserts that the split has no temporal leakage.

Nothing under `data/` is committed. A fresh clone reproduces it from those two commands.

## Running experiments

```bash
python -m src.experiments.run_main
```

Fits all nine models on train, evaluates on test, writes `results/main_results.csv` and
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
| `mf` | predicted rating | Biases + L2, SGD, early stopping on validation. Tuned on validation RMSE |
| `mf_ranking` | predicted rating | Same algorithm, tuned on validation precision@10. Much weaker L2 |

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
| Item-kNN (RMSE-tuned) | 1.0162 | 0.7992 | 0.0357 | 0.0246 | 0.3378 | 0.6708 | 0.9130 |
| Item-kNN (ranking-tuned) | 1.1153 | 0.8754 | 0.0455 | 0.0367 | 0.3740 | 0.8133 | 0.9097 |
| Matrix factorisation (RMSE-tuned) | 0.9877 | 0.7812 | 0.0267 | 0.0261 | 0.1460 | 0.5816 | 0.9727 |
| Matrix factorisation (ranking-tuned) | 1.0102 | 0.7991 | 0.0451 | 0.0403 | 0.1410 | 0.8976 | 0.9690 |

Test split: 20000 held-out ratings; ranking metrics averaged over the 906 users with at least one relevant held-out item. Seed 20260903, run 20260905T071948.

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

**precision@10 measures popularity, not accuracy.** Across the nine models, precision@10
correlates with popularity bias at r = 0.70 and with RMSE at r = 0.08. Dropping the two
baselines whose ranking is degenerate by construction, that becomes **r = 0.91 against
popularity** and 0.49 against RMSE. Whatever precision@10 is rewarding on this dataset, it
is much closer to "shows popular films" than to "predicts ratings well".

The tuning experiment makes this concrete rather than correlational. Selecting the *same
algorithm* on precision instead of RMSE moves it towards the popularity end:

| | RMSE-tuned | ranking-tuned |
|---|---|---|
| Matrix factorisation, P@10 | 0.0267 | **0.0451** |
| Matrix factorisation, popularity percentile | 0.5816 | **0.8976** |
| Item-kNN, P@10 | 0.0357 | **0.0455** |
| Item-kNN, popularity percentile | 0.6708 | **0.8133** |

Both variants buy their precision with popularity. Neither reaches most-popular's 0.0657,
which sits at a popularity percentile of 0.9935 — the ordering on precision is close to the
ordering on popularity bias throughout.

**No personalised model beats most-popular on precision@10**, and the bootstrap says that
is real rather than noise — every interval excludes zero (`results/bootstrap_ci.csv`). On
precision alone the correct conclusion from this table would be "do not personalise". That
is the strongest available argument for reporting catalogue behaviour beside accuracy, and
it is an argument the data made rather than one we assumed.

**The accuracy and catalogue families genuinely pull apart.** Matrix factorisation wins RMSE
at 0.9877 while showing 15% of the catalogue; content-based is 15% worse on RMSE at 1.1422
and reaches 46%. Item-kNN sits between them on both and is the best all-rounder — second
on RMSE, twice MF's coverage.

**Hyperparameter selection is a reportable finding, not a preliminary.** Within a single
algorithm, validation RMSE and validation precision@10 pick different configurations, and
pick them strongly: item-kNN loses 46% of its precision when selected on RMSE, and even
switches centring convention (item centring wins accuracy, user centring wins ranking).
Both sweeps show the same anti-correlation — r = +0.67 for MF over 32 configs, r = +0.80
for kNN over 48. See `results/tuning_*.csv`.

One caveat on the correlations above: nine models is a small sample, and the models are not
independent draws. Treat r = 0.91 as a description of this table rather than an estimate of
a population parameter. The tuning sweeps, with 32 and 48 configurations each, are the
better-powered version of the same claim.

### Are these gaps real?

`results/bootstrap_ci.csv` resamples the 906 scored users 1000 times, paired across models,
and puts a 95% interval on every pairwise difference. Resampling users rather than ratings
matters: one person's 700 ratings are not 700 independent observations, and resampling
ratings would produce intervals far too narrow.

Every claim the results section leans on excludes zero.

| Comparison | Difference | 95% interval |
|---|---|---|
| MF − most-popular, P@10 | −0.0390 | [−0.0455, −0.0327] |
| Item-kNN − most-popular, P@10 | −0.0300 | [−0.0366, −0.0234] |
| Best personalised (kNN-ranking) − most-popular, P@10 | −0.0203 | [−0.0271, −0.0131] |
| MF − item-kNN, RMSE | −0.0283 | [−0.0368, −0.0208] |
| MF − MF-ranking, P@10 | −0.0185 | [−0.0234, −0.0141] |
| MF − MF-ranking, RMSE | −0.0226 | [−0.0286, −0.0153] |
| Item-kNN − kNN-ranking, RMSE | −0.0995 | [−0.1217, −0.0778] |

So: most-popular's advantage on precision is real against every personalised model
including the ranking-tuned ones; matrix factorisation's RMSE win over item-kNN is real;
and the selection-criterion trade-off is real in both directions — the ranking-tuned
variants really do rank better and really are less accurate.

### Cold start

`results/cold_start.csv` and `figures/cold_start.pdf`. 200 users have their training
history truncated to 1, 3, 5, 10 and 20 ratings; the other 743 are left intact so the
model's view of the catalogue stays normal, and evaluation is restricted to the truncated
users. Every model is refitted at every level.

**The prediction was wrong.** Content-based scoring was expected to hold up best with
almost no history, since it can score an item from its features without needing co-raters.
It is in fact the **worst** model at one rating, on both metrics.

| Test RMSE | 1 rating | 20 ratings | degradation |
|---|---|---|---|
| Matrix factorisation | **1.0538** | 1.0010 | +0.053 |
| Item-kNN | 1.2536 | 1.0400 | +0.214 |
| Content-based | **1.4761** | 1.1262 | +0.350 |
| Global mean | 1.2074 | 1.2086 | — |

At one rating, content-based is worse than predicting the global mean for everybody. Its
precision@10 is also last of the nine (0.0147 against most-popular's 0.0406).

Matrix factorisation is the most robust of the personalised models, and the reason is its
bias terms. With one rating there is nothing to learn about a user's taste, but `mu + b_u +
b_i` is still a competent predictor, so the model degrades to a good baseline rather than
to noise. Content-based has no such floor: with one rated film the profile *is* that film's
feature vector, and a globally-fitted calibration turns that into confident, wrong
predictions.

The one thing content-based does keep is reach — 43% catalogue coverage at a single rating,
against 3% for matrix factorisation. So the honest version of the original claim is that
content-based survives cold start in *coverage*, not in accuracy or ranking.

A detail that acts as a correctness check: at one rating, item-kNN (ranking-tuned) scores
exactly the same RMSE as the item-mean baseline, 1.0752. It uses user centring, so with a
single rating the user's mean equals that rating, every deviation is zero, and the model
correctly falls back to its shrunk item mean.

**This undercuts the hybrid as specified.** The plan was content-heavy for sparse profiles
and CF-heavy for dense ones. On this data content is worst exactly where it was meant to
be strongest, so a density-weighted blend towards content would make sparse users worse,
not better. The hybrid needs rethinking before it is built rather than after.

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
- The two `*_ranking` models are the same code as their twins with different
  hyperparameters, not different algorithms. They exist to make the selection-criterion
  effect a row in this table rather than a claim in the prose.

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

`tests/test_models.py` is parameterised over every model family — all nine pass the same
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

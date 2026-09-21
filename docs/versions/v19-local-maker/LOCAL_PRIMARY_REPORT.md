# G19-05 local-goal readout screen — 21 September 2026

Does a predictive bank help recover local goals and operations? Its advantage reverses with evidence: across four training budgets, the bank lowers logarithmic loss by 0.13193 nats with final artifacts alone, but raises it by 0.83956 nats when every operation is witnessed, relative to direct history. With artifacts alone it reaches the exact-reference loss plus 0.02 nats at 512 labels; with full witnesses direct history reaches that target at 2,048 labels and the bank does not. This is an exploratory constructed-method result, miniature — architecture untested; joint process correspondence and human intent are not established.


Logarithmic loss measures categorical prediction error in nats; lower is better.
The first table reports the learned bank's difference from each fitted comparator,
averaged by normalized area over log label budget. Negative values favor the bank.
Brackets are 95% paired development-lineage intervals conditional on these fits.
Each evidence row receives the same sampled trajectories and training teachers.

| Evidence | Bank minus direct history | Bank minus frozen state |
|---|---:|---:|
| Final artifact only | -0.13193 [-0.14690, -0.11794] | -0.13155 [-0.14656, -0.11746] |
| Artifact plus truthful context | -0.25897 [-0.28097, -0.23566] | -0.20584 [-0.22793, -0.18203] |
| Sparse operation witnesses | -0.23429 [-0.27704, -0.19203] | -0.05791 [-0.09010, -0.02417] |
| Complete operation witnesses | 0.83956 [0.81668, 0.86384] | 0.42111 [0.39113, 0.45101] |

The next table reports absolute loss averaged equally over three local-goal and
three operation questions. Columns count nested target-labelled training histories.
Each arm also has the same 2,048 old-outcome training histories per draw. The
full-data mean uses all 2,048 target labels at every budget, so only its last column
is label-matched. The supplied-law reference is privileged; the exact-bank arm
still has to learn its target head.

| Evidence / readout | 32 labels | 128 labels | 512 labels | 2,048 labels |
|---|---:|---:|---:|---:|
| Final artifact only / Direct history | 1.99889 | 1.45753 | 1.35331 | 1.33618 |
| Final artifact only / Learned bank | 1.50654 | 1.34567 | 1.32400 | 1.31933 |
| Final artifact only / Frozen old-task state | 1.98682 | 1.45783 | 1.35582 | 1.34038 |
| Final artifact only / Exact bank with learned head | 1.56293 | 1.34481 | 1.32118 | 1.31520 |
| Final artifact only / Full-data mean | 1.34718 | 1.34718 | 1.34718 | 1.34718 |
| Final artifact only / Supplied-law exact reference | 1.31137 | 1.31137 | 1.31137 | 1.31137 |
| Artifact plus truthful context / Direct history | 2.52847 | 1.56700 | 1.33553 | 1.28563 |
| Artifact plus truthful context / Learned bank | 1.52124 | 1.32991 | 1.30103 | 1.28220 |
| Artifact plus truthful context / Frozen old-task state | 2.17247 | 1.55400 | 1.35457 | 1.31076 |
| Artifact plus truthful context / Exact bank with learned head | 1.54469 | 1.31983 | 1.27451 | 1.26222 |
| Artifact plus truthful context / Full-data mean | 1.34718 | 1.34718 | 1.34718 | 1.34718 |
| Artifact plus truthful context / Supplied-law exact reference | 1.22971 | 1.22971 | 1.22971 | 1.22971 |
| Sparse operation witnesses / Direct history | 2.99134 | 1.40335 | 0.93361 | 0.86626 |
| Sparse operation witnesses / Learned bank | 1.42180 | 1.17790 | 1.12677 | 1.09466 |
| Sparse operation witnesses / Frozen old-task state | 1.97089 | 1.30988 | 0.98405 | 0.91449 |
| Sparse operation witnesses / Exact bank with learned head | 1.51030 | 1.29835 | 1.24003 | 1.21782 |
| Sparse operation witnesses / Full-data mean | 1.34718 | 1.34718 | 1.34718 | 1.34718 |
| Sparse operation witnesses / Supplied-law exact reference | 0.68972 | 0.68972 | 0.68972 | 0.68972 |
| Complete operation witnesses / Direct history | 0.69046 | 0.20586 | 0.14106 | 0.12857 |
| Complete operation witnesses / Learned bank | 1.30460 | 1.07625 | 1.04185 | 1.00943 |
| Complete operation witnesses / Frozen old-task state | 1.23896 | 0.65310 | 0.51567 | 0.44708 |
| Complete operation witnesses / Exact bank with learned head | 1.48576 | 1.20154 | 1.18691 | 1.14038 |
| Complete operation witnesses / Full-data mean | 1.34718 | 1.34718 | 1.34718 | 1.34718 |
| Complete operation witnesses / Supplied-law exact reference | 0.11537 | 0.11537 | 0.11537 | 0.11537 |

The label-matched mean audit is explicitly post-screen and separately retained.
It uses exactly the original nested target rows at each budget, with the same
category floor. At 2,048 labels it equals the original full-data mean. It changes
neither the frozen science nor any original score. Full budget curves and the
separate three-goal/three-operation means are in the independent review records.

The next table separates goal and operation loss at 2,048 labels. Each column
averages the three time-local questions of that kind; these marginal forecasts
are not joint process-hypothesis probabilities. In the complete-witness condition,
the operations themselves are supplied as evidence, so their accurate prediction
is an evidence-use check, not inference of unknown actions.

| Evidence / readout | Local-goal loss | Operation loss |
|---|---:|---:|
| Final artifact only / Direct history | 1.07974 | 1.59263 |
| Final artifact only / Learned bank | 1.07519 | 1.56347 |
| Final artifact only / Frozen old-task state | 1.08132 | 1.59943 |
| Final artifact only / Exact bank with learned head | 1.07259 | 1.55780 |
| Final artifact only / Supplied-law exact reference | 1.07289 | 1.54985 |
| Artifact plus truthful context / Direct history | 1.03895 | 1.53230 |
| Artifact plus truthful context / Learned bank | 1.04471 | 1.51969 |
| Artifact plus truthful context / Frozen old-task state | 1.05731 | 1.56421 |
| Artifact plus truthful context / Exact bank with learned head | 1.03260 | 1.49185 |
| Artifact plus truthful context / Supplied-law exact reference | 1.01729 | 1.44213 |
| Sparse operation witnesses / Direct history | 0.75794 | 0.97458 |
| Sparse operation witnesses / Learned bank | 0.94240 | 1.24692 |
| Sparse operation witnesses / Frozen old-task state | 0.79796 | 1.03102 |
| Sparse operation witnesses / Exact bank with learned head | 1.03950 | 1.39614 |
| Sparse operation witnesses / Supplied-law exact reference | 0.66182 | 0.71762 |
| Complete operation witnesses / Direct history | 0.25067 | 0.00647 |
| Complete operation witnesses / Learned bank | 0.94085 | 1.07802 |
| Complete operation witnesses / Frozen old-task state | 0.47348 | 0.42069 |
| Complete operation witnesses / Exact bank with learned head | 0.99873 | 1.28203 |
| Complete operation witnesses / Supplied-law exact reference | 0.23074 | 0.00000 |

The aggregate bank advantage in sparse witnesses reverses at the higher budgets;
an area average must not hide that crossover. With complete witnesses, the bank
loses to direct history in every retained feature-seed/training-draw combination.
The old-task bank predicts fresh-episode endpoints from persistent maker state;
the new questions concern transient goals and operations in the preceding episode.
Even its exact version need not be a sufficient statistic for those questions.
This is a live structural rival, not an exact nonidentification certificate.
The separately prepared sufficiency diagnostic must establish an actual witness
before any information-loss claim. Poor readout geometry, finite labels and
regularization remain additional rivals; a probability-valid successor is prepared.

This screen uses sixteen development coefficient lineages, two fixed random-feature
seeds and two independent training draws. Each draw has 128 training lineages with
16 histories each and 32 development histories per development lineage. Its 43,008
score rows aggregate repeated questions, seeds, budgets and evidence projections;
they are not independent worlds. Seed and draw variability stays inside paired
lineage. No test or confirmation lineage was consumed, and two random-feature
seeds do not satisfy the later five-trained-initialization requirement.

All readouts have the same target-head coefficient count and the same available
training teachers. Their representation geometry and total capacity differ.
Local-goal teacher labels come from evaluator records on training examples only.
The exact reference additionally knows the law. The frozen ridge output uses its
declared clipping, category floor and normalization; no scoring repair changed
after observing outcomes. The full record retains invalid raw forecasts, squared
probability errors, old-task capability, shuffled-label controls and component costs.

**Disposition:** exploratory advantage under artifact/context evidence, a
budget-dependent crossover under sparse witnesses, and reversal under full
witnesses. The artifact-only threshold crossing is descriptive, not confirmation
or proof of a twofold label saving against a comparator that never crosses within
the observed range. G-P1 remains open beyond this fixed-feature instrument.
**Pursuit:** test readout regularization and process sufficiency independently;
continue the already executed frame and jointness leaves through their own reviews.
No tiny reader or causal interchange is admitted by this screen.

**Evidence roles:** original reader-transport IDs remain in the evaluator capsule.
Their class coordinate is constant in this local batch, but all pairing metadata
is excluded from the separate blind reader archive. Training files contain only
features and explicitly declared teacher labels; development files contain only
features. Exact banks, targets, full trajectories, forecasts and fitted coefficients
are scientific/evaluator material. The original arrays and source remain unchanged.

**Validation:** all 43,008 score rows and every aggregate cell were reconstructed
independently from saved predictions and targets, with zero discrepancy. The paired
area intervals reproduce using the independent four-budget weighting formula.
Adjacent replay and a full replay from freshly extracted source match all deterministic
files, including model arrays, forecasts and raw trajectories. All six scoped fixture
controls pass. These checks cannot establish joint path compatibility, calibration
of unsupported mental assertions, exact bank sufficiency, human intent, a universal
training-population interval, or architectural robustness.

[Scientific records and export](../../../results/v19/G19-05-local-primary-1/EXPORT_MANIFEST.json).
[Blind reader archive](../../../results/v19/G19-05-local-primary-1/READER.zip).
[Prepared independent successors](NEXT_DESIGNS_2026-09-21.md).

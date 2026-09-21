# Bounded optimizer comparison — independently verified

Does reducing optimization error change the predictive-bank comparison? Under the unchanged penalized objective, the bank lowers old-world logarithmic loss by 0.09829 nats relative to direct history, with a 95% paired lineage interval of [0.06739, 0.13098]. In the local world, artifacts alone give a difference smaller than the declared practical margin, added context favors the bank, and complete operation witnesses favor direct history by 0.77324 nats. All forecasts and gradients verify, but no fit reaches the declared gradient tolerance. This is an exploratory constructed-method advantage with reversals, miniature — architecture untested; process correspondence and human intent remain unestablished.

Logarithmic loss measures forecast error in nats; lower is better. The normalized
area averages loss over the logarithm of nested training budgets. Negative bank-
minus-rival differences favor the bank. Intervals resample paired coefficient
lineages, retaining both feature seeds, both training draws and repeated questions
within lineage. They quantify conditional development variation, not general
training uncertainty or independent confirmation. Old-world areas retain all four
classes with equal weights; the history-inert class is not removed.

The frozen comparison changes only the optimizer, retaining the earlier softmax
objective, penalty, features, zero initialization, labels and populations. The old
world has eight development lineages and budgets 32/128. The local world has sixteen
development lineages and budgets 32/128/512/2048. Fixed random-feature seeds are not
five independently trained sequence models. Exact-bank and exact-reference arms
have explicit law privilege. Local-goal training labels are evaluator supervision.

The tables give normalized loss differences and 95% conditional intervals. Rows
name the evidence condition and columns name each rival. The old-world table uses
the equal-class population; the first local table averages goal and operation
questions. The next two keep those target roles separate.

| Evidence | Bank minus direct history | Bank minus frozen latent | Bank minus matched-label mean |
|---|---:|---:|---:|
| Old artifact history, equal classes | -0.09829 [-0.13098, -0.06739] | -0.09860 [-0.12464, -0.07551] | -0.18143 [-0.22196, -0.14679] |

Local questions together:

| Evidence | Bank minus direct history | Bank minus frozen latent | Bank minus matched-label mean |
|---|---:|---:|---:|
| Final artifact | -0.01356 [-0.01679, -0.01033] | -0.00081 [-0.00215, 0.00057] | 0.00642 [0.00423, 0.00877] |
| Artifact and context | -0.03149 [-0.03727, -0.02538] | -0.00374 [-0.00472, -0.00273] | 0.00429 [0.00249, 0.00620] |
| Sparse operation witnesses | 0.25045 [0.23994, 0.26189] | 0.01596 [0.01460, 0.01744] | -0.00707 [-0.01007, -0.00406] |
| Complete operation witnesses | 0.77324 [0.76057, 0.78580] | 0.05362 [0.05123, 0.05609] | -0.02683 [-0.03065, -0.02315] |

Local-goal questions:

| Evidence | Bank minus direct history | Bank minus frozen latent | Bank minus matched-label mean |
|---|---:|---:|---:|
| Final artifact | -0.01002 [-0.01243, -0.00743] | -0.00297 [-0.00421, -0.00161] | -0.00061 [-0.00077, -0.00044] |
| Artifact and context | -0.02771 [-0.03357, -0.02166] | -0.00257 [-0.00362, -0.00150] | -0.00133 [-0.00159, -0.00108] |
| Sparse operation witnesses | 0.14711 [0.13462, 0.16056] | 0.00812 [0.00631, 0.00997] | -0.01378 [-0.01474, -0.01286] |
| Complete operation witnesses | 0.52034 [0.50024, 0.54040] | 0.03593 [0.03228, 0.03963] | -0.02945 [-0.03172, -0.02719] |

Operation questions:

| Evidence | Bank minus direct history | Bank minus frozen latent | Bank minus matched-label mean |
|---|---:|---:|---:|
| Final artifact | -0.01710 [-0.02195, -0.01226] | 0.00135 [-0.00081, 0.00330] | 0.01345 [0.00906, 0.01823] |
| Artifact and context | -0.03527 [-0.04199, -0.02839] | -0.00490 [-0.00609, -0.00374] | 0.00992 [0.00640, 0.01374] |
| Sparse operation witnesses | 0.35379 [0.34411, 0.36446] | 0.02380 [0.02234, 0.02533] | -0.00036 [-0.00585, 0.00529] |
| Complete operation witnesses | 1.02613 [1.01774, 1.03457] | 0.07131 [0.06950, 0.07329] | -0.02422 [-0.02992, -0.01843] |

Relative to direct history, the artifact-only local area remains entirely within
the prespecified ±0.02-nat practical margin. Context now favors the bank beyond
that margin. Sparse and full witnesses strongly favor direct history. Yet bank
areas with artifact/context alone remain slightly worse than the matched-label
mean. A relative improvement therefore does not by itself establish useful primary
capability. The old bank remains 1.54405 nats above the privileged reference on
the log-budget average. All class and budget results, absolute curves and separate
draw/feature-seed comparisons are in the independent review.

All 288 final objectives decrease relative to the previous fixed schedule.
Independently reconstructed maximum absolute gradients range from 0.00000011434
to 0.00004982853. None meets the frozen 0.0000001 tolerance: 280 successful solver
exits reflect relative objective reduction, while eight reach the iteration limit.
These are substantially smaller gradients, not a certificate of the exact optimum.
The limited optimizer comparison is accepted as executed; the strict convergence
claim remains unmet. Additional changes need an identified scientific cause, not
repeated tuning against these development outcomes.

The earlier scoring deviation is preserved: fitted softmax probabilities are
unfloored, matching the predecessor, while references and ridge outputs receive
the fixed repair. The predecessor's separate floor sensitivity bound does not
automatically transfer to these newly fitted probabilities. No floor-adjusted
effect is substituted here. This is a same-objective optimizer diagnostic, not an
isolated proof that features lack information or that a particular process occurred.

**Validation:** independent known-loss, known-gradient, constant-label placebo,
finite-difference and normalization controls pass. All 40,704 scores reconstruct
exactly, and 432 forecast files reconstruct within 2.6e-15 from saved heads and
features. All 288 objectives and gradients, equal-class weights, nested means and
paired curves verify. Both complete replays match all 843 deterministic output
files; each execution's timing hash is checked separately. Source, plan, numerical
inputs, environment and failed/limited attempts remain retained. Test and
confirmation lineages are untouched. These checks cannot establish general
architecture validity, joint process correspondence or human intent.

**Warrant:** exploratory method advantage/reversal; conditional local artifact-only
practical equivalence; strict gradient tolerance unmet. **Pursuit:** preserve this
bounded optimizer result and advance the independent recursive-update and task
assessment branches. The primary learning question and the week remain open.

[Scientific and evaluator evidence](../../../results/v19/G19-A-convergence-1/EXPORT_MANIFEST.json).

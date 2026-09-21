# Joint-process readout: negligible artifact gains and witnessed-history reversal

Does a learned predictive bank help recover the maker’s complete local goal-and-operation sequence? No practical bank advantage was found. Its learning-curve gain over raw history is below 0.007 nats with artifacts or context alone; with sparse and complete operation witnesses, loss increases by 0.09365 and 0.27545 nats. Independent reconstruction and both full replays verify this exploratory constructed-method result. Process compatibility remains poor; historical uniqueness and human intent are not established.

Logarithmic loss measures how much probability the reader assigns to complete
three-step goal-and-operation histories, in nats; lower is better. The primary
summary is the area under loss versus log label budget, divided by the log-budget
span. Rows below identify the evidence received. Columns subtract the comparator
from the learned bank; brackets are 95% paired coefficient-lineage bootstrap
intervals, conditional on the retained fits. The practical margin is 0.02 nats.

| Evidence | Bank minus raw history | Bank minus latent state |
|---|---:|---:|
| Final artifact | -0.00265 [-0.00284, -0.00246] | -0.00056 [-0.00061, -0.00052] |
| Artifact and context | -0.00676 [-0.00715, -0.00638] | -0.00074 [-0.00081, -0.00069] |
| First operation witnessed | +0.09365 [+0.08855, +0.09896] | +0.00231 [+0.00221, +0.00241] |
| All operations witnessed | +0.27545 [+0.26134, +0.29018] | +0.00701 [+0.00674, +0.00729] |

Artifact/context contrasts and every bank-versus-latent contrast lie inside the
practical margin throughout these conditional intervals. The two witnessed-history
bank-versus-raw reversals exceed the margin throughout. These are sufficiently
precise conditional estimates, not population-wide equivalence or confirmation.
No positive bank advantage is promoted to untouched confirmation on this result.

The next table preserves every nested label budget and evidence condition with
the same paired conditional intervals. No pooled evidence mixture replaces them.

| Evidence | Label budget | Bank minus raw history | Bank minus latent state |
|---|---:|---:|---:|
| Final artifact | 32 | -0.00761 [-0.00816, -0.00710] | -0.00100 [-0.00110, -0.00091] |
| Final artifact | 128 | -0.00284 [-0.00307, -0.00262] | -0.00058 [-0.00063, -0.00053] |
| Final artifact | 512 | -0.00103 [-0.00110, -0.00097] | -0.00042 [-0.00045, -0.00039] |
| Final artifact | 2048 | -0.00053 [-0.00056, -0.00050] | -0.00037 [-0.00039, -0.00035] |
| Artifact and context | 32 | -0.01962 [-0.02079, -0.01849] | -0.00089 [-0.00097, -0.00081] |
| Artifact and context | 128 | -0.00809 [-0.00860, -0.00761] | -0.00080 [-0.00086, -0.00073] |
| Artifact and context | 512 | -0.00212 [-0.00220, -0.00203] | -0.00067 [-0.00072, -0.00062] |
| Artifact and context | 2048 | -0.00051 [-0.00053, -0.00049] | -0.00064 [-0.00070, -0.00060] |
| First operation witnessed | 32 | +0.04040 [+0.03808, +0.04284] | +0.00207 [+0.00197, +0.00217] |
| First operation witnessed | 128 | +0.08802 [+0.08317, +0.09310] | +0.00243 [+0.00232, +0.00255] |
| First operation witnessed | 512 | +0.11208 [+0.10603, +0.11838] | +0.00225 [+0.00217, +0.00233] |
| First operation witnessed | 2048 | +0.12132 [+0.11493, +0.12799] | +0.00244 [+0.00234, +0.00253] |
| All operations witnessed | 32 | +0.11165 [+0.10574, +0.11786] | +0.00591 [+0.00571, +0.00611] |
| All operations witnessed | 128 | +0.27018 [+0.25601, +0.28492] | +0.00750 [+0.00717, +0.00784] |
| All operations witnessed | 512 | +0.32731 [+0.31069, +0.34464] | +0.00698 [+0.00672, +0.00724] |
| All operations witnessed | 2048 | +0.34607 [+0.32880, +0.36415] | +0.00719 [+0.00693, +0.00745] |

The absolute losses below use the largest 2,048-label budget. The frequency reader
gets the same target-label prefix, and the exact oracle gets the supplied native
law. Its much lower loss quantifies substantial remaining headroom.

| Evidence | Raw history | Latent state | Learned bank | Same-budget frequencies | Exact oracle |
|---|---:|---:|---:|---:|---:|
| Final artifact | 5.60918 | 5.60902 | 5.60865 | 5.59257 | 4.09015 |
| Artifact and context | 5.60883 | 5.60897 | 5.60832 | 5.59257 | 3.47205 |
| First operation witnessed | 5.48494 | 5.60382 | 5.60625 | 5.59257 | 2.03924 |
| All operations witnessed | 5.25646 | 5.59534 | 5.60252 | 5.59257 | 0.68049 |

Process correspondence is a separate result. The next table reports the largest
budget: probability assigned to compatible histories, exact true-posterior mass
covered by the reader's nominal 90% candidate set, frequency of an incompatible
most-probable history, and mean correctly localized goals and operations across
the three positions. Each entry first weights packets by their native probability,
then averages lineages and fits. Candidate coverage is not asserted calibration.

| Evidence | Reader | Compatible probability | True mass in 90% candidate set | Incompatible modal process | Goal accuracy | Operation accuracy |
|---|---|---:|---:|---:|---:|---:|
| Final artifact | Raw history | 32.05% | 83.47% | 78.71% | 37.96% | 28.18% |
| Final artifact | Learned bank | 32.05% | 83.32% | 78.71% | 37.96% | 28.18% |
| Artifact and context | Raw history | 18.49% | 83.33% | 84.54% | 37.96% | 28.18% |
| Artifact and context | Learned bank | 18.48% | 83.86% | 84.54% | 37.96% | 28.18% |
| First operation witnessed | Raw history | 5.55% | 84.19% | 86.15% | 43.91% | 34.00% |
| First operation witnessed | Learned bank | 4.78% | 83.92% | 91.51% | 37.96% | 28.18% |
| All operations witnessed | Raw history | 1.94% | 85.72% | 92.52% | 55.12% | 44.77% |
| All operations witnessed | Learned bank | 1.12% | 84.19% | 96.45% | 37.96% | 28.18% |

All learned target heads abstain from a single-process assertion at this budget
because their maximum probability stays below 0.5. Incompatible modes therefore
describe their distributions, not confident assertions that the policy emitted.
Complete witnesses still leave goal ambiguity, but the bank's roughly 1.12%
compatible mass is far from the reference. Useful historical recovery is not
established by the small artifact-only relative gain.

The fixed 5,832-tuple universe retains the same scored question at all budgets.
Only training-prefix labels enter the learned alphabet. At 2,048 labels the
unseen true mass is about 2.44%, while the fixed reserved unknown mass is about
0.049%. This is a coverage/calibration limitation of the declared model, not
missing evaluator labels silently supplied at test time. No post-outcome repair
or alternative alphabet replaces the original scores.

All 480 fits report solver success. Independent gradients meet the declared
1e-7 tolerance in 290; the other 190 remain recorded, with maximum 1.152e-6.
The schedule and criteria were not changed. This does not prove that optimization
is irrelevant or that the observed representation ranking is structural.

The comparison uses 32 training and 16 development coefficient lineages, two
interleaved training draws and five paired fixed-feature seeds. Seeds are not
five independently trained sequence models. Every arm has the same target labels
and 2,048 auxiliary observable distribution teachers; target-head parameter
counts match, representation geometry does not. Areas average draws/seeds within
lineage before 10,000 bootstrap resamples. The following table keeps fit variation
separate: the two areas condition on each training draw, while the range spans the
five feature seeds after averaging draws. Neither is a universal training interval.

| Evidence | Comparator | Two draw-specific areas | Range across five feature-seed areas |
|---|---|---|---:|
| Final artifact | Raw history | -0.00225, -0.00304 | -0.00291 to -0.00237 |
| Final artifact | Frozen latent state | -0.00056, -0.00057 | -0.00073 to -0.00035 |
| Artifact and context | Raw history | -0.00579, -0.00773 | -0.00768 to -0.00624 |
| Artifact and context | Frozen latent state | -0.00078, -0.00070 | -0.00087 to -0.00057 |
| First operation witnessed | Raw history | +0.09722, +0.09009 | +0.08372 to +0.10203 |
| First operation witnessed | Frozen latent state | +0.00237, +0.00225 | +0.00152 to +0.00296 |
| All operations witnessed | Raw history | +0.27991, +0.27100 | +0.26072 to +0.29185 |
| All operations witnessed | Frozen latent state | +0.00610, +0.00792 | +0.00549 to +0.00813 |

Independent reconstruction verifies 663,552 native paths across training and
development, all public projections and weights, label selections, auxiliary
models, 480 saved heads and 10,240 scored strata. Maximum saved-forecast error is
4.45e-16 and maximum score discrepancy 1.14e-13. Known-process, null, corruption,
finite-difference, hidden-field, unseen-label and undo controls pass. Discrete
candidate decisions are reconstructed from saved probabilities to retain the
actual numerical tie ordering. Both complete replays match all 2,023 deterministic
files; timing records verify separately. Validation does not prove robustness to
another architecture, new training population, or human behavior.

Anonymous evidence, explicit training teachers, scientific summaries and evaluator
truth have separate roles. The accepted scientific export includes source, plan,
native paths and score records. Full model/forecast arrays remain retained in
the evaluator capsule with every identity published in its manifest; they are
not reader input. The source and complete replay proofs support regeneration.

**Warrant:** exploratory constructed method; practically negligible conditional
artifact/context gain, witnessed-history reversal, poor process compatibility;
miniature — architecture untested. No historical uniqueness or human-intent claim.

**Pursuit:** no positive-bank confirmation promotion and no seed expansion.
Reserved test/confirmation lineages remain untouched. The crossed-rule checker
and independent support/holdout result remain active alternatives. The broader
primary architectural claim stays open; this finite joint-label comparison is
complete and the research week remains active.

[Frozen contrast](JOINT_READOUT_PROTOCOL.md), [independent protocol](JOINT_REVIEW_PROTOCOL.md),
[scientific records](../../../results/v19/G19-05-joint-readout-1/EXPORT_MANIFEST.json).

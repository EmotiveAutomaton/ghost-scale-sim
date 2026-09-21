# Native factor fixtures and the cost of whole-state replacement

Can a reader's internal state change one maker factor without importing an unrelated skill change? The saved readers remain predictively capable, but full-state replacement is not selective: after 32 observations it raises stay-case loss by 0.18134 nats for the transformer and 0.20203 for the recurrent reader. Exact native fixtures, all 41,600 scores and both full replays verify this constructed-method diagnostic; selective access and human intent are not established.

Logarithmic loss measures forecast error in nats; lower is better. This table
compares the saved readers with their unchanged training-target mean and the
exact history-conditional reference on the new balanced-maker challenge. Rows
identify the architecture and observation prefix. Brackets are 95% paired
coefficient-lineage bootstrap intervals, conditional on the retained fits.

| Reader | Observations | Reader loss | Mean loss | Exact conditional loss | Reader minus mean, 95% interval |
|---|---:|---:|---:|---:|---:|
| Transformer | 8 | 1.92412 | 1.98881 | 1.90973 | -0.06468 [-0.07654, -0.05478] |
| Transformer | 32 | 1.90032 | 1.99344 | 1.87514 | -0.09312 [-0.10189, -0.08449] |
| Recurrent | 8 | 1.91870 | 1.98881 | 1.90973 | -0.07010 [-0.08197, -0.06099] |
| Recurrent | 32 | 1.88847 | 1.99344 | 1.87514 | -0.10497 [-0.11306, -0.09745] |

All sixteen per-fit capability decisions pass: each of two architectures, two
draws and two seeds at both lengths beats the mean by at least 0.02 nats, recovers
at least half its exact-reference headroom, and makes nonconstant predictions.
These are the prior pilot's engineering thresholds, reapplied without adjustment.
Mean gains range from 0.05453 to 0.10852 nats across the individual checks. This
does not identify the maker with certainty or recover a preceding local process.

The next table reports full-donor loss minus unchanged-recipient loss against
the same known-maker hybrid target, in nats with the same conditional intervals.
Positive differences are damage. Change donors flip the selected purpose or
belief factor and skill; stay donors change only skill. The exact hybrid changes
only the selected factor. Stay rows for the two factors are the same paired
comparison, so they are shown once, not counted as independent evidence.

| Reader | Observations | Purpose change | Belief change | Factor stay (either factor) |
|---|---:|---:|---:|---:|
| Transformer | 8 | +0.05929 [+0.04937, +0.06874] | -0.04974 [-0.05966, -0.03886] | +0.12124 [+0.10450, +0.13703] |
| Transformer | 32 | +0.10886 [+0.09593, +0.12058] | -0.07305 [-0.08366, -0.06376] | +0.18134 [+0.16456, +0.19613] |
| Recurrent | 8 | +0.06242 [+0.04638, +0.07458] | -0.05517 [-0.06461, -0.04362] | +0.12429 [+0.10632, +0.13987] |
| Recurrent | 32 | +0.09806 [+0.08660, +0.10848] | -0.08863 [-0.09932, -0.08062] | +0.20203 [+0.18407, +0.21828] |

Whole-state replacement improves belief-change prediction but worsens purpose
change and both stay comparisons. Each displayed interval excludes zero; the
stay and purpose damage cannot be hidden by pooling with the belief improvement.
The comparator is an unchanged recipient reader, not an oracle or a selective
intervention. These effects do not show that a suitable direction cannot exist.

The exact stay distribution is identical to the recipient's. The mean total
variation of true purpose and belief changes is 0.22727 and 0.26334, respectively.
Total variation is half the sum of absolute probability differences, from zero
for identical distributions to one for disjoint distributions. All 1,024 scored
development contexts per change factor have nonzero effects; none was filtered
out after inspection. Full donor replacement adds 0.21150 mean total variation
from the unwanted skill change in either factor. Repeated draws do not turn the
same law into independent evidence.

The packet covers 32 original training and eight development coefficient
lineages, both original history draws and all sixteen makers. Each paired maker
receives the same contexts and operation-selection uniforms for 32 episodes.
The native controller chooses each local goal before its operation and preserves
artifact, undo buffer and step. In total, 40,960 complete episodes and 5,120 factor
pairs produce 41,600 score rows. No model or alignment is fitted. The governing
purpose and belief factor are persistent maker properties, not relabelled local
goals. Test and confirmation lineages remain untouched.

Independent complete-path enumeration agrees with the producer's dynamic-program
laws. Every sampled transition, selected goal, common uniform, anonymized mapping,
posterior, hybrid and score is checked. Log-likelihood posterior reconstruction
differs by at most 1.32e-14; score reconstruction by at most 1.34e-15. The saved
model equations reuse the prior independently validated NumPy reconstruction;
this pass is not a second independent implementation of the network architecture.
All 548 deterministic files match original, adjacent and full extracted-source
replays. Execution timings have separately verified hashes. A failed review
receipt serialization is retained and charged; it changed no scientific data.

Each interval resamples eight paired development lineage means 10,000 times,
after averaging makers, contexts, draws and seeds. Sixteen checks are not sixteen
independent trained architectures; two draws do not estimate universal training
uncertainty. Interventions are scored against known-maker hybrids, whereas normal
capability uses history-conditional targets. Their different information access
is essential, not an interchangeable reference choice.

Anonymous token codes and novelty flags are the only blind reader inputs.
Pairing labels, native trajectories, makers, laws, hidden states and predictions
are scientific/evaluator evidence. The unchanged saved models, original input
capsule, exact source and all raw arrays are preserved separately from reader
inputs. No human or natural-language result follows.

**Warrant:** verified constructed-method capability and nonselective full-state
replacement diagnostic; miniature — architecture untested. No selective
mediation, historical process correspondence, or confirmation is established.

**Pursuit:** freeze and admit the [rank-one access comparison](INTERCHANGE_ALIGNMENT_PROTOCOL.md)
at the existing final hidden site, with training-only factor labels and equal-rank
controls. It uses the same two consumed model settings without retraining them.
F1 local rollouts and D1 probabilistic cue correction remain independent prepared
alternatives. The primary local comparison and research week remain open.

[Frozen fixtures](INTERCHANGE_FIXTURE_PROTOCOL.md),
[independent scores](../../../results/v19/G19-C1-fixtures-1/INDEPENDENT_REVIEW.json),
and [role-separated exports](../../../results/v19/G19-C1-fixtures-1/EXPORT_MANIFEST.json).

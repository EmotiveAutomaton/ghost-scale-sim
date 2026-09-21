# Factor decoding and selective coordinate replacement

Can a direction that predicts a maker factor selectively change its consequences? No practically meaningful selective benefit was found for the ridge-probe direction: its 32-observation gains are 0.00273–0.00728 nats, below the declared 0.02-nat margin. The class-mean direction gives larger purpose gains with small stay costs, but its belief swap causes large stay damage. All 57,344 scores and both full replays verify this exploratory constructed-method result; joint selective access and human intent are not established.

Logarithmic loss measures forecast error in nats; lower is better. The following
table reports ridge-probe replacement loss minus unchanged-recipient loss against
the same known-maker hybrid target. Negative numbers improve prediction. Rows
separate architecture, evidence prefix and factor; change donors flip the selected
factor and skill, while stay donors change skill only. Brackets are 95% paired
coefficient-lineage bootstrap intervals, conditional on the retained fits.

| Reader | Observations | Factor | Change loss difference | Stay loss difference |
|---|---:|---|---:|---:|
| Transformer | 8 | Purpose | -0.00182 [-0.00239, -0.00122] | -0.00010 [-0.00020, -0.00001] |
| Transformer | 8 | Belief | -0.00474 [-0.00531, -0.00420] | -0.00010 [-0.00024, +0.00001] |
| Transformer | 32 | Purpose | -0.00273 [-0.00353, -0.00194] | +0.00006 [+0.00001, +0.00010] |
| Transformer | 32 | Belief | -0.00604 [-0.00651, -0.00552] | -0.00023 [-0.00036, -0.00011] |
| Recurrent | 8 | Purpose | -0.00213 [-0.00272, -0.00154] | -0.00001 [-0.00006, +0.00004] |
| Recurrent | 8 | Belief | -0.00368 [-0.00410, -0.00329] | -0.00013 [-0.00031, +0.00004] |
| Recurrent | 32 | Purpose | -0.00513 [-0.00608, -0.00425] | -0.00007 [-0.00011, -0.00002] |
| Recurrent | 32 | Belief | -0.00728 [-0.00778, -0.00684] | -0.00023 [-0.00031, -0.00013] |

All change intervals exclude zero, but even their largest supported improvements
remain below 0.02 nats. This is a small directional effect, not a practical
selective-access win. Stay differences are very small; the transformer purpose
cell has a positive difference at 32 observations. A small safe edit can also
mean that the decoder barely uses the edited direction.

Probe-label accuracy is the fraction of balanced maker labels classified
correctly; chance is 50%. This table separates decodability from intervention
success, with architecture and observation prefix as row labels.

| Reader | Observations | Purpose accuracy | Belief accuracy |
|---|---:|---:|---:|
| Transformer | 8 | 71.48% | 69.92% |
| Transformer | 32 | 78.91% | 78.91% |
| Recurrent | 8 | 66.80% | 71.09% |
| Recurrent | 32 | 86.52% | 87.30% |

The probes fit at 32 observations only. The eight-observation result is transfer
to a shorter prefix. At 32 observations, per-fit purpose accuracies range from
76.56% to 80.47% for the transformer and 83.59% to 89.06% for the recurrent
reader; belief ranges are 77.34%–80.47% and 85.94%–89.06%. None of those accuracy
figures proves the coordinate has a selective causal role.

The next table gives ridge-probe loss minus each named control in change cases
at 32 observations. Negative values favor the ridge probe. Brackets have the
same paired-lineage interpretation as above; random means one frozen equal-rank
direction per model/factor, not a distribution over arbitrary directions.

| Reader | Factor | Random direction | Shuffled labels | Wrong factor | Class-mean direction |
|---|---|---:|---:|---:|---:|
| Transformer | Purpose | -0.00212 [-0.00333, -0.00072] | -0.00285 [-0.00370, -0.00203] | -0.00190 [-0.00276, -0.00107] | +0.04852 [+0.03935, +0.05862] |
| Transformer | Belief | -0.00108 [-0.00209, +0.00010] | -0.00598 [-0.00645, -0.00546] | -0.00591 [-0.00662, -0.00505] | +0.04269 [+0.03242, +0.05279] |
| Recurrent | Purpose | -0.00168 [-0.00256, -0.00104] | -0.00513 [-0.00608, -0.00424] | -0.00386 [-0.00473, -0.00303] | +0.06852 [+0.05675, +0.08116] |
| Recurrent | Belief | +0.00242 [+0.00157, +0.00336] | -0.00732 [-0.00775, -0.00694] | -0.00702 [-0.00762, -0.00649] | +0.07155 [+0.06288, +0.08190] |

The learned probe beats shuffled and wrong-factor controls in these change
cells, but the recurrent belief probe loses to the random direction by 0.00242
nats. The transformer belief comparison with random includes zero. These are
conditional control contrasts; they neither establish a general random-direction
baseline nor support a pooled claim of selective access.

The training class-mean direction is also rank one and uses exactly the same
auxiliary labels. This table reports its loss minus unchanged-recipient loss,
with the same row definitions and interval procedure.

| Reader | Observations | Factor | Change loss difference | Stay loss difference |
|---|---:|---|---:|---:|
| Transformer | 8 | Purpose | -0.04506 [-0.05993, -0.03020] | +0.00236 [+0.00027, +0.00499] |
| Transformer | 8 | Belief | -0.04981 [-0.06301, -0.03473] | +0.11035 [+0.09316, +0.12863] |
| Transformer | 32 | Purpose | -0.05124 [-0.06207, -0.04141] | +0.00644 [+0.00490, +0.00790] |
| Transformer | 32 | Belief | -0.04872 [-0.05910, -0.03829] | +0.21567 [+0.19423, +0.23597] |
| Recurrent | 8 | Purpose | -0.04218 [-0.05180, -0.02957] | +0.00209 [+0.00110, +0.00323] |
| Recurrent | 8 | Belief | -0.05219 [-0.06313, -0.04029] | +0.11622 [+0.10003, +0.13170] |
| Recurrent | 32 | Purpose | -0.07365 [-0.08710, -0.06098] | +0.00829 [+0.00716, +0.00937] |
| Recurrent | 32 | Belief | -0.07883 [-0.08963, -0.06989] | +0.20380 [+0.18831, +0.21638] |

At 32 observations the class-mean purpose direction improves change loss by
0.05124 and 0.07365 nats for transformer and recurrent readers, with stay costs
0.00644 and 0.00829. This is a limited purpose-only candidate under the stated
practical margin, not evidence that the ridge probe won. The class-mean belief
direction improves change cases but raises stay loss by 0.21567 and 0.20380
nats. Its belief effect is nonselective. A candidate in one factor does not
authorize joint purpose/belief composition, and no untouched confirmation or
historical local-process claim follows.

The following ranges show the four fixed combinations of two training draws and
two original model seeds for ridge-probe differences at 32 observations. These
are observed per-fit ranges, not confidence intervals or independent worlds.

| Reader | Factor | Change difference range across fits | Stay difference range across fits |
|---|---|---:|---:|
| Transformer | Purpose | [-0.00503, -0.00131] | [-0.00002, +0.00011] |
| Transformer | Belief | [-0.01250, -0.00288] | [-0.00041, -0.00012] |
| Recurrent | Purpose | [-0.00796, -0.00277] | [-0.00014, +0.00004] |
| Recurrent | Belief | [-0.00943, -0.00515] | [-0.00042, -0.00014] |

Every estimate averages makers and contexts, then draws and seeds within each
of eight paired development coefficient lineages before 10,000 resamples with
seed 190501. Full per-fit values, all seven arms, total-variation distances and
all shorter-prefix control contrasts remain in the numerical review. Total
variation is half the sum of absolute probability differences; zero means
identical forecasts and one means disjoint forecasts. Two training draws and
two seeds do not estimate universal training uncertainty. No multiplicity-
adjusted confirmatory claim is made.

The sixteen probes each use 512 training states from 32 original lineages and
all sixteen makers. The capacity is one direction in the unchanged 32- or
38-coordinate final hidden state. Closed-form centered ridge uses the frozen
penalty, with no development tuning. Signed purpose and belief labels are
privileged auxiliary supervision, supplied equally to both architectures. Normal
reader weights and prediction heads remain unchanged. This is not optimized
nonlinear alignment or DAS, and ordinary reader training did not receive these
factor labels. The two shared sequence-model settings remain consumed.

Independent augmented least squares reproduces the producer's normal-equation
fits to 3.83e-13. Independently formed coordinate edits and softmax probabilities
agree within 1.70e-14; all 57,344 finite score rows agree within 5.78e-15. All
256 probe-accuracy rows agree exactly. Within-lineage permutations, mean and
random directions, donor/hybrid mappings and full/no-op replacements verify.
All 982 deterministic files match original, adjacent and extracted-source full
replays; execution timings are hash-checked separately. The native world and
saved hidden states are bound to the prior fully verified fixture, not claimed
as another independent network implementation.

The first reviewer did not explicitly reject nonfinite oracle-entropy checks
at zero target probability. Its script and receipt are retained, and the complete
review was repeated with the correct zero-mass convention and finite assertions.
The corrected second review is controlling. No producer, score or frozen source
was changed. Review attempts are included in the resource accounting.

Blind reader inputs remain anonymous observed codes and novelty flags from the
fixture packet. Factor labels and fitted directions are auxiliary-training
evidence, while pair mappings, laws, hidden states, hybrids and scores are
evaluator/reproduction evidence. Interventions use known-maker targets;
predictive capability used history-conditional targets. Their different privilege
cannot be erased by a good score. Purpose and belief are persistent factors,
not relabelled historical local goals. Test and confirmation lineages are unused.

**Warrant:** practically small ridge-probe effects, a limited class-mean purpose
candidate and damaging class-mean belief access; exploratory constructed method,
miniature — architecture untested. No joint selective mediation, human intent,
or preceding-process correspondence.

**Pursuit:** continue the independent [probabilistic cue-correction experiment](CUE_CORRECTION_PROTOCOL.md).
F1 learned local rollouts and one bounded alternative transformer site remain
prepared, with no unimplemented handler represented as runnable. The primary
local comparison, confirmation and research week remain open.

[Frozen protocol](INTERCHANGE_ALIGNMENT_PROTOCOL.md),
[controlling numerical review](../../../results/v19/G19-C1-alignment-1/INDEPENDENT_REVIEW_2.json),
[role-separated exports](../../../results/v19/G19-C1-alignment-1/EXPORT_MANIFEST.json).

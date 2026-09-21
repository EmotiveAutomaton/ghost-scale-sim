# Uncertain reports, correction and repeated evidence

Can a process reader recover from a wrong report without counting its repetition twice? Source-aware correction restores the appropriate posterior; treating one wrong report as two independent reports adds 1.92615 nats of joint process loss under the neutral prior. All 24,576 scores and both full replays verify the probability-based result. Best-process localization is sensitive to numerical ties. This is a constructed-method result with supplied laws and report reliability, not evidence of human intent or new understanding from rearranging old evidence.

Logarithmic loss measures error in the probability assigned to the actual complete
local-goal and operation sequences; lower is better, in nats. The table gives each
report condition's joint loss minus the same case's base loss. Negative values
improve prediction. Columns separate the uniform maker prior from the stipulated
self-like prior, which weights purpose zero three times as heavily. All report
likelihoods in this table use reliability 0.9. Brackets are 95% paired
coefficient-lineage bootstrap intervals, conditional on this sampled case roster.

| Report handling | Neutral prior | Self-like prior |
|---|---:|---:|
| Truthful | -0.51482 [-0.53994, -0.49001] | -0.51892 [-0.54288, -0.49485] |
| Wrong | +1.40560 [+1.36620, +1.44673] | +1.41169 [+1.37234, +1.45175] |
| Repeated wrong, same source | +1.40560 [+1.36620, +1.44673] | +1.41169 [+1.37234, +1.45175] |
| Repeated wrong, treated as independent | +3.33175 [+3.25117, +3.42039] | +3.33965 [+3.25848, +3.42583] |
| Wrong, then retracted | +0.00000 [+0.00000, +0.00000] | +0.00000 [+0.00000, +0.00000] |
| Wrong, then replaced by truthful | -0.51482 [-0.53994, -0.49001] | -0.51892 [-0.54288, -0.49485] |
| Opposing independent reports | +0.00000 [-0.00000, +0.00000] | +0.00000 [-0.00000, +0.00000] |

The neutral base loss is 3.35574 nats; the self-like base loss is 3.40998.
Relative to one wrong report, treating its duplicate as independent adds
1.92615 [1.88246, 1.97557] nats under the neutral prior and
1.92796 [1.88415, 1.97724] under the self-like prior. These contrasts exceed the
descriptive 0.02-nat margin. Correcting the source improves loss relative to the
wrong report by 1.92043 and 1.93061 nats, respectively.

Replacement discards the old source likelihood and substitutes the corrected
one. Retraction removes it. The two opposing independent reports cancel because
the relation is binary and their stipulated reliabilities are equal. These are
algebraic identities of this update rule. They do not show that real sources
have known reliability or that the reader can infer which report is true.

The next table separates logarithmic losses for complete local-goal sequences
and operation sequences. Each cell is report minus base, with paired lineage
intervals. Neither marginal score establishes unique joint historical recovery.

| Prior | Report | Local-goal sequence loss difference | Operation sequence loss difference |
|---|---|---:|---:|
| Neutral | Truthful | -0.51482 [-0.53994, -0.49001] | -0.33963 [-0.37772, -0.30230] |
| Neutral | Wrong | +1.40560 [+1.36620, +1.44673] | +0.90999 [+0.81120, +1.00313] |
| Neutral | Repeated wrong, independent | +3.33175 [+3.25117, +3.42039] | +2.15313 [+1.91714, +2.37248] |
| Self-like | Truthful | -0.51892 [-0.54288, -0.49485] | -0.33591 [-0.37599, -0.29614] |
| Self-like | Wrong | +1.41169 [+1.37234, +1.45175] | +0.91047 [+0.80804, +1.00732] |
| Self-like | Repeated wrong, independent | +3.33965 [+3.25848, +3.42583] | +2.15446 [+1.91403, +2.37634] |

All twelve arms at reliability 0.5 reproduce base probabilities. Literal
rearrangement, equal-length reread, irrelevant and unknown reports add no accepted
proposition and reproduce base at either reliability. Every supported case retains
the true trajectory in its joint support, with zero incompatible posterior mass
and no false single-account assertion. The outside-alphabet fixture is unknown
in all 192 checks; it does not test a plausible missing cause inside the alphabet.

The relation itself concerns whether the first and last selected local goals are
equal. Truthful and wrong report arms were selected with evaluator truth. They
are additional teacher information, not implications of the final artifact or a
reorganization of identical known propositions. The exact complete hypothesis
family is also supplied. Reliability 0.9 is assumed even in the deliberately all-
wrong stress test; it is not an empirically calibrated source model.

The following table separates the two original sampling seeds. Values are report
minus base joint loss, averaged over all eight lineages. Seeds are repeated case
draws, not independent fitted models or independent worlds.

| Prior | Report | Difference at sampling seed one | Difference at sampling seed two |
|---|---:|---:|
| Neutral | Truthful | -0.51094 | -0.51871 |
| Neutral | Wrong | +1.39537 | +1.41583 |
| Neutral | Repeated wrong, independent | +3.30896 | +3.35454 |
| Self-like | Truthful | -0.51361 | -0.52423 |
| Self-like | Wrong | +1.40072 | +1.42265 |
| Self-like | Repeated wrong, independent | +3.31621 | +3.36308 |

The population is all 512 previously admitted E1 cases, eight coefficient
lineages times two original sampling seeds times 32 cases, with repetitions and
original sampling weights retained. Each case crosses two priors, two reliability
values and twelve report arms: 24,576 score rows. Within each contrast, cases and
seeds are averaged inside lineage before 10,000 resamples with seed 190501.
There are no new trajectories, model fits, training draws or confirmation worlds.

Independent reconstruction regroups all 110,592 native hypotheses by visible
artifact, initial state and stated request, verifies the actual-path mappings,
then computes the report odds with separate closed-form exponents. Every saved
joint posterior agrees within 2.23e-16 and all probability-based scores within
5.33e-15. All 17 deterministic files match both the adjacent replay and a complete
replay from extracted source. Execution measurements are hash-checked separately.
Native executor probabilities rely on the previously verified D1 source; this is
not another independent implementation of that architecture.

The first review stopped on modal localization disagreement. A diagnostic review
isolated roundoff-sensitive near ties; the complete final review verifies the
recorded choices against saved posterior arrays and qualifies that measure.
Across 184 score rows, 316 localized-goal or localized-operation accuracy
entries change under independent arithmetic, despite posterior agreement to
machine precision. The recorded values are preserved, but no localization
advantage is accepted. Full-joint loss, support and report-handling conclusions
are unaffected. Both failed review attempts and the final review are retained
and charged. No producer, score definition or frozen source was changed.

**Warrant:** a verified exploratory constructed-method result about provenance
and correction under supplied likelihoods; miniature — architecture untested.
No calibrated human-source inference, invented cause discovery, learned
reorganization or human-intent claim. Modal localization remains qualified.

**Pursuit:** the single [alternative transformer site](ALTERNATE_SITE_PROTOCOL.md)
is prepared for admission. Learned native rollouts and an in-alphabet missing-cause
fixture remain independent alternatives requiring their own frozen manifests.
The broader primary and research week remain open.

[Frozen cue protocol](CUE_CORRECTION_PROTOCOL.md),
[independent reconstruction](../../../results/v19/G19-D1-cue-correction-1/INDEPENDENT_REVIEW.json),
[role-separated evidence](../../../results/v19/G19-D1-cue-correction-1/EXPORT_MANIFEST.json).

# Learned forward prediction — 21 September 2026

Does composing learned transitions improve endpoint prediction beyond a direct table trained on the same paths? Sixteen learned rollouts lose to direct prediction at all four training budgets; at 2,048 paths the loss penalty is 0.06868 nats. Exact propagation removes a further 0.09813 nats of rollout error, but its 0.02945-nat apparent advantage over direct prediction has a paired lineage interval crossing zero. All 552,960 native paths, 327,680 scores and both full replays verify this exploratory constructed-method result. Supplied mechanics metadata, unequal model capacities and count-dependent smoothing prevent a general simulation or process-reconstruction claim.

Logarithmic loss measures endpoint probability error in nats; lower is better.
Rows distinguish direct lookup, sampled computation and exact propagation.
Columns give the number of complete labelled training paths. Native query mass is
weighted within each development lineage; eight lineages and two training draws
then receive equal weight. These are forward predictions of proposed operations.

| Predictor | 32 paths | 128 paths | 512 paths | 2,048 paths |
|---|---:|---:|---:|---:|
| Direct probabilities | 2.04043 | 1.93020 | 1.58613 | 0.96174 |
| 16 sampled direct endpoints | 2.34327 | 2.19348 | 1.78973 | 1.06828 |
| 1 forward path | 2.43561 | 2.43719 | 2.22668 | 1.78061 |
| 4 forward paths | 2.67884 | 2.64627 | 2.20555 | 1.33659 |
| 16 forward paths | 2.40417 | 2.31499 | 1.83043 | 1.03042 |
| Exact learned propagation | 2.06592 | 1.98936 | 1.62432 | 0.93229 |
| 16 disturbed-law paths | 2.35307 | 2.36793 | 2.37000 | 2.53294 |
| Exact known mechanics | 0.00000 | 0.00000 | 0.00000 | 0.00000 |

The 95% paired coefficient-lineage interval for the largest-budget sampled-forward
penalty is [0.03816, 0.09800] nats. Exact learned propagation minus direct prediction
is -0.02945 [-0.06198, 0.00211] nats: an inconclusive advantage, not a practical
win. The two training draws give -0.02655 and -0.03234 nats separately. Repeated
queries and draws remain inside their lineage; these intervals condition on two
fixed draws and do not estimate general training-population uncertainty.

Sixteen rollouts improve on one by 0.75019 nats at 2,048 training paths, yet remain
0.09813 nats [0.09583, 0.10033] worse than exact propagation of the identical
learned transition table. Sampling direct endpoints likewise adds 0.10654 nats to
its exact table. Four paths are worse than one at the two smallest budgets, and
the disturbed-law control is slightly better at 32 paths before becoming much
worse at the largest budget. Extra sampling is not uniformly beneficial. The
fixed total-one endpoint pseudocount changes its relative weight with sample count;
the observed changes combine computational randomness with changed smoothing.

Both models receive the same 2,048 nested complete labelled paths per draw,
sampled from 32 original training lineages. Transition prediction uses intermediate
states and endpoint prediction uses endpoint sufficient statistics. Every query
supplies initial artifact, proposed operations, skill and belief mechanics bits;
it does not supply targets or actual intermediate states. The forward table has
12,288 count parameters and supplied undo-buffer evolution. The direct table has
eight counts per visited complete query. Their factorization, parameter counts,
statistical regularization and uses of available supervision differ. Sixteen direct
samples match sample count, not total arithmetic cost. Component timing combines
all forecast arms with scoring, so it cannot establish per-arm runtime savings.

Native choices select the operation-sequence distribution. All 640 distinct queries
have a unique true endpoint under supplied mechanics; the exact-law result is
therefore a deterministic ceiling. Recovering that endpoint is not identifying
the goal or historical process that selected those operations. No inverse process,
retrieval, changed-law transfer or capacity-matched effort comparison is completed
by this pilot. The broader F family remains open.

Independent verification reconstructs all 552,960 trajectories and native choice
probabilities, both deterministic training selections, all count tables, undo
bookkeeping, and the exact sum over all 512 intermediate artifact paths per query.
It rebuilds shared-prefix Monte Carlo forecasts, direct sampling, the specified
disturbance, all 327,680 scores, and naturally weighted cells. Maximum forecast
discrepancy is 1.45e-15; score discrepancy is 3.11e-15. All 63 deterministic files
match both full replays. Source, environment, plan and separate timing hashes verify.

Reader export contains only anonymous queries and declared mechanics metadata.
Auxiliary training supervision is a separate archive. Native identities, path
weights, evaluation endpoints, count models, forecasts and scores have scientific/
evaluator roles. No evaluator truth enters the reader query archive.

**Warrant:** exploratory constructed-method estimate, miniature — architecture
untested. Sampled rollout advantage is reversed; the exact learned advantage is
inconclusive. The chosen world, regularizer and two training draws limit transfer.
No confirmation or human-intent claim follows.

**Pursuit:** implement the prepared joint goal-operation readout to test historical
correspondence directly. Crossed-rule composition and a controlled support/compute
diagnostic remain independent alternatives. More rollout seeds alone are not the
next explanatory challenge.

[Frozen protocol](ROLLOUT_PROTOCOL.md),
[independent values](../../../results/v19/G19-F1-rollout-1/INDEPENDENT_REVIEW.json),
[role-separated exports](../../../results/v19/G19-F1-rollout-1/EXPORT_MANIFEST.json).

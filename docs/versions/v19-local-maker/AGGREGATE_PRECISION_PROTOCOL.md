# Finite precision in an exact report state: prepared alternative

Does storing an otherwise sufficient aggregate report state at lower precision
destroy useful retrospective updates, and are rare reports especially sensitive?
Use the accepted single-report uniform-source state, independently of whether
the source-prior basis extension is admitted. Freeze float64, float32 and float16
storage before native outcomes. Quantize the group and joint masses, then perform
queries in float64. Copied-endpoint counts stay integer. Record the storage used,
normalization drift, underflow and every changed support flag before any repair.

Compare direct cast with one declared mass-preserving reconstruction: normalize
the cast group probabilities, and normalize each nonempty cast joint row to its
group mass. Preserve exact zeros. A positive group with an all-zero cast joint
row is an explicit reconstruction failure; do not insert evaluator mass or a
probability floor. Unsupported forecasts and defined forecast errors are separate
outcomes. The exact float64 state is the same-evidence supplied-law reference.

Use all 28,672 saved posteriors, eight laws, both evidence conditions, two draws,
five copying probabilities, all endpoint reports and every future coordinate.
Report probability-weighted errors and fixed report-probability strata
[0, 0.000001), [0.000001, 0.001), [0.001, 1], with exact-zero support separate.
These are descriptive conditioning strata, not fitted cutoffs or a practical
importance threshold. No new observation, fit or protected lineage.

Prepared, unimplemented and unadmitted. Require exact self/constant-law/certain-copy
controls; explicit rounding and underflow fixtures; scalar Bayes and proper-loss
checks; source/endpoint permutations; support-mask and reconstruction failures;
and complete-support timing. Freeze source and plans before native outcomes.
The prospective inclusive cap is 3,600 CPU seconds, including checks, failures,
three executions, independent reconstruction and original-row regroup within
the existing ceilings. No learned access, universal optimality, minimum-storage,
historical correspondence or human-intent claim follows.

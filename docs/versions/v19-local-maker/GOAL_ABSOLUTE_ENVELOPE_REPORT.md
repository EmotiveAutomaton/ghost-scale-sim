# Absolute residual envelope: independently verified

We tested whether averaging within probability bins hides frame-level absolute errors. At 2,048 labels, even twenty bins conceal 7.7–13.3% of the predictive bank’s absolute probability error across positions under native weighting. Independent reconstruction, all 5,508 paired estimates and both complete replays verify this constructed-method diagnostic. Historical goal correspondence and human intent remain unestablished.

For each witnessed frame and local goal, the absolute residual is the difference
between the forecast probability and native reference probability, without its
sign. Summing these weighted residuals before bin averaging gives an upper bound
on the absolute discrepancy reported after bin averaging. The difference measures
error hidden by cancellation. Nested five-, ten- and twenty-bin partitions reduce
that gap; they do not alter the forecasts or improve prediction.

All sixteen development laws, two training draws, five fit seeds, four label
budgets, four readouts, two forecasts, three positions and three goals are retained.
Native weighting uses each law's frame distribution; equal-frame weighting gives
each of 704 public frames equal mass. No new fit, sample or protected lineage is used.

The table describes the predictive bank's support-restricted forecast at 2,048
labels. Rows name the population and process position. Errors sum the three goal
components in absolute probability units. The hidden share divides the twenty-bin
gap by the unbinned error using population means; it has no separately estimated
confidence interval.

| Population | Position | Unbinned absolute error | Five-bin discrepancy | Ten-bin discrepancy | Twenty-bin discrepancy | Hidden share at twenty bins |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| native | 1 | 0.06682360 | 0.05381948 | 0.05568249 | 0.05796364 | 13.26% |
| native | 2 | 0.08147170 | 0.06542712 | 0.07030687 | 0.07447863 | 8.58% |
| native | 3 | 0.04143294 | 0.03706014 | 0.03739348 | 0.03824571 | 7.69% |
| equal-frame | 1 | 0.06902082 | 0.05827819 | 0.05926358 | 0.06118301 | 11.36% |
| equal-frame | 2 | 0.07682007 | 0.06648025 | 0.06882897 | 0.07156619 | 6.84% |
| equal-frame | 3 | 0.03757791 | 0.03387732 | 0.03417709 | 0.03450152 | 8.19% |

Every goal-specific mean, envelope interval, readout contrast and label-budget
curve is retained, including reversals. Intervals condition on retained fits:
10,000 resamples of sixteen paired laws keep both draws and all five seeds inside
each law. Draw and seed variation remain separate. No practical-impact threshold
or architecture winner was specified.

All 1,296 deterministic outputs match original, adjacent and extracted-source
executions. Source, plan, environment, input, output and timing bindings verify.
The independent checker reconstructs 92,160 score rows, 320 group bundles,
320 raw residual bundles, 225,280 frame forecasts and 5,120 parent identities.
Maximum score/array discrepancy is 1.01e-14 and parent discrepancy 8.89e-16.
All 1,581,952 empty-bin cells have matching undefined means. Native self is exact
zero. Negative roundoff is retained; no gap violates the frozen -1e-12 tolerance.

Independent scalar absolute and signed-bin sums rebuild all raw contributions,
bin weights and native target/forecast sums. A third direct-index regroup checks
576 means, 1,728 envelope estimates, 3,024 budget contrasts and 756 normalized
log-budget areas within 9.94e-15. Scalar base-three sums check native matrix
projection while preserving executed binary64 bin boundaries. This does not
independently audit the numerical library.

The optimized checker passed 81 isolated controls, including opposing and constant
residuals, unequal/zero weights, native self, permutations, exact and adjacent
boundaries, probability one, empty means, complete synthetic execution, missing
or duplicate rosters, and deliberate corruption. The first failed singleton
corruption fixture and earlier inclusive-budget blocker remain preserved. Removing
scalar array allocations resolved that cost cause without changing scores,
population, partitions, tolerance, independence, safety margin or the 1,800-second
inclusive card ceiling. No completed scientific execution was restarted.

Reader packets remain separate from inherited scientific inputs, evaluator truth,
forecasts, residuals and scores. This exploratory constructed-method diagnostic is
miniature — architecture untested beyond the declared roster. It does not establish
historical goal correspondence or human intent. Exact forecast collisions and
exhaustive retrospective updating remain independent prepared questions.

[Review protocol](GOAL_ABSOLUTE_ENVELOPE_REVIEW_PROTOCOL.md).
[Numerical acceptance](../../../results/v19/G19-D-goal-absolute-envelope-1/FINAL_REVIEW.json).
[Complete regroup](../../../results/v19/G19-D-goal-absolute-envelope-1/INDEPENDENT_REGROUP.json).

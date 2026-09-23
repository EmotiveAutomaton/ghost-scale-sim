# Squared residual decomposition: independently verified

We tested whether small average errors inside probability bins conceal larger frame-level errors. At 2,048 labels, variation within ten bins accounts for 24.6–46.6% of the predictive bank’s squared probability error across positions under native weighting. Independent reconstruction, all 2,772 paired estimates and both complete replays verify this constructed-method diagnostic. Historical goal correspondence and human intent remain unestablished.

Ten probability bins are fixed at integer tenths, with probability one in the
last bin. For each frame and constructed goal, the residual is forecast probability
minus the native reference probability. The weighted squared error equals the
weighted squared mean residual within each bin plus residual variance within bins.
The latter term disappears if one reports only bin-average residuals. This is
an identity applied to the frozen forecasts, not a new predictive improvement.

All sixteen development laws, two training draws, five fit seeds, four label
budgets, four readouts, two forecasts, three positions and three goals remain
paired. Native weighting uses each law's frame distribution; equal-frame weighting
gives each of the 704 public frames equal mass. No new fit, sample, protected
lineage or model setting is used.

This table reports the predictive bank's support-restricted forecast at 2,048
labels. Each row names the weighting population and process position. The first
three columns sum over the three constructed goals and use squared probability
units. The final column is within-bin variance divided by total squared error,
computed from the displayed population means; it has no separately estimated
confidence interval.

| Population | Position | Squared bin-average residual | Within-bin variance | Total squared error | Within-bin share |
| --- | ---: | ---: | ---: | ---: | ---: |
| native | 1 | 0.00560971 | 0.00490039 | 0.01051009 | 46.63% |
| native | 2 | 0.00859221 | 0.00525917 | 0.01385138 | 37.97% |
| native | 3 | 0.00486701 | 0.00158709 | 0.00645411 | 24.59% |
| equal-frame | 1 | 0.00623365 | 0.00820904 | 0.01444269 | 56.84% |
| equal-frame | 2 | 0.00850858 | 0.00651520 | 0.01502378 | 43.37% |
| equal-frame | 3 | 0.00429584 | 0.00144254 | 0.00573839 | 25.14% |


The reported shares are descriptive and conditional on retained fits. Every
goal-specific mean, component interval, readout contrast and label-budget curve
is retained in the complete regroup receipt, including reversals. The 95% intervals
resample sixteen paired laws with both draws and five fit seeds within each law.
Training-draw and fit-seed variation are retained separately, not treated as new
independent laws. No practical threshold or architecture winner was specified.

All 974 deterministic outputs match original, adjacent and extracted-source
executions. Source, plan, environment, input, output and timing bindings verify.
Independent reconstruction covers 92,160 metric rows, 320 moment bundles,
225,280 frame forecasts, 704 packets and 5,120 parent loss/accuracy identities.
Maximum array discrepancy is 7.33e-15, row discrepancy 1.12e-15 and parent
discrepancy 8.89e-16. The 387,264 empty learned bin cells have undefined
conditional moments; the empty-bin patterns match exactly. Native self-forecasts
have zero error. Tiny negative producer variance from roundoff remains preserved;
none breaches the frozen -1e-12 tolerance.

The independent checker uses scalar memberships and accurate scalar sums.
Variance is recomputed from centered residuals instead of subtracting moments.
A third direct-index regroup verifies 576 means, 1,152 within-readout component
estimates, 1,296 paired budget contrasts and 324 normalized log-budget areas within
8.89e-16. Seed 191020 fixes 10,000 paired-law resamples. Forty-seven isolated
checker controls passed, including opposing/constant residuals, unequal weights,
ordered-pair variance identities, boundaries, zero weights, empty moments,
permutations, complete synthetic execution, roster and corruption failures.
Scalar base-three marginal sums check the recreated native matrix projection,
which preserves the executed binary64 bin boundaries. This does not independently
audit the numerical library itself.

Rebuilt arrays, native targets, inherited inputs, forecasts and scores remain
scientific/evaluator records, separate from reader packets. This is an exploratory
constructed-method diagnostic, miniature — architecture untested beyond the
declared roster. It does not establish historical goal correspondence or human
intent. The unbinned absolute-error envelope asks a distinct question about error
hidden even by finer partitions; exhaustive retrospective updating is independent.

[Review protocol](GOAL_SQUARED_DECOMPOSITION_REVIEW_PROTOCOL.md).
[Numerical acceptance](../../../results/v19/G19-D-goal-squared-decomposition-1/FINAL_REVIEW.json).
[Complete regroup](../../../results/v19/G19-D-goal-squared-decomposition-1/INDEPENDENT_REGROUP.json).

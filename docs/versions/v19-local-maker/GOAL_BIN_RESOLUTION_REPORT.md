# Fixed probability-bin resolution: independently verified

We tested whether finer probability bins expose hidden local-goal forecast errors. At 2,048 labels, moving from ten to twenty bins reveals up to 0.31400 additional percentage points of absolute discrepancy for the predictive bank under native weighting. Independent reconstruction, all 4,392 paired estimates and both complete replays verify this constructed-method diagnostic. No practical-impact threshold was specified; historical goal correspondence and human intent remain unestablished.

The frozen five-, ten- and twenty-bin partitions use integer-ratio binary64 edges,
with probability one assigned to the last bin. All sixteen development laws,
two training draws, five fit seeds, four budgets, four readouts, two forecasts,
three positions and three goals remain paired. Native weighting preserves each
law's frame distribution; equal-frame weighting gives each of the 704 public
frames equal mass. No new fit, sample, protected lineage or model setting is used.

The table describes the predictive bank's support-restricted forecast at 2,048
labels. Each row names a weighting population, position and constructed goal.
Discrepancy sums the absolute weighted forecast-minus-native probability inside
each bin. The last column gives the ten-to-twenty-bin increase and its conditional
95% paired-law interval. All values are percentage points. Meaning, dependency
and presentation are the controller's three goal types.

| Population | Position | Goal | Five bins | Ten bins | Twenty bins | Increase [interval] |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| native | 1 | meaning | 1.68120 | 1.68120 | 1.74932 | 0.06813 [0.05540, 0.08230] |
| native | 1 | dependency | 1.87052 | 2.04831 | 2.17095 | 0.12263 [0.10933, 0.13535] |
| native | 1 | presentation | 1.83023 | 1.83874 | 1.87609 | 0.03735 [0.02916, 0.04580] |
| native | 2 | meaning | 2.16662 | 2.34755 | 2.66155 | 0.31400 [0.27060, 0.35864] |
| native | 2 | dependency | 2.16659 | 2.33764 | 2.37721 | 0.03957 [0.03444, 0.04447] |
| native | 2 | presentation | 2.20950 | 2.34550 | 2.40910 | 0.06361 [0.05372, 0.07446] |
| native | 3 | meaning | 1.84398 | 1.86089 | 1.90618 | 0.04530 [0.02094, 0.07198] |
| native | 3 | dependency | 0.01538 | 0.01538 | 0.01538 | 0.00000 [0.00000, 0.00000] |
| native | 3 | presentation | 1.84665 | 1.86308 | 1.90301 | 0.03993 [0.01842, 0.06379] |
| equal-frame | 1 | meaning | 1.93632 | 1.93632 | 1.93632 | 0.00000 [0.00000, 0.00000] |
| equal-frame | 1 | dependency | 2.07727 | 2.15677 | 2.29156 | 0.13479 [0.12897, 0.14066] |
| equal-frame | 1 | presentation | 1.81424 | 1.83328 | 1.89043 | 0.05715 [0.05328, 0.06081] |
| equal-frame | 2 | meaning | 2.25750 | 2.27138 | 2.47346 | 0.20208 [0.16007, 0.24543] |
| equal-frame | 2 | dependency | 2.27638 | 2.36347 | 2.39741 | 0.03395 [0.02866, 0.03920] |
| equal-frame | 2 | presentation | 2.11415 | 2.24805 | 2.28575 | 0.03770 [0.03395, 0.04180] |
| equal-frame | 3 | meaning | 1.67619 | 1.69369 | 1.71069 | 0.01700 [0.00511, 0.03256] |
| equal-frame | 3 | dependency | 0.02806 | 0.02806 | 0.02806 | 0.00000 [0.00000, 0.00000] |
| equal-frame | 3 | presentation | 1.68348 | 1.69595 | 1.71140 | 0.01545 [0.00572, 0.02795] |

The largest displayed native-weighted increase is at the second position's meaning
goal. The third position's dependency goal has zero increase. Near-roundoff
increments are not scientific effects. The change exposes error cancellation
within coarser bins; it does not improve the forecasts. Unbinned squared error
is unchanged. No practical-impact threshold or architecture winner was specified.
All budgets and readout comparisons, including reversals, remain in the complete
regroup receipt. Intervals condition on the retained fits and resample sixteen
paired laws with both draws and five seeds inside each law. Draw and fit-seed
variation is retained separately, not counted as independent laws.

All 974 deterministic outputs match original, adjacent and extracted-source
executions. Source, plan, environment, input, output and timing bindings verify.
Independent reconstruction covers 92,160 metric rows, 320 sufficient-sum bundles,
225,280 frame forecasts, 704 packets and 5,120 parent loss/accuracy identities.
Maximum array discrepancy is 7.33e-15; row discrepancy is 4.33e-15 and parent
discrepancy 8.89e-16. There are 1,581,952 empty learned bin cells; means are
undefined exactly at zero mass. Native self-forecasts have zero discrepancy.
No refinement increment breaches the frozen negative-roundoff tolerance.

The independent checker uses ordered scalar memberships and accurate scalar
sums, then checks nesting, coarsening and mass conservation. A third direct-index
regroup verifies 576 means, 1,152 within-readout refinement estimates, 2,592 paired
budget contrasts and 648 normalized log-budget areas within 4.20e-15. Seed 191019
fixes 10,000 paired-law resamples. Seventy-one isolated checker controls passed,
including opposing and identical errors, all exact and adjacent boundaries,
probability one, native self, empty means, faulty assignments, synthetic execution,
roster failures and corruption cases. Scalar base-three sums check marginals;
recreated native matrix projection preserves executed binary64 bin boundaries.
This does not independently audit the numerical library itself.

Rebuilt arrays, native targets, inherited inputs, forecasts and scores are
scientific/evaluator records, separate from reader packets. This is an exploratory
constructed-method diagnostic, miniature — architecture untested beyond the
declared roster. Historical goal correspondence and human intent remain
unestablished. Squared-error decomposition asks a distinct question about variation
inside bins; exhaustive retrospective updating remains independent.

[Review protocol](GOAL_BIN_RESOLUTION_REVIEW_PROTOCOL.md).
[Numerical acceptance](../../../results/v19/G19-D-goal-bin-resolution-1/FINAL_REVIEW.json).
[Complete regroup](../../../results/v19/G19-D-goal-bin-resolution-1/INDEPENDENT_REGROUP.json).

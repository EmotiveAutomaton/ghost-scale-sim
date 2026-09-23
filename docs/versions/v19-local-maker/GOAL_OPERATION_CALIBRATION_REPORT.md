# Operation-conditioned goal calibration: independently verified

We tested whether pooling witnessed operations hides opposing errors in local-goal probabilities. At 2,048 labels, pooling hides 0.00927–0.01149 percentage points of absolute calibration discrepancy at the first two positions; the third shows none beyond rounding. Independent reconstruction, all 2,196 paired estimates and both complete replays verify this constructed-method diagnostic. Historical goal correspondence and human intent remain unestablished.

The frozen diagnostic retains sixteen development laws, two training draws, five
fit seeds, four label budgets, four readouts, two forecasts, three positions and
three goal classes under native and equal-frame weighting. Every one of the six
legal operations is retained, including empty groups. No fitting or recalibration
occurs. Group membership uses only the witnessed operation at the scored position.
Native weights preserve the generator's frame distribution; equal-frame weights
give each of the 704 public evidence frames equal mass.

The table shows the predictive bank at 2,048 labels with the restricted forecast.
Each row identifies a population, position and local goal. Pooled discrepancy
takes the absolute forecast-minus-native probability mass in each fixed bin after
summing operations. Grouped discrepancy takes absolute values before summing
operations and bins. The difference measures cancellation. All numerical entries
are percentage points; intervals are 95% paired-law bootstrap intervals conditional
on the retained fits. Goals are meaning, dependency and presentation, respectively. This table does not average opposing goals together.

| Population | Position | Goal | Pooled discrepancy | Grouped discrepancy | Cancellation | Conditional interval |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| native | 1 | meaning | 1.68120 | 1.69047 | 0.00927 | [0.00910, 0.00945] |
| native | 1 | dependency | 2.04831 | 2.05895 | 0.01063 | [0.01047, 0.01081] |
| native | 1 | presentation | 1.83874 | 1.85015 | 0.01141 | [0.01130, 0.01151] |
| native | 2 | meaning | 2.34755 | 2.35697 | 0.00942 | [0.00924, 0.00960] |
| native | 2 | dependency | 2.33764 | 2.34912 | 0.01149 | [0.01137, 0.01159] |
| native | 2 | presentation | 2.34550 | 2.35646 | 0.01096 | [0.01083, 0.01110] |
| native | 3 | meaning | 1.86089 | 1.86089 | 0.00000 | [-0.00000, 0.00000] |
| native | 3 | dependency | 0.01538 | 0.01538 | 0.00000 | [-0.00000, 0.00000] |
| native | 3 | presentation | 1.86308 | 1.86308 | -0.00000 | [-0.00000, -0.00000] |
| equal-frame | 1 | meaning | 1.93632 | 1.95211 | 0.01580 | [0.01580, 0.01580] |
| equal-frame | 1 | dependency | 2.15677 | 2.17658 | 0.01981 | [0.01981, 0.01981] |
| equal-frame | 1 | presentation | 1.83328 | 1.85196 | 0.01868 | [0.01868, 0.01868] |
| equal-frame | 2 | meaning | 2.27138 | 2.28908 | 0.01770 | [0.01770, 0.01770] |
| equal-frame | 2 | dependency | 2.36347 | 2.37784 | 0.01437 | [0.01437, 0.01437] |
| equal-frame | 2 | presentation | 2.24805 | 2.26519 | 0.01714 | [0.01714, 0.01714] |
| equal-frame | 3 | meaning | 1.69369 | 1.69369 | -0.00000 | [-0.00000, 0.00000] |
| equal-frame | 3 | dependency | 0.02806 | 0.02806 | 0.00000 | [0.00000, 0.00000] |
| equal-frame | 3 | presentation | 1.69595 | 1.69595 | 0.00000 | [-0.00000, 0.00000] |

The cancellation is small relative to the approximately 1.68–2.36 percentage
points of discrepancy at the first two positions. No practical threshold was
frozen for this diagnostic, so the result is measured cancellation, not a claim
of material impact. Third-position differences are roundoff near 1e-19; a bootstrap
interval excluding zero at that scale is not a scientific effect. Under equal-frame
weighting the cancellation is effectively constant across laws, so extremely narrow
conditional intervals do not quantify training uncertainty. Draw-specific and
seed-specific means remain in the full record. In one first-two-position cell,
one training draw has no cancellation and the other supplies the entire mean.

All 653 deterministic outputs match original, adjacent and extracted-source runs.
Source, plan, environment, input, output and timing bindings verify. Independent
scalar reconstruction covers 92,160 metric rows, 320 array bundles, 225,280 frame
forecasts, 704 reader packets and 5,120 parent loss/accuracy identities. Maximum
array discrepancy is 4.67e-15; maximum score discrepancy is 1.78e-15. Original and
rebuilt rows are regrouped separately. A third direct-index bootstrap reproduces
576 means, 576 within-readout gaps, 1,296 budget contrasts and 324 log-budget areas
within 1.78e-15. Seed 191016 fixes 10,000 paired resamples of sixteen laws, with
both draws and five seeds kept inside each law. Fits remain conditional evidence.

There are 4,360,640 empty group/bin cells and 61,440 empty group cells across the
learned bundles. Their means are NaN exactly when denominators are zero. Native
self-forecasts have zero discrepancy. No triangle inequality fails beyond the
declared rounding tolerance. Forty-six isolated checker controls pass, including
scalar known cancellation, identical error, mass conservation, empty cells, exact
and adjacent bin edges, permutations, full execution and corruptions. The earlier
producer fixture's non-JSON boolean failure remains retained.

The independent checker validates saved binary64 marginals with scalar base-three
sums before binning. The unsaved native matrix projection is recreated and checked
against scalar sums to preserve exact bin membership; this is not a numerical
library audit. The original producer, its replays and all frozen sources remain
unchanged. This is exploratory miniature method evidence, architecture untested
beyond the admitted roster, with no human-intent or historical-goal conclusion.

Reader witnesses are separate from forecasts, evaluator targets, group arrays and
reconstruction archives. Both shared tiny settings are consumed; all confirmation
and test lineages remain untouched. Endpoint-artifact reliability and retrospective
updating remain independent alternatives. V19 is active.

[Review protocol](GOAL_OPERATION_CALIBRATION_REVIEW_PROTOCOL.md).
[Numerical acceptance](../../../results/v19/G19-D-goal-operation-calibration-1/FINAL_REVIEW.json).
[Complete regroup](../../../results/v19/G19-D-goal-operation-calibration-1/INDEPENDENT_REGROUP.json).
[Evidence roles](../../../results/v19/G19-D-goal-operation-calibration-1/EVIDENCE_ROLES.json).

# Endpoint-artifact calibration: independently verified

We tested whether pooling visible endpoint artifacts hides opposing errors in local-goal probabilities. At 2,048 labels, endpoint pooling hides up to 0.23066 percentage points of absolute calibration discrepancy in the predictive bank under native weighting. Independent reconstruction, all 2,196 paired estimates and both complete replays verify this constructed-method diagnostic. No practical-impact threshold was specified; historical goal correspondence and human intent remain unestablished.

The frozen diagnostic retains all eight binary three-unit endpoint artifacts,
sixteen development laws, two training draws, five fit seeds, four label budgets,
four readouts, two frozen forecasts, three positions and three local goals.
Grouping uses only the visible final artifact and never a hidden purpose, goal
or score. Native weighting preserves each generator's public-frame distribution;
equal-frame weighting gives each of the 704 visible frames equal mass. No new
fit, sample, tiny setting or protected lineage is used.

The table shows the predictive bank's restricted forecast at 2,048 labels. Each
row gives a population, position and goal. Pooled discrepancy takes the absolute
forecast-minus-native probability mass after summing endpoint groups inside each
fixed bin; grouped discrepancy takes absolute values before that summation.
Their difference measures cancellation. Values and conditional 95% paired-law
bootstrap intervals are percentage points. Goal names denote the constructed
controller's meaning, dependency and presentation objectives.

| Population | Position | Goal | Pooled discrepancy | Grouped discrepancy | Cancellation | Conditional interval |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| native | 1 | meaning | 1.68120 | 1.74715 | 0.06595 | [0.05351, 0.08237] |
| native | 1 | dependency | 2.04831 | 2.17092 | 0.12261 | [0.11268, 0.13307] |
| native | 1 | presentation | 1.83874 | 2.04821 | 0.20947 | [0.17062, 0.25679] |
| native | 2 | meaning | 2.34755 | 2.57821 | 0.23066 | [0.19595, 0.26780] |
| native | 2 | dependency | 2.33764 | 2.37473 | 0.03709 | [0.02839, 0.04531] |
| native | 2 | presentation | 2.34550 | 2.56069 | 0.21520 | [0.18484, 0.24631] |
| native | 3 | meaning | 1.86089 | 1.91610 | 0.05522 | [0.03792, 0.07372] |
| native | 3 | dependency | 0.01538 | 0.01538 | 0.00000 | [-0.00000, 0.00000] |
| native | 3 | presentation | 1.86308 | 1.91063 | 0.04756 | [0.02730, 0.06873] |
| equal-frame | 1 | meaning | 1.93632 | 1.96558 | 0.02927 | [0.02647, 0.03344] |
| equal-frame | 1 | dependency | 2.15677 | 2.33822 | 0.18145 | [0.17398, 0.18935] |
| equal-frame | 1 | presentation | 1.83328 | 2.04052 | 0.20725 | [0.17330, 0.24705] |
| equal-frame | 2 | meaning | 2.27138 | 2.38212 | 0.11074 | [0.08740, 0.13753] |
| equal-frame | 2 | dependency | 2.36347 | 2.39605 | 0.03258 | [0.02047, 0.04577] |
| equal-frame | 2 | presentation | 2.24805 | 2.28063 | 0.03258 | [0.02921, 0.03751] |
| equal-frame | 3 | meaning | 1.69369 | 1.74100 | 0.04731 | [0.02875, 0.06844] |
| equal-frame | 3 | dependency | 0.02806 | 0.02806 | 0.00000 | [0.00000, 0.00000] |
| equal-frame | 3 | presentation | 1.69595 | 1.73076 | 0.03481 | [0.01779, 0.05392] |

The largest displayed native-weighted cancellation is at the second position's
meaning goal. This diagnoses opposing errors hidden by aggregation; it does not
recalibrate a forecast or establish a practically important effect. The third
position's dependency goal has no cancellation beyond rounding. Intervals at
roundoff scale are not evidence of an effect. Equal-frame results can have very
narrow law intervals because the frozen forecasts and frame weights are shared;
they do not measure full training uncertainty. Draw and seed means remain in
the complete regroup record rather than treating them as independent worlds.

All 653 deterministic outputs match the original, adjacent replay and complete
extracted-source replay. Sources, plans, environment fingerprints, inherited inputs,
outputs and timing bindings verify. The independent scalar checker reconstructs
92,160 rows, 320 full group-array bundles, 225,280 frame forecasts, 704 reader
packets and 5,120 parent loss/accuracy identities. Maximum array discrepancy is
9.99e-16, maximum row discrepancy 3.05e-16 and maximum parent discrepancy 8.88e-16.
Every reconstructed group/bin weight, forecast sum, native-correctness sum, mass,
conditional signed error and empty mean is retained. There are 4,007,872 empty
group/bin cells and no empty whole groups across these learned bundles; the
empty-bin means are NaN exactly at zero weight. Native self-forecasts have zero
calibration discrepancy, and no cancellation gap violates the triangle inequality
beyond the frozen rounding tolerance.

Original and reconstructed rows receive independent scalar regrouping. A third
direct-index bootstrap of the original rows verifies all 576 means, 576 within-arm
cancellation estimates, 1,296 paired budget contrasts and 324 normalized log-budget
areas within 2.50e-16. Seed 191017 fixes 10,000 paired resamples of sixteen laws,
with both draws and five seeds inside each law. This is conditional exploratory
evidence, not confirmation on untouched laws.

All 48 isolated checker controls passed. They include opposite and identical
errors, native self, empty groups, all endpoint assignments, permutations, bin
boundaries, mass conservation, target rejection and deliberate corruptions.
Scalar base-three sums verify saved marginals before binning. The unsaved native
matrix projection is recreated and independently checked to preserve executed
binary64 bin assignments; this does not audit the numerical library itself.
Frozen sources and earlier failed attempts remain preserved.

Reader packets contain only public evidence and remain separate from scientific
forecasts, native truth, inherited inputs and rebuilt arrays. This is a constructed
method diagnostic, miniature — architecture untested beyond the admitted roster.
Neither historical goal correspondence nor human intent follows. V19 remains
active; declared-purpose calibration and retrospective updating are independent
next alternatives.

[Review protocol](GOAL_ARTIFACT_CALIBRATION_REVIEW_PROTOCOL.md).
[Numerical acceptance](../../../results/v19/G19-D-goal-artifact-calibration-1/FINAL_REVIEW.json).
[Complete regroup](../../../results/v19/G19-D-goal-artifact-calibration-1/INDEPENDENT_REGROUP.json).
[Evidence roles](../../../results/v19/G19-D-goal-artifact-calibration-1/EVIDENCE_ROLES.json).

# Complete goal-class reliability: independently verified

We tested whether confidence in a selected local goal hides errors in the other goal probabilities. At 2,048 labels, the predictive bank's full three-goal squared probability error is 0.00645–0.01385 across positions under native weighting. Independent reconstruction, all 2,700 paired estimates and both complete replays verify this constructed-method diagnostic. Selected-goal confidence alone does not measure this error; historical goal correspondence and human intent remain unestablished.

The frozen diagnostic retains sixteen development laws, two training draws, five
fit seeds, four label budgets, four readouts, two forecasts, three positions and
three goal classes under native and equal-frame weighting. No fitting or
recalibration occurs. Native weights preserve the generator's frame distribution;
equal-frame weights give each of the 704 evidence frames equal mass. The two
forecasts share goal marginals up to floating-point roundoff, although their path dependence differs.

The table shows the predictive bank at 2,048 labels with the restricted forecast.
Each row is one population and operation position. Multiclass Brier loss is the
expected sum of squared errors against the three possible one-hot goal outcomes.
It decomposes into native uncertainty plus summed squared probability error.
The last column averages absolute discrepancy over ten fixed probability bins,
separately for each class and law, then averages the three classes; units are
percentage points. It does not pool opposite class errors.

| Population | Position | Multiclass Brier loss | Native uncertainty | Squared probability error | Mean class-bin discrepancy, points |
| --- | ---: | ---: | ---: | ---: | ---: |
| native | 1 | 0.177280 | 0.166770 | 0.010510 | 1.85608 |
| native | 2 | 0.179920 | 0.166069 | 0.013851 | 2.34356 |
| native | 3 | 0.100540 | 0.094086 | 0.006454 | 1.24645 |
| equal-frame | 1 | 0.149991 | 0.135548 | 0.014443 | 1.97545 |
| equal-frame | 2 | 0.149916 | 0.134893 | 0.015024 | 2.29430 |
| equal-frame | 3 | 0.091396 | 0.085658 | 0.005738 | 1.13924 |

Individual native-weighted class-bin discrepancies range from 0.01538 to 2.34755
percentage points. The near-zero case is the second goal at the third position,
whose native uncertainty is also essentially zero. Signed class errors are much
smaller and cancel when summed across all goals because probabilities normalize;
that identity is not calibration. Selected-goal confidence measures a different
quantity, so neither it nor the signed class average replaces the complete
probability diagnostic. No new practical threshold or architecture winner is
declared. All budget contrasts and intervals remain in the full regroup.

All 653 deterministic outputs match the original and both complete replays.
Source, plan, environment, input, output and timing bindings verify. Independent
scalar reconstruction covers 92,160 class rows, 288 native rows, 30,720 Brier
decompositions, 225,280 frame forecasts, 704 unchanged packets and 5,120 parent
identities. Score discrepancies are at most 7.33e-15. Original and reconstructed
rows separately regroup; the third direct-index bootstrap agrees within 7.11e-15
on 2,160 budget contrasts, 540 areas, 576 class means and 192 equal-class means.
Seed 191015 fixes 10,000 paired law resamples; draws and seeds remain inside laws.
Intervals condition on the retained fits. Equal-class means are descriptive.

There are 387,264 empty learned-row bins, whose means remain undefined. Tiny
roundoff above one affects 38,400 learned rows and 120 native rows. Raw calibration
preserves those values; the frozen proper-loss rule clips them while retaining
counts and masses. Maximum normalization discrepancy is 2.34e-15; maximum Brier
identity discrepancy is 2.23e-16. No contrast is undefined. Native self-forecasts
have zero calibration and squared probability error but retain native uncertainty.
Pooled law-bin discrepancy remains separate from average law-bin discrepancy.

Forty-seven isolated checker controls pass. The retained first attempt had one
missing portable-route failure among 47 controls; the fixed snapshot passes all.
Scalar base-three marginal sums validate saved binary64 probabilities before
binning and clipping. Explicit one-hot expectation checks the Brier decomposition.
The unsaved native projection primitive is recreated and checked against scalar
sums; this is not an independent numerical-library audit. Inherited fits and
native targets remain previously verified inputs. This is exploratory miniature
evidence, architecture untested beyond the admitted roster, with no historical
goal or human-intent inference. Confirmation and test lineages remain untouched.

Reader witnesses are separate from forecasts, evaluator targets and reconstruction
archives. Both shared tiny settings are consumed. Operation-conditioned reliability
and retrospective updating remain independent alternatives; the campaign is active.

[Review protocol](GOAL_CLASS_RELIABILITY_REVIEW_PROTOCOL.md).
[Numerical acceptance](../../../results/v19/G19-D-goal-class-reliability-1/FINAL_REVIEW.json).
[Full regroup](../../../results/v19/G19-D-goal-class-reliability-1/INDEPENDENT_REGROUP.json).
[Evidence roles](../../../results/v19/G19-D-goal-class-reliability-1/EVIDENCE_ROLES.json).

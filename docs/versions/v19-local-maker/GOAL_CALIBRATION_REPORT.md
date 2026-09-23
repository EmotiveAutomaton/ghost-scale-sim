# Local-goal confidence: independently verified overstatement

We tested whether confidence in local-goal reports matches their native expected correctness. At 2,048 labels, the predictive bank overstates correctness by 1.73–2.97 percentage points across the three positions under native weighting. Independent reconstruction, all 900 paired estimates and both complete replays verify this constructed-method calibration finding. Historical goal correspondence and human intent remain unestablished.

The frozen diagnostic retains sixteen development laws, two training draws, five
fit seeds, four label budgets, four readouts, two forecasts, all three positions
and both native and equal-frame weighting. The illustrated original restricted
forecast uses public complete witnesses. The separate goal-product forecast is
also completely scored; no new fitting or recalibration is performed.

The table reports the predictive bank at 2,048 labels. Each row is one position
and one population. Confidence is the forecast probability of its selected goal;
correctness is the native probability of that same report. Signed excess is
confidence minus correctness in percentage points. Bin discrepancy averages each
lineage's absolute discrepancy over ten fixed confidence bins, also in percentage
points. Squared error compares confidence with native correctness. Brier loss is
the expected squared error for the binary event that the selected report is right.
The logarithmic loss for that binary event is measured in nats.

| Population | Position | Confidence, % | Correctness, % | Signed excess, points | Bin discrepancy, points | Squared error | Brier loss | Binary log loss, nats |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| native | 1 | 86.51017 | 84.19039 | 2.31978 | 2.32619 | 0.005234 | 0.064710 | 0.187167 |
| native | 2 | 87.14645 | 84.18067 | 2.96577 | 2.98276 | 0.007127 | 0.066609 | 0.186460 |
| native | 3 | 93.00236 | 91.26872 | 1.73364 | 1.80805 | 0.003225 | 0.050268 | 0.139111 |
| equal-frame | 1 | 89.01329 | 86.66626 | 2.34702 | 2.35274 | 0.007194 | 0.054288 | 0.166833 |
| equal-frame | 2 | 89.63196 | 86.76371 | 2.86825 | 2.90121 | 0.007608 | 0.055107 | 0.158264 |
| equal-frame | 3 | 93.45987 | 91.91969 | 1.54018 | 1.61848 | 0.002867 | 0.045696 | 0.126262 |

Matched frequencies have smaller signed excess in every illustrated position:
1.22–2.22 percentage points with native weights, compared with 1.73–2.97 for the
bank. Raw-history excess is 2.15–3.60 points; frozen-latent excess is 1.70–2.97.
This descriptive cell does not establish a general architecture ranking. All
budget-level contrasts, paired intervals and log-budget areas remain in the full
regroup. There is no declared practical threshold for these calibration metrics.

All 652 deterministic outputs match the original and both complete replays.
Source, plan, environment, frozen input, output and native timing bindings verify.
Independent scalar reconstruction covers 30,720 learned strata, 96 native rows,
225,280 frame forecasts, 704 reader packets and 5,120 parent identities. Maximum
score disagreement is 1.09e-14. Original and reconstructed rows separately regroup;
a third direct-index bootstrap agrees within 9.55e-15 on all 720 budget contrasts,
180 areas and 192 means. Seed 191014 fixes 10,000 paired law resamples. Two draws
and five fit seeds remain inside each law; intervals condition on these fits.

There are 161,472 empty learned-row bins; their means remain undefined. No learned
row has infinite binary loss and no contrast is undefined. Tiny roundoff values
above one occur in 20,880 learned strata and 74 native strata: the raw values are
preserved for calibration, while proper losses use the frozen clipping rule and
retain clipping counts and masses. These counts identify affected strata, not a
substantive probability error. All native self-forecasts have zero signed, binned
and squared calibration errors, while proper losses retain native uncertainty.
Pooled bin discrepancy and average lineage-bin discrepancy differ because pooling
can cancel opposite errors. Complete numerators, denominators and both summaries
remain available. Exact binary64 bins/modes are checked after saved probabilities
are validated; this does not independently audit the numerical library.

Thirty-seven checker controls pass, including twenty corruption cases, complete
synthetic execution, exact/adjacent bin boundaries, empty bins, clipping, impossible
support, zero-weight impossibility, within-bin cancellation and native uncertainty.
The binary proper scores judge the selected goal only. They do not assess the
distribution over all three goals, historically realized intentions, or humans.
This is exploratory miniature evidence, architecture untested beyond its admitted
roster. Reserved confirmation/test lineages remain untouched.

Reader witnesses are separate from scientific forecasts, evaluator targets and
reconstruction archives. Both shared tiny settings remain consumed. Complete
class-wise reliability and retrospective updating are independently prepared;
the campaign remains active with 65 accepted batches and unchanged resource limits.

[Review protocol](GOAL_CALIBRATION_REVIEW_PROTOCOL.md).
[Numerical acceptance](../../../results/v19/G19-D-goal-calibration-1/FINAL_REVIEW.json).
[Full paired regroup](../../../results/v19/G19-D-goal-calibration-1/INDEPENDENT_REGROUP.json).
[Evidence roles](../../../results/v19/G19-D-goal-calibration-1/EVIDENCE_ROLES.json).

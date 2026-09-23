# Goal decisions: independently verified objective mismatch

We tested whether choosing a whole goal path sacrifices stepwise accuracy with the forecast held fixed. At 2,048 labels, separate step decisions lower the predictive bank's native-weighted step accuracy by 0.08349 percentage points despite improving its own forecast objective. Independent reconstruction, all 240 paired estimates and both complete replays verify this constructed-method reversal. It does not establish historical goal correspondence or human intent.

The fixed forecast describes 27 possible three-step goal paths after all actual
operations have been witnessed. Choosing its most probable path maximizes its
own whole-path success probability. Choosing a separate modal goal at every step
minimizes its own average step error. Native posterior weights independently score
both decisions; they never select or restrict a learned report.

At 2,048 labels, the predictive bank's separate decisions reduce its own forecast
mean step error from 0.112831 to 0.111137. Native-weighted step accuracy instead
falls from 86.63008% to 86.54659%; whole-path accuracy falls from 65.41383% to
65.33642%. The two decisions disagree on 4.96939% of native packet mass. Prediction
loss is unchanged because neither decision changes the forecast.

The table compares separate-step minus whole-path decisions at 2,048 labels for
the original operation-restricted forecasts. Entries are percentage-point changes
in native expected correct-step fraction and correct-whole-path fraction, with
paired 95% lineage bootstrap intervals. Negative numbers indicate lower accuracy.
All draws and seeds remain paired within a lineage.

| Readout | Step accuracy change, percentage points | Whole-path accuracy change, percentage points |
| --- | ---: | ---: |
| Raw history | -0.04447 [-0.05024, -0.03843] | -0.00892 [-0.01306, -0.00485] |
| Frozen latent state | -0.11011 [-0.12447, -0.09494] | -0.12168 [-0.13723, -0.10516] |
| Predictive bank | -0.08349 [-0.09410, -0.07220] | -0.07741 [-0.08765, -0.06660] |
| Matched frequencies | -0.15015 [-0.16926, -0.12985] | -0.23157 [-0.26175, -0.19966] |

All four readouts lose both forms of native accuracy at the largest budget.
That is not a universal preference for whole-path decisions. The bank's step
changes at 32/128/512/2048 labels are 0, -0.00620, +0.07250 and -0.08349 percentage
points. Its normalized area across log training budgets is +0.00819 percentage
points [0.00720, 0.00915], hiding the largest-budget reversal. No accuracy-specific
practical margin was declared; the separate 0.02-nat prediction margin does not
apply to percentage points. These small, conditional effects do not promote the
primary bank claim.

Native posterior decisions coincide on every packet and lineage: both reach
87.47757% step accuracy and 67.55871% whole-path accuracy. Thus this roster does
not exhibit a native objective tradeoff, though synthetic opposing-optimum and
off-support fixtures validate that the instrument can detect one. No learned
chosen tuple has zero native support on this roster; that is weaker than knowing
which historical path occurred.

For the product-of-goal-marginals forecasts, raw-history, frozen-state and bank
decisions agree at the largest budget. Matched frequencies differ: separate
choices gain 0.02900 step and 0.05218 path percentage points. Saved binary64 product
probabilities can distinguish nominally tied paths. These numerical choices do
not demonstrate new marginal information; exact and near ties are separately
controlled, with the smallest numeric goal label winning exact ties.

The inherited roster has sixteen development coefficient lineages, two training
draws, five fit seeds, four readouts and four nested budgets. All 10,240 learned
score rows, 32 native rows, 225,280 frame forecasts, 704 public packets and 5,120
parent loss strata verify. Original, adjacent and extracted-source executions
match all 828 deterministic outputs. All source, plan, environment, input, output
and timing bindings verify, including the independent checker.

The checker reconstructs forecasts by scalar integer-label sums and separately
recomputes probabilities, marginals, exact decisions, ties, margins, forecast
risks and native scores. Maximum probability error is 2.11e-15; decision-array
error is 2.56e-15 and score error is 1.03e-14. Scalar grouping with bootstrap
multiplicities checks both original and rebuilt rows. A third direct-index
bootstrap reproduces 192 budget contrasts, 48 normalized log-budget areas and
64 means within 1.34e-15. Seed 191011 fixes 10,000 paired sixteen-lineage resamples;
training-draw and fit-seed variation remain separate. Intervals condition on these
fits and are not universal training-population uncertainty.

Thirty-four isolated controls passed before checker dispatch, including opposing
objectives, point/uniform laws, equal marginals with different joint modes,
exact/near ties, full synthetic execution and deliberate corruption of every
saved decision field, native scores, reader fields, rosters and bindings. The
original fitted forecasts and native target laws remain inherited accepted
inputs, not newly trained or independently discovered human mechanisms.

This is exploratory constructed-method evidence, miniature; architecture untested
beyond the admitted roster. Optimizing an imperfect forecast's internal objective
need not optimize its performance under the native law. Reader exports contain
only anonymous public artifacts, context, operation witnesses and tool proposals;
all forecasts, choices, targets and scores remain scientific/evaluator records.
No fit, sample, protected lineage or tiny setting is consumed. Both shared tiny
settings remain used. Neither historical correspondence, human intent, selective
causal access nor confirmation of the primary bank claim follows.

Sixty-two scientific batches have independent acceptance. Goal-report abstention
and retrospective evidence updating remain independent alternatives. The 96-hour
total, protected 16-hour reserve and immutable reporting times remain unchanged.

[Review protocol](GOAL_DECISION_REVIEW_PROTOCOL.md).
[Accepted reconstruction](../../../results/v19/G19-D-goal-decision-1/FINAL_REVIEW.json).
[Full regroup](../../../results/v19/G19-D-goal-decision-1/INDEPENDENT_REGROUP.json).
[Evidence roles](../../../results/v19/G19-D-goal-decision-1/EVIDENCE_ROLES.json).

The fixed-cost goal-abstention comparison is source admitted after 47 isolated controls. The existing serial queue dispatched it and both complete replays at verified below-normal priority. Numerical acceptance awaits completion and independent reconstruction. Retrospective evidence updating and less specific goal reporting remain two prepared independent alternatives. No new fit or protected lineage is used.

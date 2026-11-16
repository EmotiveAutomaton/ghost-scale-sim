# D: prospective diary calibration

Question: what do corrected correlations of noisy individual slopes recover when
the true reactivities share one dial, two dials, or a dial plus contest readiness?
This is a method calibration within a stipulated diary world.

The full packet Cartesian grid has 5,184 cells. Start with ordinary floors,
untied levels/rates, binary events, 2,272 people and six-prompt diaries; then daily
diaries, other sample sizes and stress cells. Each cell has an excluded 40-world
pilot and 200 fresh worlds in eight blocks of 25. Reserve at most 20 CPU-hours,
including setup, failed attempts, replay and reconstruction. Stop at the last
complete cell. Pilot separation exclusions do not apply to correlation calibration;
availability failures are reported rather than interpreted as scientific nulls.

Nuisance parameters follow the supplied preflight: usual level 0.6 with SD 0.5,
shared prompt noise SD 0.5, three item errors of SD 0.8 per family. Prompt event
rates are Beta(2,8), scaled by requested rate/0.2 and clipped to [0.02,0.9]; daily
rates use Beta(4,6), rate/0.4 and [0.05,0.95]. A tied rate adds log-dial to its
log odds. A tied level adds 0.4 times the mean standardized log reactivity.
Graded positive events are 0.5, 1 or 1.5 equally. Mild ratings are rounded to
0..5 (prompt) or 0..4 (daily); the additive origin is irrelevant to slopes.
No-floor ratings remain continuous. A heavy-floor offset is frozen by a separate
32,768-person integration panel targeting two thirds at zero, with Gaussian item
and shared noise integrated analytically. Its fixed seed never enters science.

Odd/even means odd/even days. Prompt halves require three event and three
non-event reports; daily halves require one each. Cross-half correlations are
averaged before division by the square root of the two positive reliabilities.
The same-half diagnostic uses matched correlations, not covariances. Estimates
outside [-1,1] are retained. Nonpositive reliability or insufficient observations
is unavailable, never floored to a tiny denominator.

Estimator three is a correlated random-slope method of moments: estimate each
person's full-record slope, subtract its residual-derived sampling covariance
from the between-person covariance, and standardize if both corrected variances
are positive. This is a working linear random-effects estimator; flooring can
violate its assumptions. It is not an exact censored-likelihood reference.

Per-world intervals use an eight-group delete-group person jackknife. The ES2
one-dial rule requires a lower bound of at least 0.90; the multiple-dial rule an
upper bound at most 0.85. Unavailable worlds never count as settled. Across-world
means and availability use eight independent blocks, with explicit denominators.
Conditional estimates retain their unavailable-world counts and world spread.

Raw evidence retains per-person regression sufficient statistics and true slopes,
not individual diary items. Whole-seed replay regenerates every item and statistic.
Independent reconstruction solves the normal equations and residual covariance;
it shares the stored sufficient statistics and summary conventions. Controls
check exact slopes, zero/one latent targets, and the rival's analytic correlation.
Supplied diary scripts and saved tables replay exactly; retained-path comparisons
cover both shapes and every correlation. D.b is a separate extension.

Route: Sounding Line ES2/MI2 receives bias, spread, availability and the rival's
landing point. Results do not estimate any human trait or prove questionnaire validity.

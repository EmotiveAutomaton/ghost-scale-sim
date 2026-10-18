# N17: lean, sensitivity and taste, declared before sampling

Sixteen redrawn constructed laws, 64 makers per setting/law. Cross n=20,50,100,250
with taste absent/present and payoff/belief lean. Five offers plus dismiss.
Offer quality and a taste feature are independent Gaussian coordinates. The
quality coordinate is supplied to these synthetic readers; this does not assert
that CoAuthor provides an equivalent gold quality measure.

For offer i, logit = (c-log(5)+d*q_i+w*f_i)/temperature; dismiss logit zero.
Criterion c is in {-2,-1,0,1,2}, sensitivity d in {0.25,0.75,1.5,2.5}, taste w
in {-1.5,-0.75,0,0.75,1.5}. Truth uses this same finite support. Each law redraws
quality/feature scale, temperature multiplier and the criterion population law.
Estimate the parameter posterior from the first half of ordinary records; freeze
it before predicting the second half. Readers: c with d=w=0; c+d with w=0;
c+d+w; oracle. All integrate the same declared uniform parameter priors; no
evaluator parameters reach a fitted reader. Report parameter error and held-out
proper log scores, including all unfavorable outcomes.

Fit global criterion-population location/spread and temperature on an independent
synthetic aggregate bank, using a fixed grid and equal scaled squared deviations
from Sounding Line targets: within agreement .4217687075, cross agreement .3645,
chance approximately .41, within kappa .0890826932, cross kappa .0271148623,
acceptance ICC .093337938 and acceptance split-half correlation .780262258.
Use a separate validation bank and retain every target, achieved value and error.
These statistics have different populations/channels in Sounding Line; a six-action
synthetic channel cannot be asserted equivalent to its handling categories.
The near-.41 chance target is approximate, not algebraically substituted for the
chance value implied by either kappa. No raw human records are copied.

False taste detection uses a prospective finite-null e-value: likelihood of the
held-out sequence under the frozen full reader divided by the **maximum** held-out
sequence likelihood over all registered c+d null parameters. Declare detection
only when this ratio is at least 20. The denominator dominates the true null law
on this finite support; the numerator is a normalized predictive distribution
conditioned on training. Hence the null false-positive probability is at most
5% by Markov's inequality, without tuning to discovery. Still require the
empirical no-taste rate <=5% at each n, pooled equally over laws/lean mechanisms;
retain law-level and finite-count uncertainty. Do not weaken a failed gate.

Primary: full-reader minus c+d held-out log-loss gain with nonzero taste at n=100.
Criterion/sensitivity are declared separated only if c+d beats lean-only with
positive law-block lower bound and both parameter mean-square errors improve on
their prior-mean estimates at n=100. Report both component errors and intervals.
The SDT1 route also requires the no-taste gate, and remains conditional on the
reported calibration mismatch and availability of a real quality measure.

Payoff and belief mechanisms have identical ordinary logits. A stakes probe
doubles the bad-accept cost: subtract log(2) from payoff c; belief stays unchanged.
A base-rate announcement removes the belief intercept (c=0), leaving payoff c.
The two probe menus are independent of the fitting/test halves. Integrate the
frozen parameter posterior under each mechanism and update after each response.
Measure gain in true-mechanism log probability from the pair. These selective
interventions are constructed assumptions, not an established human response law.

Known answers: feature permutation leaves w=0 likelihood unchanged; split fitting
never reads held-out responses; fixed-null likelihood is bounded by the maximized
null; exact enumeration on a short binary fixture validates the e-value bound;
ordinary mechanisms alias; probe laws differ; independent log-score/likelihood
reconstruction. Full replay and law-block bootstrap (4096 draws) precede acceptance.
Cap three CPU-hours. Shed n=250 first, then use the first eight independent law
draws. Freeze reductions in a new excluded pilot before discovery. No controls shed.

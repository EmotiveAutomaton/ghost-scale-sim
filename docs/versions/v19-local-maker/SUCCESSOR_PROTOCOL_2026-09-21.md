# Probability-valid readout and process sufficiency admission

These are the executable versions of the two prepared comparisons in
[the successor design](NEXT_DESIGNS_2026-09-21.md). They preserve the original
readout and alternative results. Plans bind every retained input and source
before execution. Each has an adjacent replay and a separately extracted-source
replay in the existing single-worker queue. Test and confirmation lineages stay
reserved; neither comparison trains a sequence model or uses a tiny-model setting.

## Retained-feature probability head

Reuse G19-04 and G19-05 training and development arrays, old heads, latent bases,
and saved target-feature transformations. Reconstruct their feature coordinates
exactly from the saved transformations; do not generate a new feature seed or
refit a representation. Each of the four arms retains 128 tanh coordinates plus
an intercept and the same available teachers. Fit multinomial softmax heads with
129 coefficients per target category, zero initialization, L2 penalty 0.01 on
non-intercept coefficients, 200 full-batch gradient steps of size 0.1, and no
checkpoint selection. The objective sums categorical cross-entropies across
questions, averages over histories, and adds half the penalty times squared
coefficient norm. Each categorical head normalizes independently.

Use both retained feature seeds and training draws, the original old-world
budgets 32/128 and local budgets 32/128/512/2048. Include an explicitly prespecified
mean of exactly the same nested new labels and the existing privileged exact
reference, with the original fixed category floor. Local goal labels are evaluator
supervision on training only; no development truth influences fitting. Save all
forecasts, coefficients, initial/intermediate/final training objectives and gradient
norms. Report convergence limits rather than tuning the fixed schedule afterward.
Preserve separate local-goal and operation scores, each old class, equal-class
population, and paired lineage area comparisons. The original ridge arrays are
retained as the comparison; this does not replace G-P1 or establish trained-model
capability. Initial cap including admission and both replays: three CPU hours.

Controls precede dispatch: separable known labels, constant-target placebo,
finite-difference gradient, and multiple independent categorical normalizers.
Input hashes include the parent plans and retained data/model file identities.

## Persistent maker posterior versus preceding operation order

Use the first eight admitted local development lineages and their retained complete
enumeration. Regroup all four evidence projections into visible packets, retaining
their masses, 16-maker posteriors and local goal/operation target distributions.
Screen posterior bins at 8, 10 and 12 decimal places deterministically, retaining
posterior residual and target gap. Bins are a numerical candidate screen, not
proof of exact equality or an exhaustive nearest-neighbor search.

A source-derived reachable witness is frozen before this analysis: for each of
the four contexts, compare fully witnessed operation sequences
`replace-presentation, inspect, inspect` and
`inspect, replace-presentation, inspect`. Presentation changes only the display
unit. Goal probabilities depend on claim/evidence, purpose, skill, belief, routine
and request, and do not depend on display or the undo buffer. First/second-step
inspection sums all local-goal branches. At the final step dependency uses undo,
so final inspection sums meaning and presentation. For every maker, both paths
therefore have the same product: presentation-action probability, first/second
inspection probability, and final-inspection probability, times the common prior.
The context and maker prior contributes 1/64. The retained enumeration must agree
with this expression, not merely with another approximately equal posterior.

Consequently a successful certificate is conditional on these executor rules and
means equal persistent-maker posterior with different preceding operation order.
It applies to exact fresh-episode banks that factor through that posterior. A
learned bank may encode extra history in its approximation errors, so the theorem
does not establish that a learned bank cannot answer those questions. It makes no
claim that all evidence tiers or histories have such a witness.

Keep the proof, hidden posterior and target distributions in evaluator evidence;
export the paired witnessed packets separately. A synthetic exact alias is the
positive fixture and an invertible near-singular bank is the negative fixture.
Check the analytic factorization against the full transient executor on a separate
fixture law before dispatch. Initial cap including admission and both replays:
two CPU hours. No new sampling or fitting is required.

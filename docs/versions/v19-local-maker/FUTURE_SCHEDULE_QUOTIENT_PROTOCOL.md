# Lossless future-schedule quotient: prepared alternative

Can grouping complete hypotheses by their remaining state schedule reduce storage
without losing any declared future forecast? This B-family alternative uses the
existing supplied-law unknown-time/type roster at horizons 32 and 128 and fixed
checkpoints 8/16/17/20/32, where in range. The handler is implemented; isolated
controls and source admission precede execution. No outcome is claimed.
It is independent of primitive feedback and needs no
compression or marginal-transition advantage.

Before execution, fix a signature as the complete maker-index sequence from the
checkpoint through the horizon. Group all hypotheses with identical signatures,
sum their saved posterior weights and preserve the original membership map.
Use all eight development lineages and both saved observation draws, every maker,
source condition and actual-change condition available at the selected checkpoints.
No probability cutoff, selected capacity, refit, new observations or protected
lineage is permitted. If a requested checkpoint is absent from saved posteriors,
report it as unavailable; do not regenerate observations to fill it.

For every remaining time and four contexts, verify weighted endpoint forecasts
against the original complete hypothesis distribution using a separate scalar
sum. Report component counts and float64-weight/int32-map bytes, counting shared
schedule metadata separately. Include a stationary roster, hypotheses that merge
only after their change, and unequal future schedules with identical current
maker. Deliberately wrong merges must fail the future-forecast check. Equal-law
forecasts do not license merging different state schedules in this design.

This is a lossless representation and cost question under supplied laws. It
does not infer schedules, establish reachability of arbitrary simplex witnesses,
or prove historical process correspondence. Source, inputs and controls must be
frozen before execution. Initial cap one CPU hour including tests/replay/review
within the existing B/global budget, with no automatic family extension. Retain
bounded primitive feedback as the other prepared alternative.

Implementation freezes float64 weights and int32 membership/schedule arrays.
All remaining times are scored, with four eight-endpoint forecast vectors per
time and posterior row. The independent calculation reconstructs each original
hypothesis's state by tuple transitions, sums its unmerged weights into sixteen
current states, and uses a scalar sum across those states for every endpoint.
The grouped forecast must agree within an absolute 1e-12; small controls also
expand the direct full-hypothesis sum. This numerical tolerance is not a claim
of binary-identical arithmetic. Save all vectors and per-time discrepancies.

Retained parent metadata supplies all five requested checkpoints at horizon 32,
but only 16 and 32 at horizon 128. Record 8, 17 and 20 as unavailable there.
This is a pre-execution input inventory; no observations are regenerated.
Original and source-identity-omitted parents remain paired on both observation
draws, all makers and actual-change/source conditions. Parent inference validity
is inherited from its separately bound independent reviews. The native fixture
checks complete parent mapping, missing checkpoints and corrupted forecasts.

[Bounded primitive feedback](PRIMITIVE_FEEDBACK_PROTOCOL.md) and
[retrospective evidence after grouping](RETROSPECTIVE_QUOTIENT_PROTOCOL.md)
remain independently prepared alternatives, requiring separate implementation
and source admission; neither depends on this comparison winning.

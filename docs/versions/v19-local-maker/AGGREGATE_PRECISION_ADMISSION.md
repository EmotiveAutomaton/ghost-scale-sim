# Aggregate storage precision: prospective admission

Does lower precision damage the sufficient state for a later report about the
past? The prepared contrast fixes float64, float32 and float16 storage, each
queried by direct casting or the one mass-preserving reconstruction. All query
arithmetic uses float64. Copied-endpoint counts remain int32. The supplied law
and future schedule remain exact and are charged separately from stored state.

The full population remains 28,672 saved posteriors, 505,856 source candidates,
eight laws, two evidence conditions, two paired draws, five copying probabilities,
eight endpoint reports and every future coordinate. No new fit, observation or
protected lineage is used. All six storage/reconstruction combinations are kept.

The exact group and joint-report masses also serve as the stored float64 state.
Float32 and float16 arrays are retained in their actual dtypes. Query arrays are
reconstructed deterministically from those saved arrays. Normalization drift and
underflow are recorded before reconstruction. A positive group whose quantized
joint row is empty invalidates the reconstructed state; no probability floor or
evaluator mass is inserted. Exact zero entries stay zero.

For each method, retain raw report mass, support and reconstruction-failure flags.
Forecast errors exist only where both reference and approximation support the
report and the reconstructed state is usable. Report lost support, unusable
reference mass and errors separately. Defined error contributions are weighted
by exact report probabilities; they are not unconditional scores when mass is
unusable. The three fixed positive-report strata are below one millionth, from
one millionth to below one thousandth, and at least one thousandth. Exact-zero
reports are separate. Retain each stratum's mass, defined mass and error sum;
an empty stratum has no conditional mean. Squared error equals excess expected
one-hot squared loss for normalized forecasts. No practical margin is asserted.

Controls compare scalar source-level Bayes, every future coordinate and explicit
proper losses under dense, sparse, constant and disjoint laws. Rounding, positive
empty-row failure, lost rare-report support, dtype/storage counts, permutations,
mask corruption and complete synthetic integration are required. Both native and
portable dispatch use the same source-frozen handler. Complete-support synthetic
timing includes all seven structures, both source densities, three context
patterns, full 32-row raw batches and all thirty summary rows per posterior.

The first prototype duplicated stored float64 masses and six reconstructed query
arrays in its raw export. Its controls passed, but timing exceeded the inclusive
cap. The one bounded repair removes only those recoverable duplicates; original
and quantized state remain. Both source snapshots, controls and timing remain.
No scientific data was sampled to choose this repair or estimate service cost.

Admission requires the full original and two replays, independent numerical
reconstruction and separate original-row regroup to fit 3,600 CPU seconds,
including checks, failures, setup and operations. The family and global ceilings
and protected reserve are unchanged. Completion establishes only a supplied-law
constructed-method storage diagnostic, not learned access, minimum storage,
historical process correspondence or human intent.

Highest-probability source retention and precision of the shared endpoint law
remain independent prepared alternatives. Neither is an implemented job.

[Frozen question](AGGREGATE_PRECISION_PROTOCOL.md).

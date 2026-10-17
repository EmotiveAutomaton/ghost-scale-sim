# T1 cost profile: pinned engine optimizer failure

The native writing twin reached the actual ToMpathy choice backbone and failed
while fitting a maker offset: `ValueError: MAP fit failed: ABNORMAL:`. The affected
profile is T1-0001, world 0. No complete cost profile or replay is accepted, and
this is not a scientific null. The preceding retained artifacts are unchanged.

[Failure evidence](../../results/v23/provisional-continuation/T1_PROFILE_FAILURE.json)
binds the complete retained prefix and synthetic public reader. The committed
engine remains `82789fdb50d9366ab9352129b01ee37622554dfa`; the current checkout's
`_fit` function is structurally identical to the pinned one. Other uncommitted
ToMpathy work was inspected read-only and was not adopted.

The reproducing call is `Backbone.fit(reader_choices(load_reader(reader)[0]),
until=datetime.fromisoformat(cutoff))`, using the recorded synthetic reader and
cutoff with the pinned engine/environment. The failure occurs in the final
per-maker offset fit, after penalty selection. A useful bounded diagnosis should
retain the optimizer exit, objective, gradient and candidate coefficients, then
check numerical stationarity independently. A repaired solver needs its own
known-answer checks and a new pinned successor before T1 admission. Ignoring the
failure flag, changing labels or quietly accepting unconverged coefficients is
not a repair. No message was sent and no sibling repository was modified.

N4c is complete and published. T3/T4 cost profiles continue; their outcomes remain
uninterpreted until complete review. T2 and the T4 sensitivity roster remain missing.

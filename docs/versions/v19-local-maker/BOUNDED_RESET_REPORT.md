# Fixed resets reduce error without rescuing the rolled-in updater

Does a fixed reset stabilize the failed updater? It reduces forecast loss by 2.31120 nats at 128 independent observations, but the rolled-in updater remains 3.47050 nats worse than the reset one-step baseline. Both complete replays and independent reconstruction pass. This is a bounded constructed-method result, not process correspondence or human intent.

Logarithmic loss is prediction error in nats; lower is better. A reset replaces
the current predictive bank with the frozen cached-history predictor before the
next unique observation after every eight unique sources. It uses public evidence,
not an exact bank. No fitting or reset-interval selection occurs. The original
32-observation prefixes and forecasts are preserved, then extended to 128 episodes
for the same stationary maker. There are eight paired development lineages,
two retained training draws, two feature seeds and eight streams per lineage.

The table gives logarithmic loss at 128 observations. Rows distinguish independent
episodes from copied adjacent pairs; the latter contain 64 unique sources.
Columns compare unchanged heads with their fixed-reset versions and the history
cache. Lower loss is better; all forecasts receive the same declared probability
repair before scoring.

| Evidence stream | One-step | One-step with resets | Rolled-in | Rolled-in with resets | Cached history |
|---|---:|---:|---:|---:|---:|
| Independent | 1.93640 | 1.92975 | 7.71144 | 5.40025 | 1.93378 |
| Copied pairs | 1.92252 | 1.92741 | 7.85074 | 5.71435 | 1.93471 |

For the rolled-in head, reset-minus-free loss is -2.31120 nats with a 95% paired
lineage interval [-2.80742, -1.87602] in independent streams, and -2.13640
[-2.58999, -1.72485] in copied streams. Reset rolled-in loss still exceeds reset
one-step loss by 3.47050 [3.19224, 3.74119] and 3.78694 [3.48963, 4.11420],
respectively. Roughly 99.4% and 99.6% of reset rolled-in query distributions at
128 observations have invalid raw probabilities before repair. This is partial
error reduction, not recovered predictive competence.

For the original one-step head, the reset effect is small and changes sign:
-0.00664 [-0.01848, 0.00213] for independent episodes and +0.00489
[-0.00319, 0.01176] for copied pairs. Both conditional intervals are inside the
declared 0.02-nat practical margin. At 32 independent observations resetting
instead increases one-step loss by 0.00634 nats. At eight observations no reset
has yet occurred, so the reset and free arms match exactly. The full 8/32/128
strata and all four draw/seed differences remain in the independent review.
Repeated streams, questions, feature seeds and training draws do not become
independent worlds; intervals resample the eight lineages conditional on these fits.

The original one-step head and cached history differ by only +0.00262
[-0.00532, 0.01290] nats at 128 independent observations. Resetting lowers the
one-step head relative to the cache by 0.00402 [0.00069, 0.00723] nats, also inside
the practical margin. No end-to-end computational advantage is established.

At 128 positions, resets add 15 direct-head evaluations per independent stream
or seven per copied stream, plus a maintained 32-bin history cache. They retain
the original models' sunk fitting costs. The count feature still divides by
log(33); at the end it reaches 1.38990 for independent streams and 1.19387 for
copies, outside its training range. Length therefore changes both accumulated
updates and this numerical extrapolation. The experiment does not isolate them.
All three scored prefixes precede any next reset; a score is not the freshly
substituted cache output.

Teacher-forced outputs at copied scored positions are exact pass-throughs, not
learned updates. The exact-filter comparison itself uses the common probability
floor, so its tiny difference from an unmodified exact posterior is numerical.
Initialization privilege, reference-law privilege and free-running error remain
separate limitations. Only stationary makers are tested; transient goals, skill
changes and hidden source dependence remain open.

**Validation:** independent arithmetic reconstructs all 64 complete 128-position
forecast files, 32,768 exact-filter prefixes from retained endpoint laws, every
reset decision/cache forecast, and all 1,536 score rows. Maximum score discrepancy
is below 2e-15. Parent prefixes and 8/32 free forecasts are byte-identical; copies
never update or trigger resets. Adjacent and separately extracted-source complete
replays match all 113 deterministic files. Input/source/environment hashes and
separate timing records verify. The review neither refits heads nor independently
re-enumerates the endpoint laws. This is miniature — architecture untested.

**Warrant:** exploratory advantage for this reset relative to its failed rolled-in
parent, with a large remaining reversal against one-step prediction; one-step
reset differences are practically small conditional on these fits.
**Pursuit:** this single bounded reset strategy is resolved. No interval search
or seed padding follows. The [persistent practice comparison](PRACTICE_PROTOCOL.md)
is an independent successor; portfolio and whole-stream calibration designs remain
prepared. The primary neural comparison and causal-access instrument remain open.

[Scientific evidence and separate reader export](../../../results/v19/G19-B2-bounded-reset-1/EXPORT_MANIFEST.json).

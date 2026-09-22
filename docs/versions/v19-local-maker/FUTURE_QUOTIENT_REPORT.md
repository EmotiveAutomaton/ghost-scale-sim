# Future-schedule grouping: independently verified

We tested whether grouping identical remaining maker-state schedules preserves forecasts while reducing storage. All 4,620,288 saved forecast vectors reconstruct within 1e-12, with both complete replays verified. Float64 weights plus shared integer metadata use 23.96% fewer bytes on available checkpoints, or 22.53% fewer including unused requested-checkpoint metadata. This is a constructed-method storage result; online updating, workspace, speed, historical process correspondence and human intent remain unestablished.

All eight development laws, both saved observation draws and paired source-aware/
source-omitted histories remain. The 112 cells contain 28,672 checkpoint posteriors
and 14,336 paired source rows. Sixteen makers, purpose/skill changes, actual change/
stationarity and independent/copied observations retain their original pairing.
No new observations, fits or untouched confirmation lineages were used.

The table describes each horizon/checkpoint pair. Original hypotheses count full
change histories; schedule groups count distinct maker-state sequences from that
checkpoint through the horizon, including zero-weight groups. Available posterior
rows count saved distributions across all laws and conditions. Strict break-even
rows are the smallest number whose weight savings exceed the shared int32
membership and schedule bytes; common parent metadata is excluded from both sides.

| Horizon | Checkpoint | Original hypotheses | Schedule groups | Available posterior rows | Strict break-even rows |
|---|---|---|---|---|---|
| 32 | 8 | 560 | 560 | 4096 | Never |
| 32 | 16 | 560 | 304 | 4096 | 12 |
| 32 | 17 | 560 | 272 | 4096 | 9 |
| 32 | 20 | 560 | 176 | 4096 | 4 |
| 32 | 32 | 560 | 16 | 4096 | 1 |
| 128 | 8 | 3632 | 3632 | 0 | Never |
| 128 | 16 | 3632 | 3376 | 4096 | 753 |
| 128 | 17 | 3632 | 3344 | 0 | 657 |
| 128 | 20 | 3632 | 3248 | 0 | 466 |
| 128 | 32 | 3632 | 2864 | 4096 | 184 |

Horizon 128 has no saved posterior at checkpoints 8, 17 or 20. Their metadata is
retained and charged separately, without inventing missing observations. The
32-step checkpoint at step 8 merges nothing and adds 58,240 shared bytes. At the
final 32-step checkpoint, sixteen current maker states are sufficient for the
remaining endpoint forecasts. This does not imply that they preserve the past.

Original weights occupy 329,777,152 float64 bytes. Grouped weights occupy
247,988,224 bytes; adding 2,780,736 shared bytes for available checkpoints gives
250,768,960 bytes, a 23.9581% saving. Including all 7,496,448 requested metadata
bytes gives 255,484,672 bytes, a 22.5281% saving. These are explicitly enumerated
array-storage costs, not a benchmark of Python workspace, retained archive size,
or a deployable total-memory footprint. Saved forecasts and reconstruction
artifacts are separate outputs, not part of this weight-representation comparison.

The independent checker reconstructs every hypothesis, future schedule, tuple
membership, grouped weight and unmerged-state forecast. Maximum absolute errors
are 4.44089e-16 for weights and 3.16414e-15 for forecast coordinates, below 1e-12.
All row mappings, population pairs, unavailable checkpoints and summary counts
verify. A separate scalar regroup independently verifies each storage total and
the strict amortization formula. Both complete original replays reproduce all
88 deterministic outputs. The checker binds 504 sources and 90 inputs; thirteen
isolated controls include a native fixture and deliberate corruptions.

**Warrant:** constructed-method storage advantage for this retained population,
with numerical forecast preservation and exact replay. Miniature — architecture
untested. Parent law/posterior likelihood validity is inherited from separately
bound independent reviews. This establishes no learned access, online update
guarantee, speed gain, reachable-simplex theorem, historical correspondence or
human-intent result. No sampling interval is attached to deterministic byte counts.

**Pursuit:** test newly supplied retrospective evidence after grouping; the
complete lossless enumeration still needs tractable implementation and admission.
Primitive alias pooling is an independent prepared alternative. Neither needs
this result to win. Fixed primitive feedback has executed and awaits review.

All new raw arrays, row maps and schedules remain in the scientific archives.
Sixteen large inherited joint arrays remain separately hash-bound in the retained
evaluator capsule. No new reader inputs are created: schedules, laws, hidden
hypotheses and posterior weights remain scientific/evaluator material.

[Protocol](FUTURE_SCHEDULE_QUOTIENT_PROTOCOL.md),
[independent checker](FUTURE_QUOTIENT_REVIEW_PROTOCOL.md),
[evidence](../../../results/v19/G19-B-future-quotient-1/README.md).

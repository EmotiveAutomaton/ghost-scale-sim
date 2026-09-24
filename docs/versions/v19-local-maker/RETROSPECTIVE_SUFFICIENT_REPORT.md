# Joint retrospective state: independent numerical acceptance

We tested whether retaining joint past-state information permits exact updates from a new report about the past. It reproduces all 18,481,152 report updates within rounding error, but its factored storage exceeds full hypothesis weights after the earliest checkpoint. Independent reconstruction, all 224 paired strata and both complete replays verify this supplied-law constructed-method result. Minimal state, learned access, historical process correspondence and human intent remain unestablished.

A future-schedule group merges hypotheses with identical maker states from the
checkpoint onward. A report about the past can distinguish members of that group.
Keeping the joint probability of future group and past maker state supplies the
numerator needed to update the group. This test checks that algebra on every
available retained posterior and report, with supplied emission laws.

The census contains 28,672 posterior rows: eight development laws, two source
conditions, seven available horizon/checkpoint cells, two observation draws and
128 rows per draw. All 18,481,152 enumerated report queries have positive probability
on this roster. Impossible reports remain exercised by zero/disjoint synthetic
controls. The 48 missing 128-horizon checkpoint strata are explicitly unavailable;
their structural maps were checked without regenerating posterior observations.
Both draws remain paired within each law. This finite census needs no sampling interval.

The producer's largest total variation is 2.81576e-16; independent summation gives
4.52216e-16, both below the frozen 1e-12 tolerance. Total variation is half the
sum of absolute differences in updated group probabilities. It bounds the error
of every remaining-time endpoint forecast by contraction through the common
supplied law. It is not a measured maximum forecast error. Different summation
orders are not asserted to give bit-identical rounding-scale maxima.

Rows below are the seven available structural cells, each shared by 4,096
posteriors. Full weights, future groups, logical joint and factored values count
float64 entries per posterior, so multiplying by eight gives bytes. Logical joint
counts every supported group/past-state cell across past times. Factored values
count group weights plus every mixed-group joint cell, including zero masses;
singleton masses reuse their identical group weight. The last three columns are
shared int32 structure bytes per cell, not per posterior.

| Horizon | Checkpoint | Full weights | Future groups | Logical joint | Factored values | Shared support bytes | Shared schedule bytes | Shared membership bytes |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 8 | 560 | 560 | 4,480 | 560 | 53,760 | 56,000 | 2,240 |
| 32 | 16 | 560 | 304 | 5,344 | 1,040 | 78,592 | 20,672 | 2,240 |
| 32 | 17 | 560 | 272 | 5,136 | 1,056 | 79,168 | 17,408 | 2,240 |
| 32 | 20 | 560 | 176 | 4,128 | 1,104 | 77,824 | 9,152 | 2,240 |
| 32 | 32 | 560 | 16 | 1,280 | 1,296 | 81,920 | 64 | 2,240 |
| 128 | 16 | 3,632 | 3,376 | 54,496 | 4,112 | 668,416 | 1,525,952 | 14,528 |
| 128 | 32 | 3,632 | 2,864 | 92,640 | 4,368 | 1,206,016 | 1,111,232 | 14,528 |

The earliest checkpoint ties full-weight storage at 560 values. Later factored
tables use 1.13216 to 2.31429 times the full-weight count. Thus this implementation
restores single-report sufficiency without demonstrating compression against full
hypothesis weights. Logical joint storage and shared maps are reported separately;
neither unused zeros nor integer structure is hidden. No minimality theorem follows.

All 981 original deterministic outputs match both complete replays. The source
archives, 782 producer and 785 checker source files, 76 producer and 989 checker
inputs, environments and all output bindings verify. Independent bit transitions
reconstruct all ten support maps; indexed accumulation rebuilds joint masses;
direct hypothesis products and separately accumulated past-state marginals check
every report denominator. Singleton identities are exact. A separately coded
regroup reproduces all 224 law/source/checkpoint/draw strata and every reconstructed
summary row from original raw records. All 896 timing batches per execution bind.

The measured construction-plus-query regions take 131.734375, 154.578125 and
137.718750 CPU seconds in original, adjacent and portable execution. Each covers
all report queries in 32-row batches. Dividing by query count gives amortized cost,
not single-query latency; serialization and parent loading are excluded from those
timed regions and retained in total native accounting. This is no speed comparison
against a separately timed full-weight implementation.

Fifty isolated controls passed, including deliberate support, mass, denominator,
storage and timing corruption. The two preliminary fixture/serialization failures
and all source snapshots remain retained and charged. Existing scientific and
evaluator archives preserve full raw arrays; the independent reconstruction has
its own checked export. No reader inputs, fitted model, observation, protected
lineage or service owner were added. The miniature is architecture untested beyond
the declared roster; historical correspondence and human intent are not measured.

V19 remains active with 76 independently accepted scientific batches. Single-report joint-state sufficiency and its storage cost are independently verified. The source-mixture comparison passed 34 isolated controls and was observed executing through the existing serial queue, with both complete replays queued. Numerical acceptance awaits completion-event review. Two-report dependence and bounded provenance intervals remain prepared alternatives; the prior-conditional reachable rival remains timing-blocked.

[Execution protocol](RETROSPECTIVE_SUFFICIENT_STATE_PROTOCOL.md).
[Independent review](SUFFICIENT_REVIEW_PROTOCOL.md).
[Numerical acceptance](../../../results/v19/G19-B-retrospective-sufficient-1/FINAL_REVIEW.json).
[Complete regroup](../../../results/v19/G19-B-retrospective-sufficient-1/INDEPENDENT_REGROUP.json).

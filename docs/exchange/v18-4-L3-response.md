# V18.4 Frozen-state readouts: recoverable signal without a demonstrated composition advantage

Can an added decoder recover useful answers from a frozen memory that failed on new questions? The tested readouts recover aligned predictive signal, but they do not establish a memory advantage on taught compositions: at the larger label budget, the question-and-world-only nonlinear rival has lower logarithmic prediction loss than every frozen-memory readout. Untouched farther questions still resist reliable transfer. This is a descriptive constructed-method result, miniature — architecture untested; failure of this bounded decoder family does not prove information loss.

The encoders remain frozen. Fresh equal training labels teach the five original
and three composed questions to linear ridge and fixed-random-feature nonlinear
readouts. The two farther question forms remain absent. The two nested budgets
are 384 and 768 histories per architecture cell; all 160 readout fits, both
parent support regimes, all fit seeds and context-stratified shuffled controls
remain in the evidence. No test result selected a decoder or penalty.

The table reports mean logarithmic prediction loss in nats, lower being better,
at the larger label budget. Rows specify available representation; columns cross
decoder family with taught compositions or untouched farther questions. Frozen
state rows average the two equally sized parent-support regimes and covered
columns average both maker halves. These averages are descriptive paired summaries.

| Representation | Linear, covered | Nonlinear, covered | Linear, farther | Nonlinear, farther |
|---|---:|---:|---:|---:|
| Direct frozen state | 1.76211 | 1.60010 | 2.11660 | 2.19937 |
| Flat frozen state | 1.77644 | 1.66325 | 2.13945 | 2.23210 |
| Split frozen state | 1.78788 | 1.68076 | 2.15315 | 2.26525 |
| Full raw history | 1.91958 | 2.04079 | 2.29169 | 2.16762 |
| Question and world only | 1.95580 | 1.21823 | 2.32466 | 6.96048 |

Aligned frozen-state readouts improve on their shuffled-target versions and on
the linear question-only comparison. The nonlinear question-only result shows
that supplied world and question features can explain much of the taught-task
score without a history. Its farther-query failure also shows why a good covered
score cannot be treated as systematic extrapolation. Doubling label count does
not supply a general decoder-capacity theorem. Linear input dimensions differ;
nonlinear fitted widths match while fixed projections have different sizes.

Exact-reference farther loss is 0.75827. The remaining gap does not identify
whether a better objective, decoder or optimization procedure would close it.
This fresh readout lineage differs from the original training test lineage;
before/after losses from those two lineages are not a paired causal effect.
Neither extra-supervision recovery nor a shuffled-label difference establishes
that the original predictor used the recoverable information.

All 1,005 means and 51,200 fixed forecast rows replay independently with zero
discrepancy. The evaluator independently checks 288 targets/exact forecasts and
2,445 scalar scores. Source, inputs, copied encoders, fitted readouts and raw
hashes match; every portable file reassembled. This is bounded forecast replay,
not full refitting or architecture severity. No learning-control failures occur.
Science used 524.6875 charged CPU seconds and 545.893751 wall seconds;
verification added 51.34375 charged CPU seconds and 58.830537 wall seconds.
Historical V15 C11/M01 instrument failures remain unchanged.

[Portable scientific evidence](../../results/v18/exploratory-loop/L3-frozen-decoders-1/SCIENTIFIC_MANIFEST.json).

# V18.4 Frozen prediction banks: useful access with unstable raw inversion

Can old-question predictions answer new questions when given the finite world law? They can improve substantially under the fixed truncated bank map: untouched farther-question logarithmic losses are 0.84141 for flat memory and 0.82760 for split memory, versus their original decoders at 2.98964 and 3.06669. But the raw maps frequently violate probability constraints, and truncation trades exact span for stability. This is a descriptive constructed-method result with extra supplied-law access, miniature — architecture untested; it is neither learned law discovery nor evidence that the unrepaired map is a valid probability model.

This paired decoder intervention reuses the verified full-maker-support,
old-question weights and test cases. It performs no training or new label access.
The supplied world law maps the five old-query forecasts to each requested
question. Fixed relative singular-value cutoffs of 1e-10 and 1e-3 define full
and truncated maps; a passive-only bank remains a reduced-coverage comparator.

The table gives mean logarithmic prediction loss in nats, lower being better.
Rows specify the learned reader; columns cross decoder with composed or farther
question families. Composed columns average the two equally sized maker halves.

| Reader | Original, composed | Full bank, composed | Truncated, composed | Original, farther | Full bank, farther | Truncated, farther |
|---|---:|---:|---:|---:|---:|---:|
| Direct history | 1.71639 | 2.32156 | 1.46050 | 1.80569 | 1.43142 | 0.91709 |
| Flat memory | 2.84959 | 2.14881 | 1.40731 | 2.98964 | 1.20910 | 0.84141 |
| Split memory | 2.98549 | 2.04212 | 1.31057 | 3.06669 | 1.20324 | 0.82760 |

The full bank spans the composed and farther target laws to about 1.2e-12 maximum
absolute residual, but its condition number reaches 83,259.54. The maximum map
norm is 642.39 for compositions. Truncation reduces the maximum condition number
to 994.50 while leaving target-span residuals as large as 0.18251 for compositions
and 0.01396 for farther questions. The passive bank has residuals up to 0.65333.
Finite mathematical span is not a guarantee of stable access from learned inputs.

The scored bank forecasts floor entries at 1e-8 and normalize. Across all three
fit seeds, the following diagnostic table retains invalidity and repair, rather
than treating the repaired score as a valid raw probability construction. Invalid
means any negative/out-of-range entry or normalization failure under the frozen
tolerances; repair is mean absolute change summed over the 16 outcomes.

| Reader and map | Invalid composed rows (%) | Mean composed repair | Invalid farther rows (%) | Mean farther repair |
|---|---:|---:|---:|---:|
| direct full | 99.96021 | 2.14324 | 99.94575 | 0.32138 |
| direct truncated | 99.94575 | 0.42010 | 99.91319 | 0.02752 |
| direct passive | 99.61661 | 0.89739 | 98.33984 | 0.59107 |
| flat full | 99.94936 | 1.62777 | 99.97830 | 0.23458 |
| flat truncated | 99.94936 | 0.40865 | 99.97830 | 0.02132 |
| flat passive | 99.69256 | 1.33829 | 98.84983 | 0.91828 |
| split full | 99.99277 | 1.45481 | 100.00000 | 0.21406 |
| split truncated | 99.99277 | 0.31808 | 99.95660 | 0.01633 |
| split passive | 99.79384 | 1.23044 | 98.62196 | 0.82872 |

The truncation benefit does not remove the constraint violations. A supplied-law
simplex-constrained bank fit and an exact-bank reference would separate clipping
effects, approximation bias and forecast error. A prior-only supplied-law rival
is also needed before attributing the entire gain to retained history.

All 225 score means, 75 law-map diagnostic summaries, 11,520 fixed forecast rows
and every raw mapped probability replay independently with zero discrepancy.
The evaluator also checks 288 targets/exact forecasts and 585 scalar scores.
Source, parent inputs, selected weights and portable reassembly verify. This is
paired test reuse with bounded scored-forecast replay, not independent replication,
retraining or unrestricted recursive closure. No learning-control failures occur.
The initial bank plan was superseded before execution to preserve whole-file
batch arithmetic during replay; no failed scientific bank run is hidden.
Science used 165.96875 charged CPU seconds and 170.984152 wall seconds;
verification added 81.921875 charged CPU seconds and 84.835655 wall seconds.
Historical V15 C11/M01 instrument failures remain unchanged.

[Portable scientific evidence](../../results/v18/exploratory-loop/L2a-frozen-bank-1-r2/SCIENTIFIC_MANIFEST.json).

# V21 predictive compression and witnessed history

Does preserving future predictions also preserve witnessed history? The 40-byte predictive head preserves the full-history forecast but retains no order or inspection witnesses. At 1,024 bytes, the hybrid preserves prediction and 23.12% of queried witnesses, versus 25.00% for the ledger alone. This is a constructed-method result; no general hybrid advantage, learned compression or confirmation is established.

Eight reused development coefficient worlds each supply 16 independently drawn makers with 512 observed episodes, 32 fresh endpoint queries and 32 uniformly sampled historical queries. The static eight-state endpoint channel is supplied. Each old episode also contains observed order and inspection bits, independently drawn fair coins that do not affect endpoint generation. This deliberately tests whether predictive equivalence preserves witnessed history; it does not model the relationship between human memory and behavior. Worlds are the paired uncertainty units, with 4,096 bootstrap resamples. Streams, future cases, historical queries and four byte settings are not independent worlds. No fitted seeds or confirmation worlds are used.

The source-bound [admission](B05_MEMORY_ADMISSION.json) fixes actual serialized per-maker ceilings at 256, 1,024, 4,096 and 16,384 bytes. Raw history stores each recent episode in 24 bytes including a 16-byte source digest. The compact ledger needs eight bytes per episode and reconstructs that source link from the public case identity and episode index. Both include an eight-byte header. The full-history maker posterior needs 72 bytes; its four-probability endpoint head needs 40. The hybrid stores the posterior and as many recent compact records as fit. Every retained record is selected before the randomly located history query is used.

The table fixes the 1,024-byte ceiling. Rows are the five stored representations. Bytes count actual payload, future loss is uncapped natural-log loss, future Brier is the sum of squared errors across the four endpoint outcomes, and history loss scores the four order/inspection combinations. Witness coverage is the fraction of historical queries whose exact record remains. The last columns report separate order and inspection accuracy. Lower proper losses are better.

| Representation | Bytes | Future loss | Future Brier | History loss | Witness coverage | Order correct | Inspection correct |
|---|---:|---:|---:|---:|---:|---:|---:|
| Raw history | 1016 | 1.20218 | 0.65395 | 1.27088 | 8.33% | 53.15% | 54.35% |
| Maker posterior | 72 | 1.19353 | 0.64941 | 1.38629 | 0.00% | 48.66% | 49.83% |
| Predictive head | 40 | 1.19353 | 0.64941 | 1.38629 | 0.00% | 48.66% | 49.83% |
| Episode ledger | 1024 | 1.19555 | 0.65049 | 1.03972 | 25.00% | 61.84% | 63.01% |
| Predictor plus ledger | 1024 | 1.19353 | 0.64941 | 1.06578 | 23.12% | 61.04% | 61.99% |

The predictive head, maker posterior and hybrid produce the same future distribution at every ceiling. Endpoint-identical twins with different historical witnesses have identical predictive states: this is a finite counterexample to equating predictive memory with historical memory. Without a retained record the two independent witness bits receive their correct uniform prior, giving history loss log(4) = 1.38629 nats; chance-level realized bit accuracies are sampling variation. The 40-byte head also dominates the 72-byte posterior for these endpoint queries alone. Other future questions could require the posterior; none was tested here.

The next table varies the byte ceiling. Columns give queried-witness coverage for raw records, compact records and the hybrid. The predictive-only representations retain zero witnesses at every ceiling.

| Byte ceiling | Raw history | Compact ledger | Hybrid |
|---|---:|---:|---:|
| 256 | 1.73% | 6.18% | 4.54% |
| 1024 | 8.33% | 25.00% | 23.12% |
| 4096 | 33.91% | 99.95% | 98.49% |
| 16384 | 100.00% | 100.00% | 100.00% |

The hybrid trades eight compact records for its 64-byte posterior. At 1,024 bytes it keeps 119 records versus 127 for the ledger and 42 for raw history. At 4,096 bytes those counts are 503, 511 and 170. Once all 512 records fit, raw history, ledger and hybrid have identical predictions and perfect historical recovery, at 12,296, 4,104 and 4,168 bytes respectively. The ledger then dominates the hybrid's storage for these targets. At 4,096 bytes the ledger's realized future loss is slightly lower than full-history prediction; that small finite-sample reversal is retained, not called a better information ceiling.

The contrast table compares the hybrid against each rival at 1,024 bytes. Differences are hybrid minus rival; negative favors the hybrid. Brackets are descriptive paired-world 95% intervals, not multiplicity-adjusted confirmation. Historical gain against predictive-only memory coexists with worse historical coverage than the compact ledger.

| Rival | Future log-loss difference | Historical log-loss difference |
|---|---:|---:|
| Raw history | -0.00866 [-0.01289, -0.00486] | -0.20510 [-0.21796, -0.19393] |
| Maker posterior | +0.00000 [+0.00000, +0.00000] | -0.32051 [-0.33371, -0.30799] |
| Predictive head | +0.00000 [+0.00000, +0.00000] | -0.32051 [-0.33371, -0.30799] |
| Episode ledger | -0.00202 [-0.00531, +0.00046] | +0.02606 [+0.02200, +0.02978] |

All byte limits, separate target losses, full Brier, confidence errors, bit accuracies, paired intervals, record counts and readable cases are in the [complete memory record](../../../results/v21/memory/MEMORY.json). Every reader is native-only: no native-to-shift change is available. Distinct episodes enter the posterior once; this consumer has no copied-evidence product-of-factors arm. Repeated predictive states across budgets are controls, not extra evidence. Earlier learned duplicate-removal controls retain their separate source bindings.

The shared exact endpoint channel, codec, schema and case identity are outside the per-maker payload charge. Raw-record digest overhead is an encoding choice; compact ledger performance is not an optimal compression theorem. Supplied channel knowledge is distinct from support privilege, uncertainty about an unknown law, fit capacity and validation convergence. This exact consumer fits no trainable model and inherits no original-world headroom. Historical aliasing is planted independently of the predictive channel. Learned heads, dynamic changes and query-aware retention remain conditional. Miniature — architecture untested.

All 246 immutable complete packets bind to their actual plans, source capsules, archives, acceptance and environment. Every retained file and 9,140,224 forecast scores verify, reusing 244 exact prior score receipts. B05 contributes 163,840 original forecasts. Its full frozen scientific replay matches 3,137 files, excluding only CPU timing in SUMMARY. Independent elementary endpoint channels, count-likelihood maker posteriors and byte parsing reconstruct all forecasts within 1.06e-13. All 2,560 payloads, 393,216 retained source links and 1,024 predictive twin controls verify. One scientific block is accepted, bringing the campaign total to 82; execution/numerical counts are not counts of findings. [Validity](../../../results/v21/memory/VALIDITY.json).

The original consumer's measured CPU is 21.79688 seconds including generation, independent checking and export. These shared counters do not establish per-reader latency or peak working-memory efficiency. The first independent review stopped on a 1.29e-12 maker-posterior discrepancy after 512 sequential log-likelihood additions. A separate 60-decimal-digit count reference bounds this within 0.52% of the standard sequential-summation forward-error bound; the strict 1e-12 forecast check still passes. The failed review and its CPU remain retained. B05's pre-dispatch schema repair remains its only family repair; no scientific source, score, acceptance criterion or executed identity changed in this review.

The next independent B06 branch is implemented and admitted under [its frozen contract](B06_FACTORS_ADMISSION.json). It separates binary knowledge, skill and permission while holding the goal fixed in paired counterfactuals. Joint inference is matched to an exact direct lookup table under identical supplied information; learned direct fitting remains conditional. Three focused controls and two complete 27-file engineering consumers pass. A one-world full-grid pilot costs 0.31250 CPU seconds, forecasting 7.50 for original plus full replay with 50% margin, under a 900-second aggregate cap. Scientific acceptance awaits that distinct full-replay event.

E04, candidate-specific controls and omission, decisive-selection and retraction fixtures remain useful conditional work. No new F03 category or live Sounding Line inference is claimed. The fixed twelve-hour checkpoint and discovery, confirmation, science-cutoff and Friday reporting clocks remain unchanged. The campaign is active; the finite forest is not exhausted.

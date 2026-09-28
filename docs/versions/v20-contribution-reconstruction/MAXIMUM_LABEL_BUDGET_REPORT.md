# V20 maximum-label-budget review

We tested whether 131,072 training labels give the structured reader an advantage across retained histories and evidence tiers. All five new context comparisons and fourteen other evidence diagnostics fail the declared comparison rule. All 8,716,288 forecasts verify, with no exact-zero truth probabilities. These are constructed-method results; confirmation remains untouched.

Each block uses the same 32 discovery coefficient worlds, eight training worlds, one fixed fit seed, seven readers and 2,048 test queries per world. Histories contain 0, 1, 4, 16 or 64 past episodes. The 64-episode complete-record match remains pending. Complete observation still withholds private aims and leaves inspection latent. Only the context tier is primary; the other tiers cannot replace it.

Capped joint log loss is the negative natural logarithm of the saved probability on the true seven-part answer, floored at one trillionth; lower is better. It is not an uncapped proper score. The comparison rule requires improvement of at least 0.02 natural-log units over both direct rivals and the legal template, a positive lower descriptive whole-world interval, and no increase in confidently wrong attribution.

Table: rows identify evidence access and retained episode count, all at 131,072 training labels. Columns give mean capped joint log loss across the same 32 worlds for each reader.

| Evidence | Episodes | Structured | Frequencies | Joint neural | Independent bits | Legal template | Oracle |
|---|---:|---:|---:|---:|---:|---:|---:|
| Permitted context | 0 | 3.59240 | 3.50774 | 3.57983 | 3.60026 | 3.58216 | 3.40445 |
| Permitted context | 1 | 3.38328 | 3.66874 | 3.36373 | 25.16057 | 3.37592 | 3.13541 |
| Permitted context | 4 | 3.04916 | 3.22021 | 3.02506 | 23.98892 | 3.08879 | 2.72526 |
| Permitted context | 16 | 2.85144 | 2.80166 | 2.75115 | 21.49221 | 2.73679 | 2.46121 |
| Permitted context | 64 | 3.14760 | 2.58608 | 2.59458 | 13.71546 | 2.52141 | 2.39208 |
| Artifact only | 0 | 3.96283 | 3.92015 | 3.96064 | 4.04266 | 4.14532 | 3.84758 |
| Sparse record | 0 | 3.24591 | 3.16110 | 3.20657 | 3.24023 | 3.23565 | 3.05794 |
| Complete record | 0 | 1.65528 | 1.61453 | 1.63225 | 1.68365 | 1.64786 | 1.56031 |
| Artifact only | 1 | 3.80115 | 3.71093 | 3.79035 | 3.87165 | 3.99116 | 3.62277 |
| Sparse record | 1 | 3.03753 | 3.50755 | 2.98497 | 3.11304 | 3.02941 | 2.78890 |
| Complete record | 1 | 1.43886 | 2.64859 | 1.38411 | 13.34702 | 1.39966 | 1.26393 |
| Artifact only | 4 | 3.57905 | 4.20258 | 3.54592 | 21.74921 | 3.83026 | 3.26405 |
| Sparse record | 4 | 2.70443 | 2.75744 | 2.63095 | 21.92298 | 2.74228 | 2.37875 |
| Complete record | 4 | 1.16243 | 1.08406 | 1.06925 | 14.83230 | 1.08387 | 0.91450 |
| Artifact only | 16 | 3.58154 | 3.60144 | 3.47418 | 24.29716 | 3.72986 | 2.96788 |
| Sparse record | 16 | 2.51015 | 2.38483 | 2.33204 | 21.45455 | 2.39028 | 2.11470 |
| Complete record | 16 | 0.91755 | 0.79973 | 0.80248 | 11.51266 | 0.79957 | 0.72226 |
| Artifact only | 64 | 4.01876 | 3.54452 | 3.47325 | 19.87340 | 3.68975 | 2.79698 |
| Sparse record | 64 | 2.78414 | 2.16928 | 2.17126 | 20.35624 | 2.17490 | 2.04557 |

Table: each row pairs the structured reader with a required rival in the five primary context conditions. Improvement is rival minus structured capped loss; positive favors structured reading. Intervals are descriptive 95% intervals from 4,096 whole-world bootstrap samples. Confident wrong increase is structured minus rival wrong attribution at confidence of at least 90%.

| Episodes | Rival | Improvement | World interval | Confident wrong increase |
|---:|---|---:|---|---:|
| 0 | Frequencies | -0.08466 | [-0.09182, -0.07725] | 0.00000 |
| 0 | Joint neural | -0.01257 | [-0.02028, -0.00482] | 0.00000 |
| 0 | Legal template | -0.01024 | [-0.02436, 0.00372] | 0.00000 |
| 1 | Frequencies | 0.28546 | [0.27060, 0.30024] | -0.00188 |
| 1 | Joint neural | -0.01955 | [-0.02879, -0.00994] | 0.00000 |
| 1 | Legal template | -0.00737 | [-0.02421, 0.00900] | 0.00000 |
| 4 | Frequencies | 0.17105 | [0.13941, 0.20398] | -0.00183 |
| 4 | Joint neural | -0.02411 | [-0.04148, -0.00583] | 0.00000 |
| 4 | Legal template | 0.03963 | [0.00727, 0.07186] | 0.00000 |
| 16 | Frequencies | -0.04978 | [-0.09169, -0.00633] | 0.00000 |
| 16 | Joint neural | -0.10029 | [-0.13128, -0.06866] | 0.00000 |
| 16 | Legal template | -0.11465 | [-0.15844, -0.07092] | 0.00000 |
| 64 | Frequencies | -0.56153 | [-0.66560, -0.45997] | 0.00026 |
| 64 | Joint neural | -0.55303 | [-0.65577, -0.45289] | 0.00026 |
| 64 | Legal template | -0.62619 | [-0.72828, -0.52553] | 0.00026 |

No condition passes all three required rival comparisons. Fits are nested within shared training worlds; queries and repeated fits are not independent worlds. These intervals describe coefficient sensitivity conditional on this training allocation. They do not establish that every structured method fails. All fourteen nonprimary comparisons and secondary metrics are retained in the [aggregates](../../../results/v20/max-budget-wave-1/AGGREGATES.json).

No saved truth probability is exactly zero. 13 method-condition arms contain positive probabilities below the capped log-score floor; every fraction is retained in the [zero-mass check](../../../results/v20/max-budget-wave-1/ZERO_MASS_CHECK.json). These remain separate from previously documented numerical underflow and forced omission. Joint squared probability loss, a proper score, accuracy, component scores and confident wrong attribution were independently checked; no secondary replaces the failed primary.

All completion, plan, frozen source, environment, fit/decode costs and blind/evaluator roles verify. Only reader/ is blind input. Full scientific opening replays include G1, G4, G5 and G6; those and later scoring-repair, history, evidence-tier and interpretation replays are reused by immutable hash. No new implementation family, repair or changed scientific interpretation requires another replay. Same-host validation does not establish fresh-installation or cross-platform reproducibility.

These blocks bring the verified total to 144. Review used 285.96875 CPU seconds. All scientific attempts, review, documentation and a conservative 180-second control-plane allowance are charged inside the original 90-hour ceiling, with 18 hours protected. All prior failed sources and attempts remain retained.

Measured remaining demand was 22.14 hours, recalibrated to 22.89 hours before refill. Forty predeclared matched shift comparisons restore 33.88 hours across 238 discovery blocks. The 131072-label native comparisons fail in all five context histories and fourteen completed artifact/sparse/complete diagnostics. The earlier lower-budget native curves also fail, while longer histories introduce numerical concentration. Admit the forty frozen first-seed, 131072-label matched execution-law and presentation-shift comparisons across histories 0, 1, 4, 16 and 64 and all four evidence tiers. They test whether the native large-label limitation persists under the two declared distribution shifts, anchored by native matches and already admitted lower-budget shift controls. The last native complete-record 64-history match remains admitted and pending. Changed-tool changes the global execution law under both routes, not tool-specific skill. No added seeds, settings, criteria or confirmation exposure.

Twelve frozen diagnostic confirmation blocks remain untouched. The 629 conditional designs have 68.05 forecast hours, not promised execution. Combined finite demand meets the requested wall-window backlog margin; forecasts have not been inflated. Execution, numerical acceptance, failed discovery criteria and untouched confirmation remain distinct. No simulator result establishes human intent.

[Validity](../../../results/v20/max-budget-wave-1/VALIDITY.json), [costs](../../../results/v20/max-budget-wave-1/COSTS.json), [replay reuse](../../../results/v20/max-budget-wave-1/REPLAY_REUSE.json), [forecast](../../../results/v20/max-budget-wave-1/FORECAST.json), [refill](../../../results/v20/max-budget-wave-1/REFILL.json).

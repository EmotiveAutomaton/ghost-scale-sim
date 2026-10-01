# V20 small-label paired long-history shifts

We tested whether structured reading gains an advantage under execution or presentation shifts with 64 retained episodes and 128 or 512 training labels across both registered fits. All eleven completed paired comparisons fail. All 7,798,784 new forecasts verify; 3,004 exact-zero truth probabilities trace to numerical underflow. Five further conditions still lack their second fit. These are constructed-method results; confirmation remains untouched.

Seventeen new blocks contain six first fits at 512 labels and eleven second fits at 128 or 512 labels. The eleven completed pairs cover four context conditions and seven nonprimary evidence-tier conditions. Five 512-label noncontext conditions still await their admitted second fits. The total is 761 verified packets. The earlier maximum-label context candidate and three artifact diagnostics remain separate.

Both fits are averaged inside each of 32 whole coefficient worlds before methods are paired and 4,096 fixed-seed descriptive world-bootstrap resamples are taken. Fits and rows are not independent worlds; intervals remain conditional on the shared training worlds. Capped joint log loss is minus the natural logarithm of probability assigned to the true seven-part answer, with a one-trillionth floor; lower is better. It is not an uncapped proper score. The registered advantage requires at least 0.02 natural-log units against both direct rivals and the legal template, a positive lower world interval and no increase in wrong attribution at 90% confidence.

Table: completed paired comparisons, all with 64 retained episodes. Rows identify training-label count, evidence access and shifted law. Reader columns give capped joint log loss. The last column is structured improvement over joint neural reading and its descriptive whole-world interval; negative means structured is worse. Execution changes the global execution law under both proposal routes; presentation changes the presentation law.

| Labels | Evidence | Shift | Structured | Joint neural | Frequencies | Legal template | Improvement [interval] |
|---:|---|---|---:|---:|---:|---:|---|
| 128 | context | Execution | 23.75443 | 3.69168 | 2.75021 | 2.52400 | -20.06276 [-20.13895, -19.99122] |
| 128 | context | Presentation | 24.13607 | 3.51287 | 2.74648 | 2.52416 | -20.62320 [-20.66573, -20.57722] |
| 512 | context | Execution | 22.05839 | 2.99393 | 2.66974 | 2.52400 | -19.06446 [-19.19242, -18.93375] |
| 512 | context | Presentation | 21.82220 | 2.82916 | 2.67708 | 2.52416 | -18.99304 [-19.18596, -18.80807] |
| 128 | artifact | Execution | 25.48500 | 4.55693 | 3.78620 | 3.69118 | -20.92808 [-20.99989, -20.85792] |
| 128 | artifact | Presentation | 25.11152 | 4.49118 | 3.78210 | 3.68887 | -20.62033 [-20.66157, -20.57933] |
| 128 | sparse | Execution | 22.66450 | 3.33281 | 2.36902 | 2.17592 | -19.33170 [-19.40804, -19.25323] |
| 128 | sparse | Presentation | 22.82846 | 3.08302 | 2.36246 | 2.17520 | -19.74544 [-19.79504, -19.69227] |
| 128 | complete | Execution | 11.43805 | 1.29507 | 0.82100 | 0.69745 | -10.14297 [-10.22599, -10.06009] |
| 128 | complete | Presentation | 11.36416 | 1.08211 | 0.81923 | 0.69765 | -10.28205 [-10.35941, -10.21021] |
| 512 | artifact | Execution | 23.36502 | 3.75457 | 3.65562 | 3.69118 | -19.61045 [-19.74517, -19.48085] |

Every completed paired condition fails the joint criterion. All rival contrasts and confidence checks are retained in the evidence, including any individual contrast that differs from the overall disposition.

Table: secondary scores for structured / joint neural readers in the same paired conditions. Squared probability loss sums squared errors across the 128 probabilities and is proper; lower is better. Accuracy is the fraction whose most-probable joint answer is correct. Confident wrong attribution is the fraction wrong at at least 90% confidence.

| Labels | Evidence | Shift | Squared loss | Accuracy | Confident wrong attribution |
|---:|---|---|---|---|---|
| 128 | context | Execution | 1.76114 / 1.03277 | 0.09728 / 0.09506 | 0.8124008 / 0.0088196 |
| 128 | context | Presentation | 1.77915 / 1.02504 | 0.09830 / 0.09997 | 0.8532028 / 0.0114594 |
| 512 | context | Execution | 1.72188 / 0.97112 | 0.10987 / 0.11324 | 0.7757416 / 0.0015564 |
| 512 | context | Presentation | 1.71827 / 0.94982 | 0.10985 / 0.11150 | 0.7686310 / 0.0008926 |
| 128 | artifact | Execution | 1.81447 / 0.99314 | 0.04292 / 0.04457 | 0.7489395 / 0.0000000 |
| 128 | artifact | Presentation | 1.83172 / 0.98608 | 0.05077 / 0.04863 | 0.8115082 / 0.0000000 |
| 128 | sparse | Execution | 1.70183 / 1.05793 | 0.12964 / 0.14011 | 0.7965851 / 0.0385818 |
| 128 | sparse | Presentation | 1.70536 / 1.03627 | 0.13377 / 0.13998 | 0.8151779 / 0.0255585 |
| 128 | complete | Execution | 0.94948 / 0.75461 | 0.49576 / 0.49484 | 0.4450607 / 0.2195587 |
| 128 | complete | Presentation | 0.94497 / 0.70370 | 0.49776 / 0.49734 | 0.4423141 / 0.1491470 |
| 512 | artifact | Execution | 1.75491 / 0.97219 | 0.05762 / 0.05379 | 0.6738586 / 0.0000000 |

Table: six newly verified first-fit blocks, all with 64 episodes and 512 labels. Columns give capped loss by reader. All fail individually. The execution-shift artifact condition is also paired above; the other five await their second fit, so these are not six additional independent conditions.

| Evidence | Shift | Structured | Joint neural | Frequencies | Legal template |
|---|---|---:|---:|---:|---:|
| artifact | Execution | 23.74869 | 3.83073 | 3.68315 | 3.69118 |
| artifact | Presentation | 23.93384 | 3.73301 | 3.68245 | 3.68887 |
| sparse | Execution | 21.03658 | 2.58703 | 2.27595 | 2.17592 |
| sparse | Presentation | 21.09035 | 2.40363 | 2.27871 | 2.17520 |
| complete | Execution | 11.41595 | 1.04257 | 0.76004 | 0.69745 |
| complete | Presentation | 11.13062 | 0.97046 | 0.75939 | 0.69765 |

All eleven matched native/shift pairs are available. Same-fit training examples are byte-identical. Both current and retained observations shift; every reader receives exact legal support under each law. Native and shifted evaluation rows differ, so comparisons pair only whole worlds. This does not test recovery of an unknown law.

Table: changes in capped loss from each matched native reference, averaging fits inside worlds. Positive values mean the shift worsens loss; the structured interval is a descriptive paired-world interval.

| Labels | Evidence | Shift | Structured change [interval] | Joint neural change | Frequency change | Template change |
|---:|---|---|---|---:|---:|---:|
| 128 | context | Execution | -0.11058 [-0.18522, -0.03135] | 0.01132 | 0.00636 | 0.00259 |
| 128 | context | Presentation | 0.27105 [0.18973, 0.35479] | -0.16749 | 0.00263 | 0.00275 |
| 512 | context | Execution | -0.10285 [-0.17855, -0.02870] | -0.00728 | -0.00132 | 0.00259 |
| 512 | context | Presentation | -0.33904 [-0.44828, -0.22460] | -0.17205 | 0.00601 | 0.00275 |
| 128 | artifact | Execution | -0.16567 [-0.22197, -0.10821] | 0.01424 | 0.00578 | 0.00142 |
| 128 | artifact | Presentation | -0.53916 [-0.62336, -0.45328] | -0.05151 | 0.00168 | -0.00088 |
| 128 | sparse | Execution | -0.10698 [-0.19399, -0.01859] | 0.01171 | 0.00548 | 0.00102 |
| 128 | sparse | Presentation | 0.05698 [-0.02591, 0.14040] | -0.23807 | -0.00109 | 0.00030 |
| 128 | complete | Execution | -0.06568 [-0.17123, 0.03867] | 0.00197 | 0.00245 | -0.00003 |
| 128 | complete | Presentation | -0.13957 [-0.24589, -0.03520] | -0.21099 | 0.00067 | 0.00017 |
| 512 | artifact | Execution | -0.17066 [-0.26471, -0.07950] | -0.00124 | -0.00161 | 0.00142 |

Table: every new block containing exact-zero truth probabilities. All occur in structured artifact reading at 512 labels with 64 retained episodes. The count is out of 65,536 evaluated truths per reader block; the log-odds range compares truth with the modal answer. Every truth remains inside supplied legal support and its finite learned odds underflow when exponentiated. This is a numerical representation failure, distinct from logical candidate omission.

| Packet | Exact-zero truths | Truth-to-mode log-odds range |
|---|---:|---|
| v20-0756-g1 | 1,510 | -778.53852 to -745.33514 |
| v20-0757-g1 | 61 | -770.58339 to -745.14063 |
| v20-0772-g1 | 1,433 | -790.95985 to -745.17941 |

The zero audit also records every below-floor fraction. Capped loss, exact zero mass, proper squared loss and confident errors remain separate. No probabilities or scientific criteria were repaired.

Every forecast, stored secondary score and summary reconstructs. COMPLETE, plan, source archive and environment bindings, fitting/decoding costs and attempted failures reconcile. Only reader/ contains blind input; casebooks and summaries are evaluator artifacts. Execution, numerical acceptance, failed discovery criteria and untouched confirmation remain distinct. Full opening G1/G4/G5/G6, scorer repair, histories, shifts, numerical-underflow interpretation and positive discovery replays are reused by immutable hashes. No new family, repair or interpretation requires a new replay. Validation does not check fresh installation, cross-platform replay, universal reader performance, real-text correspondence or human intent.

Original blocks used 2701.34375 native CPU seconds; numerical review used 160.70312, and paired comparisons/zero diagnosis used 3.76562. Operating, documentary and publication work is charged separately. Exited-service and reviewer-descendant CPU remains imperfectly measured with retained conservative allowances.

Initial publication staging was stopped for excessive CPU cost. Its failed attempt is preserved with a conservative charge of 6213.01609 CPU seconds; scientific source and results were unchanged. Publication resumed with exact-file index assembly and single-thread index settings. The operating review reconciles the added overhead before releasing science.

No new designs are admitted at this boundary. The 77 useful queued blocks provide 3.36 measured hours; their execution/review plus two hours of future operating/publication allowance projects 69.77 discovery CPU hours before this publication. A full-day refill projects 85.06, above the 72-hour ceiling. The 20.64-hour shortfall is explicit; the useful frontier remains unexhausted and the 18-hour reserve stays protected. Reassess residual capacity after publication and before the queue drains.

At October 1 08:00 UTC, twelve frozen diagnostics have first priority, then original V20-C01 across both fits within its inclusive 4,000-second cap, reserve and cutoff. Confirmation and challenge lineages remain untouched. Science stops October 2 04:00 UTC, final preparation is 10:00, and delivery/service closure is 12:00.

[Validity](../../../results/v20/shift-wave-28/VALIDITY.json), [paired fits](../../../results/v20/shift-wave-28/PAIRED_FITS.json), [matched native shifts](../../../results/v20/shift-wave-28/PAIRED_SHIFTS.json), [all scores](../../../results/v20/shift-wave-28/AGGREGATES.json), [zero audit](../../../results/v20/shift-wave-28/ZERO_MASS_AUDIT.json), [replay reuse](../../../results/v20/shift-wave-28/REPLAY_REUSE.json), [costs](../../../results/v20/shift-wave-28/COSTS.json), [forecast](../../../results/v20/shift-wave-28/FORECAST.json).

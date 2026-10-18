# N17: lean, sensitivity and taste

N17 separates lean, sensitivity and taste in its constructed menus. At 100 records, the taste reader improves held-out log loss by 0.10906 nats (95% whole-law interval 0.09491 to 0.12433). False taste claims are 0/2,048 at 20, 50 and 100 records, and 2/2,048 at 250; all frozen no-taste gates pass. All 16,384 cases, complete replay and independent reconstruction verify. Calibration only partly matches Sounding Line, so these record counts are not validated human-data requirements.

The question was whether menu choices distinguish a readiness to accept, responsiveness to quality and a taste direction unrelated to quality. Readers fit only the first half of each maker's records and forecast the second half. Quality coordinates are known in this simulation. The taste claim requires a held-out e-value of at least 20 against the maximum finite-null likelihood, as frozen before discovery.

Each row below averages 64 makers in each of 16 laws. Gain is the reduction in log loss, in nats per held-out choice; negative values mean the larger reader is worse. Detection is the fraction clearing the frozen taste threshold. Sensitivity gain compares the criterion-plus-sensitivity reader with lean alone. Probe gain is the increase in log probability of the true payoff/belief mechanism after both interventions.

| Records | Taste in truth | Lean mechanism | Taste gain | Taste detection | Sensitivity gain | Two-probe gain |
|---:|---|---|---:|---:|---:|---:|
| 20 | absent | payoff | -0.025555 | 0.000% | 0.140329 | 0.063244 |
| 20 | absent | belief | -0.026765 | 0.000% | 0.134101 | 0.002401 |
| 20 | present | payoff | 0.078244 | 3.906% | 0.141992 | 0.056071 |
| 20 | present | belief | 0.082061 | 4.199% | 0.121559 | 0.013975 |
| 50 | absent | payoff | -0.019423 | 0.000% | 0.165890 | 0.053205 |
| 50 | absent | belief | -0.018386 | 0.000% | 0.167467 | 0.018568 |
| 50 | present | payoff | 0.111578 | 28.516% | 0.137783 | 0.055695 |
| 50 | present | belief | 0.101202 | 27.734% | 0.142161 | 0.036607 |
| 100 | absent | payoff | -0.009369 | 0.000% | 0.161360 | 0.043772 |
| 100 | absent | belief | -0.009565 | 0.000% | 0.160169 | 0.018679 |
| 100 | present | payoff | 0.106699 | 51.562% | 0.141803 | 0.057509 |
| 100 | present | belief | 0.111412 | 51.367% | 0.144781 | 0.045085 |
| 250 | absent | payoff | -0.003169 | 0.000% | 0.173907 | 0.052550 |
| 250 | absent | belief | -0.003327 | 0.195% | 0.168218 | 0.038325 |
| 250 | present | payoff | 0.111813 | 80.273% | 0.142351 | 0.046329 |
| 250 | present | belief | 0.112979 | 80.859% | 0.154003 | 0.034696 |

At 100 records the taste-detection rate in nonzero-taste worlds is 51.46%, despite the positive average prediction gain. Positive average gain does not mean every maker's taste is identified. Adding the taste parameter slightly worsens ordinary prediction when taste is absent, as the table shows. The no-taste gate still passes at every record count. Zero observed events do not establish a zero population error rate; the finite-null e-value bound applies only to the registered likelihood family.

Criterion-plus-sensitivity improves held-out prediction over lean alone by 0.15203 nats at 100 records (95% law interval 0.12764 to 0.17716). The full reader's criterion and sensitivity mean-square errors improve over their declared uniform-prior mean estimates by 3.27401 and 0.61486 parameter units squared. These are not comparisons with a population-optimal pooled estimator. Both criterion/sensitivity checks pass the prospective synthetic route to SDT1; SDT1 itself and any engine admission still require their own evidence.

The stakes and base-rate probes gain 0.03979 nats in true-mechanism log probability (95% law interval 0.03484 to 0.04491). Their selective effects on payoff and belief are stipulated in the generator. This supports a constructed probe witness for the owner card, not a demonstrated causal response in people.

Calibration is deliberately visible beside the record-count findings. The following independent **baseline calibration-bank** values precede the separate 16-law robustness redraw. Targets come from different real cohorts/channels; six synthetic actions do not reproduce Sounding Line's handling categories. The reliability bank uses 96 synthetic makers and two independent response draws on the same 256 menus, not the empirical session chronology.

| Quantity | Sounding Line target | Baseline validation | Difference |
|---|---:|---:|---:|
| Within-writer agreement | 0.42177 | 0.42908 | +0.00731 |
| Cross-writer agreement | 0.36450 | 0.37987 | +0.01537 |
| Chance agreement (approximate target) | 0.41000 | 0.34422 | -0.06578 |
| Within-writer Cohen kappa | 0.08908 | 0.12940 | +0.04032 |
| Cross-writer Cohen kappa | 0.02711 | 0.05437 | +0.02726 |
| Acceptance intraclass correlation | 0.09334 | 0.03104 | -0.06230 |
| Acceptance split-half correlation | 0.78026 | 0.87057 | +0.09031 |

Agreement is close; several other targets are not. In particular, the baseline acceptance intraclass correlation is 0.031 rather than 0.093. The additional law redraw varies criterion populations, feature scales and temperature, so baseline calibration is not a claim that every discovery law matches those targets. Transfer of the 100-record result requires a better calibrated real-data design and an available quality instrument.

Verification covers all 16 settings, all 16,384 cases, public/evaluator separation, canonical-ledger transport, frozen parameter definitions, proper scores, complete fresh-process replay, finite-null controls and independent law-block aggregates/intervals. The source-bound checker normalizes integer readout keys through JSON; producer AST is unchanged. Runtime plus independent review charged 936.8125 CPU-seconds. No cells were shed; no V23 confirmation or engine-module admission is claimed.

[Protocol](N17_PROTOCOL.md), [aggregate target provenance](N17_AGGREGATE_TARGETS.json), [summary](../../../results/v23/n17/SUMMARY.json), [proof](../../../results/v23/n17/PROOF.json), [raw hashes](../../../results/v23/n17/RAW_INDEX.json).

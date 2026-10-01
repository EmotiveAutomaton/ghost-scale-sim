# V20 final short-history shift comparisons

We tested whether structured reading gains an advantage under execution or presentation shifts with four or sixteen retained episodes and 2,048 training labels. All twelve paired comparisons fail. All 11,010,048 new forecasts verify; no truth receives exactly zero probability. These are constructed-method results; confirmation remains untouched.

The twenty-four packets contain both registered fits for artifact, sparse-record and complete-record evidence under each shift, at four and sixteen retained episodes. They add twelve nonprimary diagnostics, with 794 packets now verified in 61 named reviews. Both fits are averaged inside each of 32 whole coefficient worlds before methods are paired and 4,096 fixed-seed world-bootstrap resamples are taken. Rows and fits are not independent worlds. Intervals are descriptive and conditional on shared training worlds.

Capped joint log loss is minus the natural logarithm of the probability assigned to the true seven-part answer, with a one-trillionth floor; lower is better. This is not an uncapped proper score. The declared advantage requires an improvement of at least 0.02 natural-log units against each direct rival and the legal template, a positive lower world interval, and no increase in wrong attribution at 90% confidence.

Table: comparisons at 2,048 labels. Each row identifies evidence access, retained episode count and changed evaluation law. Reader columns give capped joint log loss. Improvement is joint neural loss minus structured loss, with its descriptive whole-world interval; negative favors the neural reader. Execution changes the global execution law under both proposal routes.

| Shift | Structured | Joint neural | Frequencies | Legal template | Improvement [interval] |
|---|---:|---:|---:|---:|---|
| Artifact only / 4 episodes / Execution | 4.86348 | 3.63326 | 3.72266 | 3.83271 | -1.23023 [-1.25621, -1.20453] |
| Artifact only / 4 episodes / Presentation | 5.95712 | 4.13570 | 4.77545 | 3.82909 | -1.82141 [-1.84519, -1.79765] |
| Sparse record / 4 episodes / Execution | 3.76069 | 2.72783 | 2.76267 | 2.74603 | -1.03286 [-1.06171, -1.00492] |
| Sparse record / 4 episodes / Presentation | 5.17461 | 3.17020 | 2.75981 | 2.74022 | -2.00441 [-2.02859, -1.98146] |
| Complete record / 4 episodes / Execution | 1.97013 | 1.16617 | 1.10231 | 1.08350 | -0.80397 [-0.82483, -0.78250] |
| Complete record / 4 episodes / Presentation | 2.04762 | 1.14002 | 1.09875 | 1.08038 | -0.90760 [-0.93115, -0.88299] |
| Artifact only / 16 episodes / Execution | 8.48554 | 3.56156 | 3.61933 | 3.73141 | -4.92398 [-4.99895, -4.85215] |
| Artifact only / 16 episodes / Presentation | 8.98457 | 3.85406 | 3.62269 | 3.72836 | -5.13050 [-5.21347, -5.04769] |
| Sparse record / 16 episodes / Execution | 6.29484 | 2.46141 | 2.41012 | 2.39389 | -3.83343 [-3.90735, -3.75836] |
| Sparse record / 16 episodes / Presentation | 7.67053 | 2.68539 | 2.40728 | 2.38982 | -4.98514 [-5.04654, -4.92292] |
| Complete record / 16 episodes / Execution | 3.21756 | 0.88848 | 0.81796 | 0.80124 | -2.32908 [-2.36137, -2.29851] |
| Complete record / 16 episodes / Presentation | 3.11240 | 0.87004 | 0.81696 | 0.80088 | -2.24236 [-2.28657, -2.19708] |

All twelve conditions fail the all-rival rule. Every required rival contrast and confidence check remains in the paired evidence.

Table: structured / joint neural secondary scores in the same conditions. Squared probability loss sums squared errors over 128 outcomes and is proper; lower is better. Accuracy is the fraction with the correct most-probable joint answer. Confident wrong attribution is the fraction wrong at at least 90% confidence.

| Shift | Squared loss | Accuracy | Confident wrong attribution |
|---|---|---|---|
| Artifact only / 4 episodes / Execution | 1.00858 / 0.96324 | 0.05894 / 0.05803 | 0.0000000 / 0.0000000 |
| Artifact only / 4 episodes / Presentation | 1.04976 / 0.97719 | 0.04733 / 0.04755 | 0.0000000 / 0.0000000 |
| Sparse record / 4 episodes / Execution | 1.05185 / 0.91437 | 0.10960 / 0.11081 | 0.0166092 / 0.0000000 |
| Sparse record / 4 episodes / Presentation | 1.13687 / 0.93319 | 0.09393 / 0.09745 | 0.0227432 / 0.0000000 |
| Complete record / 4 episodes / Execution | 0.88296 / 0.68085 | 0.38109 / 0.37865 | 0.1396713 / 0.0095749 |
| Complete record / 4 episodes / Presentation | 0.89755 / 0.67041 | 0.38625 / 0.38274 | 0.1593475 / 0.0040741 |
| Artifact only / 16 episodes / Execution | 1.17651 / 0.96347 | 0.06513 / 0.05473 | 0.0145340 / 0.0000000 |
| Artifact only / 16 episodes / Presentation | 1.23182 / 0.97091 | 0.05868 / 0.05476 | 0.0310974 / 0.0000000 |
| Sparse record / 16 episodes / Execution | 1.29216 / 0.89878 | 0.13289 / 0.13239 | 0.1848907 / 0.0000000 |
| Sparse record / 16 episodes / Presentation | 1.35475 / 0.91115 | 0.12246 / 0.12695 | 0.2189484 / 0.0000000 |
| Complete record / 16 episodes / Execution | 0.91910 / 0.60136 | 0.46677 / 0.46639 | 0.3506622 / 0.0104294 |
| Complete record / 16 episodes / Presentation | 0.91239 / 0.59233 | 0.46889 / 0.46687 | 0.3460922 / 0.0026169 |

All twelve matched native/shift comparisons are available, with identical training bytes for each same-fit comparison. Exact legal support is supplied under each law. Evaluation rows differ; only whole worlds are paired. This does not test unknown-law recovery.

Table: change in capped loss from the matched native reference, averaging fits within worlds. Positive means the shift worsens loss; the structured interval is descriptive.

| Shift | Structured change [interval] | Joint neural change | Frequency change | Template change |
|---|---|---:|---:|---:|
| Artifact only / 4 episodes / Execution | -0.24459 [-0.28716, -0.20419] | 0.00013 | -1.29578 | 0.00244 |
| Artifact only / 4 episodes / Presentation | 0.84904 [0.80645, 0.89015] | 0.50258 | -0.24299 | -0.00117 |
| Sparse record / 4 episodes / Execution | -0.31574 [-0.34094, -0.29206] | 0.00195 | 0.00384 | 0.00375 |
| Sparse record / 4 episodes / Presentation | 1.09817 [1.05933, 1.13772] | 0.44432 | 0.00098 | -0.00207 |
| Complete record / 4 episodes / Execution | -0.21139 [-0.23945, -0.18466] | -0.00122 | -0.00023 | -0.00037 |
| Complete record / 4 episodes / Presentation | -0.13390 [-0.15516, -0.11185] | -0.02737 | -0.00380 | -0.00349 |
| Artifact only / 16 episodes / Execution | -0.60012 [-0.71165, -0.49697] | -0.00376 | 0.00218 | 0.00155 |
| Artifact only / 16 episodes / Presentation | -0.10109 [-0.21740, 0.00836] | 0.28874 | 0.00555 | -0.00150 |
| Sparse record / 16 episodes / Execution | -0.55032 [-0.61182, -0.49049] | -0.00078 | 0.00426 | 0.00362 |
| Sparse record / 16 episodes / Presentation | 0.82537 [0.71051, 0.93693] | 0.22320 | 0.00142 | -0.00045 |
| Complete record / 16 episodes / Execution | -0.25115 [-0.29821, -0.20506] | -0.00539 | 0.00199 | 0.00167 |
| Complete record / 16 episodes / Presentation | -0.35632 [-0.39679, -0.31327] | -0.02383 | 0.00099 | 0.00131 |

Every saved truth probability was inspected; none is exactly zero. Fractions below the log-loss floor are recorded separately for every arm. This does not change earlier numerical-underflow or logical-omission findings. Every forecast, secondary score and summary reconstructs. COMPLETE, frozen archive, plan, environment, fitting/decoding costs and blind/evaluator separation verify. Only reader/ is blind input; summaries and casebooks contain evaluator truth.

Complete opening G1/G4/G5/G6, repaired scoring, history, shifted-law, underflow-interpretation and positive reference replays are reused by immutable hashes. The implementation and interpretation have not changed, so no new full replay is required. Validation does not establish fresh-installation or cross-platform replay, universal reader superiority, real-text process correspondence or human intent.

Original blocks used 2065.60938 native CPU seconds; independent numerical review used 234.84375, and paired comparisons/zero audit used 1.65625. Operating, documentary and publication work is charged separately. Historical exited-service and descendant CPU uncertainty remains recorded.

All 24 final admitted discovery packets are complete and reviewed. No discovery remains eligible; the 44 prior resource deferrals retain their frozen plans and raw evidence. The useful conditional forest remains unexecuted, with no exhaustion claim. Current charged work plus two hours of future operating allowance projects 71.435 discovery CPU hours before publication, below the 72-hour ceiling. No new work is admitted; the 18-hour reserve stays protected. The existing supervisor will wait for frozen confirmation.

Discovery freezes October 1 at 06:00 UTC. The twelve original diagnostics open at 08:00, followed by both original frozen V20-C01 fits, independent review and one full replay within the inclusive 4,000-second bound. Confirmation remains untouched; diagnostic confirmation cannot confirm the failed native primary. Science stops at 11:30 UTC; delivery targets 12:00, latest 13:00, with exact owned-service closure verified separately. Final synthesis/export drafts are refreshed with confirmation explicitly pending.

[Validity](../../../results/v20/shift-wave-31/VALIDITY.json), [paired fits](../../../results/v20/shift-wave-31/PAIRED_FITS.json), [matched shifts](../../../results/v20/shift-wave-31/PAIRED_SHIFTS.json), [all scores](../../../results/v20/shift-wave-31/AGGREGATES.json), [zero audit](../../../results/v20/shift-wave-31/ZERO_MASS_AUDIT.json), [replay reuse](../../../results/v20/shift-wave-31/REPLAY_REUSE.json), [costs](../../../results/v20/shift-wave-31/COSTS.json), [forecast](../../../results/v20/shift-wave-31/FORECAST.json), [confirmation readiness](../../../results/v20/shift-wave-31/CONFIRMATION_READINESS.json).

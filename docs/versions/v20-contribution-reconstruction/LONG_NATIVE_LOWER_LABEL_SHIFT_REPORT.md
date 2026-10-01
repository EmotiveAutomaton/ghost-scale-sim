# V20 long native histories and lower-label context shifts

We tested whether structured reading gains an advantage with long native histories or lower-label shifted context. All seven native comparisons fail across both fits, and all thirteen shifted-context comparisons fail in their first fit. All 9,175,040 new forecasts verify; 94,777 exact-zero truth probabilities come from numerical underflow. These are constructed-method results; confirmation remains untouched.

The seven new native second-fit blocks comprise complete-record evidence with 64 past episodes and 8,192 labels, plus artifact-only, sparse-record and complete-record evidence with 256 past episodes at 128 and 512 labels. The thirteen new first-fit context blocks test both execution and presentation shifts at zero and one past episode with 128, 512 and 2,048 labels, then the execution shift at four episodes and 128 labels. All original fits and native counterparts are hash-bound verified evidence. The total is 607 verified packets.

For repeated native fits, average both fits inside each of 32 whole coefficient worlds before pairing readers and applying 4,096 fixed-seed descriptive world-bootstrap resamples. The shifted-context results use one registered fit and pair each shifted world with its native counterpart; evaluation rows differ and are not paired. Neither rows nor fits are independent worlds. Uncertainty remains conditional on shared training worlds.

Capped joint log loss is minus the natural logarithm of probability assigned to the true seven-part answer, with probabilities floored at one trillionth; lower is better. The declared advantage requires at least 0.02 natural-log units over both direct rivals and the legal template, a positive lower world interval and no increase in wrong attribution at 90% confidence. Exact zero mass is reported separately; this capped loss is not an uncapped proper score.

Table: native comparisons averaged over both fits. Rows identify history length, label count and evidence tier. Reader columns give capped loss; the final column gives structured improvement over the joint neural reader and its descriptive world interval. Negative improvement means worse structured reading.

| History | Labels | Evidence | Structured | Joint neural | Frequencies | Legal template | Improvement [interval] |
|---:|---:|---|---:|---:|---:|---:|---|
| 64 | 8,192 | Complete | 2.55527 | 0.72189 | 0.70127 | 0.69748 | -1.83338 [-1.86525, -1.80218] |
| 256 | 128 | Artifact | 25.97721 | 4.60079 | 3.77946 | 3.68868 | -21.37642 [-21.45830, -21.29666] |
| 256 | 128 | Sparse | 23.32859 | 3.12043 | 2.33350 | 2.14531 | -20.20816 [-20.27477, -20.14122] |
| 256 | 128 | Complete | 11.98631 | 1.40800 | 0.81387 | 0.69315 | -10.57832 [-10.65078, -10.51214] |
| 256 | 512 | Artifact | 25.42640 | 3.72892 | 3.65614 | 3.68868 | -21.69748 [-21.78907, -21.59905] |
| 256 | 512 | Sparse | 22.85538 | 2.45789 | 2.23674 | 2.14531 | -20.39749 [-20.51112, -20.28243] |
| 256 | 512 | Complete | 13.05342 | 0.94044 | 0.75477 | 0.69315 | -12.11298 [-12.18121, -12.04807] |

Every required rival wins all seven native comparisons, with every loss-improvement interval below zero. There are 0 complete matched native/shift comparisons across both fits; 14 lack both verified shifted fits. The availability receipt records exact counterparts; no unmatched-budget or unmatched-fit estimate substitutes for missing evidence.

Table: shifted-context first-fit results. Rows identify history, labels and the shifted law. Reader columns give capped loss; improvement compares structured with joint neural reading. Execution means the global execution law changes under both proposal routes; presentation means a changed presentation law.

| History | Labels | Shift | Structured | Joint neural | Frequencies | Legal template | Improvement [interval] |
|---:|---:|---|---:|---:|---:|---:|---|
| 0 | 128 | Execution | 6.76041 | 5.35958 | 4.61355 | 3.58538 | -1.40083 [-1.43101, -1.37000] |
| 0 | 128 | Presentation | 6.85462 | 5.41525 | 5.52766 | 3.58355 | -1.43938 [-1.48635, -1.39505] |
| 0 | 512 | Execution | 5.06279 | 4.03575 | 5.10533 | 3.58538 | -1.02704 [-1.04997, -1.00446] |
| 0 | 512 | Presentation | 5.24626 | 4.10980 | 5.69282 | 3.58355 | -1.13646 [-1.16200, -1.11077] |
| 0 | 2,048 | Execution | 3.80557 | 3.65674 | 6.40600 | 3.58538 | -0.14883 [-0.15745, -0.14026] |
| 0 | 2,048 | Presentation | 4.28991 | 3.84187 | 5.03524 | 3.58355 | -0.44804 [-0.45938, -0.43658] |
| 1 | 128 | Execution | 9.89219 | 5.50017 | 3.59104 | 3.38051 | -4.39201 [-4.45615, -4.32595] |
| 1 | 128 | Presentation | 10.15810 | 5.47793 | 3.81008 | 3.37683 | -4.68018 [-4.75721, -4.60764] |
| 1 | 512 | Execution | 6.47100 | 3.88880 | 3.55960 | 3.38051 | -2.58220 [-2.63287, -2.53291] |
| 1 | 512 | Presentation | 6.68203 | 3.99638 | 4.14790 | 3.37683 | -2.68565 [-2.72960, -2.64043] |
| 1 | 2,048 | Execution | 3.84247 | 3.44902 | 3.51629 | 3.38051 | -0.39345 [-0.40865, -0.37839] |
| 1 | 2,048 | Presentation | 4.54430 | 3.69361 | 4.61658 | 3.37683 | -0.85070 [-0.86866, -0.83381] |
| 4 | 128 | Execution | 15.79447 | 4.80403 | 3.28933 | 3.09412 | -10.99044 [-11.10481, -10.88169] |

All thirteen shifted-context conditions fail the required joint comparison in their first fit. These are lower-label sensitivity comparisons, not new confirmation candidates. Training examples are byte-identical to each matched native reference. Both current and retained observations use the shifted law, and every reader receives exact common legal support under that law. This does not test recovery of an unknown law.

Table: secondary scores for structured / joint neural readers. Squared probability loss is the sum of squared joint-probability errors and is proper; lower is better. Accuracy is the fraction whose most-probable joint answer is correct. Confident wrong attribution is the fraction wrong at at least 90% confidence. Rows separate two-fit native conditions from first-fit shifted conditions.

| History | Labels | Evidence/law | Squared loss | Accuracy | Confident wrong attribution |
|---:|---:|---|---|---|---|
| 64 | 8,192 | complete / native | 0.85864 / 0.52390 | 0.50051 / 0.49978 | 0.3396683 / 0.0000000 |
| 256 | 128 | artifact / native | 1.88394 / 0.99038 | 0.04514 / 0.03320 | 0.8999100 / 0.0000000 |
| 256 | 128 | sparse / native | 1.71669 / 0.98616 | 0.13328 / 0.13364 | 0.8375702 / 0.0293350 |
| 256 | 128 | complete / native | 0.94746 / 0.73231 | 0.49937 / 0.49850 | 0.4460526 / 0.2122421 |
| 256 | 512 | artifact / native | 1.85480 / 0.96925 | 0.05984 / 0.04866 | 0.8875122 / 0.0000000 |
| 256 | 512 | sparse / native | 1.69228 / 0.92334 | 0.14714 / 0.15190 | 0.8273621 / 0.0020065 |
| 256 | 512 | complete / native | 0.99421 / 0.64437 | 0.50025 / 0.50149 | 0.4940414 / 0.0788956 |
| 0 | 128 | context / changed-tool | 1.13016 / 1.10516 | 0.04059 / 0.03911 | 0.0164337 / 0.0023804 |
| 0 | 128 | context / presentation-shift | 1.12599 / 1.08554 | 0.04764 / 0.04866 | 0.0080872 / 0.0037079 |
| 0 | 512 | context / changed-tool | 1.01559 / 0.98598 | 0.05211 / 0.04384 | 0.0023804 / 0.0000000 |
| 0 | 512 | context / presentation-shift | 1.01364 / 0.98182 | 0.04930 / 0.04816 | 0.0000000 / 0.0000000 |
| 0 | 2,048 | context / changed-tool | 0.97336 / 0.96145 | 0.04649 / 0.04912 | 0.0000000 / 0.0000000 |
| 0 | 2,048 | context / presentation-shift | 0.97142 / 0.95938 | 0.05426 / 0.05527 | 0.0000000 / 0.0000000 |
| 1 | 128 | context / changed-tool | 1.29784 / 1.18552 | 0.05121 / 0.05478 | 0.0754700 / 0.0228119 |
| 1 | 128 | context / presentation-shift | 1.31519 / 1.18024 | 0.05510 / 0.05817 | 0.0788116 / 0.0167236 |
| 1 | 512 | context / changed-tool | 1.08690 / 0.99308 | 0.06277 / 0.06020 | 0.0136414 / 0.0008087 |
| 1 | 512 | context / presentation-shift | 1.09114 / 0.99041 | 0.05861 / 0.05431 | 0.0050507 / 0.0000000 |
| 1 | 2,048 | context / changed-tool | 0.98414 / 0.95405 | 0.06155 / 0.06461 | 0.0001526 / 0.0000000 |
| 1 | 2,048 | context / presentation-shift | 0.99441 / 0.95458 | 0.05997 / 0.06110 | 0.0000305 / 0.0000000 |
| 4 | 128 | context / changed-tool | 1.58090 / 1.18898 | 0.06400 / 0.07173 | 0.3854675 / 0.0269623 |

Table: every newly reviewed reader with truth probability below the log-score floor. Percentages use 65,536 truths per reader, which are not independent worlds. Exact zeros are separate from nonzero below-floor probabilities.

| History | Labels | Evidence/law | Reader | Below floor | Exact zero |
|---:|---:|---|---|---:|---:|
| 64 | 8,192 | complete / native | Independent bits | 0.16937% | 0.00000% |
| 64 | 8,192 | complete / native | Structured | 0.20752% | 0.00000% |
| 256 | 128 | artifact / native | Independent bits | 22.51434% | 0.00000% |
| 256 | 128 | artifact / native | Structured | 89.91852% | 58.28400% |
| 256 | 128 | sparse / native | Structured | 81.67267% | 41.06293% |
| 256 | 128 | complete / native | Structured | 43.52112% | 15.55634% |
| 256 | 512 | artifact / native | Independent bits | 35.63385% | 0.00000% |
| 256 | 512 | artifact / native | Structured | 88.98315% | 15.49988% |
| 256 | 512 | sparse / native | Independent bits | 0.00610% | 0.00000% |
| 256 | 512 | sparse / native | Structured | 79.73633% | 10.61249% |
| 256 | 512 | complete / native | Structured | 45.59631% | 3.60260% |
| 1 | 128 | context / changed-tool | Independent bits | 78.87421% | 0.00000% |
| 1 | 128 | context / presentation-shift | Independent bits | 78.84369% | 0.00000% |
| 1 | 512 | context / changed-tool | Independent bits | 77.95410% | 0.00000% |
| 1 | 512 | context / presentation-shift | Independent bits | 79.08783% | 0.00000% |
| 1 | 2,048 | context / changed-tool | Independent bits | 77.93732% | 0.00000% |
| 1 | 2,048 | context / presentation-shift | Independent bits | 82.30743% | 0.00000% |
| 4 | 128 | context / changed-tool | Independent bits | 90.34424% | 0.00000% |
| 4 | 128 | context / changed-tool | Structured | 28.76892% | 0.00000% |

All 94,777 exact-zero truths remain inside supplied legal support. Independently reconstructed finite truth-to-modal log odds underflow on exponentiation for every zero. This retains the established numerical-underflow interpretation, distinct from forced candidate omission or logical impossibility. The zero audit gives counts and log-odds ranges per affected packet.

Every stored forecast, secondary score and summary reconstructs. COMPLETE, plans, frozen source archives, environment, fit/decode costs and attempted failures reconcile. Only reader/ is blind input; casebooks and summaries are evaluator artifacts. Execution, numerical acceptance, discovery criteria and untouched confirmation remain distinct. Full opening G1/G4/G5/G6, score-repair, history, shift, underflow-interpretation and both-fit positive replays are reused by immutable hashes. There is no new implementation family, repair or scientific interpretation requiring another replay. Validation does not establish fresh installation, cross-platform replay, universal reader performance, real-text process correspondence or human intent.

Original blocks used 3428.65625 native CPU seconds; numerical review used 206.28125 and the paired-comparison/zero audit 14.95312. Forecasts, documents, publication preparation and operating work are charged separately. Retained shell/native-inspection failures did not alter scientific source or criteria. Historical exited-service and reviewer-descendant CPU remains imperfectly measured with conservative allowances.

No designs were admitted. The 163 pending useful discovery blocks have a measured forecast of 12.65 hours. Restoring a measured day projects 78.38 discovery CPU hours, above 72 including required reviews and two hours of future operating/publication allowance. The 11.35-hour shortfall is explicit; the useful frontier remains unexhausted and the separate 18-hour reserve protected.

At October 1 08:00 UTC, the twelve frozen diagnostics have first priority, followed by original V20-C01 across both fits within its inclusive 4,000-second cap, reserve and cutoff. Confirmation and challenge lineages remain untouched. Science ends October 2 04:00 UTC, final preparation is 10:00, and delivery/service closure is 12:00.

[Validity](../../../results/v20/native-reference-wave-7/VALIDITY.json), [paired fits](../../../results/v20/native-reference-wave-7/PAIRED_FITS.json), [native shift availability](../../../results/v20/native-reference-wave-7/PAIRED_SHIFTS.json), [first-fit shifts](../../../results/v20/native-reference-wave-7/FIRST_FIT_SHIFTS.json), [scores](../../../results/v20/native-reference-wave-7/AGGREGATES.json), [zero audit](../../../results/v20/native-reference-wave-7/ZERO_MASS_AUDIT.json), [replay reuse](../../../results/v20/native-reference-wave-7/REPLAY_REUSE.json), [costs](../../../results/v20/native-reference-wave-7/COSTS.json), [forecast](../../../results/v20/native-reference-wave-7/FORECAST.json).

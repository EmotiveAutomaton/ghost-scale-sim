# V20 small-label long-history shifts

We tested whether structured reading gains an advantage under execution or presentation shifts with 64 retained episodes and 128 or 512 training labels. All ten first-fit comparisons fail. All 4,587,520 new forecasts verify, with no exact-zero truth probabilities. Both-fit comparisons remain incomplete. These are constructed-method results; confirmation remains untouched.

These ten blocks use the first registered fit with 64 retained episodes. Permitted-context evidence is tested at 128 and 512 labels; artifact, sparse-record and complete-record evidence at 128 labels. Each condition includes execution and presentation shifts. The campaign now has 744 verified packets. Second fits remain queued; this batch completes no two-fit condition.

Methods are paired within 32 whole coefficient worlds, using 4,096 fixed-seed descriptive world-bootstrap resamples. Neither evaluation rows nor repeated fits are independent worlds. Once both fits exist, they must be averaged within each world. Intervals remain conditional on the shared training worlds.

Capped joint log loss is minus the natural logarithm of the probability assigned to the true seven-part answer, with probabilities floored at one trillionth; lower is better. The registered advantage requires at least 0.02 natural-log units over both direct rivals and the legal template, a positive lower world interval and no increase in wrong attribution at 90% confidence. Exact zero mass is separate; this capped loss is not an uncapped proper score.

Table: first-fit comparisons. Rows identify evidence tier, number of training labels and shift; every row uses 64 past episodes. Reader columns give capped joint log loss. The final column gives the improvement of structured over joint neural reading and its descriptive world interval; negative values mean structured reading is worse. Execution changes the execution law under both proposal routes; presentation changes the presentation law.

| Evidence | Labels | Shift | Structured | Joint neural | Frequencies | Legal template | Improvement [interval] |
|---|---:|---|---:|---:|---:|---:|---|
| context | 128 | Execution | 24.21418 | 3.60772 | 2.73813 | 2.52400 | -20.60646 [-20.76506, -20.45741] |
| context | 128 | Presentation | 24.49317 | 3.34051 | 2.73878 | 2.52416 | -21.15267 [-21.27258, -21.03486] |
| context | 512 | Execution | 22.26311 | 2.97249 | 2.67861 | 2.52400 | -19.29062 [-19.40717, -19.17296] |
| context | 512 | Presentation | 22.43722 | 2.81465 | 2.68847 | 2.52416 | -19.62257 [-19.83301, -19.42221] |
| artifact | 128 | Execution | 26.32430 | 4.41800 | 3.75780 | 3.69118 | -21.90631 [-22.06596, -21.75041] |
| artifact | 128 | Presentation | 25.32450 | 4.39872 | 3.75700 | 3.68887 | -20.92578 [-21.05284, -20.79654] |
| sparse | 128 | Execution | 22.93832 | 3.08381 | 2.34831 | 2.17592 | -19.85451 [-19.98704, -19.72520] |
| sparse | 128 | Presentation | 22.99564 | 3.00996 | 2.34608 | 2.17520 | -19.98568 [-20.10193, -19.86667] |
| complete | 128 | Execution | 10.76188 | 1.28418 | 0.79482 | 0.69745 | -9.47770 [-9.57637, -9.38444] |
| complete | 128 | Presentation | 10.69570 | 1.07594 | 0.79429 | 0.69765 | -9.61976 [-9.72570, -9.51205] |

All ten first-fit conditions fail the required comparison. Every required rival contrast and confidence check remains in the evidence; this is discovery sensitivity, not confirmation.

Table: secondary scores for structured / joint neural reading in the same first-fit conditions. Squared probability loss sums squared errors across 128 joint probabilities and is proper; lower is better. Accuracy is the fraction with the correct most-probable joint answer. Confident wrong attribution is the fraction wrong at at least 90% confidence.

| Evidence | Labels | Shift | Squared loss | Accuracy | Confident wrong attribution |
|---|---:|---|---|---|---|
| context | 128 | Execution | 1.78889 / 0.99686 | 0.08861 / 0.10146 | 0.8406830 / 0.0005035 |
| context | 128 | Presentation | 1.80544 / 1.00220 | 0.08438 / 0.10860 | 0.8633881 / 0.0008545 |
| context | 512 | Execution | 1.71284 / 0.96673 | 0.11076 / 0.11540 | 0.7580261 / 0.0027924 |
| context | 512 | Presentation | 1.74224 / 0.94207 | 0.11115 / 0.11722 | 0.8197784 / 0.0008850 |
| artifact | 128 | Execution | 1.88048 / 0.98848 | 0.03755 / 0.04335 | 0.8682556 / 0.0000000 |
| artifact | 128 | Presentation | 1.83544 / 0.98277 | 0.04663 / 0.04375 | 0.8059692 / 0.0000000 |
| sparse | 128 | Execution | 1.72012 / 1.03938 | 0.12276 / 0.14081 | 0.8158569 / 0.0261383 |
| sparse | 128 | Presentation | 1.72688 / 1.02436 | 0.12148 / 0.14285 | 0.8231659 / 0.0209503 |
| complete | 128 | Execution | 0.94073 / 0.74582 | 0.49774 / 0.49556 | 0.4378357 / 0.2097473 |
| complete | 128 | Presentation | 0.93749 / 0.70087 | 0.49789 / 0.49593 | 0.4351807 / 0.1471558 |

There are 10 verified same-fit native/shift counterparts and 0 unavailable counterparts. Reused COMPLETE and summary hashes match earlier acceptance; same-fit training examples are byte-identical. Native and shifted evaluation samples differ, so comparisons pair whole worlds. Current observations and retained histories shift together, and exact legal support under each law is supplied to every learned reader. This does not test recovery of an unknown law.

Table: first-fit changes in capped loss relative to matched native references. Positive means worse under the shift. Rows identify evidence, label budget and shift; the structured column includes its descriptive paired-world interval.

| Evidence | Labels | Shift | Structured change [interval] | Joint neural change | Frequency change | Template change |
|---|---:|---|---|---:|---:|---:|
| context | 128 | Execution | -0.07728 [-0.17494, 0.02968] | 0.00103 | 0.00347 | 0.00259 |
| context | 128 | Presentation | 0.20171 [0.12386, 0.28623] | -0.26618 | 0.00412 | 0.00275 |
| context | 512 | Execution | -0.05415 [-0.15193, 0.04343] | 0.00188 | -0.00094 | 0.00259 |
| context | 512 | Presentation | 0.11996 [-0.02257, 0.26730] | -0.15595 | 0.00893 | 0.00275 |
| artifact | 128 | Execution | -0.06461 [-0.12529, 0.00229] | 0.00506 | 0.00203 | 0.00142 |
| artifact | 128 | Presentation | -1.06442 [-1.13731, -0.98598] | -0.01422 | 0.00123 | -0.00088 |
| sparse | 128 | Execution | -0.09302 [-0.20901, 0.02503] | 0.00500 | 0.00197 | 0.00102 |
| sparse | 128 | Presentation | -0.03570 [-0.13379, 0.06236] | -0.06884 | -0.00027 | 0.00030 |
| complete | 128 | Execution | -0.11705 [-0.25298, 0.01438] | 0.00616 | 0.00159 | -0.00003 |
| complete | 128 | Presentation | -0.18323 [-0.30774, -0.06382] | -0.20207 | 0.00107 | 0.00017 |

No new forecast assigns exactly zero probability to truth. Below-floor probabilities were checked separately; their absence or presence does not establish calibration. Every forecast, stored secondary score and summary reconstructs. Source, plan, environment, costs and attempts reconcile. Only reader/ is blind input; casebooks and summaries are evaluator artifacts. Full opening G1/G4/G5/G6, scoring-repair, history, shift, underflow-interpretation and both-fit positive replays remain hash-bound evidence. No new family, repair or scientific interpretation requires fresh replay. Validation does not establish fresh installation, cross-platform reproducibility, real-text correspondence or human intent.

Original blocks used 1549.10938 native CPU seconds; numerical review used 100.01562, and native-pair/zero-mass review used 0.98438. Operating, documentary and publication work is charged separately. Historical exited-service and reviewer-descendant CPU remains imperfectly measured with conservative allowances.

No refill is needed at this checkpoint: 94 admitted useful discovery blocks remain, with 4.91 measured hours. Existing queue, required reviews and two hours of future operating/publication allowance project 70.02 discovery CPU hours. Restoring a full measured day projects 84.16 hours, above 72. The 19.09-hour shortfall is explicit; useful conditional work remains unexhausted and the 18-hour reserve stays protected.

On October 1 at 08:00 UTC, the twelve frozen diagnostic blocks take priority, followed by original V20-C01 on untouched lineages 96–127 across both fits within its inclusive 4,000-second cap. Challenge lineages remain untouched. Science cutoff is October 2 at 04:00 UTC, final preparation at 10:00 and delivery/service closure at 12:00.

[Validity](../../../results/v20/health-20260930T2230/VALIDITY.json), [scores](../../../results/v20/health-20260930T2230/AGGREGATES.json), [matched native shifts](../../../results/v20/health-20260930T2230/PAIRED_SHIFTS.json), [zero audit](../../../results/v20/health-20260930T2230/ZERO_MASS_AUDIT.json), [replay reuse](../../../results/v20/health-20260930T2230/REPLAY_REUSE.json), [costs](../../../results/v20/health-20260930T2230/COSTS.json), [forecast](../../../results/v20/health-20260930T2230/FORECAST.json).

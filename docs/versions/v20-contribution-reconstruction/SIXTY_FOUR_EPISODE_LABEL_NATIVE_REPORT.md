# V20 sixty-four-episode native label comparisons

We tested whether structured reading gains an advantage under native rules with 64 past episodes at three training-label budgets. All eight paired-fit evidence-tier comparisons fail against every required rival. All 3,670,016 new forecasts verify; 1,437 exact-zero truth probabilities come from numerical underflow. These are constructed-method results; confirmation remains untouched.

The eight new second-fit blocks cover artifact-only, sparse-record and complete-record evidence at 128 and 512 labels, and artifact-only and sparse-record evidence at 8,192 labels. Each uses 64 past episodes. Their first fits were already verified. The total is 587 verified packets. These are nonprimary diagnostics; the 8,192-label complete-record counterpart remains outside this reviewed batch. No new context primary or confirmation outcome is claimed.

Both registered fits are averaged inside each of 32 coefficient worlds before pairing readers and applying 4,096 fixed-seed descriptive world-bootstrap resamples. Neither rows nor fits are independent worlds. Uncertainty is conditional on shared training worlds. Capped joint log loss is minus the natural logarithm of probability assigned to the true seven-part answer, floored at one trillionth; lower is better. The declared advantage requires at least 0.02 natural-log units over both direct rivals and the legal template, a positive lower world interval, and no increase in wrong attribution at 90% confidence.

Table: native paired-fit capped log loss with 64 past episodes. Rows identify training-label count and evidence tier. Reader columns give mean loss; the final column gives structured improvement over joint neural reading with its descriptive world interval. Negative improvement means structured reading is worse.

| Training labels | Evidence | Structured | Joint neural | Frequencies | Legal template | Improvement [interval] |
|---:|---|---:|---:|---:|---:|---|
| 128 | Artifact | 25.65067 | 4.54269 | 3.78042 | 3.68975 | -21.10798 [-21.18043, -21.03667] |
| 128 | Sparse | 22.77148 | 3.32109 | 2.36354 | 2.17490 | -19.45039 [-19.50255, -19.39360] |
| 128 | Complete | 11.50373 | 1.29310 | 0.81856 | 0.69748 | -10.21063 [-10.28428, -10.13860] |
| 512 | Artifact | 23.53567 | 3.75581 | 3.65722 | 3.68975 | -19.77986 [-19.90283, -19.64489] |
| 512 | Sparse | 20.95026 | 2.59572 | 2.26653 | 2.17490 | -18.35453 [-18.50442, -18.19708] |
| 512 | Complete | 11.47560 | 1.02487 | 0.75936 | 0.69748 | -10.45072 [-10.52455, -10.37764] |
| 8,192 | Artifact | 8.87423 | 3.52855 | 3.55456 | 3.68975 | -5.34568 [-5.63466, -5.07215] |
| 8,192 | Sparse | 5.79784 | 2.19545 | 2.17720 | 2.17490 | -3.60239 [-3.72150, -3.48872] |

Structured reading loses to every required rival in all eight conditions, with every gain interval below zero and increased confident wrong attribution. Sixteen execution/presentation shift conditions lack both verified fits at the matched budgets and evidence tiers; available first fits are recorded, with no unmatched-fit or unmatched-budget shift estimate substituted.

Table: native paired-fit secondary scores. Squared probability loss sums squared joint-probability errors and is proper; lower is better. Accuracy is the fraction with the correct most-probable joint answer. Confident wrong attribution is the fraction wrong at at least 90% confidence. Each cell compares structured / joint neural reading. Other readers and aim/tool/order/inspection scores remain in the checked aggregates.

| Training labels | Evidence | Squared loss | Accuracy | Confident wrong attribution |
|---:|---|---|---|---|
| 128 | Artifact | 1.85431 / 0.99249 | 0.04395 / 0.04572 | 0.8360214 / 0.0000000 |
| 128 | Sparse | 1.70411 / 1.05474 | 0.13182 / 0.14118 | 0.8076859 / 0.0374527 |
| 128 | Complete | 0.94850 / 0.74992 | 0.49765 / 0.49908 | 0.4449692 / 0.2173386 |
| 512 | Artifact | 1.77756 / 0.97227 | 0.05875 / 0.05317 | 0.7273560 / 0.0000000 |
| 512 | Sparse | 1.66039 / 0.95282 | 0.14514 / 0.14545 | 0.7609558 / 0.0039062 |
| 512 | Complete | 0.98571 / 0.68309 | 0.49957 / 0.49716 | 0.4837723 / 0.1190338 |
| 8,192 | Artifact | 1.21252 / 0.96235 | 0.07472 / 0.05271 | 0.0308762 / 0.0000000 |
| 8,192 | Sparse | 1.31552 / 0.86707 | 0.14185 / 0.14771 | 0.2174606 / 0.0000000 |

Table: new second-fit truth probabilities below the declared log-score floor and exact zero, reported separately. Rows list every reader with any below-floor truth. Percentages use the 65,536 evaluated truths for that reader; they are not independent worlds.

| Training labels | Evidence | Reader | Below floor | Exact zero |
|---:|---|---|---:|---:|
| 128 | Artifact | Independent bits | 16.89758% | 0.00000% |
| 128 | Artifact | Structured | 86.81030% | 0.00000% |
| 128 | Sparse | Independent bits | 35.46448% | 0.00000% |
| 128 | Sparse | Structured | 77.57111% | 0.00000% |
| 128 | Complete | Independent bits | 0.14648% | 0.00000% |
| 128 | Complete | Structured | 42.01355% | 0.00000% |
| 512 | Artifact | Independent bits | 29.82330% | 0.00000% |
| 512 | Artifact | Structured | 72.16644% | 2.19269% |
| 512 | Sparse | Independent bits | 21.96808% | 0.00000% |
| 512 | Sparse | Structured | 63.51776% | 0.00000% |
| 512 | Complete | Structured | 34.01489% | 0.00000% |
| 8,192 | Artifact | Independent bits | 67.57507% | 0.00000% |
| 8,192 | Artifact | Structured | 5.00946% | 0.00000% |
| 8,192 | Sparse | Independent bits | 28.28979% | 0.00000% |
| 8,192 | Sparse | Structured | 0.36774% | 0.00000% |

All 1,437 exact-zero truths occur in structured reading with artifact-only evidence at 512 labels. Every zero truth remains inside supplied legal support. Independent reconstruction gives finite learned log odds relative to the modal answer from -798.609 to -745.631; exponentiation underflows to zero in every case. This retains the established numerical-underflow interpretation, distinct from candidate omission or logical impossibility. The capped score conceals penalties beyond its floor and is not an uncapped proper score.

Every stored forecast, secondary score and summary reconstructs. Completion, plan, frozen source archive, environment, fit/decode costs and attempted failures reconcile. Only reader/ is blind input; casebooks and summaries are evaluator artifacts. Execution, numerical acceptance, discovery criteria and untouched confirmation remain separate. Full opening G1/G4/G5/G6, repaired scoring, history and shift references, the prior underflow interpretation replay, and both-fit positive references are reused by immutable hashes. There is no new implementation family, scientific repair or changed interpretation requiring another replay. Validation does not establish fresh installation, cross-platform replay, universal structured-reader performance, real-text correspondence or human intent.

Original blocks used 1196.48438 native CPU seconds; numerical review used 73.06250 and the supplement 1.34375. Forecasting, documents, publication preparation and operations are charged separately. A failed operational receipt used an overly narrow state assertion; its failure is retained and corrected without changing scientific source, model settings, scores or criteria. Historical exited-service and reviewer-descendant accounting limitations remain.

No new designs were admitted. The 183 pending useful discovery blocks have a measured forecast of 14.75 hours. Restoring a full day would project 77.28 discovery CPU hours, above the 72-hour ceiling including required reviews and two hours of future operating/publication allowance. The 9.25-hour runway shortfall is explicit; the useful frontier remains unexhausted. The separate 18-hour reserve remains protected. Reassess at the next exact result boundary; a forecast is not execution or occupancy.

At October 1 08:00 UTC, the twelve frozen diagnostic confirmation blocks have first priority, followed by the original two-fit context extension within its inclusive 4,000-second cap, reserve and cutoff. Confirmation and challenge lineages remain untouched. Science ends October 2 04:00 UTC; final preparation at 10:00 and delivery/service closure at 12:00 remain fixed.

[Validity](../../../results/v20/native-reference-wave-6/VALIDITY.json), [paired fits](../../../results/v20/native-reference-wave-6/PAIRED_FITS.json), [shift availability](../../../results/v20/native-reference-wave-6/PAIRED_SHIFTS.json), [all scores](../../../results/v20/native-reference-wave-6/AGGREGATES.json), [zero audit](../../../results/v20/native-reference-wave-6/ZERO_MASS_AUDIT.json), [replay reuse](../../../results/v20/native-reference-wave-6/REPLAY_REUSE.json), [costs](../../../results/v20/native-reference-wave-6/COSTS.json), [forecast](../../../results/v20/native-reference-wave-6/FORECAST.json).

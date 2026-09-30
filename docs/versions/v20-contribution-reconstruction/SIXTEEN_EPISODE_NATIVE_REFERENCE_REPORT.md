# V20 sixteen-episode native references

We tested whether structured reading gains an advantage under native rules with sixteen past episodes and 32,768 training labels. All three paired-fit evidence-tier comparisons fail. All 1,376,256 new forecasts verify, with no exact-zero truth probabilities. These are constructed-method results; confirmation remains untouched.

Three new second-fit native blocks cover artifact-only, sparse-record and complete-record evidence. Their first native fits and both corresponding shifted fits were already verified. There are now 564 verified packets. The permitted-context primary is unchanged.

Both fits are averaged inside each of 32 coefficient worlds before pairing readers and applying 4,096 fixed-seed descriptive world-bootstrap resamples. Rows and fits are not independent worlds; uncertainty is conditional on shared training worlds. Capped joint log loss is minus the natural logarithm of probability assigned to the true seven-part answer, floored at one trillionth; lower is better. The declared advantage requires at least 0.02 natural-log units improvement over both direct rivals and the legal template, a positive lower world interval, and no increase in incorrect attribution at 90% confidence. These evidence tiers are nonprimary diagnostics.

Table: native paired-fit capped log loss. Rows identify available evidence; all use sixteen past episodes and 32,768 labels. Reader columns show mean loss; the last column is structured improvement over joint neural reading with its descriptive world interval. Negative improvement means structured reading is worse.

| Evidence | Structured | Joint neural | Frequencies | Legal template | Improvement [interval] |
|---|---:|---:|---:|---:|---|
| Artifact | 3.73650 | 3.44786 | 3.59188 | 3.72986 | -0.28863 [-0.36267, -0.22030] |
| Sparse | 2.70095 | 2.33102 | 2.38511 | 2.39028 | -0.36993 [-0.40123, -0.33805] |
| Complete | 1.06048 | 0.81532 | 0.80044 | 0.79957 | -0.24516 [-0.27053, -0.22007] |

Structured reading loses to joint neural reading in all three conditions, with world intervals below zero. Sparse and complete records also lose to both other required rivals and increase confident wrong attribution. Artifact-only reading loses to frequencies; its difference from the legal template is unresolved. None passes the all-rival criterion.

Table: matched shifts, averaging both fits inside each world. Each loss change is shifted minus native loss for the same reader; negative means lower loss under the changed rule. The last column gives the change in structured advantage over joint neural reading, with its world interval; positive means relative advantage grew.

| Evidence | Change | Structured loss change | Neural loss change | Advantage change [interval] |
|---|---|---:|---:|---|
| Artifact | Execution | -0.28726 | -0.00290 | 0.28436 [0.23183, 0.34095] |
| Artifact | Presentation | 4.50902 | 0.29841 | -4.21061 [-4.31677, -4.09773] |
| Sparse | Execution | -0.15408 | 0.00503 | 0.15911 [0.13942, 0.18070] |
| Sparse | Presentation | 2.13202 | 0.29398 | -1.83804 [-1.93599, -1.73509] |
| Complete | Execution | -0.11207 | -0.00172 | 0.11035 [0.09124, 0.13086] |
| Complete | Presentation | -0.01042 | -0.00190 | 0.00853 [0.00198, 0.01576] |

The execution change lowers structured loss in all three evidence conditions; joint neural changes remain unresolved. Presentation change strongly harms artifact-only and sparse-record structured reading, while complete-record loss falls slightly. These contrasts retain the established shift interpretation and do not rescue any of the three native comparisons.

Training examples are byte-identical within each same-fit native/shift pair. Current and retained episodes shift together, and evaluation samples differ, so rows are not paired. The execution change reverses repair under both proposal routes. All learned readers receive exact legal support from each supplied law. This does not test recovery of an unknown law; shifts can change target difficulty as well as reader behavior.

Table: paired-fit native secondary scores. Squared probability loss sums squared joint-probability errors and is proper; lower is better. Accuracy is the fraction with the correct most-probable joint answer. Confident wrong attribution is the fraction wrong at at least 90% confidence. Each cell compares structured / joint neural reading. All other arms and goal/tool/order/inspection scores remain in the checked aggregates.

| Evidence | Squared loss | Accuracy | Confident wrong attribution |
|---|---|---|---|
| Artifact | 0.97343 / 0.95610 | 0.07434 / 0.07328 | 0.0000000 / 0.0000000 |
| Sparse | 0.94985 / 0.87535 | 0.13329 / 0.14194 | 0.0016632 / 0.0000000 |
| Complete | 0.66724 / 0.54775 | 0.46831 / 0.46469 | 0.0456848 / 0.0000000 |

Table: new second-fit truth probabilities below the declared log-score floor. Only independent-bit reading has below-floor truths here. Exact-zero truth mass is separately zero for every new arm.

| Evidence | Independent-bit truths below floor |
|---|---:|
| Artifact | 64.56451% |
| Sparse | 29.05884% |
| Complete | 3.76434% |

Capping hides the full penalty of tiny positive probabilities; capped log loss is not an uncapped proper score. Every stored forecast, secondary score and summary reconstructs. Only reader/ is blind input; casebooks and summaries are evaluator artifacts. Completion, plan, source, environment, fit/decode costs and attempted failures reconcile. Execution, numerical acceptance, discovery criteria and confirmation remain separate.

Full opening G1/G4/G5/G6, repaired scoring, long-history and shift reference replays are reused by immutable hashes, including both-fit positive shifted references. Completing matched second native fits adds no implementation family, repair or changed interpretation; no new replay is required. Validation does not cover fresh installation, cross-platform replay, universal structured-reader performance, real-text history or human intent.

Original blocks used 451.45312 native CPU seconds. Independent numerical review used 29.79688, and the matched-fit/shift supplement 0.98438. Forecasting, documents, publication and operating overhead are charged separately. Earlier failures and accounting limitations remain retained. Total/discovery ceilings remain 90/72 CPU hours with 18 protected.

No new designs were admitted at this checkpoint. The existing queue contains 206 useful discovery blocks, forecast at 21.02 hours. Restoring a full measured day would project discovery costs beyond the 72-hour ceiling after including required reviews and two hours of future operating/publication allowance. The 2.98-hour shortfall is explicit; useful conditional designs remain unexhausted. The separate 18-hour confirmation reserve stays protected. Reassess at the next named boundary; forecasts are not execution or occupancy.

At October 1 08:00 UTC the twelve frozen diagnostic confirmation blocks have first priority, followed by the original two-fit context extension within its inclusive 4,000-second cap, reserve and cutoff. Confirmation and challenge lineages remain untouched. Midpoint reporting, science cutoff and Friday delivery retain their fixed times.

[Validity](../../../results/v20/native-reference-wave-2/VALIDITY.json), [paired fits](../../../results/v20/native-reference-wave-2/PAIRED_FITS.json), [matched shifts](../../../results/v20/native-reference-wave-2/PAIRED_SHIFTS.json), [all scores](../../../results/v20/native-reference-wave-2/AGGREGATES.json), [zero audit](../../../results/v20/native-reference-wave-2/ZERO_MASS_AUDIT.json), [replay reuse](../../../results/v20/native-reference-wave-2/REPLAY_REUSE.json), [costs](../../../results/v20/native-reference-wave-2/COSTS.json), [forecast](../../../results/v20/native-reference-wave-2/FORECAST.json).

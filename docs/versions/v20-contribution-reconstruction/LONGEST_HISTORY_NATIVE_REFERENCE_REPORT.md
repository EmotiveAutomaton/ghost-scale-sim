# V20 longest-history native references

We tested whether structured reading gains an advantage under native rules with 256 past episodes and 8,192 training labels. All three paired-fit evidence-tier comparisons fail against every required rival. All 1,376,256 new forecasts verify, with no exact-zero truth probabilities. These are constructed-method results; confirmation remains untouched.

Three new second-fit native blocks cover artifact-only, sparse-record and complete-record evidence. Their first native fits and both corresponding shifted fits were already verified. There are now 567 verified packets. The permitted-context primary is unchanged.

Both fits are averaged inside each of 32 coefficient worlds before pairing readers and applying 4,096 fixed-seed descriptive world-bootstrap resamples. Rows and fits are not independent worlds; uncertainty is conditional on shared training worlds. Capped joint log loss is minus the natural logarithm of probability assigned to the true seven-part answer, floored at one trillionth; lower is better. The declared advantage requires at least 0.02 natural-log units improvement over both direct rivals and the legal template, a positive lower world interval, and no increase in incorrect attribution at 90% confidence. These evidence tiers are nonprimary diagnostics.

Table: native paired-fit capped log loss. Rows identify available evidence; all use 256 past episodes and 8,192 labels. Reader columns show mean loss; the last column is structured improvement over joint neural reading with its descriptive world interval. Negative improvement means structured reading is worse.

| Evidence | Structured | Joint neural | Frequencies | Legal template | Improvement [interval] |
|---|---:|---:|---:|---:|---|
| Artifact | 14.24004 | 3.55406 | 3.55350 | 3.68868 | -10.68598 [-11.36512, -10.04454] |
| Sparse | 12.45570 | 2.17101 | 2.14766 | 2.14531 | -10.28470 [-10.68743, -9.88818] |
| Complete | 5.50544 | 0.71807 | 0.69693 | 0.69315 | -4.78737 [-4.85616, -4.72040] |

Structured reading loses to every required rival in all three conditions, with all world intervals below zero and increased confident wrong attribution. None passes the all-rival criterion.

Table: matched shifts, averaging both fits inside each world. Each loss change is shifted minus native loss for the same reader; negative means lower loss under the changed rule. The last column gives the change in structured advantage over joint neural reading, with its world interval; positive means relative advantage grew.

| Evidence | Change | Structured loss change | Neural loss change | Advantage change [interval] |
|---|---|---:|---:|---|
| Artifact | Execution | -2.08604 | 0.00332 | 2.08937 [1.52923, 2.70114] |
| Artifact | Presentation | 5.42350 | 0.00597 | -5.41752 [-5.67361, -5.15586] |
| Sparse | Execution | -0.79213 | 0.00219 | 0.79432 [0.70227, 0.88268] |
| Sparse | Presentation | 1.79270 | 0.00613 | -1.78657 [-2.15064, -1.41353] |
| Complete | Execution | -0.34136 | 0.00111 | 0.34247 [0.28284, 0.40632] |
| Complete | Presentation | 0.16542 | -0.00050 | -0.16592 [-0.22815, -0.10328] |

The execution change lowers structured loss in all three evidence conditions; joint neural changes remain unresolved. Presentation change raises structured loss in all three evidence conditions. These contrasts retain the established shift interpretation and do not rescue any of the three native comparisons.

Training examples are byte-identical within each same-fit native/shift pair. Current and retained episodes shift together, and evaluation samples differ, so rows are not paired. The execution change reverses repair under both proposal routes. All learned readers receive exact legal support from each supplied law. This does not test recovery of an unknown law; shifts can change target difficulty as well as reader behavior.

Table: paired-fit native secondary scores. Squared probability loss sums squared joint-probability errors and is proper; lower is better. Accuracy is the fraction with the correct most-probable joint answer. Confident wrong attribution is the fraction wrong at at least 90% confidence. Each cell compares structured / joint neural reading. All other arms and goal/tool/order/inspection scores remain in the checked aggregates.

| Evidence | Squared loss | Accuracy | Confident wrong attribution |
|---|---|---|---|
| Artifact | 1.52517 / 0.96355 | 0.07668 / 0.05098 | 0.3375626 / 0.0000000 |
| Sparse | 1.53629 / 0.86562 | 0.14112 / 0.14417 | 0.5290451 / 0.0000000 |
| Complete | 0.93583 / 0.52303 | 0.50111 / 0.50217 | 0.4298172 / 0.0000000 |

Table: new second-fit truth probabilities below the declared log-score floor. Independent-bit and structured reading have below-floor truths here. Exact-zero truth mass is separately zero for every new arm.

| Evidence | Reader | Truths below floor |
|---|---|---:|
| Artifact | Independent bits | 64.56451% |
| Artifact | Structured | 28.51257% |
| Sparse | Independent bits | 4.56696% |
| Sparse | Structured | 13.30109% |
| Complete | Structured | 5.16663% |

Capping hides the full penalty of tiny positive probabilities; capped log loss is not an uncapped proper score. Every stored forecast, secondary score and summary reconstructs. Only reader/ is blind input; casebooks and summaries are evaluator artifacts. Completion, plan, source, environment, fit/decode costs and attempted failures reconcile. Execution, numerical acceptance, discovery criteria and confirmation remain separate.

Full opening G1/G4/G5/G6, repaired scoring, long-history and shift reference replays are reused by immutable hashes, including both-fit positive shifted references. Completing matched second native fits adds no implementation family, repair or changed interpretation; no new replay is required. Validation does not cover fresh installation, cross-platform replay, universal structured-reader performance, real-text history or human intent.

Original blocks used 1637.53125 native CPU seconds. Independent numerical review used 30.31250, and the matched-fit/shift supplement 0.96875. Forecasting, documents, publication and operating overhead are charged separately. Earlier failures and accounting limitations remain retained. Total/discovery ceilings remain 90/72 CPU hours with 18 protected.

No new designs were admitted at this boundary. The existing queue contains 203 useful discovery blocks, forecast at 19.87 hours. Restoring a full measured day would project discovery costs beyond the 72-hour ceiling after including required reviews and two hours of future operating/publication allowance. The 4.13-hour shortfall is explicit; useful conditional designs remain unexhausted. The separate 18-hour confirmation reserve stays protected. Reassess at the next named boundary; forecasts are not execution or occupancy.

At October 1 08:00 UTC the twelve frozen diagnostic confirmation blocks have first priority, followed by the original two-fit context extension within its inclusive 4,000-second cap, reserve and cutoff. Confirmation and challenge lineages remain untouched. Midpoint reporting, science cutoff and Friday delivery retain their fixed times.

[Validity](../../../results/v20/native-reference-wave-3/VALIDITY.json), [paired fits](../../../results/v20/native-reference-wave-3/PAIRED_FITS.json), [matched shifts](../../../results/v20/native-reference-wave-3/PAIRED_SHIFTS.json), [all scores](../../../results/v20/native-reference-wave-3/AGGREGATES.json), [zero audit](../../../results/v20/native-reference-wave-3/ZERO_MASS_AUDIT.json), [replay reuse](../../../results/v20/native-reference-wave-3/REPLAY_REUSE.json), [costs](../../../results/v20/native-reference-wave-3/COSTS.json), [forecast](../../../results/v20/native-reference-wave-3/FORECAST.json).

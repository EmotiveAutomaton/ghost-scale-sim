# V20 longest-history maximum-label native references

We tested whether structured reading gains an advantage under native rules with 256 past episodes and 32,768 training labels. All three paired-fit evidence-tier comparisons fail against every required rival. All 1,376,256 new forecasts verify, with no exact-zero truth probabilities. These are constructed-method results; confirmation remains untouched.

Three new second-fit native blocks cover artifact-only, sparse-record and complete-record evidence. Their first native fits and both corresponding shifted fits were already verified. There are now 573 verified packets. The permitted-context primary is unchanged.

Both fits are averaged inside each of 32 coefficient worlds before pairing readers and applying 4,096 fixed-seed descriptive world-bootstrap resamples. Rows and fits are not independent worlds; uncertainty is conditional on shared training worlds. Capped joint log loss is minus the natural logarithm of probability assigned to the true seven-part answer, floored at one trillionth; lower is better. The declared advantage requires at least 0.02 natural-log units improvement over both direct rivals and the legal template, a positive lower world interval, and no increase in incorrect attribution at 90% confidence. These evidence tiers are nonprimary diagnostics.

Table: native paired-fit capped log loss. Rows identify available evidence; all use 256 past episodes and 32,768 labels. Reader columns show mean loss; the last column is structured improvement over joint neural reading with its descriptive world interval. Negative improvement means structured reading is worse.

| Evidence | Structured | Joint neural | Frequencies | Legal template | Improvement [interval] |
|---|---:|---:|---:|---:|---|
| Artifact | 9.82605 | 3.52969 | 3.54613 | 3.68868 | -6.29636 [-7.21376, -5.45658] |
| Sparse | 6.82515 | 2.14709 | 2.13959 | 2.14531 | -4.67806 [-5.22031, -4.14217] |
| Complete | 2.38567 | 0.69999 | 0.69392 | 0.69315 | -1.68569 [-1.72368, -1.64901] |

Structured reading loses to every required rival in all three conditions, with all world intervals below zero and increased confident wrong attribution. None passes the all-rival criterion.

Table: matched shifts, averaging both fits inside each world. Each loss change is shifted minus native loss for the same reader; negative means lower loss under the changed rule. The last column gives the change in structured advantage over joint neural reading, with its world interval; positive means relative advantage grew.

| Evidence | Change | Structured loss change | Neural loss change | Advantage change [interval] |
|---|---|---:|---:|---|
| Artifact | Execution | -2.17968 | 0.00627 | 2.18595 [1.46154, 2.96957] |
| Artifact | Presentation | 8.43124 | 0.00320 | -8.42805 [-8.80055, -8.03304] |
| Sparse | Execution | -0.64150 | 0.00233 | 0.64383 [0.58388, 0.70088] |
| Sparse | Presentation | 4.42175 | 0.01056 | -4.41120 [-4.85301, -3.95194] |
| Complete | Execution | -0.25046 | -0.00017 | 0.25029 [0.21824, 0.28160] |
| Complete | Presentation | 1.08165 | -0.00054 | -1.08219 [-1.11709, -1.04806] |

The execution change lowers structured loss in all three evidence conditions; joint neural changes remain unresolved. Presentation change raises structured loss in all three evidence conditions. These contrasts retain the established shift interpretation and do not rescue any of the three native comparisons.

Training examples are byte-identical within each same-fit native/shift pair. Current and retained episodes shift together, and evaluation samples differ, so rows are not paired. The execution change reverses repair under both proposal routes. All learned readers receive exact legal support from each supplied law. This does not test recovery of an unknown law; shifts can change target difficulty as well as reader behavior.

Table: paired-fit native secondary scores. Squared probability loss sums squared joint-probability errors and is proper; lower is better. Accuracy is the fraction with the correct most-probable joint answer. Confident wrong attribution is the fraction wrong at at least 90% confidence. Each cell compares structured / joint neural reading. All other arms and goal/tool/order/inspection scores remain in the checked aggregates.

| Evidence | Squared loss | Accuracy | Confident wrong attribution |
|---|---|---|---|
| Artifact | 1.20605 / 0.96163 | 0.07829 / 0.06300 | 0.0323029 / 0.0000000 |
| Sparse | 1.36577 / 0.85958 | 0.13646 / 0.15257 | 0.2616196 / 0.0000000 |
| Complete | 0.85805 / 0.50670 | 0.49934 / 0.50029 | 0.3387756 / 0.0000000 |

Table: new second-fit truth probabilities below the declared log-score floor. Independent-bit and structured reading have below-floor truths here. Exact-zero truth mass is separately zero for every new arm.

| Evidence | Reader | Truths below floor |
|---|---|---:|
| Artifact | Independent bits | 56.56128% |
| Artifact | Structured | 6.52618% |
| Sparse | Independent bits | 10.25391% |
| Sparse | Structured | 0.16937% |
| Complete | Structured | 0.01831% |

Capping hides the full penalty of tiny positive probabilities; capped log loss is not an uncapped proper score. Every stored forecast, secondary score and summary reconstructs. Only reader/ is blind input; casebooks and summaries are evaluator artifacts. Completion, plan, source, environment, fit/decode costs and attempted failures reconcile. Execution, numerical acceptance, discovery criteria and confirmation remain separate.

Full opening G1/G4/G5/G6, repaired scoring, long-history and shift reference replays are reused by immutable hashes, including both-fit positive shifted references. Completing matched second native fits adds no implementation family, repair or changed interpretation; no new replay is required. Validation does not cover fresh installation, cross-platform replay, universal structured-reader performance, real-text history or human intent.

Original blocks used 3125.65625 native CPU seconds. Independent numerical review used 31.29688, and the matched-fit/shift supplement 0.93750. Forecasting, documents, publication and operating overhead are charged separately. The first summary helper incorrectly assumed the shifted counterparts were pending; its failure, source and CPU remain retained. The corrected helper reuses completed paired-fit evidence and checks the six available shifts. No scientific source or criterion changed. Earlier failures and accounting limitations remain retained. Total/discovery ceilings remain 90/72 CPU hours with 18 protected.

No new designs were admitted at this boundary. The existing queue contains 197 useful discovery blocks, forecast at 16.68 hours. Restoring a full measured day would project discovery costs beyond the 72-hour ceiling after including required reviews and two hours of future operating/publication allowance. The 7.32-hour shortfall is explicit; useful conditional designs remain unexhausted. The separate 18-hour confirmation reserve stays protected. Reassess at the next named boundary; forecasts are not execution or occupancy.

At October 1 08:00 UTC the twelve frozen diagnostic confirmation blocks have first priority, followed by the original two-fit context extension within its inclusive 4,000-second cap, reserve and cutoff. Confirmation and challenge lineages remain untouched. Midpoint reporting, science cutoff and Friday delivery retain their fixed times.

[Validity](../../../results/v20/native-reference-wave-4/VALIDITY.json), [paired fits](../../../results/v20/native-reference-wave-4/PAIRED_FITS.json), [matched shifts](../../../results/v20/native-reference-wave-4/PAIRED_SHIFTS.json), [all scores](../../../results/v20/native-reference-wave-4/AGGREGATES.json), [zero audit](../../../results/v20/native-reference-wave-4/ZERO_MASS_AUDIT.json), [replay reuse](../../../results/v20/native-reference-wave-4/REPLAY_REUSE.json), [costs](../../../results/v20/native-reference-wave-4/COSTS.json), [forecast](../../../results/v20/native-reference-wave-4/FORECAST.json).

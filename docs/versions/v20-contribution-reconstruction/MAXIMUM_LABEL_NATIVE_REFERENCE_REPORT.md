# V20 maximum-label native references

We tested whether structured reading gains an advantage under native rules at the largest training budget. All 19 paired-fit comparisons fail, including all five context primary conditions. All 8,716,288 new forecasts verify, with no exact-zero truth probabilities. These are constructed-method results; confirmation remains untouched.

Nineteen new second-fit native blocks use 131,072 training labels and zero, one, four, sixteen or sixty-four past episodes. The 64-episode complete-record native block remains queued and is excluded here. Every matching first native fit and both corresponding shifted fits was already verified. There are now 536 verified packets.

Both fits are averaged inside each of 32 coefficient worlds before pairing readers and applying 4,096 fixed-seed descriptive world-bootstrap resamples. Rows and fits are not independent worlds; uncertainty is conditional on shared training worlds. Capped joint log loss is minus the natural logarithm of probability assigned to the true seven-part answer, floored at one trillionth; lower is better. The declared advantage requires at least 0.02 natural-log units improvement over both direct rivals and the legal template, a positive lower world interval, and no increase in incorrect attribution at 90% confidence. Only context conditions are primary.

Table: each row averages both registered fits within whole worlds at the maximum training budget. Evidence names the available record; history counts past episodes. Loss columns compare structured reading, direct frequencies, joint neural reading and the legal template; lower is better. Every row fails the complete advantage rule.

| Evidence | History | Structured | Frequencies | Joint neural | Template | Primary? |
|---|---:|---:|---:|---:|---:|---|
| Context | 0 | 3.59037 | 3.50826 | 3.57851 | 3.58216 | Yes |
| Context | 1 | 3.38095 | 3.66619 | 3.36439 | 3.37592 | Yes |
| Context | 4 | 3.04725 | 3.22079 | 3.02844 | 3.08879 | Yes |
| Context | 16 | 2.85115 | 2.80258 | 2.74970 | 2.73679 | Yes |
| Context | 64 | 3.14585 | 2.58703 | 2.60009 | 2.52141 | Yes |
| Artifact | 0 | 3.96252 | 3.92052 | 3.96199 | 4.14532 | No |
| Sparse | 0 | 3.24385 | 3.16168 | 3.20689 | 3.23565 | No |
| Complete | 0 | 1.65388 | 1.61492 | 1.63154 | 1.64786 | No |
| Artifact | 1 | 3.80104 | 3.71145 | 3.79406 | 3.99116 | No |
| Sparse | 1 | 3.03516 | 3.50981 | 2.98487 | 3.02941 | No |
| Complete | 1 | 1.43762 | 2.64936 | 1.38453 | 1.39966 | No |
| Artifact | 4 | 3.57879 | 4.20313 | 3.54914 | 3.83026 | No |
| Sparse | 4 | 2.70250 | 2.75697 | 2.63202 | 2.74228 | No |
| Complete | 4 | 1.16119 | 1.08410 | 1.07104 | 1.08387 | No |
| Artifact | 16 | 3.58192 | 3.60283 | 3.46541 | 3.72986 | No |
| Sparse | 16 | 2.50950 | 2.38471 | 2.33478 | 2.39028 | No |
| Complete | 16 | 0.91737 | 0.79976 | 0.80882 | 0.79957 | No |
| Artifact | 64 | 4.01864 | 3.54524 | 3.49727 | 3.68975 | No |
| Sparse | 64 | 2.78167 | 2.16917 | 2.17183 | 2.17490 | No |

The mean structured loss is worse than joint neural reading in every condition. Eighteen corresponding world intervals exclude zero; artifact-only reading with no history has an unresolved difference. Better performance against an individual frequency or template rival in some rows does not satisfy the all-rival rule. Long histories do not repair the native decoder failure.

Table: all matched shifts, averaging both fits inside each world. Each change is shifted loss minus native loss for the same reader; negative means lower loss under the changed rule. The final column is the change in structured advantage over joint neural reading, with its descriptive world interval; positive means the relative advantage grew. This separates within-reader change from comparative advantage.

| Evidence | History | Change | Structured loss change | Neural loss change | Advantage change [interval] |
|---|---:|---|---:|---:|---|
| Context | 0 | Execution | 0.00141 | 0.01378 | 0.01237 [0.00457, 0.02012] |
| Context | 0 | Presentation | 1.28548 | 0.21680 | -1.06868 [-1.08763, -1.04865] |
| Context | 1 | Execution | -0.01787 | 0.01279 | 0.03066 [0.02385, 0.03746] |
| Context | 1 | Presentation | 1.49769 | 0.28064 | -1.21705 [-1.23828, -1.19464] |
| Context | 4 | Execution | -0.04197 | 0.01469 | 0.05666 [0.04628, 0.06765] |
| Context | 4 | Presentation | 2.24915 | 0.46165 | -1.78750 [-1.82068, -1.75216] |
| Context | 16 | Execution | -0.08205 | 0.00473 | 0.08679 [0.06993, 0.10565] |
| Context | 16 | Presentation | 3.40522 | 0.19003 | -3.21519 [-3.33864, -3.08674] |
| Context | 64 | Execution | -0.09405 | 0.00260 | 0.09665 [0.07831, 0.11539] |
| Context | 64 | Presentation | 4.86751 | -0.05000 | -4.91751 [-5.07613, -4.75479] |
| Artifact | 0 | Execution | 0.04608 | 0.02531 | -0.02078 [-0.02804, -0.01348] |
| Artifact | 0 | Presentation | 1.31331 | 0.22619 | -1.08712 [-1.10664, -1.06665] |
| Sparse | 0 | Execution | -0.00021 | 0.01143 | 0.01164 [0.00410, 0.01932] |
| Sparse | 0 | Presentation | 1.26221 | 0.21839 | -1.04382 [-1.06269, -1.02368] |
| Complete | 0 | Execution | -0.02298 | 0.00479 | 0.02778 [0.02030, 0.03508] |
| Complete | 0 | Presentation | -0.00627 | -0.00325 | 0.00303 [-0.00244, 0.00831] |
| Artifact | 1 | Execution | 0.01969 | 0.02097 | 0.00128 [-0.00391, 0.00660] |
| Artifact | 1 | Presentation | 1.55977 | 0.31077 | -1.24901 [-1.27007, -1.22670] |
| Sparse | 1 | Execution | -0.01949 | 0.01186 | 0.03136 [0.02534, 0.03743] |
| Sparse | 1 | Presentation | 1.44353 | 0.28771 | -1.15583 [-1.17666, -1.13381] |
| Complete | 1 | Execution | -0.04379 | 0.00370 | 0.04749 [0.04224, 0.05278] |
| Complete | 1 | Presentation | -0.00373 | -0.00127 | 0.00246 [-0.00367, 0.00845] |
| Artifact | 4 | Execution | -0.03947 | 0.01787 | 0.05734 [0.04222, 0.07366] |
| Artifact | 4 | Presentation | 2.52958 | 0.55771 | -1.97187 [-2.00338, -1.93883] |
| Sparse | 4 | Execution | -0.04310 | 0.01261 | 0.05571 [0.04580, 0.06639] |
| Sparse | 4 | Presentation | 2.05006 | 0.48227 | -1.56778 [-1.59943, -1.53430] |
| Complete | 4 | Execution | -0.06677 | 0.00256 | 0.06933 [0.05727, 0.08183] |
| Complete | 4 | Presentation | -0.00166 | -0.00233 | -0.00067 [-0.00720, 0.00632] |
| Artifact | 16 | Execution | -0.24103 | 0.00552 | 0.24655 [0.19241, 0.30368] |
| Artifact | 16 | Presentation | 5.19974 | 0.33186 | -4.86788 [-4.98436, -4.74501] |
| Sparse | 16 | Execution | -0.08287 | 0.00418 | 0.08705 [0.06973, 0.10630] |
| Sparse | 16 | Presentation | 2.43202 | 0.30837 | -2.12366 [-2.23457, -2.00821] |
| Complete | 16 | Execution | -0.07849 | -0.00016 | 0.07833 [0.05998, 0.09816] |
| Complete | 16 | Presentation | -0.00809 | -0.00111 | 0.00698 [0.00102, 0.01325] |
| Artifact | 64 | Execution | -0.72080 | 0.00173 | 0.72253 [0.48929, 0.98187] |
| Artifact | 64 | Presentation | 10.42557 | 0.03066 | -10.39491 [-10.53382, -10.25071] |
| Sparse | 64 | Execution | -0.09593 | 0.00240 | 0.09832 [0.07926, 0.11793] |
| Sparse | 64 | Presentation | 2.10480 | 0.01772 | -2.08707 [-2.25413, -1.91820] |

The four-episode context exception is specific to the execution change: native structured loss is 3.04725 versus joint neural 3.02844. Under the changed rule, structured loss falls by 0.04197 while neural loss rises by 0.01469. Relative structured advantage grows by 0.05666, with world interval 0.04628 to 0.06765. The earlier artifact exceptions at four, sixteen and sixty-four episodes likewise involve lower structured loss under the execution change. None establishes a native-rule advantage. The frozen confirmation candidate remains unchanged.

Training data are byte-identical within each same-fit native/shift pair. Current and retained episodes shift together, and evaluation samples differ, so rows are not paired. The execution change reverses repair under both proposal routes. All learned readers receive exact legal support from each supplied law. These comparisons do not test learning an unknown law, and a change in loss can reflect changed target difficulty as well as reader behavior.

Table: paired-fit secondary scores for the native comparisons. Squared probability loss sums squared errors across joint probabilities and is proper; lower is better. Accuracy is the fraction with the correct most-probable joint answer; higher is better. Confident wrong attribution is the fraction wrong at at least 90% confidence. Each cell compares structured / joint neural reading. All other arms and goal/tool/order/inspection scores remain in the checked aggregates.

| Evidence | History | Squared loss | Accuracy | Confident wrong attribution |
|---|---:|---|---|---|
| Context | 0 | 0.95316 / 0.95350 | 0.06248 / 0.06650 | 0.0000000 / 0.0000000 |
| Context | 1 | 0.94317 / 0.94406 | 0.07580 / 0.08136 | 0.0000000 / 0.0000000 |
| Context | 4 | 0.92924 / 0.92905 | 0.08970 / 0.09667 | 0.0000000 / 0.0000000 |
| Context | 16 | 0.92525 / 0.91424 | 0.10061 / 0.10600 | 0.0000000 / 0.0000000 |
| Context | 64 | 0.98146 / 0.90555 | 0.10400 / 0.11204 | 0.0002136 / 0.0000000 |
| Artifact | 0 | 0.96837 / 0.96907 | 0.05562 / 0.05510 | 0.0000000 / 0.0000000 |
| Sparse | 0 | 0.93763 / 0.93553 | 0.08254 / 0.08825 | 0.0000000 / 0.0000000 |
| Complete | 0 | 0.77922 / 0.77416 | 0.26195 / 0.25515 | 0.0000000 / 0.0000000 |
| Artifact | 1 | 0.96369 / 0.96481 | 0.06543 / 0.06358 | 0.0000000 / 0.0000000 |
| Sparse | 1 | 0.92457 / 0.92165 | 0.09940 / 0.10661 | 0.0000000 / 0.0000000 |
| Complete | 1 | 0.72683 / 0.71625 | 0.31873 / 0.31336 | 0.0000000 / 0.0000000 |
| Artifact | 4 | 0.95898 / 0.95836 | 0.07433 / 0.07310 | 0.0000000 / 0.0000000 |
| Sparse | 4 | 0.90637 / 0.89858 | 0.11910 / 0.12722 | 0.0000000 / 0.0000000 |
| Complete | 4 | 0.65452 / 0.63081 | 0.39738 / 0.38911 | 0.0000000 / 0.0000000 |
| Artifact | 16 | 0.95991 / 0.95726 | 0.07680 / 0.07172 | 0.0000000 / 0.0000000 |
| Sparse | 16 | 0.90298 / 0.87529 | 0.13424 / 0.14085 | 0.0000153 / 0.0000000 |
| Complete | 16 | 0.58982 / 0.54315 | 0.47072 / 0.46520 | 0.0027161 / 0.0000000 |
| Artifact | 64 | 0.97382 / 0.95981 | 0.08060 / 0.06177 | 0.0000000 / 0.0000000 |
| Sparse | 64 | 0.97744 / 0.86100 | 0.13844 / 0.15163 | 0.0024185 / 0.0000000 |

Table: percentage of new second-fit truth probabilities below the log-score floor, by native condition. Exact-zero truth mass is separately zero for every new arm. Only independent-bit reading has below-floor truths; the table omits its zero entries.

| Evidence | History | Independent-bit truths below floor |
|---|---:|---:|
| Context | 1 | 73.94714% |
| Context | 4 | 58.43506% |
| Context | 16 | 16.92505% |
| Context | 64 | 39.94293% |
| Artifact | 1 | 92.17529% |
| Sparse | 1 | 67.78259% |
| Complete | 1 | 15.31677% |
| Artifact | 4 | 89.66980% |
| Sparse | 4 | 71.06934% |
| Complete | 4 | 1.23901% |
| Artifact | 16 | 67.97638% |
| Sparse | 16 | 42.12646% |
| Artifact | 64 | 62.31689% |
| Sparse | 64 | 29.87823% |

Capping hides the full penalty of tiny positive probabilities; capped log loss is not an uncapped proper score. Every stored forecast, secondary and summary reconstructs. Only reader/ is blind input; casebooks and summaries are evaluator artifacts. COMPLETE, plan, source, environment, fit/decode costs and attempted failures reconcile. Execution, numerical acceptance, discovery criteria and confirmation remain separate.

Full opening G1/G4/G5/G6, repaired scoring, long-history and shift reference replays are reused by immutable hashes, including both-fit positive shifted references. Native-to-shift interpretation was established in the first-fit reports; completing matched second native fits adds no implementation family, repair or changed interpretation. No additional replay is required. Validation does not cover fresh installation, cross-platform replay, universal structured-reader performance, real-text history or human intent.

Original blocks used 6834.46875 native CPU seconds. Independent numerical review used 198.00000, and the matched-fit/shift supplement 3.81250. Forecasting, documents, publication and operating overhead are charged separately. Earlier failures and accounting limitations remain retained. Total/discovery ceilings remain 90/72 CPU hours with 18 protected.

Thirty-two frozen 128/512-label, 64-episode comparisons were admitted across all four evidence tiers, both shifts and both fits. They test budget dependence beside admitted 2,048/8,192/32,768-label counterparts. Measured useful runway rises from 19.76 to 22.67 hours across 230 discovery blocks. The 1.33-hour shortfall from one day preserves two hours of future operating/publication headroom inside the discovery ceiling. This headroom is additional to protected confirmation reserve. Remaining conditional designs are not exhausted; eighteen redundant acquisition descriptions remain excluded. The broader 1.25 planning cushion is short by 10.50 hours. Forecasts are not execution or occupancy. Further refill must pass both wall and CPU limits.

At October 1 08:00 UTC the twelve frozen diagnostic confirmation blocks have first priority, followed by the original two-fit context extension within its inclusive 4,000-second cap, reserve and cutoff. Confirmation and challenge lineages remain untouched. Midpoint reporting, science cutoff and Friday delivery retain their fixed times.

[Validity](../../../results/v20/native-reference-wave-1/VALIDITY.json), [paired fits](../../../results/v20/native-reference-wave-1/PAIRED_FITS.json), [matched shifts](../../../results/v20/native-reference-wave-1/PAIRED_SHIFTS.json), [all scores](../../../results/v20/native-reference-wave-1/AGGREGATES.json), [zero audit](../../../results/v20/native-reference-wave-1/ZERO_MASS_AUDIT.json), [replay reuse](../../../results/v20/native-reference-wave-1/REPLAY_REUSE.json), [positive replay reuse](../../../results/v20/native-reference-wave-1/POSITIVE_REPLAY_REUSE.json), [costs](../../../results/v20/native-reference-wave-1/COSTS.json), [forecast](../../../results/v20/native-reference-wave-1/FORECAST.json).

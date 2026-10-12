# V21 calibrated rival for partial answers

The test for a maker-specific partial-answer advantage finds none at fixed specificity against the calibrated rival in the native artifact condition. At four, five and six asserted fields, the maker's false-claim differences are +0.293, +0.146 and -0.244 percentage points, with every interval crossing zero. A small adaptive matched-error gain remains: at penalty five both make no observed false claims, while the maker asserts 0.033 more fields per case. These are verified constructed-method development results, not confirmation or evidence of maker-specific causation.

The question is whether the earlier native-artifact partial-answer result survives a strong generic correction for dependence. The new rival selects an episode-likelihood power on validation worlds from 0.25, 0.5 and 1, using the registered independent and copied-history conditions. Its power is frozen here. This supplement uses its zero-artificial-copying predictions in the same six selected native/changed conditions as the [original answer-policy study](ANSWER_POLICY_REPORT.md); it introduces no training labels, fits, worlds or probability refitting.

Each condition has eight original-law development worlds, 256 cases per world and two fitted seeds averaged within each world. Artifact uses 8,192 training labels; context and complete evidence use 512. There are eight training laws and four same-tier past episodes, with inspection hidden. The changed condition alters current and historical execution, unlike the current-only B07 adaptation intervention. The same query truths match the new rival to every original policy within a condition. Native/changed comparisons pair worlds, not individual cases. The 4,096 paired whole-world bootstrap draws give descriptive 95% intervals; selection and multiple contrasts are not multiplicity-adjusted confirmation.

Every reader has the same finite action library: assert any subset of seven binary fields, including none. Realized loss is the withheld fraction of seven fields plus a penalty of 1, 2, 5 or 10 if any asserted field is wrong. Expected risk under a forecast is distinct from realized risk. A fixed number of fields matches specificity and answers every case, but can select different fields across readers. Retained alternatives count all finite assignments compatible with the assertion, including zero-probability assignments; they do not count supported historical routes.

This table reports native-artifact maker minus calibrated-rival false claims in percentage points at each fixed number of asserted fields. Negative favors the maker; brackets are paired-world descriptive 95% intervals. Zero fields is abstention. Fixed-cardinality decisions are identical across penalties.

| Asserted fields | False-claim difference [interval], percentage points |
|---|---|
| 0 | +0.000 [+0.000, +0.000] |
| 1 | +0.000 [+0.000, +0.000] |
| 2 | -0.317 [-1.245, +0.659] |
| 3 | -0.488 [-1.465, +0.562] |
| 4 | +0.293 [-0.610, +1.050] |
| 5 | +0.146 [-0.415, +0.757] |
| 6 | -0.244 [-0.854, +0.342] |
| 7 | +0.342 [-0.342, +1.050] |

The earlier four-to-six-field advantage over the original rivals was real for that rival set. The calibrated rival removes the established fixed-specificity benefit. This is not an equivalence proof: eight-world intervals permit small differences. The smaller adaptive improvement remains a separate result. At penalty five both native-artifact readers have zero observed false claims and answer something in 64.795% of cases; the maker asserts 1.294 versus 1.261 fields per case. Its loss is lower by 0.00467, with interval [-0.00600, -0.00331], entirely from the extra correct assertions in this finite sample. No empirical zero-error rate guarantees the same error on fresh worlds. At penalties one, two and ten its adaptive loss is also lower than the calibrated rival by 0.02148, 0.02403 and 0.00188 respectively, but coverage and specificity differ; these are distinct operating points.

The next table fixes the false-specificity penalty at five. Rows identify evidence and execution. Loss is lower-is-better; the two numbers in each paired column are maker / calibrated rival. The last column is the best observed adaptive same-label rival among all seven implemented rivals, chosen descriptively on these development worlds. The extra-label fresh-law diagnostic is excluded from that choice.

| Condition | Maker loss | Calibrated loss | Asserted fields, maker / rival | False claims, maker / rival | Best observed matched rival |
|---|---:|---:|---:|---:|---|
| Artifact, native | 0.81515 | 0.81982 | 1.294 / 1.261 | 0.000% / 0.000% | Calibrated episode weighting |
| Artifact, changed | 1.32652 | 1.06592 | 1.730 / 1.863 | 11.475% / 6.641% | Fixed-schedule network |
| Context, native | 0.79440 | 0.68520 | 2.302 / 2.212 | 2.466% / 0.024% | Exact table |
| Context, changed | 0.67536 | 0.66420 | 2.358 / 2.351 | 0.244% / 0.000% | Exact table |
| Complete, native | 0.36140 | 0.22674 | 5.513 / 5.438 | 2.979% / 0.073% | Exact table |
| Complete, changed | 0.22524 | 0.22405 | 5.432 / 5.432 | 0.024% / 0.000% | Exact table |

Several other individual contrasts favor the maker against calibrated weighting, including changed artifact at four through seven fields and changed context at two and three. They do not establish superiority over the entire matched rival set: the strongest original rival still prevents the shifted fixed-specificity claim. Changed complete evidence has a small adaptive gain over calibrated weighting at penalty one, but not a general advantage across penalties. All cardinalities, ratios and strongest-rival paired contrasts remain in the complete analysis; none is silently promoted.

This table reports every reader's changed-minus-native adaptive loss at penalty five. Positive means worsening. Columns correspond to artifact, context and complete evidence. Other penalties, fixed cardinalities, error, coverage, specificity and alternatives are retained in the aggregate record.

| Reader | Artifact change | Context change | Complete change |
|---|---:|---:|---:|
| Learned maker | +0.51137 | -0.11904 | -0.13616 |
| Validation-stopped direct | +0.56773 | -0.02497 | -0.05608 |
| Exact table | -0.98936 | -0.02030 | +0.00070 |
| Fixed-schedule network | -0.01381 | -0.02427 | -0.05350 |
| Extra-label fresh-law direct | +1.43119 | +0.05284 | +0.02515 |
| Grouped empirical | -0.17578 | -0.02434 | +0.00070 |
| Naive Bayes | -0.03582 | +0.02958 | -0.09204 |
| Naive Bayes, current duplicate removed | -0.00105 | +0.11332 | -0.04265 |
| Calibrated episode weighting | +0.24609 | -0.02100 | -0.00269 |

The original naive-Bayes arm and removal of its current duplicate factor remain distinct controls, with improvements and reversals preserved. The calibrated rival discounts whole episodes and is not that duplicate-removal intervention. No artificially copied query is added here; the known-copy-root arm equals the latent arm at zero artificial copying and supplies no distinct forecast. Common exact current support remains a supplied privilege. Direct fitting is validation-stopped; the maker has fixed-step mixture fitting and different capacity. This policy comparison adds no law-marginal information and cannot resolve historical aliases hidden in the unchanged forecast. Fresh-law direct receives extra generated labels and remains a separate diagnostic. These are miniature constructed-method results; architecture generality is untested.

All 26 input/completion bindings pass, reusing 19 exact score receipts. Seven new policy packets include the complete frozen native-artifact replay, matching all 577 scientific files byte for byte. Independent enumeration evaluates all 2,187 partial assignments for each of 24,576 new rival forecasts, verifies 884,736 original policy decisions and all summaries within 7.55e-15, and checks another 147,456 replay decisions by exact correspondence. The 589,824 repeated fixed-ratio rows are controls. The original 196,608 forecasts and 7,077,888 decisions retain their verified hash bindings; reusing them is not fresh corroboration. No new scientific fit, family repair or confirmation occurred.

The prior probability candidates also retain their separate learned-persistence limitation: their shifted gains did not establish maker-specific historical-content use. No mechanism promotion is frozen by this result. A diagnostic separating information limits from reader failure remains the fallback for the fixed confirmation gate. Reserved confirmation and challenge worlds are unopened.

The next registered simple-stopping D04 extension passed three focused controls but failed its first complete engineering export on a NumPy Boolean serialization error. Its source, partial output and CPU cost are retained. D04's single repair allowance was already consumed by its earlier zero-prior repair, so this extension is stopped without a second repair or scientific admission. It supplies no result about stopping or CPU advantage. Other unimplemented and resource-deferred scope remains visible; the campaign is not closed or exhausted. The existing supervisor stays held, with no ready scientific work, until a scheduled operating decision or fixed gate.

[Validity](../../../results/v21/answer-calibrated/VALIDITY.json), [all compact contrasts](../../../results/v21/answer-calibrated/ANALYSIS.json), [complete paired analysis](../../../results/v21/answer-calibrated/FULL_ANALYSIS.json.gz), [source and replay bindings](../../../results/v21/answer-calibrated/SOURCE_BINDINGS.json).

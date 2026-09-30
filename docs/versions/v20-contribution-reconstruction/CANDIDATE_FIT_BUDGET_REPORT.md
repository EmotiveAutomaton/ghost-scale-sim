# V20 candidate expansion: fit and label-budget sensitivity

We tested whether finite candidate expansion still helps after reducing labels or repeating training. Adding one or four ranked alternatives improves truth coverage and capped log loss in both panels; with four additions, coverage reaches 81.82% with fewer labels and 82.35% after averaging the larger-data fits. Forced choice still gives zero probability to 24.98% of truths. All 1,572,864 new forecasts verify. These are constructed-method findings; confirmation remains untouched.

Six new packets test zero, one and four added alternatives with 2,048 labels in the first registered fit and 32,768 labels in the second. The previously verified 32,768-label first fit supplies matched references. Each condition has 32 discovery coefficient worlds and 2,048 evaluation cases per world. Average both registered fits within each world before the fixed-seed 4,096-resample whole-world bootstrap. The smaller-label panel has only one fit. Rows, budgets and repeated training fits are not independent worlds.

## Common-target forecast comparison

The initial list contains 64 of 128 supplied process alternatives and omits every route-one answer. Expansion uses the learned decoder to rank omitted alternatives with a fixed stable tie rule. Preserved unknown mass is distributed uniformly over omitted alternatives only for full-universe scoring. This is a finite supplied universe, not autonomous open-world invention.

Capped log loss is negative log truth probability with a floor of 1e-12, in natural-log units; lower is better. Exact-zero truth probability is separate. Squared probability loss is also checked. Candidate-plus-unknown event loss changes its target partition as the list grows and must not rank budgets on a common target.

Table: rows are label/fit panels and counts of added alternatives. Coverage is the percentage of truths explicitly listed. Gain is the decrease in full-universe capped loss relative to preserving unknown mass without expansion; brackets are descriptive 95% whole-world intervals.

| Training panel | Added alternatives | Truth coverage | Capped loss | Loss gain [interval] |
|---|---:|---:|---:|---:|
| 2,048 labels, first fit | 0 | 75.02% | 4.00568 | 0.00000 [0.00000, 0.00000] |
| 2,048 labels, first fit | 1 | 76.97% | 3.99465 | 0.01104 [0.00888, 0.01336] |
| 2,048 labels, first fit | 4 | 81.82% | 3.94141 | 0.06427 [0.05805, 0.07071] |
| 32,768 labels, both fits | 0 | 75.02% | 3.81923 | 0.00000 [0.00000, 0.00000] |
| 32,768 labels, both fits | 1 | 77.05% | 3.79515 | 0.02408 [0.02175, 0.02650] |
| 32,768 labels, both fits | 4 | 82.35% | 3.71243 | 0.10680 [0.10122, 0.11297] |

Zero-budget expansion is exactly the unknown-mass baseline. Both nonzero budgets improve capped loss in each panel, with positive lower descriptive intervals. All squared-loss contrasts are retained in the linked aggregates; the zero-budget equality is a control, not evidence of a learned improvement.

The matched first-fit comparison separates data quantity from averaging fits. With four additions, using 2,048 instead of 32,768 labels raises capped loss by 0.22928 [0.21824, 0.23967] and changes truth coverage by -0.615 percentage points [-0.797, -0.420]. More labels improve this finite decoder; this is not a new structured-versus-direct primary comparison. The one-addition coverage difference is unresolved, with its interval spanning zero.

## Omission failures and evidence limits

Forced choice assigns exact zero to the same 16,372 of 65,536 truths in every new packet: 24.98169%. Its uncapped log loss is infinite. The six packets retain 98,232 stored zero rows, which repeat the same omissions rather than create independent replications. Every other new arm has zero exact-zero truths. The negative forced-choice result remains a failure even when its capped score is finite.

The supplied-hypothesis control uses evaluator truth to insert the right answer and is privileged. Its success does not show autonomous candidate invention. Identical training/evaluation records and baseline forecasts across expansion budgets were checked byte for byte. The two larger-data fits vary training and fitting within the same worlds; their rows are not fresh confirmation.

## Verification and continuation

All six COMPLETE, plan, frozen source/archive, environment, fit, cost and blind/evaluator bindings verify. The existing analysis reviewer independently reconstructs every stored forecast, primary/secondary score, summary and candidate-event score. A separate count-based decoder reconstruction checks every candidate ordering, mask, transported mass and saved forecast, reusing legal support only from a hash-verified full-universe reference with identical inputs.

Both full four-addition scientific blocks replay from their frozen source: the 2,048-label first fit and 32,768-label second fit. Each reproduces all 450 scientific files and its complete summary across all 32 worlds. Earlier complete G1/G4/G5/G6, scorer-repair and candidate/correction replays remain bound by immutable hashes. No producer, criterion, model setting or environment changed. Earlier failed attempts and sources remain preserved; no new scientific failure occurred.

The verified total is 438 packets. No new admission is needed: 29.61 measured eligible hours remain across 123 discovery blocks. Larger candidate budgets, correction conditions and matched fit comparisons are already admitted. The combined useful forecast misses the 1.25 planning cushion by 0.52 hours at this snapshot; the shortfall is retained without inflated estimates.

All computation and overhead share the 90-hour CPU ceiling, with discovery capped at 72 and 18 protected. Twelve frozen diagnostic confirmation blocks remain first priority on October 1 at 08:00 UTC, followed by the separately frozen context extension subject to its existing cap. No confirmation or challenge lineage was used here. Validation does not establish fresh-installation or cross-platform reproduction, universal architecture, open-world invention, or human intent.

[Validity](../../../results/v20/omission-wave-2/VALIDITY.json), [aggregates](../../../results/v20/omission-wave-2/AGGREGATES.json), [paired fits](../../../results/v20/omission-wave-2/PAIRED_FITS.json), [paired label budgets](../../../results/v20/omission-wave-2/PAIRED_BUDGETS.json), [candidate audit](../../../results/v20/omission-wave-2/CANDIDATE_AUDIT.json), [full replays](../../../results/v20/omission-wave-2/REPLAY.json), [costs](../../../results/v20/omission-wave-2/COSTS.json), [forecast](../../../results/v20/omission-wave-2/FORECAST.json).

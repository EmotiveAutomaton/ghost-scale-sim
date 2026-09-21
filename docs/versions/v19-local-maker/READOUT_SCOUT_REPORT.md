# G19-04 artifact-history readout screen — 21 September 2026

Can a learned predictive bank make new questions easier to decode from artifact histories? In the old-world screen, it lowers logarithmic loss by 1.87980 nats versus raw history and 0.68281 versus the frozen latent readout, averaged over the two training budgets and four equally weighted classes. Yet at 128 target-labelled histories its loss is 3.00938 nats, versus 2.35439 for the full-data no-history control and 0.78491 for the supplied-law exact reference. This is an exploratory constructed-method advantage among these fitted readouts, miniature — architecture untested; useful local-goal access and process correspondence are not established.


Logarithmic loss measures forecast error in nats; lower is better. Each entry below is the learned bank's difference from a comparator, averaged by the normalized area over log training budget. With two budgets this is the average of the two differences. Brackets show 95% paired coefficient-lineage intervals conditional on the retained fits. Rows retain all four world classes and their equal-weight population; endogenous choices means the maker's goal and tradeoff affect the considered option set.

| Population | Bank minus raw history | Bank minus frozen latent | Bank minus full-data mean |
|---|---:|---:|---:|
| Softmax, endogenous choices off | -1.87791 [-2.09882, -1.63535] | -0.62991 [-0.87591, -0.41443] | 0.96538 [0.58405, 1.37083] |
| Softmax, endogenous choices on | -1.01585 [-1.42901, -0.58491] | -0.64992 [-0.88974, -0.34513] | 2.51389 [2.17613, 2.89242] |
| Satisficing, endogenous choices off | -2.78965 [-3.32829, -2.32522] | -0.71572 [-1.04562, -0.37494] | 0.44659 [-0.05919, 0.92726] |
| Satisficing, endogenous choices on | -1.83579 [-2.34168, -1.35955] | -0.73571 [-1.17478, -0.26372] | 1.50719 [0.92283, 2.03912] |
| Four classes equally weighted | -1.87980 [-2.20198, -1.58886] | -0.68281 [-0.85358, -0.53045] | 1.35826 [1.07543, 1.60744] |

The next table reports absolute forecast loss for every main arm. Columns are the two nested target-label budgets; rows share 2,048 old-question training records per training draw. The full-data no-history mean also uses all 2,048 **new-target** labels, so it is a stronger-supervision capability diagnostic rather than a label-matched primary comparator. The exact bank is privileged input to a learned head; only the final exact-reference row supplies the target law itself.

| Readout | 32 target-labelled histories | 128 target-labelled histories |
|---|---:|---:|
| Raw history | 5.63701 | 5.54789 |
| Frozen old-task latent | 5.14780 | 3.64313 |
| Learned bank | 4.41593 | 3.00938 |
| Exact bank, fitted target head | 4.70716 | 3.11187 |
| Full-data no-history mean | 2.35439 | 2.35439 |
| Supplied-law exact reference | 0.78491 | 0.78491 |

The independent, explicitly post-screen label-budget audit also evaluates the mean of only the same first 32 or 128 training target records. Its losses are 2.65905 and 2.36174 nats respectively, still below the learned bank at both budgets. This diagnostic adds no sampled histories, changes no original score, and was not a preregistered arm. It motivates a probability-valid or shrinkage readout comparison rather than a claim of useful bank access.

The screen contains eight development coefficient lineages, two fixed feature seeds, two independently sampled training draws, 128 training lineages with 16 histories each, and 32 development histories per lineage per draw. The 5,120 retained score rows aggregate histories, queries, seeds, budgets and repeated class summaries; they are not 5,120 independent worlds. Both seeds and training draws remain within paired lineage. The area advantage has the same sign in each of the four seed/draw combinations. The complete record preserves each budget, class, query and fit combination. These are development results; the 32 test and 32 confirmation lineages remain untouched. Two random-feature seeds do not satisfy the later five-initialization requirement for a trained main comparison.

The model sees eight artifact observations, their public query contexts and source-duplication indicators. It receives no realized program, rule, policy matrix, hidden state or unknown numerical law parameter as a fitted input. The exact reference separately knows the world law and uses the same uniform 24-state prior that generates these histories. This differs from the earlier privileged-program A1 evidence and cannot be pooled with it as the same task. Even the old history-inert class is conditional on knowing its law; a cross-law no-history mean is not that conditional oracle.

Old-question fitting improves over its full-data mean in each of the four feature-seed/training-draw combinations. New-question decoding remains weak: at 128 target labels, the learned-bank head emits invalid raw category probabilities in 99.17% of history/query predictions before its preregistered clipping, category floor and normalization. The corresponding figures are 100% for raw history, 99.85% for frozen latent and 98.49% for the exact-bank fitted head. The scores use the declared repair; no scoring rule changed after outcomes. The many repairs and the failure of the privileged-bank fitted head locate a substantial readout/regularization rival, without isolating it from geometry, law generalization or bank estimation error. The squared probability error gives a different ranking against the no-history mean at 128 labels, so this diagnosis is specifically about logarithmic loss and its sensitivity to near-zero forecasts.

**Disposition:** exploratory relative advantage, with inadequate useful-access evidence at the tested budgets; no confirmation and no tiny-reader admission. All six scoped fixture controls pass. They check implementation properties, not useful held-out capability. **Pursuit:** retain the local primary screen already admitted under the frozen protocol, inspect its exact-bank/process mismatch separately, then freeze one bounded probability-valid or mean-shrinkage diagnostic using retained training data. Independent frame and jointness leaves remain admitted and do not depend on this screen winning.

**Evidence-role audit:** original arrays labelled as reader transport include pairing identifiers whose third coordinate reveals the hidden old-world class. Code inspection confirms that fitting uses only the feature array; these identifiers enter scoring, not model input. Preserve those originals in the evaluator/reproduction capsule. The separate reader archive removes every pairing identifier and keeps only declared features and training teacher labels. Development targets, exact banks, model forecasts and evaluator truth are excluded. This is an export correction, not a retrospective change to the scientific source or its results.

**Validation:** all 5,120 score rows were independently rebuilt from saved forecasts and evaluator targets with zero discrepancy, all original cell means and paired area intervals reproduce, and the equal-class population matches the average of its four classes within each lineage. Adjacent full replay and a second full run from freshly extracted source match every deterministic output byte, including models, forecasts and raw evidence. Timing records retain their own hashes and are not required to repeat. These checks do not establish local process correspondence, an architecture-independent bank advantage, or a training-population confidence interval.

[Scientific export and checks](../../../results/v19/G19-04-readout-scout-1/EXPORT_MANIFEST.json). [Reader-only archive](../../../results/v19/G19-04-readout-scout-1/READER.zip). [Independent reconstruction](../../../results/v19/G19-04-readout-scout-1/INDEPENDENT_REVIEW.json).

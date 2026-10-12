# Learned persistence and unrelated historical content

Do the two shifted candidates use maker-specific historical content? Neither establishes it: replacing that content independently improves the context candidate's log loss by 0.0145 nats and does not establish a loss change for the complete-evidence candidate. Their inferred same-maker rates remain 0.600 and 0.671 at nominal zero persistence. These are verified constructed-method diagnostics; personalization scale-up stops, with context information, frozen fitting and supplied support kept separate.

The question was whether the two changed-execution candidates that survived generic episode tempering actually use maker-specific content. Four original/changed context and complete-evidence conditions reuse eight development worlds, 256 cases per world and two clean-trained fits. Each fit uses 512 labels from eight training laws, a 32-state maker mixture and 40 fitting steps. Fits are averaged within worlds. These are selected diagnostics on reused V21 data, with no confirmation exposure.

The nominal same-maker retention rates are 1, 0.9, 0.5 and 0, plus one coherent foreign-author bundle. Each replacement preserves the original current case, every historical context and the four-episode count. Independent replacement draws a uniform maker conditional on the retained context; that draw can coincidentally match the original maker. The coherent bundle shares one foreign maker and lies outside the independent-episode contamination model. Historical contexts can retain maker information even after all other content is replaced.

Natural-log loss is the average negative logarithm of the probability assigned to the true seven-bit history, in nats; lower is better. Full multiclass Brier is the sum of squared errors across all 128 target probabilities; lower is better. Intervals below are descriptive 95% paired bootstrap intervals over eight worlds, using 4,096 draws. Neither cases nor fits increase the number of independent worlds.

At nominal zero retention, the inferred reader improves changed-context loss by 0.01447 nats (interval 0.00967 to 0.01966) relative to its clean-history condition. Changed complete-evidence improvement is 0.00112 (interval −0.00058 to 0.00295), which does not establish a change or equivalence. Rigid same-maker forecasts show the same direction. Unrelated historical content therefore does not eliminate the reported benefit. The nominal-zero inferred same-maker rates remain 0.60022 and 0.67115; clean rates were 0.62984 and 0.67179. Rate detection, predictive loss and useful context information are separate outcomes.

In changed context with clean history, inferred persistence improves loss by 0.03985 nats over validation-calibrated episodes, but full Brier worsens by 0.00410. In changed complete evidence, those differences are −0.04977 nats and −0.02606 Brier. These generic-rival comparisons do not establish maker-specific content use. Native conditions remain substantially worse than the best retained matched rival. No new exact-law ceiling or refitted direct-convergence check exists for contaminated cells.

The following table gives each evidence/execution condition and nominal retention rate. Columns show inferred-reader loss, rigid-reader loss, contexts-only loss, no-history loss, and the lowest-loss retained matched rival (selected descriptively on these same worlds). Lower loss is better.

| Condition | Retention | Inferred | Rigid | Contexts only | No history | Best matched rival | Rival loss |
|---|---:|---:|---:|---:|---:|---|---:|
| Native context | 1.0 | 3.65373 | 3.76878 | 3.52954 | 3.41868 | Exact-match table | 3.27788 |
| Native context | 0.9 | 3.64049 | 3.74827 | 3.52954 | 3.41868 | Exact-match table | 3.27788 |
| Native context | 0.5 | 3.60005 | 3.67370 | 3.52954 | 3.41868 | Exact-match table | 3.27788 |
| Native context | 0.0 | 3.54977 | 3.59883 | 3.52954 | 3.41868 | Pooled empirical | 3.27695 |
| Native context | shuffled-author | 3.55834 | 3.61868 | 3.52954 | 3.41868 | Exact-match table | 3.27788 |
| Changed context | 1.0 | 3.10717 | 3.09418 | 3.11817 | 3.11561 | Validation-calibrated episodes | 3.14702 |
| Changed context | 0.9 | 3.10695 | 3.09208 | 3.11817 | 3.11561 | Validation-calibrated episodes | 3.14626 |
| Changed context | 0.5 | 3.10297 | 3.08646 | 3.11817 | 3.11561 | Validation-calibrated episodes | 3.14498 |
| Changed context | 0.0 | 3.09270 | 3.07433 | 3.11817 | 3.11561 | Validation-calibrated episodes | 3.13933 |
| Changed context | shuffled-author | 3.09232 | 3.07491 | 3.11817 | 3.11561 | Validation-calibrated episodes | 3.14303 |
| Native complete evidence | 1.0 | 1.28581 | 1.27544 | 1.27999 | 1.27333 | Pooled empirical | 1.16991 |
| Native complete evidence | 0.9 | 1.28726 | 1.27937 | 1.27999 | 1.27333 | Pooled empirical | 1.16991 |
| Native complete evidence | 0.5 | 1.26719 | 1.24097 | 1.27999 | 1.27333 | Pooled empirical | 1.16991 |
| Native complete evidence | 0.0 | 1.26544 | 1.24490 | 1.27999 | 1.27333 | Pooled empirical | 1.16991 |
| Native complete evidence | shuffled-author | 1.26079 | 1.23592 | 1.27999 | 1.27333 | Pooled empirical | 1.16991 |
| Changed complete evidence | 1.0 | 1.10142 | 1.09972 | 1.10433 | 1.10447 | Validation-calibrated episodes | 1.15119 |
| Changed complete evidence | 0.9 | 1.10122 | 1.09960 | 1.10433 | 1.10447 | Validation-calibrated episodes | 1.15113 |
| Changed complete evidence | 0.5 | 1.10049 | 1.09874 | 1.10433 | 1.10447 | Validation-calibrated episodes | 1.15105 |
| Changed complete evidence | 0.0 | 1.10030 | 1.09852 | 1.10433 | 1.10447 | Validation-calibrated episodes | 1.15118 |
| Changed complete evidence | shuffled-author | 1.10068 | 1.09893 | 1.10433 | 1.10447 | Validation-calibrated episodes | 1.15112 |

## Every reader: native-to-changed natural-log loss

Rows identify the evidence condition and reader; columns give changed minus native natural-log loss at each retention intervention. Negative means the score improves under changed execution. Both current and historical execution change; coefficient worlds are paired, individual case samples differ. This is not the sparse-current adaptation experiment. Full uncertainty, accuracy, confidence-error and top-eight changes are retained in [the shift record](../../../results/v21/learned-persistence/NATIVE_TO_SHIFT.json).

| Evidence / reader | 1 | 0.9 | 0.5 | 0 | Coherent foreign author |
|---|---:|---:|---:|---:|---:|
| Context / Inferred persistence | -0.54656 | -0.53353 | -0.49708 | -0.45707 | -0.46602 |
| Context / Rigid same maker | -0.67460 | -0.65618 | -0.58724 | -0.52450 | -0.54377 |
| Context / Supplied rate | -0.67460 | -0.62573 | -0.49401 | -0.41137 | -0.41137 |
| Context / Historical contexts only | -0.41137 | -0.41137 | -0.41137 | -0.41137 | -0.41137 |
| Context / No history | -0.30307 | -0.30307 | -0.30307 | -0.30307 | -0.30307 |
| Context / Validation-stopped direct | -0.13608 | -0.14104 | -0.15706 | -0.18599 | -0.19184 |
| Context / Pooled empirical | -0.08073 | -0.08081 | -0.07800 | -0.07624 | -0.07850 |
| Context / Exact-match table | -0.07717 | -0.07717 | -0.07717 | -0.07717 | -0.07717 |
| Context / Validation-calibrated episodes | -0.36365 | -0.36277 | -0.36091 | -0.36984 | -0.37702 |
| Context / Naive Bayes | -0.55895 | -0.57629 | -0.63003 | -0.69084 | -0.77219 |
| Context / Naive Bayes, duplicate removed | -0.16164 | -0.17088 | -0.18164 | -0.25898 | -0.36106 |
| Complete / Inferred persistence | -0.18439 | -0.18604 | -0.16670 | -0.16514 | -0.16011 |
| Complete / Rigid same maker | -0.17572 | -0.17977 | -0.14223 | -0.14638 | -0.13699 |
| Complete / Supplied rate | -0.17572 | -0.18924 | -0.17507 | -0.17566 | -0.17566 |
| Complete / Historical contexts only | -0.17566 | -0.17566 | -0.17566 | -0.17566 | -0.17566 |
| Complete / No history | -0.16886 | -0.16886 | -0.16886 | -0.16886 | -0.16886 |
| Complete / Validation-stopped direct | -0.07046 | -0.06656 | -0.05659 | -0.07612 | -0.07107 |
| Complete / Pooled empirical | -0.01566 | -0.01566 | -0.01566 | -0.01566 | -0.01566 |
| Complete / Exact-match table | -0.01566 | -0.01566 | -0.01566 | -0.01566 | -0.01566 |
| Complete / Validation-calibrated episodes | -0.16529 | -0.16398 | -0.15966 | -0.16099 | -0.16185 |
| Complete / Naive Bayes | -0.42252 | -0.37462 | -0.36109 | -0.36812 | -0.38478 |
| Complete / Naive Bayes, duplicate removed | -0.22354 | -0.17084 | -0.13902 | -0.15191 | -0.15774 |

## Every reader: native-to-changed full multiclass brier

Rows identify the evidence condition and reader; columns give changed minus native full multiclass brier at each retention intervention. Negative means the score improves under changed execution. Both current and historical execution change; coefficient worlds are paired, individual case samples differ. This is not the sparse-current adaptation experiment. Full uncertainty, accuracy, confidence-error and top-eight changes are retained in [the shift record](../../../results/v21/learned-persistence/NATIVE_TO_SHIFT.json).

| Evidence / reader | 1 | 0.9 | 0.5 | 0 | Coherent foreign author |
|---|---:|---:|---:|---:|---:|
| Context / Inferred persistence | -0.10215 | -0.09822 | -0.09159 | -0.08269 | -0.08810 |
| Context / Rigid same maker | -0.14795 | -0.14360 | -0.13268 | -0.11798 | -0.12276 |
| Context / Supplied rate | -0.14795 | -0.13006 | -0.08797 | -0.06003 | -0.06003 |
| Context / Historical contexts only | -0.06003 | -0.06003 | -0.06003 | -0.06003 | -0.06003 |
| Context / No history | -0.02520 | -0.02520 | -0.02520 | -0.02520 | -0.02520 |
| Context / Validation-stopped direct | -0.01245 | -0.01306 | -0.01510 | -0.01787 | -0.01753 |
| Context / Pooled empirical | -0.00742 | -0.00743 | -0.00633 | -0.00560 | -0.00637 |
| Context / Exact-match table | -0.00583 | -0.00583 | -0.00583 | -0.00583 | -0.00583 |
| Context / Validation-calibrated episodes | -0.04140 | -0.04113 | -0.04051 | -0.04203 | -0.04286 |
| Context / Naive Bayes | -0.03626 | -0.03906 | -0.04799 | -0.04805 | -0.04801 |
| Context / Naive Bayes, duplicate removed | 0.00367 | 0.00184 | -0.00080 | -0.00677 | -0.00937 |
| Complete / Inferred persistence | -0.09324 | -0.09462 | -0.08741 | -0.08755 | -0.08493 |
| Complete / Rigid same maker | -0.07419 | -0.07609 | -0.06179 | -0.06582 | -0.05990 |
| Complete / Supplied rate | -0.07419 | -0.08325 | -0.08918 | -0.09750 | -0.09750 |
| Complete / Historical contexts only | -0.09750 | -0.09750 | -0.09750 | -0.09750 | -0.09750 |
| Complete / No history | -0.09283 | -0.09283 | -0.09283 | -0.09283 | -0.09283 |
| Complete / Validation-stopped direct | -0.01668 | -0.01817 | -0.01167 | -0.01320 | -0.01041 |
| Complete / Pooled empirical | -0.00882 | -0.00882 | -0.00882 | -0.00882 | -0.00882 |
| Complete / Exact-match table | -0.00882 | -0.00882 | -0.00882 | -0.00882 | -0.00882 |
| Complete / Validation-calibrated episodes | -0.08498 | -0.08419 | -0.08213 | -0.08431 | -0.08436 |
| Complete / Naive Bayes | -0.02348 | -0.02628 | -0.01389 | -0.00726 | 0.00617 |
| Complete / Naive Bayes, duplicate removed | -0.02712 | -0.03002 | -0.00978 | -0.00493 | 0.00393 |

## Duplicate removal, validity and limits

Both naive-Bayes arms are reported under every intervention above. The duplicate-removed arm withholds the current evidence feature from multiplication; it does not remove historical episodes. [Paired duplicate-removal contrasts](../../../results/v21/learned-persistence/DUPLICATE_REMOVAL.json), [all reader contrasts](../../../results/v21/learned-persistence/CONTRASTS.json) and [complete score summaries](../../../results/v21/learned-persistence/SUMMARY.json) preserve reversals and individual target-bit scores.

All 1,126,400 new forecast scores pass independent reconstruction. Eight unchanged score receipts are reused after every completion, plan, environment, source and file binding is rechecked. The full scientific native-context replay matches all 1,002 files, including 1,000 array files. We reproduce 98,304 original rigid/direct/table/pooled/naive forecasts at retention one, independently check 409,600 scalar-mixture forecasts and all rate posteriors, reconstruct every replacement and context, and reaggregate every summary. Maximum probability discrepancy is 6.55e-15. Costs are charged. No scientific producer or scoring criterion changed.

The shared current support and generating rate are distinct supplied privileges. The latent model has fixed capacity and fixed fitting steps; the direct fits validation-stopped on clean data only. Exact upstream headroom does not create contaminated-cell headroom. Historical aliases, uncertainty about the law, content contamination and context information remain separate. Coherent foreign histories are a model-mismatch stress test. The supplied-rate arm is not a fully correct generator-law posterior. No human authorship conclusion, mechanism promotion or fresh confirmation follows.

The registered stop condition is applied to this personalization implementation. Its prior repair allowance remains consumed. A04 inspection-record tests are the next independently implemented branch; their admission, execution and later scientific acceptance remain distinct. C06/E02/E03 and genuine resource-deferred scope remain visible.

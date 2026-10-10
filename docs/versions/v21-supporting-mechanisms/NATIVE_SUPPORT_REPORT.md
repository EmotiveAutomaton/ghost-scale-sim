# V21 native support diagnostic - 10 October 2026

Does a hard rule excluding supposedly impossible histories improve reconstruction? In this native constructed-method diagnostic, supplied exact support lowers natural-log loss from 3.8643 to 3.4505 nats, but learned support excludes truth in 1.59% of forecasts and makes uncapped log loss infinite. The eight-world paired reduction is 0.4138 nats (descriptive 95% interval 0.3844 to 0.4459); the full Brier improvement remains uncertain. This is development evidence, miniature - architecture untested, with no shifted-law or structured-reader superiority claim.

## Design and scope

This is the first native G21-C03 block: eight original-law development worlds,
256 test cases in each, with two fitted seeds averaged inside each world. Both
64-unit direct networks use 2,048 labels from eight separate training laws and
4,096 validation labels from sixteen separate laws. Both stop on validation;
fit capacity and saved weights are identical across support arms. The current
observation is artifact-only; four past episodes have complete records including
inspection. These are 2,048 test cases, 4,096 fit-case pairs and 32,768 forecasts
across eight support arms, not 32,768 independent worlds.

The recorded prerequisite has exact-law headroom with a lower bound above .02
and both fits validation-stopped. Its hashes and saved scores were checked here.
This receipt accepts the support transformation of retained predictions. It does
not accept a new law-marginal boundary: the upstream ceiling and fitting families
still need their own complete scientific replay before first acceptance.

## Support effects

Each row is one support rule applied to the same retained direct predictor.
Natural-log loss and full multiclass Brier score measure probability error;
lower is better. Accuracy is the fraction of modal answers correct. Exclusion
is the fraction assigned exactly zero probability on the true history.

| Support rule | Natural-log loss | Full Brier | Accuracy | Truth excluded |
|---|---:|---:|---:|---:|
| None | 3.86430 | 0.97275 | 6.738% | 0% |
| Broad physical rule | 3.86430 | 0.97275 | 6.738% | 0% |
| Training-learned hard rule | infinite | 0.97363 | 7.275% | 1.587% |
| Supplied exact rule | 3.45050 | 0.96987 | 8.154% | 0% |
| Exact rule, zero omission control | 3.45050 | 0.96987 | 8.154% | 0% |
| Exact rule, 1% candidate omission | infinite | 0.97003 | 8.154% | 0.293% |
| Exact rule, 5% candidate omission | infinite | 0.97218 | 8.325% | 3.442% |
| Exact rule, 20% candidate omission | infinite | 0.98899 | 8.057% | 18.262% |

The exact rule's full Brier change is -0.002883, with a descriptive world-paired
95% interval of -0.007327 to +0.001402. Its accuracy gain is 1.416 percentage
points (1.099 to 1.855). The learned rule increases accuracy by 0.537 points while
making proper log loss infinite; neither accuracy nor capped compatibility loss
rescues excluded truths. Candidate omission is frozen by candidate identity and
fit seed, independently of test labels. Its nominal rate is not the realized
fraction of true answers removed. If pruning empties a row, the original exact
support is restored; this fallback remains part of the evaluated method.

The broad physical rule is inactive in this artifact-only cell because current
tool availability is hidden. Its identity with the unrestricted arm is therefore
an observation limit, not a failure to reproduce the control. The zero-omission
arm is identical to supplied exact support. Unrestricted predictions assign
28.41% mean mass outside supplied exact support; the learned rule leaves 18.36%.
Removing incompatible probability improves modal accuracy without establishing
recovery of a unique past: supplied exact support still has only 8.15% modal
accuracy. Support privilege, law uncertainty and historical ambiguity are distinct.

## Verification and limits

All 32,768 retained forecasts passed the independent scalar score checker.
The complete scientific C03 replay, using the frozen core-2 source, original
scientific populations and retained predictor inputs, reproduces 129 files,
including all 128 raw forecast arrays, byte for byte. This is separate from the
opening engineering consumer replay. Independent regrouping reconstructs all
128 summary rows and masks, and checks all 2,048 supplied-law posteriors by
sequential law/maker Bayes calculation; maximum probability discrepancy is
1.73e-15. The two prerequisite blocks add 20,480 checked forecast scores.
Known-answer revealing/irrelevant controls pass. Raw infinities remain preserved;
JSON means use null accompanied by explicit infinite-loss fractions.

Uncertainty uses 4,096 whole-world bootstrap draws after averaging fits within
each world. The eight worlds share training fits; these are descriptive intervals
conditional on those fits, not precision from independent training replications.
No confirmation population was opened. The C03 full replay verifies this family,
not the unreplayed upstream law-marginal or optimization families.

There is only one reader architecture in this block. Method-by-support interaction,
matched shifted-law C03 results, and each reader's native-to-shift contrast remain
unaccepted here. The separate [duplicate-removal audit](OPENING_REPORT.md) is
retained; it supplies no missing C03 shift comparison. This support result does
not attribute a gain to learned maker structure or modular transfer.

## Disposition and continuation

Accept this bounded native diagnostic. Do not promote the learned hard-pruning
rule for deployment: it excludes true histories. Keep the remaining C03 support
cells and method-by-support comparisons explicit. Prioritize the registered C04
soft-reliability comparison alongside B02/B03 and C02; C04 remains conditional
until its consumer, controls, independent checker, cost pilot and new frozen
plans are admitted. No scientific repair was made to this family.

[Aggregates, paired contrasts and evaluator-only examples](../../../results/v21/native-support-00/AGGREGATE.json),
[validity](../../../results/v21/native-support-00/VALIDITY.json),
[hash bindings](../../../results/v21/native-support-00/BINDINGS.json), and
[complete replay](../../../results/v21/native-support-00/FULL_REPLAY.json).

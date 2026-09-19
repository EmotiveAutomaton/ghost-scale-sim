# V18.2 discovery results

## G0: first verified block

Can earlier work improve predictions of a maker's next artifact? Across 128 held-out maker histories, the exact persistent reader reduced average predictive log loss from 0.988 to 0.925 when it received eight process observations; the most-likely-artifact accuracy stayed at 78.52%, equal to the direct reader. This is a constructed mechanism result: history improved probabilities here, without improving the chosen prediction or establishing that an explicit maker representation is necessary. Miniature — architecture untested.

The table reports means over 128 independent maker/world lineages, with four
probes averaged inside each lineage. Log loss is the negative logarithm of the
probability assigned to the observed artifact; lower is better. Accuracy is the
fraction for which the most probable artifact matched the sampled one.

| Evidence supplied | Direct log loss | Persistent log loss | Empirical history log loss | Persistent accuracy |
|---|---:|---:|---:|---:|
| One artifact | 0.9877 | 0.9752 | 2.5621 | 78.52% |
| Eight artifacts | 0.9877 | 0.9253 | 1.9492 | 78.52% |
| Eight process observations | 0.9877 | 0.9254 | 1.9492 | 78.52% |

The evaluator oracle's log loss was 0.8847 and its accuracy was also 78.52%.
Thus modal choice is saturated on this first support. Process evidence did not
improve this observed log score over the artifact sequence. Additional draws of
the same cell are not the next scientific priority.

G0 retained 64 development and 128 test histories, 9,216 scored rows and 2,304
independently replayed executions. Fifteen independent reference posteriors
matched, and 240 selected score rows reproduced exactly under the frozen source.
All raw blocks and source are in the [proof directory](../../../../results/v18/maker-state/g0/).
The raw archive contains evaluator truth and is not blind reader input.

The first block does not establish adaptive policy tracking, neural performance,
causal meaning of network coordinates or generalization to different physics.
Those are separate admitted comparisons. G0's original schema lacks the later
explicit credible-set coverage fields; they must be derived from retained
posteriors or reported in subsequent blocks, not invented retrospectively.

## First intervention wave

The first intervention wave tested whether readers can update preferences while keeping the maker's information separate from their own. Preference updating helped after a policy change, and treating a reader-only correction as the maker's belief badly distorted prediction. The missing-skill family was disproved in 42 of 128 histories. These are constructed mechanism results; the first learned models failed the held-out task-structure test, and the original selective-learning block is withheld pending its corrected comparison. Miniature — architecture untested.

The [full core-wave response](../../../exchange/v18-2-core-wave-response.md) contains the comparison table and the retained reporting/uptake defects. The campaign remains active.

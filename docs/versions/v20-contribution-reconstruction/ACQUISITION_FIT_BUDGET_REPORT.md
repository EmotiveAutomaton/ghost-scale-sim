# V20 acquisition training-draw and label-budget sensitivity

We tested whether learned evidence acquisition keeps its advantage when training is repeated or reduced. Across both larger-data fits, independent noisy cues still make selection worse than buying the cheapest record; reliable cues favor selection. The copied-cue advantage at the cheapest fee is unresolved after averaging fits. With fewer labels and the highest fee, selection instead wins under noisy cues. All 18,874,368 new forecasts verify. These are constructed-method findings; confirmation remains untouched.

## Design and uncertainty

The 48 new packets comprise 24 first fits with 2,048 labels and 24 second fits with 32,768 labels. Each panel crosses four cue reliabilities, independent or copied cues, and three fees. All have context evidence and no retained episodes. The 24 larger-budget first fits were already independently verified; their COMPLETE and summary hashes are reused. The verified campaign total is 432 packets. No new context-primary comparison or confirmation outcome is supplied by this acquisition batch.

Each condition uses the same 32 whole discovery coefficient worlds and 2,048 cases per world. Reader and evaluator cases are byte-identical across matched training draws and label budgets; training samples differ. For the larger budget, average both fits inside each world before the fixed 4,096-resample paired world bootstrap. The smaller budget currently has one fit. Rows and fits do not add independent worlds. Intervals are descriptive 95% intervals for this finite generator, conditional on the shared training-world allocation.

Capped logarithmic loss is minus the natural logarithm of probability assigned to the true seven-part answer, with probability floored at one trillionth. Loss plus acquisition fee adds a stipulated utility price; it is not measured CPU or money. Lower is better for both. Exact-zero truth probability is inspected separately. Squared probability loss remains a proper secondary score. Wrong attribution at at least 90% confidence is another separately verified outcome.

## Repeating training preserves the independent-cue reversal

Table: rows identify cue reliability (probability of a correct report) and whether a free first context report is supplied with copied repeats. Columns show the cheapest-record loss plus fee minus the learned-selector loss plus fee at each base price, averaged over both 32,768-label fits. Positive favors selection; negative favors always buying the cheapest event record. Context and repeat prices are twice and three times the base price.

| Reliability | Free first report, copied repeats | Fee 0.005 | Fee 0.02 | Fee 0.08 |
|---|---|---:|---:|---:|
| 50% | No | -0.10381 | -0.10994 | -0.10251 |
| 50% | Yes | +0.00014 | +0.00764 | +0.03764 |
| 65% | No | -0.01267 | -0.01880 | -0.02674 |
| 65% | Yes | +0.00008 | +0.00758 | +0.03758 |
| 80% | No | +0.07729 | +0.07116 | +0.04817 |
| 80% | Yes | +0.00001 | +0.00751 | +0.03751 |
| 95% | No | +0.16817 | +0.16204 | +0.12280 |
| 95% | Yes | -0.00005 | +0.00745 | +0.03745 |

All six independent-cue conditions at 50% or 65% reliability favor the cheapest record, with their whole descriptive intervals below zero. All six at 80% or 95% favor selection, with intervals above zero. Thus the original reliability reversal survives the second registered training draw. Reliability at test time is not relearned: the cue channel in training is 80% reliable, whereas test reliability is deliberately varied. This remains a misspecification test. The supplied-law selector knows the test channel but uses the same imperfect learned readout; it is not a perfect-reader ceiling.

All four copied-cue conditions at the cheapest fee have intervals spanning zero after averaging fits. Their means range from -0.00005 to +0.00014. The first-fit advantage is therefore not established by the paired-fit comparison. The second fit alone loses in all four. At the two higher fees, all eight copied-cue comparisons favor selection after averaging fits. Copy grouping never purchases a repeated report, but copied conditions already contain a free first report, so their absolute losses do not isolate copying from initial access. This does not establish learned resistance to double counting.

## Fewer labels change high-price choices

Table: rows and price columns have the same meanings as above, now for the single 2,048-label fit. These are new training samples on the same evaluation worlds, not extra independent worlds or a matched two-fit budget comparison.

| Reliability | Free first report, copied repeats | Fee 0.005 | Fee 0.02 | Fee 0.08 |
|---|---|---:|---:|---:|
| 50% | No | -0.08775 | -0.09388 | +0.01739 |
| 50% | Yes | -0.00479 | +0.00271 | +0.03271 |
| 65% | No | -0.00484 | -0.01097 | +0.04737 |
| 65% | Yes | -0.00343 | +0.00407 | +0.03407 |
| 80% | No | +0.07556 | +0.06943 | +0.07705 |
| 80% | Yes | -0.00218 | +0.00532 | +0.03532 |
| 95% | No | +0.15817 | +0.15204 | +0.10839 |
| 95% | Yes | -0.00098 | +0.00652 | +0.03652 |

With independent cues at the highest base fee, the smaller-budget fit favors selection even at 50% and 65% reliability. The gains are +0.01739 (interval +0.01302 to +0.02197) and +0.04737 (+0.04313 to +0.05186). At 50%, the forecast-loss gain alone has an interval spanning zero; the gain after fees is supported by savings. At 65%, both forecast loss and loss plus fee favor selection. Point estimates at the lower two fees favor the cheapest record at those reliabilities. At 65% reliability and fee 0.005, however, the interval spans zero (-0.01062 to +0.00153), so that comparison is unresolved; the other three intervals are below zero. This is a label-budget-sensitive policy result, not a general finding that less training improves prediction.

The independently reconstructed high-fee policy explains the changed allocation. Out of 65,536 cases, the smaller-budget reader buys 32,767 event records and 10,049 context reports, and declines 22,720 purchases. The larger-budget second fit buys the same 32,767 event records but 23,292 context reports and declines only 9,477. Neither buys repeats. The smaller-budget reliability estimate is 0.78585, versus 0.80116 for the larger-budget second fit; their learned posteriors also differ. This comparison does not isolate which fitted component causes the decision change. The next registered smaller-budget fit is needed before attributing it to label count rather than the particular training draw.

All new policies, including fixed, random, cheapest-record and supplied-law choices, remain in the aggregate record with all secondary scores. No primary criterion is replaced by these fee comparisons. Earlier candidate-omission zeros, correction findings, failed context comparisons, the maximum-budget context candidate and artifact positives remain separate.

## Verification, replay and continuation

All 48 COMPLETE records, plan/source/environment bindings, training and decoding costs, stored forecasts, secondary scores and summaries verify. The existing numerical reviewer checks every new forecast. Independent arithmetic reconstructs learned information gains, supplied-law choices, fees and copied-source identities. Every truth probability was inspected for the log floor and exact zero; no new exact-zero truth probabilities occur. Only reader inputs are blind; training labels, evaluator cases and summaries have separate roles. Raw arrays and per-case policies remain locally retained under immutable completion hashes.

Three full scientific blocks replay from the frozen score-repair source: lower-label high-fee chance cues, larger-label second-fit high-fee chance cues, and second-fit low-fee copied cues. Every scientific file and complete summary matches. These directly cover the changed policy interpretation and training-draw sensitivity. Prior full opening G1/G4/G5/G6, repaired-scoring and reliability replays remain bound by immutable hashes. This is reproduction, not confirmation. Producer source, scientific criteria, model settings and environment were unchanged. All earlier failed sources, outputs and costs remain retained; no new scientific failure occurred. Native inspection and read-command failures are retained as operational attempts, not scientific failures.

The finite-forest refill admits 44 designs: 24 smaller-budget acquisition second fits and 20 maximum-budget native second fits corresponding to the shifted panels already admitted. The first panel tests whether the acquisition changes survive another registered training draw; the second supplies matched native comparisons for shifted successes and failures without selecting only positives. Measured eligible demand is 29.84 hours across 129 discovery blocks, with 450 conditional descriptions retained. Eighteen known flag-only acquisition duplicates remain excluded from useful demand. Forecasts are demand estimates, not evidence of execution or continuous occupancy.

The measured combined useful backlog does not meet the 1.25 planning margin; its shortfall is 0.77 hours. No estimate is inflated to hide a gap. All science, checks, replay, failed attempts and operating overhead share the 90-hour ceiling. Discovery is capped at 72 hours, protecting 18. The original twelve diagnostic confirmation blocks retain first priority at the fixed opening, followed by the already frozen two-fit context candidate extension within its inclusive 4,000-second cap. Confirmation and challenge lineages remain untouched. All existing cutoffs stand.

Validation does not establish fresh-installation or cross-platform reproduction, universal architectural success, causation by one fitted component, or human intent.

[validity](../../../results/v20/acquisition-wave-2/VALIDITY.json), [all aggregates](../../../results/v20/acquisition-wave-2/AGGREGATES.json), [paired fits](../../../results/v20/acquisition-wave-2/PAIRED_FITS.json), [budget comparisons](../../../results/v20/acquisition-wave-2/PAIRED_BUDGETS.json), [policy audit](../../../results/v20/acquisition-wave-2/POLICY_AUDIT.json), [full replays](../../../results/v20/acquisition-wave-2/REPLAY.json), [costs](../../../results/v20/acquisition-wave-2/COSTS.json), [forecast](../../../results/v20/acquisition-wave-2/FORECAST.json).

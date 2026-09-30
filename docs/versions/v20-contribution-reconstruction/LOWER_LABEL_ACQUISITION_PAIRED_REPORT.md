# V20 lower-label acquisition across repeated fits

We tested whether the lower-label acquisition advantage at high fees survives a second training draw. It does not: averaging both fits makes selection worse than buying the cheapest record under noisy independent cues, at every tested fee. All 9,895,936 new forecasts and a complete source replay verify. These are constructed-method results; confirmation remains untouched.

The 25 new packets complete 24 second fits with 2,048 labels across four report reliabilities, independent or copied reports, and three fees, plus the maximum-label native complete-record reference with 64 past episodes. The first acquisition fits and both 32,768-label fits were already verified. There are now 561 verified packets. No new context-primary or confirmation outcome is supplied.

Both registered fits are averaged inside each of 32 whole coefficient worlds before fixed-seed 4,096-resample paired world bootstraps. The budget comparison first averages fits at each budget and then pairs those world averages. Evaluation inputs and truth are byte-identical across fits and budgets; training samples differ. Rows and fits do not add independent worlds. Intervals are descriptive 95% world intervals, conditional on shared training worlds.

Capped logarithmic loss is minus the natural logarithm of probability assigned to the true seven-part answer, floored at one trillionth. Loss plus acquisition fee adds a stipulated utility price, not measured CPU or money. Lower is better. Exact-zero truth mass is inspected separately, and squared probability loss remains a proper secondary score. Wrong attribution at 90% confidence is also checked.

Table: each row identifies the report reliability and whether a free first context report accompanies copied repeats. Columns show cheapest-record loss plus fee minus learned-selection loss plus fee, averaged over both 2,048-label fits. Positive favors learned selection. Base prices apply to event records; context and repeat prices are twice and three times as high.

| Reliability | Free first report, copied repeats | Fee 0.005 | Fee 0.02 | Fee 0.08 |
|---|---|---:|---:|---:|
| 50% | No | -0.11670 | -0.12283 | -0.06447 |
| 50% | Yes | -0.00628 | +0.00122 | +0.03122 |
| 65% | No | -0.02979 | -0.03592 | -0.00941 |
| 65% | Yes | -0.00561 | +0.00190 | +0.03190 |
| 80% | No | +0.05551 | +0.04938 | +0.04527 |
| 80% | Yes | -0.00515 | +0.00235 | +0.03235 |
| 95% | No | +0.14314 | +0.13701 | +0.10170 |
| 95% | Yes | -0.00454 | +0.00296 | +0.03296 |

All six independent-cue conditions at 50% or 65% reliability favor the cheapest record, with whole intervals below zero. All six at 80% or 95% favor learned selection, with intervals above zero. The earlier smaller-budget first-fit high-fee gains do not survive the second registered training draw. At fee 0.08 the paired losses relative to cheapest-record buying are 0.06447 (interval 0.05990 to 0.06901) for chance reports and 0.00941 (0.00497 to 0.01385) at 65% reliability. This completes the previously open training-draw check; the earlier single-fit observations remain valid but cannot support a general lower-label advantage.

At the cheapest fee, all four copied-cue comparisons now lose with intervals below zero. At the middle fee, chance and 65%-reliable reports remain unresolved, while the two more reliable channels favor selection. At the highest fee all four favor selection. Copy grouping never purchases a repeated report, but copied conditions already supply a free first report; this does not isolate copying from initial evidence access or establish learned resistance to double counting.

Table: change in the learned selector advantage over cheapest-record buying when reducing the budget from 32,768 to 2,048 labels, with both fits averaged inside each world. Positive means the smaller budget increases relative advantage; it need not produce an absolute win. Rows have the same reliability and copying meanings as above.

| Reliability | Copied repeats | Fee 0.005 | Fee 0.02 | Fee 0.08 |
|---|---|---:|---:|---:|
| 50% | No | -0.01289 | -0.01289 | +0.03804 |
| 50% | Yes | -0.00643 | -0.00643 | -0.00643 |
| 65% | No | -0.01713 | -0.01713 | +0.01733 |
| 65% | Yes | -0.00568 | -0.00568 | -0.00568 |
| 80% | No | -0.02179 | -0.02179 | -0.00290 |
| 80% | Yes | -0.00516 | -0.00516 | -0.00516 |
| 95% | No | -0.02503 | -0.02503 | -0.02110 |
| 95% | Yes | -0.00450 | -0.00450 | -0.00450 |

Both budgets therefore preserve the independent-cue reliability reversal. At the highest fee, fewer labels reduce the noisy-cue disadvantage but do not reverse it across paired fits. The full interval and secondary-score records retain uncertainty; these comparisons do not isolate a causal training-label effect from all fitted components.

With independent cues at the high fee, the new smaller-budget fit declines 7,735 purchases, buys 32,767 event records and 25,034 context reports, with no repeats. Its first fit declined 22,720 and bought 10,049 context reports. Estimated training reliability changes from 0.78585 to 0.82591, and the learned posterior also changes. The test channel is not relearned: training reliability is stipulated at 80%, while testing varies it. The supplied-law selector knows test reliability but uses the same imperfect learned readout, so it is not a perfect-reader ceiling. Every policy and all secondary scores remain available in the checked aggregates.

The final native complete-record reference uses 131,072 labels and 64 past episodes. Averaging both fits, structured reading loses against every required rival; this is a nonprimary diagnostic. It completes all twenty maximum-label native paired comparisons and the forty corresponding native-to-shift comparisons across this report and the preceding native-reference report. Earlier shifted-context and artifact exceptions remain separate.

Table: native complete-record paired-fit scores. Each row is a reader. Capped joint loss and squared probability loss are lower-is-better; accuracy is the fraction with the correct most-probable answer. Confident error is wrong attribution with at least 90% confidence.

| Reader | Capped loss | Squared loss | Accuracy | Confident error |
|---|---:|---:|---:|---:|
| Structured | 0.89476 | 0.62326 | 0.49973 | 0.04788 |
| Direct frequencies | 0.69767 | 0.50174 | 0.50080 | 0.00000 |
| Joint neural | 0.70160 | 0.50519 | 0.49784 | 0.00000 |
| Legal template | 0.69748 | 0.50156 | 0.50053 | 0.00000 |

Table: matched complete-record shifts after averaging fits within worlds. Changes are shifted minus native capped loss; negative means better. Evaluation rows differ across shifts; worlds are paired. Training bytes match within each same-fit pair.

| Change | Structured loss change | Joint neural loss change |
|---|---:|---:|
| Execution | -0.05283 | -0.00025 |
| Presentation | 0.05009 | 0.00029 |

All COMPLETE, plan/source/environment bindings, fit and decoding costs, stored forecasts, secondary scores and summaries verify. Independent arithmetic reconstructs information-selection choices, supplied-law choices, fees and source groups. No new exact-zero truth probability occurs, and no new truth probability falls below the score floor. Exact zeros and capped loss are checked separately; capping is not an uncapped proper log score. Only reader inputs are blind; evaluator truth, casebooks and summaries have separate roles. Raw arrays and per-case policies remain local under immutable hashes.

The full new lower-label second-fit high-fee chance-cue packet replays from its frozen scientific source: all 514 scientific files and the complete summary match. This covers the completed paired-budget interpretation. Opening G1/G4/G5/G6, scoring repair, acquisition reliability and earlier complete sensitivity references remain hash-bound and reused. Reproduction is not confirmation. A private review helper failed after completing valid pair and policy receipts because its per-arm list was overwritten while assembling shift comparisons. The failed helper/output and 38.96875 CPU seconds are retained; the affected aggregate and zero audit were rebuilt from saved forecasts. The source producer, scores, criteria and model settings were unchanged.

Original blocks used 2160.10938 native CPU seconds. Numerical review used 207.15625, failed supplement 38.96875, repaired aggregation 26.89062, and complete replay 23.82812. All are charged, along with operating and publication overhead, inside the 90-hour ceiling and 72-hour discovery ceiling; 18 hours remain protected. Earlier failures and incomplete historical service-cost measurement remain explicit.

Four predeclared 2,048-label zero-history artifact-only comparisons were admitted across both shifts and both fits, supplying a low-budget anchor for the artifact tier. Measured useful queue demand is 21.33 hours across 209 discovery blocks. Its 2.67-hour shortfall from one day preserves two hours of future operating/publication headroom inside discovery, separate from confirmation reserve. The broader planning cushion is short by 10.58 hours. Useful conditional work is not exhausted; forecasts are not execution or occupancy. Further refill must respect CPU headroom and the fixed cutoff.

The twelve frozen confirmation diagnostics retain first priority at October 1 08:00 UTC, followed by the already frozen two-fit context candidate within its inclusive 4,000-second cap. Confirmation and challenge worlds remain untouched. Validation does not establish cross-platform or fresh-installation reproduction, causation by a single fitted component, a universal structured-reader advantage, or human intent.

[Validity](../../../results/v20/acquisition-wave-3/VALIDITY.json), [paired fits](../../../results/v20/acquisition-wave-3/PAIRED_FITS.json), [paired budgets](../../../results/v20/acquisition-wave-3/PAIRED_BUDGETS.json), [native reference](../../../results/v20/acquisition-wave-3/NATIVE_REFERENCE.json), [matched shifts](../../../results/v20/acquisition-wave-3/PAIRED_SHIFTS.json), [all scores](../../../results/v20/acquisition-wave-3/AGGREGATES.json), [policy audit](../../../results/v20/acquisition-wave-3/POLICY_AUDIT.json), [zero audit](../../../results/v20/acquisition-wave-3/ZERO_MASS_AUDIT.json), [full replay](../../../results/v20/acquisition-wave-3/REPLAY.json), [costs](../../../results/v20/acquisition-wave-3/COSTS.json), [forecast](../../../results/v20/acquisition-wave-3/FORECAST.json).

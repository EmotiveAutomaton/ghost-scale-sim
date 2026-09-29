# V20 acquisition reliability and candidate omission

We tested whether learned evidence acquisition stays useful when cues are noisy, copied or costly. With independent cues, it loses to the cheapest record after fees at 50% and 65% reliability, and wins at 80% and 95%, at every tested price. Six nominal conditions duplicate chance-cue evidence. All five new shift diagnostics fail. Forced omission still assigns zero probability to 24.98% of truths. All 14,352,384 forecasts verify. These are constructed-method results; confirmation remains untouched.

## Design and uncertainty

The 36 new packets comprise five remaining first-fit shift conditions, thirty acquisition conditions and the zero-budget candidate reference. There are now 311 verified packets. Each new condition uses 32 whole discovery coefficient worlds, 32,768 training labels and 2,048 test cases per world. The shift conditions retain 64 episodes; acquisition and omission retain none. One fit seed is shared across the new conditions, with identical acquisition training examples and cases across reliability, copying and price. These are matched conditions, not independent replications. Repeated fits are nested within worlds. All uncertainty intervals pair methods inside each world and use the frozen 4,096-resample world bootstrap; they are descriptive 95% intervals conditional on the training worlds.

Capped logarithmic loss is minus the natural logarithm of the probability assigned to the true seven-part answer, with probability floored at one trillionth. Lower is better. Loss plus acquisition fee adds a stipulated utility price, not measured CPU or storage cost. Exact zero probability is separate; squared probability loss remains a proper-score secondary. Confident wrong attribution means a wrong modal answer at at least 90% confidence. No new context primary or confirmation outcome is supplied by this batch.

## Acquisition: reliability changes the ordering

The readout is trained with cues correct 80% of the time; the selector estimates reliability as 0.80355846. It does not relearn reliability at test time. Each policy uses the same learned readout after observing its chosen evidence. The supplied-law selector knows the actual test reliability but still uses that imperfect learned readout. This is a reliability-misspecification test, not a calibrated selector confronting known chance cues.

Table: rows are distinct test conditions. Reliability is the probability that a context cue is correct; copied means one context report is already supplied free and further reports repeat it. Columns give the cheapest-record loss plus fee minus the learned-selector loss plus fee at three base fees. Positive numbers favor selection; negative numbers favor always buying the cheapest event record. Context and repeat prices are twice and three times the base fee.

| Cue reliability | Free first report, copied repeats | Fee 0.005 | Fee 0.02 | Fee 0.08 |
|---|---|---:|---:|---:|
| 50% | No | -0.09978 | -0.10591 | -0.09939 |
| 50% | Yes | +0.00232 | +0.00982 | +0.03982 |
| 65% | No | -0.00795 | -0.01408 | -0.02274 |
| 65% | Yes | +0.00222 | +0.00972 | +0.03972 |
| 80% | No | +0.08281 | +0.07668 | +0.05312 |
| 80% | Yes | +0.00211 | +0.00961 | +0.03961 |
| 95% | No | +0.17440 | +0.16827 | +0.12874 |
| 95% | Yes | +0.00198 | +0.00948 | +0.03948 |

All six independent-cue conditions at 50% or 65% reliability favor the cheapest record, with their entire descriptive intervals below zero. All six at 80% or 95% favor selection, with intervals above zero. This reverses the opening acquisition advantage under declared reliability misspecification. Every new acquisition condition still improves on its own no-acquisition reference after fees; that weaker result does not establish superiority over the cheapest useful alternative.

Table: rows use independent cues and the middle base fee, 0.02. The gain has the same direction as above; brackets show the descriptive whole-world interval.

| Reliability | Gain over cheapest record | Descriptive interval |
|---|---:|---:|
| 50% | -0.10591 | -0.10978 to -0.10228 |
| 65% | -0.01408 | -0.01818 to -0.01013 |
| 80% | +0.07668 | +0.07245 to +0.08096 |
| 95% | +0.16827 | +0.16475 to +0.17174 |

In copied conditions, the selector buys no repeated report: source grouping sets its additional information gain to zero. It mainly saves fees by declining event records when the event is already determined. Its capped forecast loss is slightly higher than always buying that record, while its loss plus fee is lower. The free first report can itself be misleading; copied-versus-independent absolute losses do not isolate copying because their initial evidence differs. This does not demonstrate that the naive-Bayes readout learned to avoid double-counting copied information.

Six labelled irrelevant conditions are byte-identical to the corresponding 50%-reliability conditions in every stored training, reader, policy and raw forecast file. They remain executed and charged records, but add no new evidence or replication. The 30 nominal acquisition conditions therefore contain 24 distinct settings. The new refill excludes those flag-only duplicates.

## Candidate omission and zero-budget expansion

Table: rows are candidate-handling policies. Full-universe loss scores all 128 possibilities; candidate-event loss instead scores each represented candidate plus one collective unknown event, so its target differs. Zero truth is the percentage of cases assigned exactly zero probability.

| Policy | Full-universe capped loss | Candidate-event capped loss | Exact-zero truth |
|---|---:|---:|---:|
| Forced choice | 9.33709 | 9.33709 | 24.98169% |
| Preserve unknown mass | 3.82024 | 2.78128 | 0.00000% |
| Expansion budget zero | 3.82024 | 2.78128 | 0.00000% |
| Supplied-truth inclusion control | 3.59492 | 3.59492 | 0.00000% |

Forced choice excludes every route-one answer and gives exactly zero probability to 24.98169% of truths; its uncapped log loss is infinite. These are logical omissions, not numerical underflow. The zero-expansion and unknown-mass forecasts and scores are exactly identical, as the control requires. Raw transport retains unknown mass; full-universe scoring distributes it uniformly over the finite omitted set. That convention does not generate an open-world hypothesis. The supplied-truth control is privileged and not an autonomous recovery. Larger candidate budgets remain queued.

## Remaining shift diagnostics

All five noncontext shift comparisons fail the unchanged practical-margin, interval and confident-error requirements. This completes the 48 first-fit shift comparisons, all failed; two previously admitted second-fit context comparisons remain pending. These diagnostics cannot replace the context primary. Training is byte-identical to each immutable native counterpart; both current and retained evidence use the shifted law. Whole worlds pair, but native and shifted test rows differ. The execution-law change acts on both routes, and all readers receive exact shifted-law legal support.

Table: rows are the five conditions, all using 64 retained episodes and 32,768 labels. Columns are mean capped loss for structured reading and its three required rivals.

| Shift and evidence | Structured | Frequencies | Joint neural | Legal template |
|---|---:|---:|---:|---:|
| Presentation, artifact | 14.66801 | 3.55211 | 3.61848 | 3.68887 |
| Execution, sparse | 3.12781 | 2.17008 | 2.17606 | 2.17592 |
| Presentation, sparse | 5.08718 | 2.16941 | 2.18997 | 2.17520 |
| Execution, complete | 1.17793 | 0.69846 | 0.70476 | 0.69745 |
| Presentation, complete | 1.45908 | 0.69861 | 0.70484 | 0.69765 |

## Verification, costs and continuation

Every new COMPLETE, plan, source archive, source file, environment and training-cost binding verifies. The existing analysis reviewer independently reconstructs every forecast, stored secondary score and summary, including acquisition fees and candidate-event scores. The supplement reconstructs learned and supplied-law selector choices, copying identities and every candidate transport. All saved truth probabilities were checked against the log floor and exact zero separately. Only forced omission produces exact-zero truths in this batch. Reader inputs exclude evaluator truth; training, evaluator cases and summaries retain their distinct roles.

The full copied chance-cue packet, reliable priced-cue packet and zero-budget candidate packet replay from their frozen source, matching every scientific file and entire summary. This adds three complete reference replays, not engineering fixtures. Earlier opening G1/G4/G5/G6, scorer-repair, history and shift evidence is reused by immutable hash. Scientific source, criteria and environment were unchanged. No new scientific failure occurred; all earlier failed sources, partial outputs and attempted checks remain retained.

The refill admits 48 frozen acquisition designs: the second registered fit at 32,768 labels and the first fit at 2,048 labels across the 24 distinct reliability/copy/price settings. This tests fit sensitivity and sample efficiency at the observed reversal. No flagged duplicate settings are repeated. Measured runnable demand is 24.64 hours across 121 discovery blocks. There are 579 remaining conditional descriptions, including 18 known redundant descriptions; they remain preserved but excluded from useful-demand forecasts. The useful combined backlog meets the requested margin. Forecast demand is not execution or uninterrupted occupancy.

All science, review, replay, failed attempts, operational overhead and publication consume the same 90-hour ceiling. Discovery remains limited to 72 hours with 18 protected. Twelve frozen diagnostic confirmation blocks remain untouched and first in priority at the fixed confirmation opening; no positive extension is justified by these failed primary outcomes. New science stops at the existing cutoff. The validation does not check fresh installation, cross-platform reproduction, universal architectural failure or human intent.

[Validity](../../../results/v20/acquisition-wave-1/VALIDITY.json), [all comparisons](../../../results/v20/acquisition-wave-1/AGGREGATES.json), [policy audit](../../../results/v20/acquisition-wave-1/POLICY_AUDIT.json), [candidate audit](../../../results/v20/acquisition-wave-1/CANDIDATE_AUDIT.json), [full replays](../../../results/v20/acquisition-wave-1/REPLAY.json), [costs](../../../results/v20/acquisition-wave-1/COSTS.json), [forecast](../../../results/v20/acquisition-wave-1/FORECAST.json).

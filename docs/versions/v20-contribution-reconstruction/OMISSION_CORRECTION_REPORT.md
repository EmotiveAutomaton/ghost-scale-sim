# V20 candidate expansion and context correction

We tested whether expanding a finite candidate list and revising context improve contribution forecasts. Candidate expansion raises truth coverage from 75.02% to 100% and lowers capped log loss from 3.82024 to 3.59492. Correcting false context helps at both training budgets; irrelevant updates harm at both. All 3,276,800 new forecasts verify. These are constructed-method results; confirmation remains untouched.

Seven new packets cover five nonzero candidate budgets at 32,768 training labels and context revision at 32,768 and 131,072 labels. Each uses 32 discovery coefficient worlds and 2,048 cases per world. Methods are paired inside worlds; intervals use the frozen 4,096-resample whole-world bootstrap. One fit is used per new condition. The five candidate budgets share identical training and evaluation records, so repeated baselines are not independent evidence. The verified zero-budget reference is reused by immutable hash.

## Finite candidate expansion

The initial list contains 64 of the 128 declared process alternatives and excludes every route-one answer. Expansion ranks omitted alternatives with the same learned decoder. Full-universe capped log loss is negative log truth probability, floored at 1e-12; lower is better, in natural-log units. Exact-zero truth mass is reported separately. Squared probability loss remains a separate proper score.

Table: each row gives the number of additional ranked alternatives, the fraction of truths explicitly represented, full-universe capped loss, and loss for the candidate-plus-unknown event. The final column changes its target partition as the list expands and cannot rank budgets on a common target.

| Added alternatives | Truth represented | Full-universe capped loss | Candidate-event capped loss |
|---:|---:|---:|---:|
| 0 (verified reference) | 75.01831% | 3.82024 | 2.78128 |
| 1 | 77.01569% | 3.79735 | 2.84508 |
| 4 | 82.43408% | 3.71213 | 2.99292 |
| 16 | 91.32690% | 3.64477 | 3.30901 |
| 32 | 96.61255% | 3.61687 | 3.49947 |
| 64 | 100.00000% | 3.59492 | 3.59492 |

Every nonzero budget improves capped loss and squared probability loss over preserved unknown mass in this comparison. Adding all 64 alternatives changes capped loss by -0.22533, with a descriptive 95% whole-world interval from -0.23232 to -0.21861. Full coverage at that endpoint follows from enumerating the entire supplied universe; it is not open-world invention. Ranked coverage at intermediate budgets is the learned comparison. Unknown mass remains separate in raw transport and is spread uniformly over omitted alternatives only for full-universe scoring.

Forced choice gives exact zero probability to 16,372 of 65,536 truths (24.98169%) in each of the five budgets and has capped loss 9.33709. Its uncapped log loss is infinite. Those five identical baseline blocks contain the same omissions, not five independent replications; the 81,860 stored zero rows count repeated forecasts. No other new arm has exact-zero truth mass. The supplied-truth inclusion control has privileged access. It matches full expansion in truth log loss because both restore the original truth probability, but their other probabilities and squared loss differ. Equal log loss does not make those forecasts identical.

## Context correction and its controls

Table: rows are training-label budgets; columns are saved-answer loss minus recomputed loss for each update. Positive values favor recomputation; negative values mean it harms prediction. Parentheses give descriptive 95% whole-world intervals.

| Training labels | False context corrected | False context retracted | Irrelevant cue added | Unchanged / true-to-true |
|---:|---:|---:|---:|---:|
| 32,768 | +1.20699 (+1.19275, +1.22064) | +0.80402 (+0.79626, +0.81146) | -0.12973 (-0.13442, -0.12482) | 0 / 0 |
| 131,072 | +1.18222 (+1.16756, +1.19632) | +0.77745 (+0.76914, +0.78572) | -0.13327 (-0.13793, -0.12840) | 0 / 0 |

Both larger label budgets retain correction and retraction gains, and both retain harm from an irrelevant cue. Corrected-context gains exceed both unchanged and irrelevant-update gains in the paired difference-of-differences audit, whose descriptive lower intervals are positive. Thus the gain is specific to the useful update within these supplied cue operations; more labels do not remove irrelevant-update harm. Within each unchanged or true-to-true control, saved and recomputed forecasts are byte-identical. Ledger and bounded-raw recomputation are also exactly identical because they use the same decoder on the same final evidence. They are a reconstruction control, not two independently learned methods or proof of a representation-specific advantage. No new context primary was tested; its prior failures remain unchanged.

## Verification, costs and continuation

Every new completion, plan, frozen source archive and file, environment, fit and decoding-cost binding verifies. The existing analysis reviewer independently reconstructs all forecasts, stored primary and secondary scores, summaries and candidate-event scores. Supplemental checks independently sort every omitted candidate with the frozen tie rule, check every mask and mass transport, verify all update views, and compare exact identities in control forecasts. Only reader inputs are blind; training labels, evaluator truth and summaries retain their separate roles.

The full 64-alternative expansion block and 131,072-label correction block replay from frozen source across all 32 discovery worlds, reproducing every scientific file and complete summary. Prior complete opening G1/G4/G5/G6, scoring-repair, history, shift and acquisition replays are reused by immutable hashes. These are scientific reference replays, not engineering fixtures or independent confirmation. Producer source, model settings and criteria are unchanged. All earlier failures, sources, partial evidence and costs remain retained; no new scientific failure occurred. An initial publication staging attempt was refused because its list included ignored evaluator mappings. That attempt is retained and charged; the corrected publication excludes those local mappings.

The reviewed total is 318 packets. The refill admits 21 finite-forest designs: six lower-label candidate budgets, six second-fit candidate budgets, and nine remaining correction conditions covering lower label counts and the second registered fit. The question is whether candidate ranking and the correction-versus-irrelevance contrast depend on data quantity or the fitted model; repeated fits remain nested inside worlds. Measured runnable demand is 25.51 hours across 135 discovery blocks. There are 558 conditional descriptions; 18 known duplicate acquisition descriptions remain frozen but excluded from useful-demand accounting. Forecast demand is not execution or continuous occupancy.

Science, review, replay, attempts, operational overhead and publication share the 90-hour ceiling; discovery is capped at 72 hours with 18 protected. Twelve frozen diagnostic confirmation blocks remain untouched and first in priority at the fixed opening. These diagnostic findings do not justify promoting a failed context primary. The validation does not establish fresh-installation or cross-platform reproduction, universal architectural validity, autonomous open-world discovery or human intent.

[Validity](../../../results/v20/omission-correction-wave-1/VALIDITY.json), [all comparisons](../../../results/v20/omission-correction-wave-1/AGGREGATES.json), [candidate audit](../../../results/v20/omission-correction-wave-1/CANDIDATE_AUDIT.json), [correction audit](../../../results/v20/omission-correction-wave-1/CORRECTION_AUDIT.json), [full replays](../../../results/v20/omission-correction-wave-1/REPLAY.json), [costs](../../../results/v20/omission-correction-wave-1/COSTS.json), [forecast](../../../results/v20/omission-correction-wave-1/FORECAST.json).

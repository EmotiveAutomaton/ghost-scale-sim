# V21 target information and the cost of acquiring records

Does buying information about the historical target beat buying unpredictable reports? With supplied laws, target-focused purchasing lowers fee-adjusted log loss by 0.1806–0.3327 nats against the strongest tested alternative across all 16 condition/fee cells. The entropy policy always buys the contrasting episode, so the irrelevant random-action trap is not the observed cause. This is a verified constructed-method result, miniature — architecture untested, with no learned acquisition or confirmation.

The finite study crosses original and widened coefficient laws with native and changed execution. Each condition uses the same eight development worlds and 256 current episodes per world. Current evidence exposes context; the target is the seven-bit historical account. There are no fitted seeds. Policies and fees share the same cases and potential reports within a condition. Native and changed execution share coefficient-world identities but use different sampled cases; pairing is by whole world, never by cases or fits.

Four purchases are available: a truthful operation-order event (relative cost 2), a complete same-maker episode at a contrasting context (cost 4), an 80%-reliable route-preference cue (cost 1), and an irrelevant eight-way random-action report (cost 1). The comparison policies maximize expected target information, maximize report entropy, always buy the cheapest cue, choose a random report, or buy nothing. Information and entropy selectors may abstain when no positive fee-adjusted value remains. The generator law, cue reliability and same-maker episode likelihood are supplied. This tests a known-law decision rule, not learned source trust or discovery of the true execution law.

Rows identify law regime, execution condition and fee. Net loss is natural-log prediction loss plus the fee times relative acquisition cost, in stipulated nat-equivalent units; lower is better. The reference is the strongest observed alternative by mean net loss in that cell. Gain is reference minus target-focused loss; its descriptive 95% interval resamples eight paired worlds 4,096 times. Selection and intervals are unadjusted across cells.

| Laws | Execution | Fee | Best alternative | Target net loss | Alternative net loss | Gain | 95% interval |
|---|---|---:|---|---:|---:|---:|---|
| original | native | 0 | Report entropy | 2.7699 | 3.0056 | 0.2357 | [0.1910, 0.2803] |
| original | native | 0.01 | Report entropy | 2.7910 | 3.0456 | 0.2546 | [0.2097, 0.2995] |
| original | native | 0.05 | Report entropy | 2.8729 | 3.2056 | 0.3327 | [0.2892, 0.3760] |
| original | native | 0.2 | No purchase | 3.1417 | 3.3940 | 0.2523 | [0.2480, 0.2570] |
| original | changed-tool | 0 | Report entropy | 2.7575 | 2.9772 | 0.2197 | [0.1941, 0.2401] |
| original | changed-tool | 0.01 | Report entropy | 2.7790 | 3.0172 | 0.2382 | [0.2119, 0.2591] |
| original | changed-tool | 0.05 | Report entropy | 2.8529 | 3.1772 | 0.3243 | [0.2990, 0.3441] |
| original | changed-tool | 0.2 | No purchase | 3.1186 | 3.3636 | 0.2450 | [0.2337, 0.2550] |
| wide | native | 0 | Report entropy | 2.7712 | 2.9519 | 0.1806 | [0.1252, 0.2387] |
| wide | native | 0.01 | Report entropy | 2.7925 | 2.9919 | 0.1994 | [0.1443, 0.2574] |
| wide | native | 0.05 | Report entropy | 2.8768 | 3.1519 | 0.2751 | [0.2154, 0.3388] |
| wide | native | 0.2 | No purchase | 3.1459 | 3.4010 | 0.2551 | [0.2465, 0.2629] |
| wide | changed-tool | 0 | Report entropy | 2.7843 | 2.9779 | 0.1935 | [0.1310, 0.2538] |
| wide | changed-tool | 0.01 | Report entropy | 2.8069 | 3.0179 | 0.2109 | [0.1467, 0.2725] |
| wide | changed-tool | 0.05 | Report entropy | 2.8940 | 3.1779 | 0.2839 | [0.2128, 0.3499] |
| wide | changed-tool | 0.2 | No purchase | 3.1730 | 3.4293 | 0.2562 | [0.2522, 0.2603] |

At the three lower prices, report entropy is the strongest alternative; at the highest price, no purchase is strongest. The target selector mainly buys the operation-order record, then stops more often as its price rises. At the highest fee it buys only that record or nothing, purchasing in 86.62–89.16% of cases. Target information never selects the irrelevant distractor. Report entropy also never selects it: the complete contrasting episode has greater report entropy at every tested fee. Thus the positive result distinguishes target-relevant from report-level uncertainty, while this finite menu does not realize the random-action-selection failure that motivated the control.

Proper-loss outcomes are separate. Target information improves log loss against every policy in every condition/fee cell, but it does not uniformly minimize full multiclass Brier (squared probability error). At fee 0.2 under wide changed laws its Brier is 0.9093, worse than report entropy at 0.9072, despite lower logarithmic and fee-adjusted loss. Modal accuracy and confident errors are retained rather than folded into a victory label. [Complete world means and contrasts](../../../results/v21/acquisition/ACQUISITION.json) preserve all metrics, every policy and every fee. [Cost and efficiency records](../../../results/v21/acquisition/CONTRASTS_AND_COSTS.json) include loss reduction per mean acquisition cost; no-purchase efficiency is undefined rather than divided by zero.

The following rows retain every reader’s native-to-changed-execution log-loss difference at every fee. Positive means deterioration; intervals pair coefficient worlds. These are supplied-law forecasts in both conditions, not zero-shot transfer by a learned native reader.

| Laws | Fee | Policy | Shift change | Lower | Upper |
|---|---:|---|---:|---:|---:|
| original | 0 | Target information | -0.0124 | -0.0702 | +0.0513 |
| original | 0 | Report entropy | -0.0284 | -0.0914 | +0.0417 |
| original | 0 | Fixed cheapest cue | -0.0290 | -0.0839 | +0.0296 |
| original | 0 | Random report | -0.0398 | -0.1054 | +0.0343 |
| original | 0 | No purchase | -0.0304 | -0.0893 | +0.0302 |
| original | 0.01 | Target information | -0.0118 | -0.0691 | +0.0512 |
| original | 0.01 | Report entropy | -0.0284 | -0.0914 | +0.0417 |
| original | 0.01 | Fixed cheapest cue | -0.0290 | -0.0839 | +0.0296 |
| original | 0.01 | Random report | -0.0398 | -0.1054 | +0.0343 |
| original | 0.01 | No purchase | -0.0304 | -0.0893 | +0.0302 |
| original | 0.05 | Target information | -0.0190 | -0.0723 | +0.0393 |
| original | 0.05 | Report entropy | -0.0284 | -0.0914 | +0.0417 |
| original | 0.05 | Fixed cheapest cue | -0.0290 | -0.0839 | +0.0296 |
| original | 0.05 | Random report | -0.0398 | -0.1054 | +0.0343 |
| original | 0.05 | No purchase | -0.0304 | -0.0893 | +0.0302 |
| original | 0.2 | Target information | -0.0181 | -0.0685 | +0.0370 |
| original | 0.2 | Report entropy | -0.0284 | -0.0914 | +0.0417 |
| original | 0.2 | Fixed cheapest cue | -0.0290 | -0.0839 | +0.0296 |
| original | 0.2 | Random report | -0.0398 | -0.1054 | +0.0343 |
| original | 0.2 | No purchase | -0.0304 | -0.0893 | +0.0302 |
| wide | 0 | Target information | +0.0131 | -0.0397 | +0.0736 |
| wide | 0 | Report entropy | +0.0260 | -0.0579 | +0.1210 |
| wide | 0 | Fixed cheapest cue | +0.0135 | -0.0321 | +0.0642 |
| wide | 0 | Random report | +0.0287 | -0.0170 | +0.0763 |
| wide | 0 | No purchase | +0.0282 | -0.0239 | +0.0866 |
| wide | 0.01 | Target information | +0.0145 | -0.0376 | +0.0741 |
| wide | 0.01 | Report entropy | +0.0260 | -0.0579 | +0.1210 |
| wide | 0.01 | Fixed cheapest cue | +0.0135 | -0.0321 | +0.0642 |
| wide | 0.01 | Random report | +0.0287 | -0.0170 | +0.0763 |
| wide | 0.01 | No purchase | +0.0282 | -0.0239 | +0.0866 |
| wide | 0.05 | Target information | +0.0155 | -0.0342 | +0.0741 |
| wide | 0.05 | Report entropy | +0.0260 | -0.0579 | +0.1210 |
| wide | 0.05 | Fixed cheapest cue | +0.0135 | -0.0321 | +0.0642 |
| wide | 0.05 | Random report | +0.0287 | -0.0170 | +0.0763 |
| wide | 0.05 | No purchase | +0.0282 | -0.0239 | +0.0866 |
| wide | 0.2 | Target information | +0.0238 | -0.0206 | +0.0764 |
| wide | 0.2 | Report entropy | +0.0260 | -0.0579 | +0.1210 |
| wide | 0.2 | Fixed cheapest cue | +0.0135 | -0.0321 | +0.0642 |
| wide | 0.2 | Random report | +0.0287 | -0.0170 | +0.0763 |
| wide | 0.2 | No purchase | +0.0282 | -0.0239 | +0.0866 |

The one-purchase decoder conditions on one report channel; it does not multiply duplicated features. Duplicate removal is therefore not an additional arm of this family. The earlier hash-bound duplicate-removal findings remain unchanged and are not counted as new evidence. Historical aliasing persists within each observation/report class. A supplied-law decision benefit does not establish fitted capacity, validation convergence, a law-marginal ceiling or B/C architecture superiority.

CPU is measured separately from the stipulated acquisition fee; no empirical conversion from seconds to nats is assumed. Each row gives the full eight-world consumer CPU, including independent checks and export, and mean per-world table setup. Amortized setup divides that measured setup over 1, 10, 100 or 1,000 episodes. These measurements do not isolate optimized per-policy latency.

| Laws | Execution | Consumer CPU seconds | Setup per world | Setup/1 episode | Setup/10 | Setup/100 | Setup/1,000 |
|---|---|---:|---:|---:|---:|---:|---:|
| original | native | 10.6875 | 0.273438 | 0.273438 | 0.027344 | 0.002734 | 0.000273 |
| original | changed-tool | 11.2969 | 0.275391 | 0.275391 | 0.027539 | 0.002754 | 0.000275 |
| wide | native | 11.3750 | 0.281250 | 0.281250 | 0.028125 | 0.002812 | 0.000281 |
| wide | changed-tool | 11.4375 | 0.283203 | 0.283203 | 0.028320 | 0.002832 | 0.000283 |

Shared decoder and selection counters are retained in the cost record, including their zeros. The platform CPU clock is too coarse to interpret a zero interval as free decoding. Each decoder pass constructs all four possible-report posteriors, and selection evaluates all policies. Their summed counters cannot establish a per-policy speed ranking or that the advantage survives an arbitrary real-world CPU price.

Verification checks 225 immutable complete packets, 8,007,680 saved forecasts and 34,247 files, reusing 220 matching prior score receipts. Five new packets contain the four conditions and full replay; the four accepted conditions have 163,840 forecasts. The complete original/native replay matches 177 files, excluding only CPU timing fields in one summary. Independent elementary-row reconstruction verifies every report channel, posterior, acquisition choice, fee-weighted loss and summary; maximum discrepancy is 9.88e-15. All prior full-family bindings remain valid. Four scientific blocks land here, bringing accepted blocks to 75. Execution packets, fee cells and repeated controls are not independent worlds.

The validation cannot establish learned acquisition, unknown cue reliability, hidden source grouping, sequential complementarity, human intent or fresh confirmation. The fixed original/wide coefficient regimes are not a random-architecture severity study: miniature — architecture untested beyond this construction. No F03 category or live Sounding Line inference is added. Confirmation remains unopened; the fixed discovery, science and Friday reporting clocks are unchanged.

The next independent branch is [D03 unknown source groups](D03_PROVENANCE_ADMISSION.json). Three focused controls and two complete 145-file engineering consumers pass. A one-world full-grid pilot costs 7.328125 CPU seconds; scaling to eight worlds with a 50% margin gives 175.875 seconds for original plus replay, within the 2,400-second total cap. The new immutable source is hash-bound before dispatch. One pre-admission repair made a control verdict JSON-serializable; failed source/output and CPU are retained, with no probability or criterion change and no second family repair. This is engineering admission, not scientific acceptance. Learned provenance, wider regimes, D04/B04/B05 and eligible new-world branches remain visible.

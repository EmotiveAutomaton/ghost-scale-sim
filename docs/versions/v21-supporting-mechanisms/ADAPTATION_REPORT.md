# V21 sparse-execution adaptation and complete-current support

Does a separate execution factor reach the fixed reconstruction target with fewer shifted labels? No reader reaches the 0.1-nat excess-loss target in any of six completed development cells. The best modular result is 0.1838 nats (descriptive 95% interval 0.1462–0.2174), so the quarter-label criterion is not met. These are verified constructed-method findings, miniature - architecture untested, with no confirmation.

Natural-log loss penalizes low probability on the actual seven-bit history; lower is better, in nats. Excess loss is loss above the exact shift-aware law-marginal reference. Full multiclass Brier is the summed squared probability error across all 128 histories; lower is better. Modal accuracy counts correct highest-probability histories. Confident error counts wrong answers given at least 90% probability. Intervals resample eight whole worlds 4,096 times after averaging the two fits inside each world. Those same eight development worlds recur across cells; fits, labels and cases are not independent worlds. Intervals are descriptive and unadjusted for the six inspected cells.

## Fixed target and adaptation curves

The native training grid crosses 8 or 64 coefficient laws with 512, 2,048 or 8,192 labels. All cells have four complete, inspection-observed native past episodes and a context-visible current artifact. Only current execution changes. The modular reader selects a mixture over a supplied native/evidence-flip kernel library. The finite direct and fresh-law diagnostic receive exactly the same 0/16/64/256 shifted labels; their adaptation uses 60 fixed Adam updates, with no shifted validation labels.

The table reports each cell’s lowest modular mean excess over the four registered adaptation budgets. Laws are native training coefficient worlds; labels are native training cases. Every minimum first occurs at 16 shifted labels and stays unchanged at 64 and 256. Neither primary rival nor either count control reaches the fixed target in any cell.

| Native laws | Native labels | Best modular excess [95% interval] | Target met |
|---:|---:|---:|---|
| 8 | 512 | 0.5705 [0.5206, 0.6266] | No |
| 8 | 2048 | 0.4911 [0.4353, 0.5436] | No |
| 8 | 8192 | 0.1909 [0.1450, 0.2394] | No |
| 64 | 512 | 0.5961 [0.5243, 0.6673] | No |
| 64 | 2048 | 0.4880 [0.4198, 0.5519] | No |
| 64 | 8192 | 0.1838 [0.1462, 0.2174] | No |

At 8,192 native labels from 64 laws, modular shifted loss falls from 3.2948 to 2.8282 nats after 16 shifted labels. Full Brier falls from 1.0149 to 0.9142, modal accuracy rises from 5.91% to 12.79%, and confident error falls from 0.049% to zero. It improves on the changed distribution, but its native loss rises from 2.8749 to 3.3133 after selecting the changed kernel. This is specialized adaptation, not retention of native performance.

The same-label direct reader worsens after adaptation: shifted loss is 3.0738 at zero labels, 3.6685 at 16 and 3.1839 at 256. Its native training validation stopped, but the fixed adaptation schedule is not a convergence claim. The fresh-law diagnostic improves from 4.3043 to 3.1200 by 256 shifted labels; its Brier initially worsens at 16 labels. Neither comparison establishes optimal direct adaptation. The modular advantage after adaptation survives the additional duplicate-removal control, but the preregistered target remains unmet.

The next table covers every reader and adaptation budget at the high-label, high-diversity setting. Loss is shifted log loss; excess subtracts the exact shifted law-marginal. Change is the same adapted reader’s shifted loss minus its native loss: positive means the shift is worse. Other cells, all intervals and secondary metrics are in [the aggregate](../../../results/v21/adaptation-controls/ADAPTATION.json).

| Reader | Shifted labels | Shifted loss | Excess | Native-to-shift change |
|---|---:|---:|---:|---:|
| modular | 0 | 3.2948 | 0.6504 | +0.4199 |
| modular | 16 | 2.8282 | 0.1838 | -0.4850 |
| modular | 64 | 2.8282 | 0.1838 | -0.4850 |
| modular | 256 | 2.8282 | 0.1838 | -0.4850 |
| direct-finetuned | 0 | 3.0738 | 0.4293 | +0.1728 |
| direct-finetuned | 16 | 3.6685 | 1.0240 | +0.0634 |
| direct-finetuned | 64 | 3.5015 | 0.8570 | +0.0813 |
| direct-finetuned | 256 | 3.1839 | 0.5395 | +0.0244 |
| fresh-law-finetuned | 0 | 4.3043 | 1.6598 | +1.5494 |
| fresh-law-finetuned | 16 | 4.0214 | 1.3769 | +0.3836 |
| fresh-law-finetuned | 64 | 3.6265 | 0.9820 | +0.0259 |
| fresh-law-finetuned | 256 | 3.1200 | 0.4755 | -0.1696 |
| naive-bayes | 0 | 3.4040 | 0.7596 | +0.0023 |
| naive-bayes | 16 | 3.4017 | 0.7573 | +0.0046 |
| naive-bayes | 64 | 3.4008 | 0.7564 | +0.0056 |
| naive-bayes | 256 | 3.3816 | 0.7372 | +0.0048 |
| naive-bayes-without-evidence | 0 | 3.2690 | 0.6245 | +0.0006 |
| naive-bayes-without-evidence | 16 | 3.2669 | 0.6225 | +0.0018 |
| naive-bayes-without-evidence | 64 | 3.2672 | 0.6227 | +0.0035 |
| naive-bayes-without-evidence | 256 | 3.2536 | 0.6091 | +0.0025 |

Naive Bayes and its evidence-feature-removed control have native-to-shift changes near zero here, with intervals crossing zero at every budget. Removing the feature lowers their absolute shifted loss but does not produce a modular-sized adaptation gain. The modular reader’s improvement is therefore not the same duplicate-cancellation signature diagnosed in the reused V20 audit. This alone does not identify an unrestricted maker mechanism: the two-kernel library is supplied.

## Capacity, convergence and information limits

Every native finite direct and policy fit validation-stopped. Each has 20,864 network parameters; the modular policy additionally uses an 8,192-cell conditional execution table and the supplied two-kernel library. This is not capacity matching. Six of twelve fresh-law fits hit the epoch cap. At 8,192 labels both fresh-law fits validation-stop, after 843,776 and 950,272 additional generated labels. Those extra native labels make it a separately costed generator-limit diagnostic. Saved fit costs, complete curves and adaptation settings are retained.

All arms receive exact target support, a separate privilege from the execution library. The public aggregate decomposes both log loss and full Brier into supplied-law uncertainty, additional law uncertainty and reader excess. It preserves per-bit accuracy/Brier, top-eight coverage, exact zeros, below-floor mass and confidence errors. Neither a well-fitted predictor nor low excess identifies a unique historical trajectory. The fixed first-case example is [evaluator-only](../../../results/v21/adaptation-controls/CASES.json).

In that fixed example, the current artifact pairs content 0 with linked evidence 1; the actual hidden history includes inspection. After 16 shifted labels, the modular reader raises the probability of the complete true history from 6.11% to 12.33%, while the finite direct reader lowers it from 7.18% to 1.07%. This illustrates one forecast, not a separate test or proof of unique recovery.

## Two additional hard-support diagnostics

The two supplied C03 events use complete current action evidence, hidden current inspection and no past history. They are different evidence regimes from the earlier sparse/context shift cells. The table compares the same retained direct predictor with no support, supplied exact support, and a learned hard mask; exclusion is the fraction of true histories removed by that learned mask.

| Current execution | No-support log loss | Exact-support log loss | Learned truth exclusion | Full Brier: no support → exact |
|---|---:|---:|---:|---:|
| Native | 1.7879 | 1.7388 | 16.43% | 0.8092 → 0.8131 |
| Changed | 1.7888 | 1.7318 | 3.34% | 0.8086 → 0.8112 |

Both learned masks have infinite uncapped log loss. Here the shift lowers learned exclusion from 16.43% to 3.34%; severe exclusion under shift is not universal across observation regimes. Supplied support improves log loss but worsens full Brier in both cells. All support masks, both fits, every world and native-to-shift contrasts are independently reconstructed in [the support aggregate](../../../results/v21/adaptation-controls/SUPPORT.json). This is a retained-predictor diagnostic, not a method-by-support comparison.

## Verification, disposition and next branch

The frozen review contains 192 complete packets, 6,336,512 checked forecasts and 26,220 retained files. It reuses 173 exact score receipts and checks 19 newer packets. The six supplements reproduce every original shifted forecast with zero maximum probability difference; independent count reconstruction differs by less than 2.4e-14, within the registered 1e-10 tolerance. A complete scientific supplement replay matches 456 files. Prior exact-ceiling, headroom, learned, adaptation, transport and support family replays are reused only with matching hashes and unchanged source. Numerical dependencies and replays are not separate findings.

This batch accepts six B07 development cells with their missing controls and two additional C03 diagnostics: eight new scientific blocks, 57 accepted campaign blocks in total. The original B07 packet and all original scores remain intact. The initial B07 grid is complete and fails its fixed target; no promotion, further fitting repair or confirmation is claimed. Larger registered label/diversity extensions remain conditional scope rather than a silently expanded target.

C04 is now implemented as the next independent reliability diagnostic. It compares no record, soft likelihood and hard masking at supplied reliability 1/0.95/0.8/0.5, using identical retained forecasts. Four targeted controls and two complete engineering consumers pass; 12 fallible-inspection-context fixtures exercise the canonical validator/exporter read-only, with correctness and posterior kept evaluator-only. Native and changed-execution packets plus full native replay have a combined 900-second CPU cap and fresh frozen source. Scientific acceptance awaits that full replay and review. B02/B03 and C02 remain next priorities. Copying, omission, decisive-selection and retraction fixtures remain conditional.

Discovery freezes October 15 at 07:00 UTC; confirmation opens at 13:00 UTC. Science stops October 16 at 17:00 UTC; final report and owned-service closure remain due October 17 at 01:00 UTC. No clocks change. This is a development batch, not campaign closure.

[Validity](../../../results/v21/adaptation-controls/VALIDITY.json) · [Frozen C04 admission](C04_RELIABILITY_ADMISSION.json)

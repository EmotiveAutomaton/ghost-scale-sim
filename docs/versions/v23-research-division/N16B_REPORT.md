# N16b: directed conflicts do not power FX1

Can directed binary conflicts give the frozen FX1 falsifier adequate power? N16b fails: 2, 1 and 2 of 16 independent law blocks clear the frozen primary at 2, 4 and 8 probes, versus 13 required. Habit-only false SUPPORTED is 0 at every history and budget. All 9,216 cases, byte-identical replay and two fresh retained-data reconstructions verify. This is a constructed-method limitation; probes do not repair FX1 here, and the unavailable 70% sensitivities remain unavailable.

The primary stays exactly as declared for N13: at planted strength one and 81 ordinary records, at least 13 of 16 complete reliability/noise-law blocks must have a positive within-law lower bound. The achieved counts are 2/16 after two probes, 1/16 after four, and 2/16 after eight. Their mean half-Brier gains over the records-only reader are 0.00027251, 0.00015736 and 0.00015930, all below the unchanged 0.00129378 margin. More directed conflicts therefore did not provide a monotone repair.

The table reports every frozen setting and probe budget. Gain is the goal reader's mean half-Brier improvement over the records-only reader, with a descriptive 95% interval across the sixteen independent laws. Verdict counts are SUPPORTED / FALSIFIED / INCONCLUSIVE out of sixteen. FALSIFIED is unavailable everywhere because the family-level power gate fails.

| Planted strength | Ordinary records | Added probes | Mean gain (95% law interval) | Frozen margin | Verdict counts S / F / I |
|---:|---:|---:|---|---:|---|
| 0m | 32 | 2 | -0.00000000 (-0.00000000, 0.00000000) | 0.00371767 | 0 / 0 / 16 |
| 0m | 32 | 4 | -0.00000000 (-0.00000000, 0.00000000) | 0.00371767 | 0 / 0 / 16 |
| 0m | 32 | 8 | 0.00000000 (-0.00000000, 0.00000000) | 0.00371767 | 0 / 0 / 16 |
| 0m | 81 | 2 | 0.00000000 (-0.00000000, 0.00000000) | 0.00127979 | 0 / 0 / 16 |
| 0m | 81 | 4 | 0.00000000 (-0.00000000, 0.00000000) | 0.00127979 | 0 / 0 / 16 |
| 0m | 81 | 8 | -0.00000000 (-0.00000000, 0.00000000) | 0.00127979 | 0 / 0 / 16 |
| 0m | 250 | 2 | -0.00000000 (-0.00000000, 0.00000000) | 0.00008219 | 0 / 0 / 16 |
| 0m | 250 | 4 | 0.00000000 (-0.00000000, 0.00000000) | 0.00008219 | 0 / 0 / 16 |
| 0m | 250 | 8 | 0.00000000 (-0.00000000, 0.00000000) | 0.00008219 | 0 / 0 / 16 |
| 1m | 32 | 2 | 0.00140326 (0.00076021, 0.00205307) | 0.00367288 | 1 / 0 / 15 |
| 1m | 32 | 4 | 0.00123332 (0.00069759, 0.00175097) | 0.00367288 | 0 / 0 / 16 |
| 1m | 32 | 8 | 0.00119904 (0.00067978, 0.00170652) | 0.00367288 | 0 / 0 / 16 |
| 1m | 81 | 2 | 0.00027251 (-0.00021146, 0.00072057) | 0.00129378 | 2 / 0 / 14 |
| 1m | 81 | 4 | 0.00015736 (-0.00034672, 0.00062716) | 0.00129378 | 1 / 0 / 15 |
| 1m | 81 | 8 | 0.00015930 (-0.00030327, 0.00062557) | 0.00129378 | 2 / 0 / 14 |
| 1m | 250 | 2 | -0.00007187 (-0.00015493, 0.00000062) | 0.00008266 | 0 / 0 / 16 |
| 1m | 250 | 4 | -0.00007485 (-0.00017200, -0.00000148) | 0.00008266 | 0 / 0 / 16 |
| 1m | 250 | 8 | -0.00007775 (-0.00016645, 0.00000189) | 0.00008266 | 0 / 0 / 16 |
| 2m | 32 | 2 | 0.00413215 (0.00317790, 0.00503850) | 0.00369949 | 6 / 0 / 10 |
| 2m | 32 | 4 | 0.00376258 (0.00280978, 0.00463773) | 0.00369949 | 5 / 0 / 11 |
| 2m | 32 | 8 | 0.00376799 (0.00290938, 0.00456464) | 0.00369949 | 8 / 0 / 8 |
| 2m | 81 | 2 | 0.00109870 (0.00027819, 0.00198878) | 0.00129275 | 3 / 0 / 13 |
| 2m | 81 | 4 | 0.00097187 (0.00029688, 0.00172232) | 0.00129275 | 3 / 0 / 13 |
| 2m | 81 | 8 | 0.00082756 (0.00023239, 0.00146696) | 0.00129275 | 3 / 0 / 13 |
| 2m | 250 | 2 | 0.00002124 (-0.00011001, 0.00014834) | 0.00008393 | 0 / 0 / 16 |
| 2m | 250 | 4 | 0.00002764 (-0.00009123, 0.00014531) | 0.00008393 | 0 / 0 / 16 |
| 2m | 250 | 8 | 0.00002082 (-0.00005635, 0.00009530) | 0.00008393 | 0 / 0 / 16 |

The secondary readers do not rescue the result. N13's validation-selected tempering exponent is one in all nine settings, so tempered-records versus goal gain matches the primary to within 3e-18. The records reader beats the unpersonalized pooled reader by 0.04885 to 0.08346 half-Brier units, and the personal oracle retains positive mean headroom of 0.00071 to 0.03762. These comparisons show that the null is not caused by the pooled baseline and that oracle information remains; they do not change the frozen power bar.

Binary probes coarsen ordinary four-category responses through the declared payoff bridge. Both readers receive the same likelihood, context, stakes and options. Swapping stakes is an exact alias under the exclusive-category law, so this batch establishes no human stake sensitivity. The 2/4/8 budgets are prefixes of one without-replacement trajectory and are additional to the original records. The outcome is specific to this policy and bridge, not a proof that all probing is ineffective.

Controls pass for bridge normalization, equal reader inputs, q=0 reader identity after probes, exact N13 zero-probe reproduction, probability-product/log-space agreement and the stake-order alias. Original and replay raw streams are byte-identical for all 36 files. The independent review rebuilt all 9,216 scores, all sixteen law-block intervals and the canonical ledger in two fresh processes.

The first independent review exhausted its 1,344-second reservation and remains failed. The admitted repair drew zero new cases, retained the original scientific bytes and criteria, and completed at 821.234375 CPU-seconds. Including the cost-pilot replay, frozen-fit step, original batch/replay/failed review and repair, N16b charged 2,176.296875 CPU-seconds (0.6045 hours), within its unchanged three-hour family cap. The original failure is not rewritten as success; the separate acceptance receipt binds its retained-data repair.

Route to Sounding Line: **P3 still fails power at every probe budget and passes the false-SUPPORTED check; FX1 waits.** Constructed-method validation, miniature—architecture untested; outside confirmation. [Protocol](N16B_PROTOCOL.md), [machine summary](../../../results/v23/n16b/SUMMARY.json), [proof](../../../results/v23/n16b/PROOF.json), [raw hashes](../../../results/v23/n16b/RAW_INDEX.json).

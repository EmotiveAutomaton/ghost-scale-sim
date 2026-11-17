# V31 queue and integrity review: November 17

The eligible queue is running in Gear 2 again after its overnight family-budget stop. At 2026-11-17T12:57:07.754091+00:00, four diary workers were active, 3,086 cells were waiting, and no new failure had occurred after restart. All currently admissible rosters are prepared. This is an ongoing finite campaign, not a completed week.

The table counts scientific settings in the two continuing grids. Completed settings may still await the next public aggregate batch while their raw evidence and completion receipts are retained locally.

| Continuing study | Completed | Total prepared | Remaining, including active |
|---|---:|---:|---:|
| Diary calibration, standard and heavy floors | 2,378 | 5,184 | 2,806 |
| Corrected colored-remainder calibration | 268 | 552 | 284 |

The separate 144-setting carry study is already complete. D runs first, then M. Based on completed D throughput and M's earlier 4.46-hour run for 268 settings, the remaining eligible work is provisionally about 10–20 elapsed hours from the 04:49 Pacific restart. This is a rough planning range, not a promise: heavy-floor settings, different M conditions, source verification and host-load backoff can change it. A family stops at its last complete cell if its fixed allocation cannot fund the next reservation. Last starts remain November 22 at 20:00 Pacific, with drain at 22:00.

The owner approved transferring 55 CPU-hours within the unchanged 160-hour total: D gains 30, M gains 20 and verification gains five; W contributes 35, T 15 and Q five. Effective ceilings are D 50, S 15, M 35, W 10, T 5, Q 15, healing 10 and reserve 20 hours. The snapshot accounts for 62.43 CPU-hours including active reservations. Original acceptance bytes, completed evidence, scientific capsules and validity gates are unchanged. [Resource receipt](../../../results/v31/RESOURCE_REALLOCATION_20261117.json).

No missing required record was found in the completed-work census: 5,686 completion receipts, 21,926 retained original/replay raw files, all plan/admission/worker/summary bindings and matched raw-manifest pairs passed. This census includes 16 retained failures and 153 excluded pilots; neither is successful science. Older raw-file contents retain their earlier full audits; this pass checked their required presence and receipt bindings. The separate overnight batch received a fresh full raw-byte replay and independent numerical audit. [Integrity evidence](../../../results/v31/morning-validation-20261117/INTEGRITY.json).

The overnight batch has 251 corrected M settings: 50,200 scientific worlds and 10,040 excluded pilots. All raw hashes, source bindings, complete seed replays and independent reconstructions passed. Together with 17 previously published settings, that brings M to 268 of 552. The first two original M settings remain superseded, not new replications. [M report](MORNING_20261117_M_R1.md); [full batch audit](../../../results/v31/morning-20261117/AUDIT.json).

Q's owner-approved directory correction ran, then failed its unchanged resolution limit: 0.29849 against 0.25 at a 2,000-step moving-group fixture. The independent implementation agrees to 2.27e-13, which does not establish particle accuracy. All 41 checked comparisons and 14 retained fixture files remain recorded. Its conditional builder correctly declined production. W hidden readers also remain validity-held; G6 still needs an operational definition. No additional Q/W repair is authorized by this resource transfer. [Q disposition](Q_REPAIR_DISPOSITION_20261117.md).

The runtime repair caches immutable job/receipt metadata and reloads changed accounting entries. Dispatch still reopens and verifies every source/admission binding. Eight targeted tests passed, covering newly admitted jobs, new failures, reservation release, external charges, conserved budgets, and refusal to admit Q after failed or changed evidence. The idle controller was replaced once; no live scientist was stopped, no completed job restarted, and no environment synchronized.

The queue was idle from its last M completion at 21:46 Pacific on November 16 until resumed at 04:49 on November 17. The event monitor stayed alive, but native agent reviews remain unavailable after a service usage-limit failure. A queued check was not a delivered review. The next operator-required check is approximately 09:00 Pacific, before expected D turnover; automatic delivery is not promised. Scientific dispatch, full replay and local receipt recording continue independently.

This checkpoint publishes the frozen 251-cell M batch, both Q validation dispositions, the integrity census, resource decision and runtime repair. New diary completions continue to accumulate locally and receive a separate aggregate landing. These constructed-method results establish no human marker or deployed reader.

The [first restart batch](D_RESTART_20261117.md) adds 82 fully audited diary settings, bringing the published diary prefix to 2,396. The [optional-recipe note](D_OPTIONAL_RECIPE_RATE_TIED_20261117.md) closes two reporting omissions without new draws; all 27 earlier qualifying triggers now have notes. Later completions remain local until their next audited batch.

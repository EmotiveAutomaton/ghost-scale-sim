# Queue health and wake repair, November 15

At 2026-11-16T00:10:46.095846+00:00 UTC the queue was running in Gear 2 with four single-thread, below-normal workers under the original controller. It had 2,486 verified receipts, 15 excluded, 21 retained failures and 366 withdrawn unstarted receipts. No new scientific failure was found. There were 2,859 prepared waiting plans; not all are guaranteed to fit the remaining allocation. All commissioned implementations and eligible conditional builders are present. P Bayesian and L middle remain validity-held; the earlier failures are preserved.

Two further 24-setting subsets are fully recorded: [BLIND live-sum weight 20](reports/V-blind-live20.md) and [BLIND rule stock two](reports/V-blind-rule2.md). Their 48 cells passed source, plan and admission bindings, known answers, excluded pilots, full replay, independent reconstruction and retained raw checks. Aggregate evidence, claims, findings, hypothesis records, exchange routes and the theory return were updated together. [Incremental validity receipt](../../../results/v30/WAKE_HEALTH_VALIDITY_20261115.json). No scientific source, threshold, environment or running worker was changed.

## Why the wake was missed

Native reviews ran at 14:05, 14:12 and 14:40 Pacific. The last successful report finished at 14:49:46, before its 14:50 deferred-event deadline. Finalization finished scanning records at 14:50:52 and treated the now-expired handoff as unresolved. It disarmed the monitor, so the intended 14:50 review never ran. Scientific computation continued throughout. Separately, the 14:05 monitor capsule omitted a late-imported shutdown module; that native monitor had already been replaced, and the default packager is now corrected too.

The repaired finalizer evaluates a successful, exact-hash handoff at report completion. If it expires during finalization, the events become immediately due for review; they are not acknowledged or marked published. Failed reviews, nonzero exits, changed evidence and invalid times remain blocked. Review subprocesses also no longer inherit the monitor's immutable import path. Twenty-one targeted checks pass. [Reconstruction of the actual 17-event failure](../../../results/v30/WAKE_FINALIZATION_REGRESSION_20261115.json) passes without acknowledging unpublished evidence.

Only the monitor was replaced. Its new native heartbeat is verified; automatic delivery with the replacement has not yet been observed. Native reviews produce local reports, not messages in this interactive conversation. Publication and exact event acknowledgements are completed after this snapshot, before the monitor is rearmed.

## Next boundary

The single next review is 16:30 Pacific, ahead of the estimated 16:30-16:45 V resource boundary. Around 16:09, V had 53,252 of its 57,600 CPU seconds charged or reserved; recent cells used roughly 400 CPU seconds with four workers. This is an estimate, not completion evidence. The admitted texture and finer-map queue follows within its separate reserve. Host backoff, reservations and cell costs can delay or limit coverage.

The approved total remains 60 CPU-hours. This snapshot charges or reserves 41.83 hours, including the prior conservative tail allowance once. New starts still end November 21 at 23:48 Pacific and drain November 22 at 01:48. These are outer limits, not reasons to wait after eligible work finishes. No finite closure is claimed.

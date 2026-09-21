# V19 bounded-reset review — 21 September 2026

Does a fixed reset stabilize the failed updater? It reduces forecast loss by 2.31120 nats at 128 independent observations, but the rolled-in updater remains 3.47050 nats worse than the reset one-step baseline. Both complete replays and independent reconstruction pass. This is a bounded constructed-method result, not process correspondence or human intent.

[Evidence, paired uncertainty and limitations](../versions/v19-local-maker/BOUNDED_RESET_REPORT.md). The unchanged heads use no new training. The practice successor changes the explanatory question; confirmation remains separate.

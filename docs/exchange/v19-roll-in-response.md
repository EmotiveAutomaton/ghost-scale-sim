# V19 roll-in review — 21 September 2026

Does training on its own predictions stabilize the updater? One roll-in pass instead increases forecast loss by 5.74016 nats at 32 independent observations. The unchanged teacher-refit control, independent numerical checks and both complete replays verify this failure. This is a constructed-method result, miniature — architecture untested; it does not establish that roll-in training fails generally.

[Evidence, controls and limitations](../versions/v19-local-maker/ROLL_IN_REPORT.md). Both full replays and independent checks pass; confirmation remains separate.

# N16 robustness opening, October 30

This separate extension tests whether N16's constructed scheduler advantage
persists across fresh balanced blocks, probe lengths and chooser errors. It is
now running in Gear 2: two below-normal, single-thread scientific workers and
62 prepared packets, for 64 fixed blocks and 245,760 cases in total. The original
175-block confirmation and all other completed V25 studies remain intact.

Correction first: the initial excluded attempt stopped at 88 cases because a
probability of 1.0000000000000002 failed an exact upper-bound guard. Its failure,
raw prefix and full reservation remain retained and charged. The separately
frozen repair passed all fourteen targeted checks, a complete 3,840-case
excluded execution, byte replay and independent numerical reconstruction.
All four retained 88-case raw stream prefixes reproduced exactly. No scientific
observation, setting or criterion changed, and no clipping was introduced.

The complete excluded original/replay/review cost was 589.640625 CPU-seconds.
The measured reservation is 923.2890625 seconds per scientific block; the full
projected allocation, including setup, protected reserve and remaining build
allowance, is 70,490.84375 seconds inside the remaining extension sublimit.
The prior N16 and failed-extension charges carry forward under the unchanged
64-hour total. The repaired root's remaining sublimit is recorded in ACCEPTANCE;
it is not a fresh 24-hour allowance. The deadline remains November 2 06:00 UTC.

From the measured full-block wall time, the first 16-block reporting milestone
is expected around **09:20-09:45 Pacific today**, and the complete 64-block run
around **13:25-14:45 Pacific today**. These forecasts include a 25% upper margin
for interference; they are not completion receipts. No scientific result from
this new extension is claimed at opening.

One native monitor is live. It stays disarmed while the operator lands this
opening and is armed as the final handoff. It requests reviews at the declared
16/32/48/64 verified-block milestones, failures, or the twelve-hour health
boundary if needed, then retires after final publication and acknowledgements.
The former V25 monitor stays stopped. Partial prefixes remain explicitly
partial; only all 64 blocks form the fixed-size analysis.

AGENTS.md now records the separately authorized extension, active repair root,
shared Gear 2 limit and finite stopping rule. Companion repositories, their
processes and environments were not modified by this setup.

Evidence: [protocol](N16_ROBUSTNESS_PROTOCOL_2026-10-30.md),
[repair diagnosis](N16_ROBUSTNESS_HEALING_2026-10-30.md),
[excluded validity](../../../results/v25/n16-robustness-20261030/EXCLUDED_VALIDITY.json),
[native opening](../../../results/v25/n16-robustness-20261030/NATIVE_OPENING.json),
[ETA](../../../results/v25/n16-robustness-20261030/ETA.json),
[current state](../../../results/v25/n16-robustness-20261030/CURRENT_STATUS.json).

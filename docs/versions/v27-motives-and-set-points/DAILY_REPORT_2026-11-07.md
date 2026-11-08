# V27 daily report — 7 November 2026

Snapshot: 2026-11-08 05:44 UTC (21:44 Pacific). This is the daily 20:00 Pacific line, landed late during the owner's active build/health review.

## Corrections and retained failures

K3-H2 passed excluded fixtures 1, 0, 2, 3 and 4, then failed prospectively at fixture 5. One 131,072-particle path differed from the 262,144-particle reference by 0.3231 nat, outside the unchanged 0.25-nat limit; posterior error and every crossing classification remained within their bars. The complete six-fixture record is retained, no threshold changed, and no dynamic K3 science was admitted.

The owner restored Gear 2 during this review: up to four below-normal, single-thread workers, dropping to two only after the packet's measured ten-minute host-load rule fires. The 760 CPU-hours and 15 November 20:00/22:00 Pacific endpoints are unchanged. The existing controller accepted the new receipt without restart and filled four ready slots.

## Daily operating line

The controller's sampled occupancy recorded 12.759 busy worker-hours. Idle capacity was 7.729 hours attributed to build or design, 21.365 hours attributed by the controller to host load or the earlier owner Gear 1 limit, and 3.151 hours attributed to initial build and validation. Admission and exit boundaries make these sampled values approximate; the controller's `host_load` bucket includes capacity withheld by Gear 1 and is not evidence that measured CPU crossed the host-load threshold.

Protocols exist for 241 of 275 commissioned cards. At the 05:44 UTC queue snapshot, 35 packets were ready and four were active under a four-worker ceiling; 34 dynamic K3 cards remained deliberately unbuilt behind the failed numerical gate. There were 139 verified, fully landed logical cards and 139 matching V27 claim-ledger rows. Three withdrawn K5-H1 descendants and the original failed K5 probe remain retained.

Complete texture counts were K1: 1, K2: 31, K3: 0 dynamic, K4: 25, K5: 15, K6: 16 and K7: 24. Complete doubled-block precision counts were K5: 6, K6: 7 and K7: 5.

Pool balances at the snapshot were 238.915 study hours, 12.365 preparation hours, 458.268 follow-up/healing/texture hours and 20.000 protected hours. Completed charges were 3,905.578 study seconds, 27,486.094 preparation seconds and 78,235.695 follow-up/healing/texture seconds. K3-H2 used 19,887.906 CPU-seconds and is charged at measured cost.

## Primary routes

- K1 is a constructed timing-scale ruler for possible Sounding Line suggestion-latency work and for deciding whether ToMpathy should display timing among identity marks. It licenses no human identity inference.
- K2 routes explaining-away requirements to ToMpathy; it has no Sounding Line route this week.
- K3's verified static comparisons route the band/lower-edge alias to Sounding Line CH4/SP and constrain ToMpathy's set-point reader. Dynamic moved/dulled/gain-up inference remains unadmitted.
- K4 routes FEAR/stress separability to Sounding Line CH2, CH3, IN2 and IN3, and to ToMpathy's decision about showing stress apart from FEAR.
- K5 routes only to ToMpathy's closed-source state. Its rereading and retroactive-tag contrasts are scale-only in this world and license no engine state; it has no Sounding Line route this week.
- K6 routes joint, sequential and guarded iterative reading to Sounding Line AT3/GO2 and to how ToMpathy's panels update one another.
- K7 routes completion, surprise and social-laughter markers to Sounding Line LA1/LA2 and to whether ToMpathy treats textual laughter as a structural cue.

All routes are scoped to constructed worlds: miniature, architecture untested. None establishes a result about people or language models.

## Verification, delivery and publication boundary

At 05:43 UTC, all 280 then-complete packet bindings passed: plan, original summary, replay and independent reconstruction. All 19,602 retained raw files matched their recorded SHA-256 hashes and byte sizes, and all five exported aggregate bindings per packet matched the local completed packet. Packets finishing after that cutoff remain for the next audit.

The seven approved recipient manifests were rehashed: all 70 files matched. The Sounding Line K4/K6/K7 and ToMpathy K2/K4/K6/K7 aggregate copies remain in their approved local destinations and outside this repository's publication set.

Eight targeted V27 contract, host-load and preparation-slot tests passed in an isolated repository import. Two earlier invocations are retained as verification failures: the first could not use the sandbox's default temporary directory; the second inherited the live source-frozen `PYTHONPATH`. Clearing that subprocess-only import override and using a repository-local temporary root produced the eight-pass result without synchronizing the environment.

## Costed ETA and next action

At 05:45 UTC, the active measured plans projected about 18 minutes for K2-T032, 32 minutes for K1-T002, 3.8 hours for K5-T016 and 5.3 hours for K5-T015-P2 on their respective single threads. These are admission projections, not completion claims. The conservatively early next review is 06:00 UTC (22:00 Pacific): inspect the short K2/K1 boundaries, verify any newly complete packets, and keep four ready slots filled unless the measured ten-minute host-load rule reduces the ceiling. A build/health check remains due no later than 06:44 UTC while ready or deliberately unbuilt cards remain.

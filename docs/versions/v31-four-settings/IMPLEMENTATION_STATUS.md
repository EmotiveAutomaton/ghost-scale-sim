# V31 implementation and queue state

Checkpoint 2026-11-18T02:00:16.124124+00:00. Gear 2 has 4 active M storage-recovery workers and 244 waiting settings. The complete diary grid is verified. Host-load backoff retains the four-to-two-worker limit.

The table counts settings and validation jobs separately. Original M results and interrupted attempts remain retained; count only corrected M successes and their successful storage successors toward the 552-setting grid.

| Family | Kind | Prepared | Recorded states |
|---|---|---:|---|
| D_CARRY | science | 144 | VERIFIED: 144 |
| D_CORE | science | 3456 | VERIFIED: 3456 |
| D_HEAVY | science | 1728 | VERIFIED: 1728 |
| G4_IDENTITY | science | 114 | EXCLUDED: 60, VERIFIED: 54 |
| G7_CAPITAL | science | 24 | VERIFIED: 24 |
| M_CORE | science | 552 | FAILED: 3, SUPERSEDED_BY_M_R1: 547, VERIFIED: 2 |
| M_R1 | science | 552 | FAILED: 4, SUPERSEDED_BY_M_STORAGE_R1: 256, VERIFIED: 292 |
| M_R1_VALIDATION | validation | 1 | VERIFIED: 1 |
| M_STORAGE_R1 | science | 260 | QUEUED: 244, RUNNING: 4, VERIFIED: 12 |
| Q_CONDITIONAL_ADMISSION_R1 | validation | 1 | VERIFIED: 1 |
| Q_EVENTS_FOLLOWON_Q_EXACT-0006 | science | 1 | VERIFIED: 1 |
| Q_EVENTS_FOLLOWON_Q_EXACT-0007 | science | 1 | VERIFIED: 1 |
| Q_EVENTS_FOLLOWON_Q_EXACT-0008 | science | 1 | VERIFIED: 1 |
| Q_EXACT | science | 45 | VERIFIED: 45 |
| Q_HIDDEN_R1 | validation | 1 | FAILED: 1 |
| Q_HIDDEN_R1_IO | validation | 1 | FAILED: 1 |
| Q_HIDDEN_V1 | validation | 6 | FAILED: 2, HELD_AFTER_FAILURE: 4 |
| Q_HIDDEN_V2 | validation | 6 | FAILED: 4, HELD_AFTER_FAILURE: 2 |
| S_CORE | science | 2160 | EXCLUDED: 93, VERIFIED: 2067 |
| S_LEARNING_R2 | science | 64 | VERIFIED: 64 |
| S_LEARNING_R2_VALIDATION | validation | 1 | VERIFIED: 1 |
| S_UNIFORM | science | 48 | VERIFIED: 48 |
| T_CORE | science | 216 | VERIFIED: 216 |
| V31_FOLLOWON_PREPARATION_20261116 | validation | 1 | VERIFIED: 1 |
| W_BURST | science | 27 | VERIFIED: 27 |
| W_COUPLING | science | 18 | VERIFIED: 18 |
| W_HIDDEN_R1 | validation | 1 | FAILED: 1 |
| W_HIDDEN_V1 | validation | 3 | FAILED: 2, HELD_AFTER_FAILURE: 1 |
| W_HIDDEN_V3 | validation | 3 | FAILED: 2, HELD_AFTER_FAILURE: 1 |
| W_MATCH_VALIDATION | validation | 108 | VERIFIED: 108 |
| W_READER_V2 | science | 24 | VERIFIED: 24 |
| W_READER_V4 | science | 60 | VERIFIED: 60 |

Corrected M has 304 unique settings complete at this runtime snapshot. The newly audited public batch contains 2,788 D settings and 24 M settings; it brings published D coverage to all 5,184 settings and the pre-recovery M prefix to 292. New recovery completions retain automatic full replay/reconstruction and separate later audit/landing.

All 260 unfinished M settings have immutable successor plans, unchanged scientific sources, seeds and gates. Four failed original attempts are preserved. Capacity admission protects 30 GiB and reserves full active-job output bounds. A storage hold can stop further starts before the complete grid fits. No results were moved to another drive.

Effective CPU ceilings remain D 50, S 15, M 35, W 10, T 5, Q 15, healing 10 and reserve 20 hours, totaling 160. The original acceptance and November 22 endpoints are unchanged. Q/W hidden readers remain validity-held and G6 is design-held; no further repair is implied.

Native agent reviews remain unavailable after the usage-limit failure. Event recording continues; a cheap cached inventory replaces repeated event rehashing, with fresh hash checks retained at actual delivery. No automatic agent wake is promised.

[Evening queue report](QUEUE_REPORT_EVENING_20261117.md), [complete diary](D_COMPLETE_REPORT_20261117.md), [complete optional recipe map](D_OPTIONAL_RECIPE_COMPLETE_20261117.md), [M prefix](M_PRE_STORAGE_REPORT_20261117.md).

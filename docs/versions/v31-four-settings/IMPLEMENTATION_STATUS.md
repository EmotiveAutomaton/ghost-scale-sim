# V31 implementation and queue state

Checkpoint 2026-11-17T01:32:49.203258+00:00. Gear 2 allows four below-normal, single-thread scientific workers, reducing to two under the inherited host-load rule. The existing controller is running the corrected M roster.

The table distinguishes scientific cells from excluded validation jobs. Counts are a dated snapshot; an execution receipt is not evidence that matching or a scientific hypothesis succeeded.

| Family | Kind | Prepared | Recorded states |
|---|---|---:|---|
| D_CARRY | science | 144 | VERIFIED: 144 |
| D_CORE | science | 3456 | RESOURCE_HELD: 1142, VERIFIED: 2314 |
| D_HEAVY | science | 1728 | RESOURCE_HELD: 1728 |
| G4_IDENTITY | science | 114 | EXCLUDED: 60, VERIFIED: 54 |
| G7_CAPITAL | science | 24 | VERIFIED: 24 |
| M_CORE | science | 552 | FAILED: 3, SUPERSEDED_BY_M_R1: 547, VERIFIED: 2 |
| M_R1 | science | 552 | QUEUED: 535, RUNNING: 3, VERIFIED: 14 |
| M_R1_VALIDATION | validation | 1 | VERIFIED: 1 |
| Q_CONDITIONAL_ADMISSION_R1 | validation | 1 | QUEUED: 1 |
| Q_EVENTS_FOLLOWON_Q_EXACT-0006 | science | 1 | VERIFIED: 1 |
| Q_EVENTS_FOLLOWON_Q_EXACT-0007 | science | 1 | VERIFIED: 1 |
| Q_EVENTS_FOLLOWON_Q_EXACT-0008 | science | 1 | VERIFIED: 1 |
| Q_EXACT | science | 45 | VERIFIED: 45 |
| Q_HIDDEN_R1 | validation | 1 | FAILED: 1 |
| Q_HIDDEN_R1_IO | validation | 1 | RUNNING: 1 |
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

All currently admissible scope is built. M_R1 has 552 original settings admitted after its retained-path repair. D is resource-limited. W hidden readers failed their first repair; the owner-approved Q directory-only successor is admitted in the existing controller. Its separately prepared conditional builder can admit production only after all validation and measured-profile gates pass. G6 remains design-held. These holds are not scientific nulls.

The 2,834-cell main snapshot contains 2,681 scientific cells and 153 excluded pilots, independently audited before publication. A separate same-evening supplement records completed M cells and the third Q follow-on. Thirty of 108 W matching validations passed; 78 were infeasible. [Repair and matching evidence](HEALING_REVIEW_20261116.md).

The single inexpensive event monitor remains live, but automatic agent reviews are unavailable. No automatic wake delivery is promised. Scientific execution and receipt recording continue independently. An operator must handle new admission or failure boundaries until review service is restored.

Original family ceilings, 160 CPU-hour total, start cutoff and drain deadline remain unchanged. Prepared work can still end resource-held; unused capacity in another family is not an authorized transfer. [Theory return](THEORY_RETURN.md).

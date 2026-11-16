# V31 implementation and queue state

Checkpoint 2026-11-16T08:16:53.515466+00:00. Gear 2 is active with 4 permitted scientific workers. Counts below are a snapshot; the queue continues.

| Family | Prepared jobs | Purpose |
|---|---:|---|
| D_CORE | 3456 | science |
| D_CARRY | 144 | science |
| D_HEAVY | 1728 | science |
| G4_IDENTITY | 114 | science |
| G7_CAPITAL | 24 | science |
| M_CORE | 552 | science |
| Q_EXACT | 45 | science |
| Q_HIDDEN_V1 | 6 | validation |
| Q_HIDDEN_V2 | 6 | validation |
| S_CORE | 2160 | science |
| S_UNIFORM | 48 | science |
| T_CORE | 216 | science |
| W_MATCH_VALIDATION | 108 | validation |
| W_BURST | 27 | science |
| W_COUPLING | 18 | science |
| W_HIDDEN_V1 | 3 | validation |
| W_HIDDEN_V3 | 3 | validation |

D covers the full declared diary grid plus reaction/carry. S has exact paired readers, a uniform-arrival texture and a built but validity-held learning extension. M has 552 distinct laws covering 1,296 logical labels through exact aliases. W has excluded matching, exact-reader admission, the approved coupling experiment and burst timing; its hidden readers are separately gated. T is built. Q has 45 exact cells and twelve moving-reader validation jobs. G4 and G7 reserves are built.

S.b is held after an excluded numerical reconstruction failure and its failed first repair; the requested second repair is a separate owner decision. G6 lacks its intervention definition. Matching-dependent W jobs are created only from passed calibration receipts. Conditional follow-ons still need trigger evaluation and their own admission; no eligible result is rerun for occupancy.

All scientific jobs require known answers, forty excluded pilots, full seed replay and independent reconstruction. Family ceilings can leave partial maps; a complete roster is not a promise of full execution. No registered claim or human marker is admitted.

[First 216 diary cells](D_DIARY_SHAPES.md) are fully audited; the opening eighteen overlap that map and are not additional replication. [Current machine-readable state](../../../results/v31/CURRENT_STATUS.json).

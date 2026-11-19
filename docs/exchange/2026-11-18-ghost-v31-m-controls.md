# M: overnight recovery and adversarial controls

**November 18 reporting correction.** The first external M report mislabeled the 0.2 planted remainder share as the refresh probability. The actual per-message refresh probability is 1/8 in both the frozen generator and protocol. All numerical results, seeds and scientific gates are unchanged.

Can the two-law reader identify colored persistence uniquely? It cannot in the completed adversarial controls: the mean share of makers favoring the colored law reaches 76.18% when the state refreshes at thread boundaries and 84.08% when it refreshes probabilistically per message. Under independent white noise, that share ranges from 0.30% to 0.59%. This is a constructed-method limitation: evidence against white noise does not uniquely identify the colored mechanism.

The 142 newly audited settings add 28,400 scientific worlds and 5,680 excluded pilots. Every setting passed source and receipt binding, known-answer controls, complete original/replay byte equality and independent numerical reconstruction. They bring audited corrected M coverage to 450 of 552 unique settings. Failed originals remain retained and do not add independent successful observations.

The table gives ranges across cell means in this new batch. Evidence shares are mean fractions of individual makers crossing the frozen three-nat evidence threshold; unresolved makers cross neither threshold. Short-gap/fading detection shares concern whole simulated worlds. Ranges across designs are not pooled effects or uncertainty intervals. Full per-cell eight-block intervals remain in the aggregate records.

| Planted law | New settings | Colored-evidence maker share | White-evidence maker share | Unresolved-maker share | Positive short-gap detection | Fading detection |
|---|---:|---|---|---|---|---|
| Clock-time persistence | 16 | 0.0000 to 0.0001 | 0.0000 to 0.0000 | 0.9999 to 1.0000 | 0.0400 to 1.0000 | 0.0400 to 1.0000 |
| Independent white noise | 12 | 0.0030 to 0.0059 | 0.0790 to 0.6906 | 0.3039 to 0.9180 | 0.0100 to 0.0750 | 0.0150 to 0.0750 |
| Refresh at thread boundaries | 105 | 0.0000 to 0.7618 | 0.0000 to 0.0068 | 0.2323 to 1.0000 | 0.0200 to 1.0000 | 0.0200 to 1.0000 |
| Probabilistic refresh per message | 9 | 0.0022 to 0.8408 | 0.0000 to 0.0033 | 0.1567 to 0.9978 | 0.3300 to 1.0000 | 0.2750 to 1.0000 |

Cumulative coverage is complete for clock-time persistence (324/324 settings) and white noise (12/12). The thread-refresh control has 105/108 settings; probabilistic refresh has 9/108. The 102 unrun settings were capacity-held, not scientific nulls. The sixteen new clock-time settings are weak-remainder conditions and leave 99.99% to 100% of makers unresolved; that range does not summarize the entire 324-setting colored grid.

The reader compares only the stipulated colored and independent-white laws. The two adversaries are outside that candidate set, so selecting the colored member is not a model-selection test against those adversaries. Nuisance removal is oracle-assisted; no real-text nuisance estimator or human marker is validated. Miniature architecture untested. Sounding Line EN1 receives this specificity boundary; ToMpathy receives the requirement to retain alternative causes and unresolved classifications.

[Full audit](../../results/v31/storage-overnight-20261118/AUDIT.json), [per-cell proof](../../results/v31/storage-overnight-20261118/M_STORAGE_R1/VERIFIED.json), [ranges](../../results/v31/storage-overnight-20261118/RANGES.json), [protocol](../versions/v31-four-settings/protocols/M.md).

## First external-storage result

Does the weak probabilistic-refresh condition identify individual makers? No maker crosses either frozen evidence threshold in this 200-world setting; all remain unresolved. Whole-world short-gap and fading detections each occur in 9.5% of worlds. This is a descriptive method result in a constructed setting, not a human marker.

Claim G-V31-M-EXTERNAL-FIRST-1. [Full scope and route](../versions/v31-four-settings/M_EXTERNAL_FIRST_REPORT_20261118.md).

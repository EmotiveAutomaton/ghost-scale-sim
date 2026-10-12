# V21 evidence-driven model expansion

Can predictive checks detect an omitted mechanism when the starting model is confidently wrong? With strong records and four expansion slots, surprise-triggered expansion reduces held-out log loss by 0.531 nats and expands only 2.69% of cases where nothing is omitted. The full-library planner predicts better when a mechanism is missing, but expands every weak-record case even when nothing is missing. These are verified constructed-method results with supplied laws, not open-world discovery or confirmation.

Eight reliability worlds pair omission-present and omission-absent cases, sixteen historical context/output records, weak or strong record noise, three successor budgets and independent query outcomes. The base and three distinct binary response laws form a supplied finite library. A fourth offered successor duplicates the base and receives no extra prior mass. The labels belief, constraint and order are stipulated roles, not measured psychological variables. No query truth enters a selection decision.

The table gives the four-slot comparison. Omission means the true response law is absent from the initial model. Log loss is negative natural-log probability of the query truth, in nats; full Brier is summed squared probability error. Lower is better for both. Accuracy is the fraction of correct modal predictions; expansion is the fraction adding any successor. Weak/strong refers only to historical record quality.

| Omission | Records | Reader | Log loss | Full Brier | Accuracy | Expansion |
|---|---|---|---:|---:|---:|---:|
| Absent | Weak | No expansion | 0.36246 | 0.20923 | 88.04% | 0.00% |
| Absent | Weak | Confidence trigger | 0.36246 | 0.20923 | 88.04% | 0.00% |
| Absent | Weak | Predictive surprise | 0.36325 | 0.21024 | 87.79% | 1.37% |
| Absent | Weak | Full-library planner | 0.44118 | 0.26957 | 86.87% | 100.00% |
| Absent | Weak | Fixed expansion | 0.45836 | 0.28316 | 87.26% | 100.00% |
| Absent | Weak | Full-library posterior | 0.45836 | 0.28316 | 87.26% | 100.00% |
| Absent | Strong | No expansion | 0.36246 | 0.20923 | 88.04% | 0.00% |
| Absent | Strong | Confidence trigger | 0.36246 | 0.20923 | 88.04% | 0.00% |
| Absent | Strong | Predictive surprise | 0.36892 | 0.21421 | 87.70% | 2.69% |
| Absent | Strong | Full-library planner | 0.37505 | 0.22048 | 86.62% | 15.33% |
| Absent | Strong | Fixed expansion | 0.37544 | 0.22056 | 86.62% | 100.00% |
| Absent | Strong | Full-library posterior | 0.37544 | 0.22056 | 86.62% | 100.00% |
| Present | Weak | No expansion | 1.05052 | 0.72493 | 53.96% | 0.00% |
| Present | Weak | Confidence trigger | 1.05052 | 0.72493 | 53.96% | 0.00% |
| Present | Weak | Predictive surprise | 1.03397 | 0.71368 | 54.15% | 2.93% |
| Present | Weak | Full-library planner | 0.70437 | 0.50813 | 55.52% | 100.00% |
| Present | Weak | Fixed expansion | 0.66630 | 0.47912 | 54.98% | 100.00% |
| Present | Weak | Full-library posterior | 0.66630 | 0.47912 | 54.98% | 100.00% |
| Present | Strong | No expansion | 1.05052 | 0.72493 | 53.96% | 0.00% |
| Present | Strong | Confidence trigger | 1.05052 | 0.72493 | 53.96% | 0.00% |
| Present | Strong | Predictive surprise | 0.51950 | 0.33181 | 79.25% | 71.34% |
| Present | Strong | Full-library planner | 0.40233 | 0.24198 | 85.11% | 98.39% |
| Present | Strong | Fixed expansion | 0.39965 | 0.24059 | 85.25% | 100.00% |
| Present | Strong | Full-library posterior | 0.39965 | 0.24059 | 85.25% | 100.00% |

The surprise trigger uses an exact lower binomial tail at the frozen 5% threshold. Its strong-record omission gain is 0.53102 nats (descriptive paired-world 95% interval 0.40050 to 0.66909). When the base is correct, it expands 2.69% and incurs a small 0.00647-nat loss increase (0.00311 to 0.01025). Weak records rarely trigger it: omission-present loss improves only 0.01655 nats. The confidence rule never expands because the supplied base reliability always exceeds its fixed 0.75 confidence threshold. This is the designed confidently-wrong control, not a claim about all confidence policies.

The full-library planner lowers strong-record omission loss another 0.11717 nats versus surprise (0.09848 to 0.13406), but expands 15.33% of strong-record correct-base cases and every weak-record case. In weak correct-base cases it worsens loss by 0.07793 nats versus surprise. Its objective is expected predictive loss under the full supplied posterior, not an omission detector; prior uncertainty itself can justify additions. Chronic unnecessary expansion therefore prevents a general selective-expansion claim. Fixed expansion/full-library inference also improve missing-model predictions, and fixed expansion slightly beats the planner on raw strong-record omission loss. All budgets and every paired cue change are retained in the linked records.

The planner evaluates all four distinct likelihoods and 5/11/16 subsets at budgets 1/2/4, even if it selects fewer models. It never adds the redundant fourth successor. Fixed four-slot expansion has exactly the full-library posterior but pays four addition fees instead of three. The declared fee of 0.01 per added hypothesis is a hypothetical utility cost, not CPU time; saved priced-loss results remain separate from raw proper scores and actual charged CPU. No cheap-search superiority follows.

All 294,912 original/replay probability scores pass. The complete frozen scientific replay matches 1,322 files byte for byte, including 1,320 arrays. Independent scalar reconstruction checks 147,456 original forecasts and policy decisions, 131,072 generated historical observations, all posterior masses and binomial tails, every selected subset and evaluation count, every summary and 24,576 redundant-successor nonselection checks. Maximum probability disagreement is 3.89e-16. Eight worlds determine uncertainty; cases and conditions are not independent worlds.

Fit capacity and validation convergence do not apply to this exact supplied-library ruler. Law uncertainty is within the supplied finite prior; law learning and unknown observation noise are untested. There is no support-pruning superiority, historical correspondence, native-to-execution-shift comparison or product-of-features duplicate arm. Strong-minus-weak records and redundant-law removal are separate controls. This purpose-built miniature remains architecture untested. The finite registered grid is complete; no promotion, fresh confirmation or unrestricted mechanism invention is accepted.

[Validity](../../../results/v21/expansion/VALIDITY.json), [all budgets and readers](../../../results/v21/expansion/SUMMARY.json), [paired contrasts](../../../results/v21/expansion/CONTRASTS.json), [world means](../../../results/v21/expansion/WORLD_ROWS.json), [contrast availability](../../../results/v21/expansion/CONTRAST_AVAILABILITY.json), [frozen admission](C06_EXPANSION_ADMISSION.json).

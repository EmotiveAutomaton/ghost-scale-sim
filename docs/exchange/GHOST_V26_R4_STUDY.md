# W4 Precision

Corrections first: November 2 operator review clarifies the registered comparator, observation unit or scope below; saved measurements and claim IDs are unchanged. See results/v26/REPORTING_CORRECTION_20261102.json.

Was the apparent decline in W4 detection a rounding artifact? At 1,024 records, ranking by unrounded complementary-tail probabilities improved detection by 0.457555, or 45.7555 percentage points, over the original rounded rule at matched false alarms. The repaired 1,024-record detection also met the declared comparison with 512 records. This validates the precision repair on the retained W4 worlds; downstream uses must preserve tail rankings, not necessarily use exact arithmetic in every implementation.

Claim `G-V26-R4-1`.

The table lists each registered primary by its frozen setting and observation stratum. Estimates use nats for score contrasts, records for time-to-evidence, probabilities for capture/unresolved shares, and distance bins for crossover decisions. Missing crossings remain censored; cases are averaged within worlds before uncertainty.

| Setting | Observation stratum | Measure | Mean | Available blocks |
|---|---|---|---:|---:|
| {"alpha": 10, "placebo": false} | {"history": 1024, "reader": "contrasts"} | primary | 0.4575553580876691 | 16 |

Derived records-needed/crossover decisions (null means not reached within the sampled grid):

```json
{}
```

Design pre-mortem: {"mean_tie_credit": 0.40446331035453614, "note": "Packet threshold1 is the median,14/16 worlds; tie mass .40 is randomized credit. Original failed all16 assumption retained; no worlds/settings changed.", "passed": true, "seed_domain": "retained-W4-identities", "threshold_mean": 0.999999948125, "threshold_one_share": 0.875}

By construction and modelling limits: Retained W4 alpha10, non-placebo setting0, sixteen original world identities. Original8/32/128/512/1024 increments regenerated unchanged. Extra levels split increments with a separately seeded multivariate hypergeometric bridge. Individual reader only; arm a original half-even rule with boundary fallback; b/c rank -log complementary tail checked at50/80 decimal digits. These are repairs on retained worlds, not fresh independent confirmation.

Full original/replay files are byte-identical. The independent review uses a separate scalar posterior/decimal aggregation route; some likelihood kernels and the generative law are shared. Known-answer tests check the model equations, not real-world validity. No universal ceiling is inferred from a supplied-library reference.

Sounding Line: no route this week.
ToMpathy B15: replace or confirm the records-needed curve after precision repair.

[All estimates](../../../results/v26/v26-r4-study-1/SUMMARY.json). [Verification](../../../results/v26/v26-r4-study-1/INDEPENDENT_REVIEW.json).

# W1 Repair

Corrections first: November 2 operator review clarifies the registered comparator, observation unit or scope below; saved measurements and claim IDs are unchanged. See results/v26/REPORTING_CORRECTION_20261102.json.

Which intervention separates ends that are aliased on passive choices? After 32 passive records, one information-chosen context switch improved true-hypothesis log score over one more passive choice by 0.072171 nats at choice precision 1 and 0.150107 at precision 3. Removing an available piece (0.003648, 0.009767) and satiating an end (0.032073, 0.026964) stayed below the 0.05 practical bar. These are constructed-method results for the declared interventions.

Claim `G-V26-R1-1`.

The table lists each registered primary by its frozen setting and observation stratum. Estimates use nats for score contrasts, records for time-to-evidence, probabilities for capture/unresolved shares, and distance bins for crossover decisions. Missing crossings remain censored; cases are averaged within worlds before uncertainty.

| Setting | Observation stratum | Measure | Mean | Available blocks |
|---|---|---|---:|---:|
| {"beta": 1} | {"history": 1, "reader": "remove"} | primary | 0.0036480255689386697 | 16 |
| {"beta": 3} | {"history": 1, "reader": "remove"} | primary | 0.009767106723329899 | 16 |
| {"beta": 1} | {"history": 1, "reader": "satiate"} | primary | 0.032073469977642766 | 16 |
| {"beta": 3} | {"history": 1, "reader": "satiate"} | primary | 0.026964256675760634 | 16 |
| {"beta": 1} | {"history": 1, "reader": "switch"} | primary | 0.07217050438124883 | 16 |
| {"beta": 3} | {"history": 1, "reader": "switch"} | primary | 0.15010692652539306 | 16 |

Derived records-needed/crossover decisions (null means not reached within the sampled grid):

```json
{}
```

Design pre-mortem: {"alias_world_share": 1.0, "floor": 0.25, "passed": true, "seed_domain": "check-only-R1"}

By construction and modelling limits: Eight sampled split-alias pairs per world, alternating their true endpoints. Full W1 210-hypothesis reader retained. Five training pieces share values on ends0/1; sixth differs. Separate adaptive remove/satiate/switch tracks up to16 interventions, each compared with equal-count passive records. Ties in expected information choose the first declared probe.

Full original/replay files are byte-identical. The independent review uses a separate scalar posterior/decimal aggregation route; some likelihood kernels and the generative law are shared. Known-answer tests check the model equations, not real-world validity. No universal ceiling is inferred from a supplied-library reference.

Sounding Line: no route this week.
ToMpathy owner-question order: corrected exact-alias intervention map.

[All estimates](../../../results/v26/v26-r1-study-1/SUMMARY.json). [Verification](../../../results/v26/v26-r1-study-1/INDEPENDENT_REVIEW.json).

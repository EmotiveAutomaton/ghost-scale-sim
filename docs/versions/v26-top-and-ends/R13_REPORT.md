# W13 Repair

Corrections first: earlier V25 findings and retained failures remain unchanged; this is a separate V26 card.

Harmful proxy learning required both a nonzero learning rate and enough misleading shifted episodes. With learning rate 0, no block reached one nat by 128 records. At misleading rate 0.5, the censored median was 69 records for learning rates 0.05 and 0.2; lower misleading rates were mostly censored. This is a finite dose map for the declared trust update, not a real-world prevalence claim.

Claim `G-V26-R13-1`.

The table lists each registered primary by its frozen setting and observation stratum. Estimates use nats for score contrasts, records for time-to-evidence, probabilities for capture/unresolved shares, and distance bins for crossover decisions. Missing crossings remain censored; cases are averaged within worlds before uncertainty.

| Setting | Observation stratum | Measure | Mean | Available blocks |
|---|---|---|---:|---:|
| {"learning_rate": 0.0, "mislead": 0.1} | {"reader": "threshold", "truth": 1} | not_within_128 | 1.0 | 16 |
| {"learning_rate": 0.0, "mislead": 0.25} | {"reader": "threshold", "truth": 1} | not_within_128 | 1.0 | 16 |
| {"learning_rate": 0.0, "mislead": 0.5} | {"reader": "threshold", "truth": 1} | not_within_128 | 1.0 | 16 |
| {"learning_rate": 0.0, "mislead": 0.1} | {"reader": "threshold", "truth": 1} | records_to_one_nat | None | 0 |
| {"learning_rate": 0.0, "mislead": 0.25} | {"reader": "threshold", "truth": 1} | records_to_one_nat | None | 0 |
| {"learning_rate": 0.0, "mislead": 0.5} | {"reader": "threshold", "truth": 1} | records_to_one_nat | None | 0 |
| {"learning_rate": 0.05, "mislead": 0.1} | {"reader": "threshold", "truth": 2} | not_within_128 | 0.8125 | 16 |
| {"learning_rate": 0.05, "mislead": 0.25} | {"reader": "threshold", "truth": 2} | not_within_128 | 0.5625 | 16 |
| {"learning_rate": 0.05, "mislead": 0.5} | {"reader": "threshold", "truth": 2} | not_within_128 | 0.0 | 16 |
| {"learning_rate": 0.05, "mislead": 0.1} | {"reader": "threshold", "truth": 2} | records_to_one_nat | 100.0 | 3 |
| {"learning_rate": 0.05, "mislead": 0.25} | {"reader": "threshold", "truth": 2} | records_to_one_nat | 78.85714285714286 | 7 |
| {"learning_rate": 0.05, "mislead": 0.5} | {"reader": "threshold", "truth": 2} | records_to_one_nat | 69.4375 | 16 |
| {"learning_rate": 0.2, "mislead": 0.1} | {"reader": "threshold", "truth": 3} | not_within_128 | 0.875 | 16 |
| {"learning_rate": 0.2, "mislead": 0.25} | {"reader": "threshold", "truth": 3} | not_within_128 | 0.5 | 16 |
| {"learning_rate": 0.2, "mislead": 0.5} | {"reader": "threshold", "truth": 3} | not_within_128 | 0.125 | 16 |
| {"learning_rate": 0.2, "mislead": 0.1} | {"reader": "threshold", "truth": 3} | records_to_one_nat | 47.5 | 2 |
| {"learning_rate": 0.2, "mislead": 0.25} | {"reader": "threshold", "truth": 3} | records_to_one_nat | 56.0 | 8 |
| {"learning_rate": 0.2, "mislead": 0.5} | {"reader": "threshold", "truth": 3} | records_to_one_nat | 63.785714285714285 | 14 |

Derived records-needed/crossover decisions (null means not reached within the sampled grid):

```json
{
  "{\"configuration\":{\"learning_rate\":0.0,\"mislead\":0.1},\"labels\":{\"reader\":\"threshold\",\"truth\":1}}": {
    "censored_median": null,
    "not_reached": 16,
    "blocks": 16
  },
  "{\"configuration\":{\"learning_rate\":0.0,\"mislead\":0.25},\"labels\":{\"reader\":\"threshold\",\"truth\":1}}": {
    "censored_median": null,
    "not_reached": 16,
    "blocks": 16
  },
  "{\"configuration\":{\"learning_rate\":0.0,\"mislead\":0.5},\"labels\":{\"reader\":\"threshold\",\"truth\":1}}": {
    "censored_median": null,
    "not_reached": 16,
    "blocks": 16
  },
  "{\"configuration\":{\"learning_rate\":0.05,\"mislead\":0.1},\"labels\":{\"reader\":\"threshold\",\"truth\":2}}": {
    "censored_median": null,
    "not_reached": 13,
    "blocks": 16
  },
  "{\"configuration\":{\"learning_rate\":0.05,\"mislead\":0.25},\"labels\":{\"reader\":\"threshold\",\"truth\":2}}": {
    "censored_median": null,
    "not_reached": 9,
    "blocks": 16
  },
  "{\"configuration\":{\"learning_rate\":0.05,\"mislead\":0.5},\"labels\":{\"reader\":\"threshold\",\"truth\":2}}": {
    "censored_median": 69.0,
    "not_reached": 0,
    "blocks": 16
  },
  "{\"configuration\":{\"learning_rate\":0.2,\"mislead\":0.1},\"labels\":{\"reader\":\"threshold\",\"truth\":3}}": {
    "censored_median": null,
    "not_reached": 14,
    "blocks": 16
  },
  "{\"configuration\":{\"learning_rate\":0.2,\"mislead\":0.25},\"labels\":{\"reader\":\"threshold\",\"truth\":3}}": {
    "censored_median": null,
    "not_reached": 8,
    "blocks": 16
  },
  "{\"configuration\":{\"learning_rate\":0.2,\"mislead\":0.5},\"labels\":{\"reader\":\"threshold\",\"truth\":3}}": {
    "censored_median": 69.0,
    "not_reached": 2,
    "blocks": 16
  }
}
```

Design pre-mortem: {"checks": [{"changed": true, "probability_at_least_one": 0.9999986099154762, "setting": {"learning_rate": 0.0, "mislead": 0.1}}, {"changed": true, "probability_at_least_one": 0.9999986099154762, "setting": {"learning_rate": 0.05, "mislead": 0.1}}, {"changed": true, "probability_at_least_one": 0.9999986099154762, "setting": {"learning_rate": 0.2, "mislead": 0.1}}, {"changed": true, "probability_at_least_one": 0.9999999999999999, "setting": {"learning_rate": 0.0, "mislead": 0.25}}, {"changed": true, "probability_at_least_one": 0.9999999999999999, "setting": {"learning_rate": 0.05, "mislead": 0.25}}, {"changed": true, "probability_at_least_one": 0.9999999999999999, "setting": {"learning_rate": 0.2, "mislead": 0.25}}, {"changed": true, "probability_at_least_one": 1.0, "setting": {"learning_rate": 0.0, "mislead": 0.5}}, {"changed": true, "probability_at_least_one": 1.0, "setting": {"learning_rate": 0.05, "mislead": 0.5}}, {"changed": true, "probability_at_least_one": 1.0, "setting": {"learning_rate": 0.2, "mislead": 0.5}}], "passed": true, "seed_domain": "enumerated-R13"}

By construction and modelling limits: W13 full type roster and softmax/lapse dynamics retained. Each positive shift record independently misleading with probability m; negative restoration changes trust only after the choice. Case check enumerates the positive-probability harmful branch at every positive rate. Finite trajectories with no harm remain in science; they are not redrawn.

Full original/replay files are byte-identical. The independent review uses a separate scalar posterior/decimal aggregation route; some likelihood kernels and the generative law are shared. Known-answer tests check the model equations, not real-world validity. No universal ceiling is inferred from a supplied-library reference.

Sounding Line: shifted-episode counts needed to distinguish persistence from proxy learning.
ToMpathy: no route this week.

[All estimates](../../../results/v26/v26-r13-study-1/SUMMARY.json). [Verification](../../../results/v26/v26-r13-study-1/INDEPENDENT_REVIEW.json).

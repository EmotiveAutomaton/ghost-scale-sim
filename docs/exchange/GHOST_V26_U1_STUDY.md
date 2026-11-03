# Opacity Gate

Corrections first: November 2 operator review clarifies the registered comparator, observation unit or scope below; saved measurements and claim IDs are unchanged. See results/v26/REPORTING_CORRECTION_20261102.json.

How many revealed source-history records let a learner approach a reader that already knows the source's selection policy? With bias strength 0.5 and 32 new claims, the track-record learner's mean probability on the false hypothesis exceeded the transparent reader's by 0.279801-0.299110 after one training record. The first sampled training length within 0.05 was 32 in all three settings. These are mean posterior-probability differences, not frequencies of categorical mistakes. Labels have no effect by construction; this is constructed source-policy learning evidence.

Claim `G-V26-U1-1`.

The table lists each registered primary by its frozen setting and observation stratum. Estimates use nats for score contrasts, records for time-to-evidence, probabilities for capture/unresolved shares, and distance bins for crossover decisions. Missing crossings remain censored; cases are averaged within worlds before uncertainty.

| Setting | Observation stratum | Measure | Mean | Available blocks |
|---|---|---|---:|---:|
| {"s": 0.5, "tau": 0.5} | {"history": 1, "reader": "track-record", "truth": 0} | excess_capture | 0.28142283309978555 | 16 |
| {"s": 0.5, "tau": 0.8} | {"history": 1, "reader": "track-record", "truth": 0} | excess_capture | 0.29910977385984855 | 16 |
| {"s": 0.5, "tau": 1.0} | {"history": 1, "reader": "track-record", "truth": 0} | excess_capture | 0.27980140334667625 | 16 |
| {"s": 0.5, "tau": 0.5} | {"history": 16, "reader": "track-record", "truth": 0} | excess_capture | 0.06441365634138276 | 16 |
| {"s": 0.5, "tau": 0.8} | {"history": 16, "reader": "track-record", "truth": 0} | excess_capture | 0.08151765559994194 | 16 |
| {"s": 0.5, "tau": 1.0} | {"history": 16, "reader": "track-record", "truth": 0} | excess_capture | 0.07076426140774343 | 16 |
| {"s": 0.5, "tau": 0.5} | {"history": 2, "reader": "track-record", "truth": 0} | excess_capture | 0.2509294609557123 | 16 |
| {"s": 0.5, "tau": 0.8} | {"history": 2, "reader": "track-record", "truth": 0} | excess_capture | 0.28052006828588716 | 16 |
| {"s": 0.5, "tau": 1.0} | {"history": 2, "reader": "track-record", "truth": 0} | excess_capture | 0.25237534436288517 | 16 |
| {"s": 0.5, "tau": 0.5} | {"history": 32, "reader": "track-record", "truth": 0} | excess_capture | 0.013111547833341496 | 16 |
| {"s": 0.5, "tau": 0.8} | {"history": 32, "reader": "track-record", "truth": 0} | excess_capture | 0.015498092200685888 | 16 |
| {"s": 0.5, "tau": 1.0} | {"history": 32, "reader": "track-record", "truth": 0} | excess_capture | 0.012722998202342493 | 16 |
| {"s": 0.5, "tau": 0.5} | {"history": 4, "reader": "track-record", "truth": 0} | excess_capture | 0.1968368226782895 | 16 |
| {"s": 0.5, "tau": 0.8} | {"history": 4, "reader": "track-record", "truth": 0} | excess_capture | 0.23262363218819912 | 16 |
| {"s": 0.5, "tau": 1.0} | {"history": 4, "reader": "track-record", "truth": 0} | excess_capture | 0.20910761193079253 | 16 |
| {"s": 0.5, "tau": 0.5} | {"history": 64, "reader": "track-record", "truth": 0} | excess_capture | 0.0007278244985335879 | 16 |
| {"s": 0.5, "tau": 0.8} | {"history": 64, "reader": "track-record", "truth": 0} | excess_capture | 0.002112138513507519 | 16 |
| {"s": 0.5, "tau": 1.0} | {"history": 64, "reader": "track-record", "truth": 0} | excess_capture | 0.00046061576034515993 | 16 |
| {"s": 0.5, "tau": 0.5} | {"history": 8, "reader": "track-record", "truth": 0} | excess_capture | 0.13011228017636742 | 16 |
| {"s": 0.5, "tau": 0.8} | {"history": 8, "reader": "track-record", "truth": 0} | excess_capture | 0.16231568379754258 | 16 |
| {"s": 0.5, "tau": 1.0} | {"history": 8, "reader": "track-record", "truth": 0} | excess_capture | 0.14712937705999576 | 16 |

Derived records-needed/crossover decisions (null means not reached within the sampled grid):

```json
{
  "0.5": 32,
  "0.8": 32,
  "1.0": 32
}
```

Design pre-mortem: {"checks": [{"passed": false, "s": 0.5, "share": 0.0}, {"passed": true, "s": 1.0, "share": 1.0}], "interpretation": "Equal shares of the two declared biased strengths; the floor applies to the grid, not each strength.", "passed": true, "pooled_biased_source_share": 0.5, "seed_domain": "exact-enumeration-U1"}

By construction and modelling limits: Thirty-two makers, balanced true hypotheses; source strength fixed across track and claims. Track records reveal all eight pool items. Posterior on source strength integrated jointly across all new claims, never resampled per claim. Menu arm reveals four random pool items plus selected sign, using exact hypergeometric marginal. Records-needed uses h0-world mean capture curve, not individual hits. Design-check share averages the two biased source strengths equally.

Full original/replay files are byte-identical. The independent review uses a separate scalar posterior/decimal aggregation route; some likelihood kernels and the generative law are shared. Known-answer tests check the model equations, not real-world validity. No universal ceiling is inferred from a supplied-library reference.

Sounding Line U1: source-model observables for co-writing tests.
ToMpathy: no route this week.

[All estimates](../../../results/v26/v26-u1-study-1/SUMMARY.json). [Verification](../../../results/v26/v26-u1-study-1/INDEPENDENT_REVIEW.json).

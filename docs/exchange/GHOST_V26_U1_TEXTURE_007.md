# Opacity Gate

Corrections first: earlier V25 findings and retained failures remain unchanged; this is a separate V26 card.

What does opacity gate identify in the declared constructed worlds? The verified texture retains 1 settings and 16 independent whole-world blocks per setting. The registered primary estimates range from 0.0028911 to 0.215847; their units and strata are listed in the report. This is descriptive constructed-method evidence, not confirmation or human evidence. Miniature, architecture untested.

Claim `G-V26-U1T7-1`.

The table lists each registered primary by its frozen setting and observation stratum. Estimates use nats for score contrasts, records for time-to-evidence, probabilities for capture/unresolved shares, and distance bins for crossover decisions. Missing crossings remain censored; cases are averaged within worlds before uncertainty.

| Setting | Observation stratum | Measure | Mean | Available blocks |
|---|---|---|---:|---:|
| {"claims": 128, "s": 0.5, "tau": 1.0} | {"history": 1, "reader": "track-record", "truth": 0} | excess_capture | 0.21584719476158967 | 16 |
| {"claims": 128, "s": 0.5, "tau": 1.0} | {"history": 16, "reader": "track-record", "truth": 0} | excess_capture | 0.05393224035019291 | 16 |
| {"claims": 128, "s": 0.5, "tau": 1.0} | {"history": 2, "reader": "track-record", "truth": 0} | excess_capture | 0.1857173955991947 | 16 |
| {"claims": 128, "s": 0.5, "tau": 1.0} | {"history": 32, "reader": "track-record", "truth": 0} | excess_capture | 0.012287926301850494 | 16 |
| {"claims": 128, "s": 0.5, "tau": 1.0} | {"history": 4, "reader": "track-record", "truth": 0} | excess_capture | 0.15759253585776656 | 16 |
| {"claims": 128, "s": 0.5, "tau": 1.0} | {"history": 64, "reader": "track-record", "truth": 0} | excess_capture | 0.002891100177000068 | 16 |
| {"claims": 128, "s": 0.5, "tau": 1.0} | {"history": 8, "reader": "track-record", "truth": 0} | excess_capture | 0.10953411822422987 | 16 |

Derived records-needed/crossover decisions (null means not reached within the sampled grid):

```json
{
  "1.0": 32
}
```

Design pre-mortem: {"checks": [{"passed": false, "s": 0.5, "share": 0.0}, {"passed": true, "s": 1.0, "share": 1.0}], "interpretation": "Equal shares of the two declared biased strengths; the floor applies to the grid, not each strength.", "passed": true, "pooled_biased_source_share": 0.5, "seed_domain": "exact-enumeration-U1"}

By construction and modelling limits: Thirty-two makers, balanced true hypotheses; source strength fixed across track and claims. Track records reveal all eight pool items. Posterior on source strength integrated jointly across all new claims, never resampled per claim. Menu arm reveals four random pool items plus selected sign, using exact hypergeometric marginal. Records-needed uses h0-world mean capture curve, not individual hits. Design-check share averages the two biased source strengths equally.

Full original/replay files are byte-identical. The independent review uses a separate scalar posterior/decimal aggregation route; some likelihood kernels and the generative law are shared. Known-answer tests check the model equations, not real-world validity. No universal ceiling is inferred from a supplied-library reference.

Sounding Line U1: source-model observables for co-writing tests.
ToMpathy: no route this week.

[All estimates](../../../results/v26/v26-u1-texture-007/SUMMARY.json). [Verification](../../../results/v26/v26-u1-texture-007/INDEPENDENT_REVIEW.json).

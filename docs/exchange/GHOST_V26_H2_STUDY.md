# Pauses Difficulty Deliberation

Corrections first: earlier V25 findings and retained failures remain unchanged; this is a separate V26 card.

After the reader modelled choice difficulty, pauses added 0.008765 nats of goal evidence per deliberation at d=0.6, q=0.2 and 32 observations, below the 0.05 practical bar. The result rejects a practical pause-only increment in this declared cell; it does not say that real pauses are uninformative.

Claim `G-V26-H2-1`.

The table lists each registered primary by its frozen setting and observation stratum. Estimates use nats for score contrasts, records for time-to-evidence, probabilities for capture/unresolved shares, and distance bins for crossover decisions. Missing crossings remain censored; cases are averaged within worlds before uncertainty.

| Setting | Observation stratum | Measure | Mean | Available blocks |
|---|---|---|---:|---:|
| {"d": 0.6, "q": 0.2} | {"history": 32, "reader": "contrasts"} | primary | 0.008765159002560037 | 16 |

Derived records-needed/crossover decisions (null means not reached within the sampled grid):

```json
{}
```

Design pre-mortem: {"declared_range": [0.5, 2.0], "floor": 0.5, "passed": true, "seed_domain": "check-only-H2", "uncertain_share": 0.75}

By construction and modelling limits: Six makers per world, each goal once with cyclic habits. Goal range .5-2, after the full-range check; narrower engineering check retained separately. Observed replacements identify a redraw even if the replacement equals the old unit. Primary is total gain divided by total deliberations within a world, not mean case ratios. Misread reader uses q=0.

Full original/replay files are byte-identical. The independent review uses a separate scalar posterior/decimal aggregation route; some likelihood kernels and the generative law are shared. Known-answer tests check the model equations, not real-world validity. No universal ceiling is inferred from a supplied-library reference.

Sounding Line S5/S6: goal evidence in pauses after modelling choice difficulty.
ToMpathy current-attention marks require an explicit difficulty model.

[All estimates](../../../results/v26/v26-h2-study-1/SUMMARY.json). [Verification](../../../results/v26/v26-h2-study-1/INDEPENDENT_REVIEW.json).

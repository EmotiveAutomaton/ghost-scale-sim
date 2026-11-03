# One Model Two Directions

Corrections first: November 2 operator review clarifies the registered comparator, observation unit or scope below; saved measurements and claim IDs are unchanged. See results/v26/REPORTING_CORRECTION_20261102.json.

Does starting from the reader's own preferences help infer another maker? With four of six pieces matching and eight observations, the self-reader's true-ends log score first fell below the population reader's at total-variation distance 0.25 with prior concentration 2, and 0.5 with concentration 8. These are sampled crossover bins for the declared readers and libraries, not a universal radius of self-based inference.

Claim `G-V26-E2-1`.

The table lists each registered primary by its frozen setting and observation stratum. Estimates use nats for score contrasts, records for time-to-evidence, probabilities for capture/unresolved shares, and distance bins for crossover decisions. Missing crossings remain censored; cases are averaged within worlds before uncertainty.

| Setting | Observation stratum | Measure | Mean | Available blocks |
|---|---|---|---:|---:|
| {"kappa": 2.0, "overlap": 4} | {"distance": 0.0, "history": 8, "reader": "contrasts"} | primary | 0.20463880141189547 | 16 |
| {"kappa": 8.0, "overlap": 4} | {"distance": 0.0, "history": 8, "reader": "contrasts"} | primary | 1.9911672215321563 | 16 |
| {"kappa": 2.0, "overlap": 4} | {"distance": 0.25, "history": 8, "reader": "contrasts"} | primary | -0.3477609364260244 | 16 |
| {"kappa": 8.0, "overlap": 4} | {"distance": 0.25, "history": 8, "reader": "contrasts"} | primary | 0.045233681018019514 | 16 |
| {"kappa": 2.0, "overlap": 4} | {"distance": 0.5, "history": 8, "reader": "contrasts"} | primary | -0.38996512922244864 | 16 |
| {"kappa": 8.0, "overlap": 4} | {"distance": 0.5, "history": 8, "reader": "contrasts"} | primary | -0.7889988845709094 | 16 |
| {"kappa": 2.0, "overlap": 4} | {"distance": 0.75, "history": 8, "reader": "contrasts"} | primary | -0.6515713151280544 | 16 |
| {"kappa": 8.0, "overlap": 4} | {"distance": 0.75, "history": 8, "reader": "contrasts"} | primary | -2.1745114180263565 | 16 |

Derived records-needed/crossover decisions (null means not reached within the sampled grid):

```json
{
  "2.0": 0.25,
  "8.0": 0.5
}
```

Design pre-mortem: {"all_15_makers_have_each_bin": true, "bin_shares": [0.25, 0.25, 0.25, 0.25], "floor": 0.2, "passed": true, "seed_domain": "enumerated-E2"}

By construction and modelling limits: Four pairs per distance bin per world, uniform maker grid and uniform reader within bin. Known beta3, no habit in this W1-style choice variant. Partial mismatch replaces effect rows of final pieces with independent grid outcomes. S/P contrast uses posterior log scores, not gains relative to different priors. Mixture uses half of each prior with the own library.

Full original/replay files are byte-identical. The independent review uses a separate scalar posterior/decimal aggregation route; some likelihood kernels and the generative law are shared. Known-answer tests check the model equations, not real-world validity. No universal ceiling is inferred from a supplied-library reference.

Sounding Line T5: when own-model versus population anchoring helps.
ToMpathy prior design: own model, population or mixture, subject to real-data validation.

[All estimates](../../../results/v26/v26-e2-study-1/SUMMARY.json). [Verification](../../../results/v26/v26-e2-study-1/INDEPENDENT_REVIEW.json).

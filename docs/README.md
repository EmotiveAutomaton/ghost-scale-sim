# The paper trail

Four kinds of document live here, and the difference between them is the whole reason they are kept
apart.

- **[specs/](specs/)** say what was going to be built and what it was going to predict. They are
  written before the code and are not edited afterwards.
- **[writeups/](writeups/)** say what actually happened, including every place the answer disagreed
  with the spec.
- **[decisions/](decisions/)** record design choices signed off before a build, with the evidence
  that motivated them.
- **[EXPERIMENTS.md](EXPERIMENTS.md)** is the consolidated plain-language table of every question
  asked and every answer that came back. The version in the top-level README is this table with a
  validation column added.

Where a spec is now known to be wrong, the correction lives in the write-up that found it, never
here. That is deliberate: a spec quietly amended after its own experiment has run is no longer a
record of what was predicted, and this project's claim to being checkable rests on those two things
staying separable.

## The build specs

| file | what it is |
|---|---|
| [specs/SPEC_V1.md](specs/SPEC_V1.md) | The original build spec. Hypotheses H1 through H6, nulls N1 through N7, the parameter table, the invariant tests. |
| [specs/SPEC_V2.md](specs/SPEC_V2.md) | Version 2. Adds the Learner observer, reader heterogeneity, biased machine-made content, and nulls N8 through N12. |
| [specs/PLAN_V2.md](specs/PLAN_V2.md) | The longer working plan behind version 2, including the reasoning that did not survive into the spec. |
| [specs/SPEC_V3.md](specs/SPEC_V3.md) | Version 3. Written to repair E8 under a finite-sample diagnosis, which version 3's own gate then refuted. Adds nulls N13 through N15. |
| [specs/SPEC_V4.md](specs/SPEC_V4.md) | Version 4. Not a repair: it changes what the model claims machine-made content is, from goal-empty to goal-foreign, and tests whether the earlier results survive. Adds nulls N16 through N20. |
| [specs/SPEC_V4_5.md](specs/SPEC_V4_5.md) | Version 4.5. The three-gate observer, and the promotion of the metabolic question from a column to a headline. |
| [specs/SPEC_V5.md](specs/SPEC_V5.md) | Version 5. Replaces rationality with model depth, makes provenance evidence rather than a parallel channel, and asks whether foreign content and an unskilled reader are the same failure. Adds N21. |
| [specs/SPEC_VALIDATION.md](specs/SPEC_VALIDATION.md) | The validation pass. Not a new version: nothing in it asks a new question about the world. Every item asks whether the answers already recorded can be trusted. Its results are in [../VALIDATION.md](../VALIDATION.md). |
| [specs/SPEC_PUBLIC_ASSETS.md](specs/SPEC_PUBLIC_ASSETS.md) | The specification for the public-facing material: the README rewrite and the distribution slides in `figures/social/`. |

## The write-ups

| file | version | the short version of what it found |
|---|---|---|
| [writeups/RESULTS_V1.md](writeups/RESULTS_V1.md) | 1 | The crash, the trust exploit, and the labelling trade-off. Seven deviations logged. |
| [writeups/RESULTS_V2.md](writeups/RESULTS_V2.md) | 2 | Reader heterogeneity moves from the prior to the likelihood. Biased machine-made content accumulates rather than averaging out. |
| [writeups/RESULTS_V3.md](writeups/RESULTS_V3.md) | 3 | The finite-sample diagnosis of the recursion leak was refuted by version 3's own gate. The estimator bias was located and fixed. E8 stayed withheld. |
| [writeups/RESULTS_V4.md](writeups/RESULTS_V4.md) | 4 | Goal-empty became goal-foreign, and the metabolic prediction inverted. The feature space had to double. |
| [writeups/RESULTS_V4_5.md](writeups/RESULTS_V4_5.md) | 4.5 | The three-gate model, the readability sweep, and four unwelcome results including the counting classifier that withdrew a claim. |
| [writeups/RESULTS_V5.md](writeups/RESULTS_V5.md) | 5 | Depth replaces effort. Two gates rather than three. And the variational solver caught misreading depth entirely, which is what the validation pass then went after across the whole body of work. |

## The decisions

| file | what it covers |
|---|---|
| [decisions/DECISIONS_V2.md](decisions/DECISIONS_V2.md) | Version 2's five signed-off decisions, with the measurements behind them. |
| [decisions/DECISIONS_V3.md](decisions/DECISIONS_V3.md) | Version 3's decisions, and an incident worth recording: a smoke run that overwrote committed output, and the two hardening changes that followed. |

## Reading order, if you are starting cold

1. The top-level [README](../README.md), for what was asked and what came back.
2. [../VALIDATION.md](../VALIDATION.md), for how much of it survived being checked.
3. [EXPERIMENTS.md](EXPERIMENTS.md), if you want the table without the validation column.
4. A write-up, then its spec, if you want to see a specific number's whole history.

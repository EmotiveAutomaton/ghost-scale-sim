# The repair pass: from demonstration to measurement

Generated from the verdict files in [results/repair/](../../../results/repair/). Regenerate with `python scripts/write_repair_md.py`.

---

## What this pass was for

The diagnostics pass named the real problem. **This apparatus was built to demonstrate and was being audited as though it measures.** Those have different standards, and most of what the audit returned is the gap between them rather than anything being wrong.

The evidence was direct. Trust looked unmeasurable because nothing in the model had ever been built to estimate trust; it is a knob you turn to ask what a reader with that disposition would do. The two parameters that recovered cleanly recovered because somebody wrote an estimator for them during the audit. They had none before either.

**The rule for this pass: every change either makes something measurable that was not, or removes something.** The model has grown every version since the first and nothing has ever been taken out, and that accretion is what produced constructions the audit could not break because the model refuses to run without them. An addition qualified only if it converted a free choice into a measured quantity.

---

## The pass at a glance

| item | what it asks | how it came back |
|---|---|---|
| **R-1 to R-4** | What do the under-powered criteria say once they have intervals? | some criteria remain undetermined |
| **R-5** | Has the uptake measure been measuring understanding? | error reduction goes negative where movement cannot |
| **R-6 / R-8a** | Which parameters are measurable, fitted to what the reader produces, and over what range? | trust is measurable over part of its range |
| **R-8b** | If trust is learned about a named source, does it recover, and can a trusting reader learn? | reputation blindness above the crossover |
| **R-11 / R-12** | Every reachable experiment, exact inference and fixed seeding. What moves? | some verdicts move under the repaired model |
| **R-13** | The two inconclusive experiments, rerun on a fair footing. | depth still inconclusive / crash survives the generous fallback |

---

## R-1 to R-4: recomputation, with intervals

Three of the project's headline conclusions were computed from a handful of numbers with no error bar, so nobody could say whether they were solid or noise. This puts an interval on each of them by resampling the readers, without running a single new simulation. It also adds a second disagreement statistic beside the existing one, and corrects a known bias that only matters at small reader counts.

| criterion | experiment | original | recomputed | 95% interval | threshold | verdict |
|---|---|---|---|---|---|---|
| update magnitude tracks recovered depth | E31 | 0.886 | 0.886 | [0.771, 0.886] | 0.700 | **determined meets** |
| depth recovery is not effort recovery | N21 | 5.984 | 5.984 | [4.492, 8.944] | 3.000 | **determined meets** |
| depth changes how much the reader takes on (movement (the measure on record)) | E30 | — | 0.500 | [-0.500, 0.500] | 0 | **undetermined** |
| depth changes how much the reader takes on (structural movement (the added secondary)) | E30 | — | 0.500 | [0.500, 0.500] | 0 | **determined meets** |

**update magnitude tracks recovered depth** (E31): recomputed 0.886, 95% interval [0.771, 0.886] against a threshold of 0.70. determined meets.

**depth recovery is not effort recovery** (N21): recomputed 5.984, 95% interval [4.492, 8.944] against a threshold of 3.00. determined meets.

**depth changes how much the reader takes on** (E30, movement (the measure on record)): 0.500, 95% interval [-0.500, 0.500]. undetermined.

**depth changes how much the reader takes on** (E30, structural movement (the added secondary)): 0.500, 95% interval [0.500, 0.500]. determined meets.

**The correction that mattered most.** The specification asked for these to be recomputed over per-reader pairs. Doing that would have manufactured a failure: within cells the two quantities are uncorrelated, so pooling readers drowns the signal rather than resolving it, and the pooled value comes out at 0.514 against the cell-level 0.886. Bootstrapping the cell means preserves the estimand, and it shows the criterion is not marginal at all: 75% of draws land on a single value and the interval excludes the threshold. **The earlier reading that this criterion could never have told either way was wrong, and it was mine.**

**The interior peak now has an error bar.** Resampling readers and recomputing the whole curve, the peak lands at 0.1 in 100% of draws, and the distribution over grid points is 0.1: 100%. The location is settled and the prediction card can quote it.

**Disagreement, read two ways.** Every cell now carries the modal-goal entropy as committed, the same figure with the reader-count bias corrected, and the mean pairwise divergence between readers' full beliefs. The bias correction is at most 0.0075 nats on these cells, which is negligible at the reader counts the headlines ran at and is the point: it is material only at the small counts the validation pass used, which is where the two were being compared. Four experiments cannot be given the divergence figure without re-running, and they are named.

### The interior peak, with an error bar

| overlap | fabrication index | share of bootstrap draws where this is the peak |
|---|---|---|
| 0 | 0.092 | 0 |
| 0.10 | 0.232 | 1.000 |
| 0.20 | 0.107 | 0 |
| 0.30 | 0.081 | 0 |
| 0.40 | 0.033 | 0 |
| 0.50 | 0.029 | 0 |
| 0.60 | 0.012 | 0 |
| 0.70 | 0.017 | 0 |
| 0.85 | 0.004 | 0 |
| 1.00 | 3.77e-04 | 0 |

### Disagreement, read three ways

| experiment | cell | uncertainty per reader | modal-goal entropy, as committed | bias corrected | pairwise divergence |
|---|---|---|---|---|---|
| E2 | CREATOR / SIG_CREATOR | 7.06e-08 | 0 | 0 | 3.63e-09 |
| E2 | CREATOR / SIG_GHOST | 0.092 | 0.009 | 0.010 | 0.013 |
| E2 | GHOST / SIG_CREATOR | 0.090 | 1.379 | 1.387 | 0.498 |
| E2 | GHOST / SIG_GHOST | 1.293 | 1.378 | 1.386 | 0.044 |
| E17 | CREATOR / claimed_human | 7.06e-08 | 0 | 0 | 3.63e-09 |
| E17 | CREATOR / truthful | 9.76e-07 | 0 | 0 | 7.10e-08 |
| E17 | CURATOR / claimed_human | 0.011 | 0.108 | 0.114 | 0.026 |
| E17 | CURATOR / truthful | 0.029 | 0.048 | 0.050 | 0.013 |
| E17 | GHOST / claimed_human | 0.089 | 1.379 | 1.387 | 0.498 |
| E17 | GHOST / truthful | 1.297 | 1.380 | 1.387 | 0.042 |
| E17 | POLISHED / claimed_human | 2.37e-06 | 0 | 0 | 1.79e-07 |
| E17 | POLISHED / truthful | 2.07e-05 | 0 | 0 | 2.27e-06 |

Experiments that cannot be given the divergence figure without re-running, named rather than skipped: **E20** drops the posterior before writing e20_points.csv; **E31** drops the posterior before writing e31_points.csv; **E32** drops the posterior before writing e32_points.csv; **E21** writes cell statistics only.

---

## R-5: what the uptake measure was actually measuring

Several experiments measure how much a reader 'takes on' from a work. The measure is a distance between what the reader believed before and after, so a reader who ends up confidently wrong scores almost as highly as one who ends up right: both moved a long way. This splits that one number into three, the important one being how much closer to the truth the reader actually got, which can go negative and which the old measure cannot express.

*the specification writes error reduction with the divergence arguments the wrong way round, which makes every term infinite against a point mass on the truth and leaves a value determined by the epsilon it is floored at. Corrected to the reduction in the surprisal of the truth.*

| experiment | cell | reads the right purpose | movement, as reported | error reduction | share confidently wrong |
|---|---|---|---|---|---|
| E2 | CREATOR / SIG_CREATOR | 1.000 | 1.386 | **1.386** | 0 |
| E2 | CREATOR / SIG_GHOST | 0.999 | 1.294 | **1.360** | 0 |
| E2 | GHOST / SIG_CREATOR | 0.238 | 1.297 | **-5.964** | 0.761 |
| E2 | GHOST / SIG_GHOST | 0.259 | 0.094 | **-0.091** | 5.00e-04 |
| E17 | CREATOR / claimed_human | 1.000 | 1.386 | **1.386** | 0 |
| E17 | CREATOR / truthful | 1.000 | 1.386 | **1.386** | 0 |
| E17 | CURATOR / claimed_human | 0.981 | 1.375 | **1.293** | 0.019 |
| E17 | CURATOR / truthful | 0.992 | 1.358 | **1.353** | 0.005 |
| E17 | GHOST / claimed_human | 0.244 | 1.297 | **-5.834** | 0.756 |
| E17 | GHOST / truthful | 0.247 | 0.089 | **-0.092** | 0.002 |
| E17 | POLISHED / claimed_human | 1.000 | 1.386 | **1.386** | 0 |
| E17 | POLISHED / truthful | 1.000 | 1.386 | **1.386** | 0 |

**The decomposition separates cells the old measure could not.** In 4 cell(s) error reduction is NEGATIVE: the reader ended further from the truth than it started. The clearest is GHOST / SIG_CREATOR in E2, where movement reads 1.297, which looks like substantial uptake, while error reduction reads -5.964, which says the reader was moved away from the answer. The measure on record cannot express that at all, because a distance has no sign.

**The trust weight is now reported separately** rather than multiplied in. It runs from 0.105 to 4.605 across the range one experiment sweeps, a factor of 44, which is larger than most effects in this project. Any sweep that varied trust and reported uptake was reporting the product of two things.

**Five experiments cannot be decomposed from committed data** and are named rather than skipped: E30 carries psi_analogue but not the posterior, so error reduction cannot be recomputed. This is the experiment the decomposition matters most for, and the rerun supplies it; E31 carries prior_drift but not the posterior; E20 carries neither the posterior nor a prior to measure movement against; E32 carries psi_analogue but not the posterior; E4 raw file not committed. The first of those is the one the decomposition matters most for.

*computed against the uniform prior each reader's own prior was perturbed from, because the committed files do not persist the per-reader prior. Declared rather than hidden: between-cell comparisons are unaffected because the reference is shared, and absolute single-reader values are approximate. The rerun carries the real prior.*

---

## R-6 and R-8a: the identifiability map

To measure something about a reader you have to look at what the reader does, not at what it was shown. The earlier pass looked at what it was shown, which is why trust came back unmeasurable. Fitted properly, trust is measurable up to a point and then stops mattering: past that point two readers with different trust behave identically, so there is nothing left to measure. That is a fact about the model rather than a limit on the measurement.

**Correction.** the diagnostics pass fitted trust to the observation tape, which the world generates and which contains no trust parameter. Its UNIDENTIFIABLE verdict answered the wrong question and is superseded here. The original verdict is retained in results/diagnostics/p1_recovery.json.

| parameter | what it was fitted to | observability | slope | identifiable range | reading |
|---|---|---|---|---|---|
| kappa (trust in the label) | the reader | narrow | 0.70 | 0.10 to 0.60 (50% of the range) | measurable over part of the range |
| kappa (trust in the label) | the reader | wide | 1.00 | 0.10 to 0.90 (100% of the range) | measurable across the range |
| omega (how readable the content is) | an analyst holding the model | — | 1.00 | 0 to 1.00 (100% of the range) | measurable across the range |
| mu (depth of the maker's thinking) | the reader | — | 0.32 | 1.00 to 2.00 (50% of the range) | measurable over part of the range |
| theta (the value-alignment gate) | an analyst holding the model | — | 0.95 | 0 to 7.00 (88% of the range) | measurable over part of the range |

**The earlier verdict was wrong and it was mine.** Trust looked unidentifiable because it was fitted to the observation tape, and the tape is generated by the world, which contains no trust parameter. Fitted to what the reader itself produces, it recovers.

**Under the narrow standard**, what the reader does and says, trust is identifiable over 50% of the swept range, from 0.10 to 0.60, and saturates above that. Above the saturation point readers with different trust are behaviourally identical: same engagement, same answer, every time.

**Under the wide standard**, which additionally lets the reader report graded confidence, it is identifiable over 100% of the range. The pair brackets what any real reporting process could deliver: the narrow figure is a floor and the wide one a ceiling.

**The saturation point is not arbitrary.** The channel-accounting check locates, in closed form, the trust level at which the label's evidence per glance overtakes the work's: 0.538. Above it the label has already won and additional trust has nothing left to change, which is exactly where behaviour stops varying. Saturation is what a parameter looks like once it has won its argument.

**The consequence is uncomfortable and it should be said plainly.** The model's default trust sits ABOVE that point. So at the setting every headline experiment runs, trust is not a measurable quantity even in principle, and claims of the form 'a reader with this much trust does X' are stipulations rather than measurements at that setting. What remains sayable, and is observable, is ordinal: this reader is in the regime where the label dominates the work.

**And it gives a human study its design.** Do not try to measure trust directly, and do not run the study with content where the label overwhelms what is on the page: everyone looks the same there and nothing is learned. Run it where the work can argue back, which is the only regime in which individual differences in trust have observable consequences.

**The rest of the map**, carried forward unchanged because their estimators did not change: omega (how readable the content is) is measurable across the range (100% of its range), fitted to an analyst holding the model; mu (depth of the maker's thinking) is measurable over part of the range (50% of its range), fitted to the reader; theta (the value-alignment gate) is measurable over part of the range (88% of its range), fitted to an analyst holding the model. The distinction between fitting to the reader and fitting to an analyst's model is not decoration: the second asks whether a value is identifiable in principle, which is strictly easier, and for readability that gap is the whole version 4 reframe.

---

## R-8b: trust as something learned about a source

Until now a reader's trust in labels was a setting: fixed when the reader was made, never revised, and so not the kind of thing that could be measured even in principle. This gives sources names and honesty rates, lets a reader build up a view of how reliable a particular source has been, and then asks two things. Can the reader work out how honest the source is? And does that ability depend on how much it trusted labels to begin with?

**Why it was built.** not for the reason the specification gives. Trust became measurable by fitting it to behaviour, which is a much smaller change and is already done. This earns its place under the pass's own rule instead: it converts a fixed disposition, which nothing in the reader's history could ever revise, into an inference with a true value and an estimate. It also makes a prediction the fixed-trust model cannot.

**What was deliberately not built.** reputation dynamics and strategic defection. Those need a source that chooses when to lie, which is an agent with its own objective, and the results would be hard to attribute. This adds a source identity and a reader-side belief about that source's honesty, and nothing else.

| trust the reader started with | how far the label outweighs the work, per glance | weight it gives its own reading | recovery of the source's honesty | identifiable range | reading |
|---|---|---|---|---|---|
| 0.200 | -1.109 | 0.988 | 1.00 | 100% | measurable across the range |
| 0.400 | -0.453 | 0.860 | 1.00 | 100% | measurable across the range |
| 0.538 | +0.001 | 0.499 | 1.00 | 100% | measurable across the range |
| 0.700 | +0.619 | 0.077 | 1.00 | 100% | measurable across the range |
| 0.900 | +1.909 | 0.000 | 1.00 | 100% | statistically separable but flat: it tracks almost none of the true variation |

**A trusting reader cannot learn that a source lies, and the threshold is the one the channel accounting already located.** Readers starting at trust 0.20, 0.40, 0.54, 0.70 recover the source's honesty. Readers starting at 0.90 do not, at any number of encounters, because the evidence never arrives: working out that a source is unreliable means noticing that the label and the work disagree, and above the crossover the label wins every such disagreement before it can be registered.

**This is not slow learning. It is learning that cannot start.** And it is the prediction the fixed-trust model was structurally incapable of making, which is what earns this addition its place: it converts a setting into an inference, and the inference has a threshold in it.

**If it holds outside this model it is an unpleasant claim about disclosure.** The readers most disposed to believe provenance labels are exactly the ones who cannot discover that a labeller is unreliable, so a disclosure regime protects them least where it fails most. It is stated here as a property of a simulation and nothing more, and it is the kind of claim the prediction card exists to hand to a human study rather than to settle.

---

## R-11 and R-12: every reachable experiment, both ways

The validation pass rechecked five experiments with the arithmetic done exactly rather than approximately, and three conclusions changed. This runs every experiment that can be run, including six that were blocked until the learning path was rebuilt, and reports every headline quantity both ways.

**Design.** matched pairs. The baseline arm is re-run now on the old code path rather than read off disk, so a difference between the pair is attributable to the change and not to anything that drifted since. Three things move together in the repaired arm and that is deliberate: separating them would need three full programmes and would still not isolate anything, because re-seeding alone re-randomises every reader.

| experiment | ran both ways | outcome, old code path | outcome, repaired | moved |
|---|---|---|---|---|
| E1 | yes | — | — | no |
| E5 | yes | — | — | no |
| E2 | yes | — | — | no |
| E17 | yes | confirmed: disagreement under a human claim rises monotonica | confirmed: disagreement under a human claim rises monotonica | no |
| E3 | yes | — | — | no |
| E4 | yes | — | — | no |
| E19 | yes | CRASH_SURVIVES | INCONCLUSIVE | yes |
| E20 | yes | INTERIOR_PEAK | INTERIOR_PEAK | no |
| E21 | yes | MACHINERY_PARTLY_NECESSARY | MACHINERY_PARTLY_NECESSARY | no |
| E32 | yes | TWO_DIMENSIONS | TWO_DIMENSIONS | no |
| E15 | yes | — | — | no |
| E28 | yes | BETA_IS_SEPARABLE_OVER_PART_OF_THE_RANGE | BETA_IS_SEPARABLE_OVER_PART_OF_THE_RANGE | no |
| E29 | yes | GATES_PARTLY_DISSOCIATE | GATES_PARTLY_DISSOCIATE | no |
| E30 | yes | DEPTH_MOVES_NOTHING | DEPTH_MOVES_NOTHING | no |
| E31 | yes | ONE_MECHANISM_OPPOSITE_SIGNS | DEPTH_IS_NOT_THE_COMMON_PATH | yes |
| E33 | yes | READS_THE_LATENT_WITHOUT_A_TRACE | READS_THE_LATENT_WITHOUT_A_TRACE | no |
| E7 | yes | — | — | no |
| E16 | yes | — | — | no |
| E9 | yes | — | — | no |
| E18 | yes | the estimator fix does NOT remove the generational contracti | the estimator fix does NOT remove the generational contracti | no |
| E13 | yes | shared_axis | shared_axis | no |
| E6 | yes | — | — | no |
| E6b | yes | NOT falsified — the bias axis produces the predicted corrupt | NOT falsified — the bias axis produces the predicted corrupt | no |
| E14 | yes | — | — | no |
| E12 | yes | — | — | no |

25 of 26 reachable experiments completed in both arms.

**These outcomes moved under the repaired model:** E19, E31. Each is reported with both values, and the original is retained.

Individual verdict components that flipped: E19.positive_control_passed; E20.any_cell_crashes; E31.update_tracks_recovered_depth.

**1 experiment(s) had not finished in both arms when this was written and are named rather than left out:** E34. They are the two ordered last on purpose, because they dominate the wall clock; the list is cheap-first so a run that is cut short still covers the majority.

The withheld experiment stays withheld and was not run.

*E8: withheld; it has never passed its own control and the repair pass does not change that.*

---

## R-13: the two reruns

Two experiments produced nothing readable. One asked whether the amount of thinking behind a work changes how much a reader takes from it, and found a flat line; the other's own success case failed one of its own checks. Both are now run again in conditions where the reader is genuinely uncertain, using a measure that can tell moving toward the truth from moving away from it.

**The difficulty regime does not transfer between geometries, and finding that out was worth the rerun on its own.** The probe located it at reader inexpertise 0.85 on the version 1 geometry, where accuracy drops to about 0.63. Calibrating the same knob against the version 5 geometry, accuracy will not come down: it sits at 0.900 even at the calibrated value, and the sweep found nothing inside the target band anywhere up to total inexpertise. **Depth makes the reader harder to confuse, because it gives the goal more than one route to the surface.** So the depth experiment cannot be run in a regime where goal recovery is genuinely uncertain, by this knob, at all.

**A three-level rank correlation cannot be settled by any amount of resampling**, which is why the primary is a contrast rather than a correlation. On three points Spearman takes only four possible values, so its bootstrap distribution is a handful of atoms and it straddles zero unless every draw agrees. That is a structural limit of the design and no regime repairs it.

**The contrast between the deepest and shallowest work, which is continuous and can be settled, bounds the effect rather than merely failing to find one.** On how much closer to the truth the reader got, deepest minus shallowest is -0.0384 with a 95% interval of [-0.1001, +0.0232]. On the measure the original used it is +0.0332, interval [-0.0087, +0.0746]. Both straddle zero, so neither direction is established, and the useful statement is the width: **any effect of depth on what the reader takes on is smaller than about a tenth of a nat.** For scale, the false-label effect on the same measure is -5.96. Depth's influence on uptake is at most a fiftieth of the label's, and is consistent with nothing at all.

Cell means: depth 1 gives accuracy 0.931, error reduction +0.923, movement 0.490; depth 2 gives accuracy 0.884, error reduction +0.886, movement 0.510; depth 3 gives accuracy 0.884, error reduction +0.885, movement 0.524. The two deepest levels remain indistinguishable from each other on every column, which the original experiment also found and which halves an already three-point design.

**The generous fallback comes back, and its original finding is restored.** The control it failed required the fallback to absorb exploratory human work WHILE the reader kept paying attention, and a reader that has correctly resolved the goal stops paying attention, so the one cell meant to demonstrate success scored 0.000 on a clause requiring 0.5. Rebuilt so that absorption is mass and convergence, with engagement measured separately as the crash signature already does elsewhere, the control passes at a mass of 0.704 while foreign content takes only 0.204. **The crash survives the most generous explanation the theory permits, under exact inference, with a control that can actually pass.** That restores a finding the validation pass had reduced to inconclusive, and it restores it on stronger footing than it originally had.

**Both original verdicts are retained** and are reported beside these. A rerun in a better regime is not permission to overwrite the record of what the first attempt said.

---

## What this pass changed about what may be claimed

1. **update magnitude tracks recovered depth (E31) is now determined and it holds**, at 0.886 with a 95% interval of [0.771, 0.886] against a threshold of 0.700. It was previously reported as possibly undecidable, and that reading was wrong.
2. **depth recovery is not effort recovery (N21) is now determined and it holds**, at 5.984 with a 95% interval of [4.492, 8.944] against a threshold of 3.000. It was previously reported as possibly undecidable, and that reading was wrong.
3. **The interior peak has an error bar and it is tight**: the same grid point in 1.000 of bootstrap draws. The prediction card can quote a location rather than a guess.
4. **Uptake has been split into movement and error reduction, and they disagree about the headline cell.** A false label reads as substantial uptake on the old measure and as a large NEGATIVE on the new one: readers are moved away from the answer, further than an honest label moves them toward it. Every claim about how much a reader 'takes on' needs to say which of the two it means.
5. **Trust is measurable after all, over part of its range.** The earlier unidentifiable verdict fitted it to the wrong data. It saturates above the channel crossover, and the model's default sits in the saturated region, so at the setting the headlines run at trust remains a stipulation rather than a measurement.
6. **A new prediction the earlier model could not make: a sufficiently trusting reader cannot learn that a source lies, at any number of encounters.** The threshold is the channel crossover. If it holds outside this model it says a disclosure regime protects trusting readers least where it fails most.
7. **These outcomes moved under the repaired model:** E19, E31.
8. **The generous-fallback result is restored**, under exact inference and with a control that can actually pass. The validation pass had reduced it to inconclusive.
9. **Depth still does not move what the reader takes on**, and the rerun bounds the effect rather than merely failing to find it. It also establishes that the difficulty regime does not transfer to that geometry at all, which is a finding about the design rather than about depth.

---

## What was retained rather than replaced

Every original number is still in the repository and is still reported beside its replacement. The old uptake measure is computed alongside the new one; the modal-goal entropy is reported beside the divergence and beside its bias-corrected form; the original per-reader seeding is still selectable, so every number produced before this pass can be regenerated by the code that produced it; and both original verdicts for the reruns are carried forward unchanged.

The withheld experiment stays withheld, its failing test stays in the suite, and the open residual stays open.

---

## What comes next

The minimal-model programme, which is the subtraction this pass's rule was written against. For each surviving result, find the smallest model that still produces it. Then the minimal models become a family, and comparison replaces single-model validation as the frame: with one model a failure is uninterpretable, and with a family it tells you which commitment was wrong.

*Generated from results/repair/ on 2026-07-31 by `scripts/write_repair_md.py`. Every number above is read out of a verdict file; none is typed in.*

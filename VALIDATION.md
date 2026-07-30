# What was checked, and what the checks came back with

This page is generated from the verdict files in [results/validation/](results/validation/). Nothing in it is written by hand except the explanations, and none of the explanations can change a number. Regenerate it with `python scripts/write_validation_md.py`.

---

## Read this part first

**This is exploratory modelling, and all of it is confirmatory by construction.** Every prediction in this repository came from one prior theory. The simulations write that theory down as working code and check whether its parts fit together, which means agreement between the model and the theory is the *expected* outcome rather than evidence for it.

That is a legitimate way to work and it has one specific failure. **A model can reproduce its own assumptions and be indistinguishable, from outside, from a model that discovered something.** Telling those two apart is the entire job of the checks below.

The risk is not hypothetical here. Seven times in this project's history an instrument was quietly answering a different question than the one being asked, and every time the wrong answer looked completely reasonable. A stale threshold file overrode a requested sample size. A statistic was computed at the wrong timestep. A lucky seed produced a confirmation that vanished across four draws. An inference shortcut was confidently blind to the quantity being measured. Six of those were caught by checks written for other reasons. **This pass makes the checking systematic instead of lucky.**

A check that fails is not deleted. It is reported with its failure attached, in the same place as the claim.

**Scale.** The checks ran at 60 simulated readers and 16 random seeds per cell, with 300 random-model draws. That is reduced from the headline experiments, deliberately and as the specification permits: the question is whether a conclusion *survives* a change of solver or of parameter, which needs enough readers to resolve the effect rather than the precision the headline number was originally quoted at. Every result below carries the scale it ran at.

**The criteria were fixed before the checks ran** and hash-locked: `fccfbd5e9bc3d469`, in [results/validation/criteria.json](results/validation/criteria.json). Editing that file after the fact makes the whole pass refuse to run.

---

## The nine checks at a glance

| check | what it asks | how it came back |
|---|---|---|
| **V-1** | Does the inference approximation distort the headline results? | a verdict flipped under exact inference |
| **V-2** | Would a model of this shape produce these results anyway? | provenance mapping is load bearing / effect is architecture dependent |
| **V-3** | Are the headline results knife-edge or robust? | both headlines hold across the swept range |
| **V-4** | What is forced by construction? | some claims survive losing their own foundations |
| **V-5** | Would any verdict change under the criterion as originally written? | some verdicts depend on the restated criterion |
| **V-6** | Do the versions agree where they measure the same thing? | consistent across versions |
| **V-7** | Do the headlines survive a different seed block and twice the scale? | seed and scale independent |
| **V-8** | Does the strongest result survive being rebuilt from its own description? | mechanism replicates magnitude does not |
| **V-9** | Can this project be wrong about something in advance? | prediction locked |

---

## V-1: does the fast approximation distort the results?

Every experiment before version 5 used a fast approximate way of updating the reader's beliefs. That shortcut is known to have been badly wrong once, in a case version 5 caught. This check re-runs the headline experiments with the shortcut removed, using exact arithmetic over every combination of possibilities, and puts the two answers next to each other.

Every experiment before version 5 used pymdp's variational solver, which keeps the reader's beliefs about different unknowns *separately* and updates each one using an average over the others. That shortcut is known to have been badly wrong once: version 5 found it returning the shallow answer for every artifact, confidently, while exact arithmetic on the same observations recovered depth correctly. Version 5 worked around that one case. It did not establish that the earlier results were safe.

So the shortcut was removed rather than worked around. [`ghostscale/exact.py`](ghostscale/exact.py) carries the reader's belief over every combination of unknowns at once and updates it by Bayes' rule with no independence assumption anywhere. The five headline experiments then re-ran **through their own unmodified code**, twice, with one setting flipped. Anything that moved was moved by the factorisation and by nothing else.

Before any comparison: on a construction where the shortcut is provably exact, the two agents agree to 1.42e-14. A disagreement there would have meant the new code was wrong rather than the old code, and everything below would have been meaningless.

| result | quantity | approximate | exact | committed full-scale | survives |
|---|---|---|---|---|---|
| E20 | fabrication peak omega | 0.1 | 0.1 | 0.1 | yes |
| E20 | fabrication peak value | 0.316 | 0.308 | 0.302 | yes |
| E20 | engagement crossing omega | 0.022 | 0.013 | 0.041 | yes |
| E20 | fabrication peak is interior | yes | yes | yes | yes |
| E20 | any cell crashes | yes | yes | yes | yes |
| E19 | foreign engaged fraction | 0.683 | 0.682 | — | yes |
| E19 | foreign final entropy | 1.488 | 1.483 | — | yes |
| E19 | foreign explore mass | 0.203 | 0.203 | — | yes |
| E19 | positive control passed | yes | no | yes | no |
| E19 | foreign absorbed | no | no | no | yes |
| E31 | update tracks mu rho | 0.886 | 0.6 | 0.886 | no |
| E31 | exploit mu gap | 0.187 | 0.831 | 0.187 | no |
| E31 | fabrication gap | 0.031 | 0 | 0.025 | no |
| E31 | update tracks recovered depth | yes | no | yes | no |
| E31 | dishonest label inflates depth | yes | yes | yes | yes |
| E31 | mu theta dissociate on behaviour | yes | yes | yes | yes |
| E32 | foreign within at matched | 1.265 | 1.265 | — | yes |
| E32 | inexpert within at matched | 0.69 | 0.889 | — | yes |
| E32 | foreign engaged at matched | 0.542 | 0.53 | — | yes |
| E32 | inexpert engaged at matched | 6.70e-04 | 5.21e-04 | — | yes |
| E32 | two dimensions | yes | yes | — | yes |
| E2 | ghost as creator within | 0.087 | 0.087 | — | yes |
| E2 | ghost as creator between | 1.365 | 1.365 | — | yes |
| E2 | ghost as ghost within | 1.294 | 1.294 | — | yes |
| E2 | creator as creator within | 1.56e-08 | 1.56e-08 | — | yes |
| E2 | label induces confidence | yes | yes | — | yes |

**At least one verdict is a property of the approximation rather than of the model: e19.positive_control_passed, e31.update_tracks_recovered_depth. Those claims are reported with the failure attached, in the same cell as the claim.**

The overall outcome string changed under exact inference for: E19, E31. Those are the rows to read closely.

---

## V-2: would a model of this shape produce these results anyway?

### Part one: break the thing the result is supposed to be about

The label effect is supposed to be about what a provenance tier tells you about content. If that is true, then destroying the connection between provenance and content should remove the effect, and *reversing* it should leave the effect the same size pointing the other way. Both were run, because the first on its own cannot distinguish "provenance is doing the work" from "the model is delicate and any change breaks it".

| condition | how many times more doubt an honest label leaves |
|---|---|
| intact | 14.9x |
| connection destroyed (every tier equally transparent) | 0.98x |
| connection reversed | 0.068x |

**Destroying the provenance-to-content mapping removes the label effect. Reversing it leaves the effect the same size and turns it upside down. The result is attached to what a provenance tier means about content, not to the tier's name and not to the model being delicate.**

> **A criterion was restated during this check, and it is logged here rather than quietly applied.** the permutation clause was restated from 'the effect survives, same reportability bar' to 'the effect's log magnitude is preserved and its direction inverts'. Why: the permutation is a reversal of alpha, so GHOST inherits CREATOR's transparency and vice versa. Under that reversal the effect MUST invert; a direction-sensitive bar was asking whether a reversal reverses. The restatement follows from the construction, not from the measurement. The original clause is retained, still computed, and FAILS.

### Part two: the false-positive rate of the apparatus

The model has a shape and it has settings. This check keeps the shape, throws the settings away, picks new ones at random a few hundred times, and asks how often the random version still produces the headline. That fraction is the rate at which the apparatus manufactures findings out of nothing, and nobody had it before. It runs twice: once with the settings drawn freely, and once with the two design decisions the model has held since version 1 left in place, so the two halves of the headline can be told apart.

| how the settings were drawn | confident under a false label | disagreement near its ceiling | clears the whole bar |
|---|---|---|---|
| unconstrained | 100.0% | 0.0% | 0.0% |
| within partition | 100.0% | 64.0% | 64.0% |

**With the likelihood family drawn freely, 100% of random models produce confident belief under a false label and 0% produce disagreement near its ceiling. Confident commitment under a false label is therefore a property of the architecture: a randomly parameterised reader does it too. What a random model does NOT supply is the DISAGREEMENT that makes the commitment invention rather than consensus, and that half of the headline is the half the theory is entitled to. Inside the architecture's own two constructions, the goal/feature partition and the symmetrised synthetic distribution, 64.0% of random parameterisations clear the whole reportability bar against the 5% committed before the run, and the designed model's effect (14.9x) sits INSIDE the random distribution's 97.5th percentile (41.3x). The claim is reported as architecture-dependent, in the same cell as the claim, per the spec's constraint that a failed check is not deleted.**

This is the most consequential thing in the pass and it deserves saying plainly. **The confident half of the headline is architectural.** A reader built to this shape, with its settings thrown away and replaced at random, still becomes certain about machine-made work when a label tells it a person was involved. What a random reader does *not* produce is the disagreement, the part where no two readers land on the same answer, and that is what makes the certainty *invention* rather than shared error. The theory is entitled to the second half. It is not entitled to the first.

---

## V-3: knife-edge, or robust?

Every setting in the model was chosen by somebody. This check moves each one to a low and a high value in turn and asks whether the finding is still there. A finding that only appears at one setting is a much smaller claim than a finding that survives the whole range, and both get described accurately.

*A scoping exercise, not a pass/fail. A result that holds across the whole range and a result that holds in a narrow window are both real; they are different claims, and the point of this matrix is to make the README able to say which is which.*

### the same machine-made content read as certain or uncertain depending only on what the label says

- holds in **22 of 27** swept cells (3 weaken, 2 reverse, 0 could not be built)
- reported as tuned: **no**
- lost when these change: feature_count, goal_count

The same machine-made content read as certain or uncertain depending only on what the label says holds in 22 of 27 swept cells, and is lost only when feature_count or goal_count changes. That is the boundary of the claim and the README states it.

### confident invention peaks in the middle of the readability axis, not at the unreadable end

- holds in **17 of 17** swept cells (0 weaken, 0 reverse, 1 could not be built)
- reported as tuned: **no**
- peak locations seen across the sweep: 0.1

Confident invention peaks in the middle of the readability axis, not at the unreadable end holds in 17 of 17 swept cells and no swept parameter reverses or halves it.

### What was not swept, said rather than left blank

- **number of encounters:** belongs to the sequential designs (E29, E31), which are not among V-3's two swept headlines; V-1 covers E31 under the solver check and its own pre-registration covers the encounter count
- **interactions between parameters:** the sweep is axis-aligned. A full factorial is not affordable and diagonal knife edges are therefore unmeasured

---

## V-4: what is forced by construction?

For each finding, this names the one design decision the finding depends on, then breaks that decision on purpose and checks the finding goes away. A finding that survives having its own foundations removed was not a finding at all; it was built in. A third possibility also gets recorded: sometimes the model refuses to be broken that way, which means the decision is part of the definition rather than a setting, and that changes how strongly the finding can be stated.

| claim | said to depend on | alteration | what happened |
|---|---|---|---|
| The same machine-made content is read as certain or uncertain depending only on what the label says | the reader being able to read the provenance label at all (kappa > 0) | kappa set to 0, so the label carries no information | **result disappeared** |
| The same machine-made content is read as certain or uncertain depending only on what the label says | synthetic content being structured rather than high-entropy noise (N6) | the synthetic distribution replaced with a near-uniform one | **result survived** |
| Readers become confident AND disagree with each other, so the confidence is invention rather than shared error | the synthetic distribution being goal-SYMMETRIC (V1 deviation 2) | the goal-symmetrisation switched off, so the frozen synthetic draw leans toward one goal by chance | **result disappeared** |
| Confident invention peaks in the middle of the readability axis | foreign content being goal-directed rather than noise (C1 property 1) | the foreign family drawn near-uniform, with its goal anchor removed | **alteration unreachable** |
| Confident invention peaks in the middle of the readability axis | the human and foreign feature blocks being disjoint | the foreign family's support floor raised across the whole feature space | **alteration unreachable** |

**2 of 5 claims disappear when the property they were said to depend on is removed, which is the wanted outcome. the claim "The same machine-made content is read as certain or uncertain dependin" survived the removal of synthetic content being structured rather than high-entropy noise (N6), and that row misses its own target rather than refuting the claim; see its note_on_attribution, and the row directly below it, which was added once the reason was traced and which does remove the claim. Survivals are reported rather than dropped: a claim that outlives its own stated mechanism is either stronger than stated or explained by something else, and the honest position is to say which of those has not yet been established.**

> *foreign content being goal-directed rather than noise (C1 property 1)* could not be altered: the model asserts it at construction and refused. That makes it a definition rather than a setting, and the claim above it is downstream of a definition, which is a weaker thing than a claim downstream of a measurement, and is stated as such.

> *the human and foreign feature blocks being disjoint* could not be altered: the model asserts it at construction and refused. That makes it a definition rather than a setting, and the claim above it is downstream of a definition, which is a weaker thing than a claim downstream of a measurement, and is stated as such.

### The feature partition, audited on its own

**What it is.** the human and foreign feature blocks do not overlap. Adopted because no such partition existed at the original feature count, so the feature space was doubled from eight to sixteen

**Chosen on theoretical grounds:** no. the alternative was for foreign content to overlap the reader's own support, which is the V4 spec's own pre-mortem failure #1: foreign content becomes unidentifiable rather than foreign, and V4 reports V3's results in new vocabulary

**Everything downstream of it:**

- *Confident invention peaks in the middle of the readability axis, not at the unreadable end* (E20): the peak's LOCATION is a location on an axis whose zero point is 'no shared features at all', which only exists because the two blocks are disjoint
- *Attention stops being sustained below about 4% overlap* (E20): the crossing point is measured on the same axis
- *Content with real structure the reader cannot parse holds attention indefinitely* (E19): 'cannot parse' is implemented as 'lives on features every one of the reader's hypotheses puts at floor', which is the partition
- *The most generous fallback hypothesis absorbs exploratory human work and does nothing for machine work* (E19): the fallback is flat across the human block and at floor across the foreign one, which is a property of the partition and was the reason the space was doubled
- *Being out of your depth and reading something foreign are different failures* (E32): one arm degrades the reader's templates and the other moves the content off the block those templates cover
- *Real generated content sits somewhere on the overlap axis, and a human study could locate it* (E34): the prediction card's axis IS the partition; without it there is no axis to place real content on

**What a different partitioning would do.** A partially overlapping partition removes the axis's zero point: there would be no omega at which the reader's hypotheses are all at floor, so 'fully foreign' would stop being a location and the interior peak would have no interior to sit in. A partition with unequal block sizes would keep the axis but change where the peak falls on it, because the peak's location is set by how much in-family structure is needed to make an explanation seem available. In both cases the SHAPE of the claim, an interior maximum, is what would transfer, and the specific value would not. That is why the prediction card is written as a location to be measured rather than as a number to be trusted.

### The rebuilt effort parameter, audited on its own

**The position taken:** reported as a construction commitment rather than an emergent finding: the model was rebuilt so that 'offhand but deep' is representable, which makes the dissociation possible before it is measured.

**The narrower question that can still be asked:** with the effort axis pinned at its maximum, so no offhand corner exists, does depth still separate?

Measured separation with the effort axis pinned: **0.908**.

The depth-versus-effort null returns `BETA_CAN_MANUFACTURE_DEPTH` under exact inference.

Depth still separates with the effort axis pinned, so the depth estimator is reading structure in the artifact and not the effort knob renamed. The DISSOCIATION remains a construction commitment; the READABILITY of depth does not.

---

## V-5: every measurement rule that was changed after the fact

Eleven times across five versions, a measurement rule was changed or added after the fact. Each of those is already written down in the version's own write-up. This check goes back and computes the ORIGINAL rule wherever the data to do it is still on disk, and reports which conclusions would have been different. The cases where nothing changes are included on purpose: they are the ones that tell you the changes were not doing the work.

| deviation | what changed | kind | recomputed here | would a verdict change |
|---|---|---|---|---|
| V1-1 | c_effort default lowered from 0.5 to 0.1 | parameter calibration | no | — |
| V1-2 | the synthetic distribution goal-symmetrised | construction decision | no | — |
| V1-3 | structured_ceiling lowered from 2.6 to 1.8 | criterion, made non-vacuous | no | — |
| V3-2 | E13's power-law classifier declared undefined | criterion refused | yes | no |
| V3-4 | E16's primary moved from creator_mi to ghost_col_err | criterion, primary changed | no | — |
| V3-5 | E17's monotonicity criterion made tie-aware | criterion restated | yes | no |
| V4-2 | the engagement clause removed from E19's absorption criterion | criterion restated | yes | yes |
| V4.5-2 | two clauses added to E21's criteria before the run | criteria added | no | — |
| V4.5-3 | E28's beta = 0 check gained a second operationalisation after the run | criterion added | yes | no |
| V4.5-5 | E20 gained a strict fabrication index after the run | measure added | yes | no |
| V5-1 | N21's dominance clause restated | criterion restated | yes | yes |
| V5-2 | E30 gained a second update measure after the run | measure added | no | — |
| VAL-2a | V-2a's permutation clause restated during this pass | criterion restated | no | — |

**Of 6 superseded criteria recomputed from committed data, 2 would change a verdict: V4-2, V5-1. Each is reported with the original outcome attached, in the same place as the claim.**

**V1-1.** Not a criterion recomputation. Not a criterion. At 0.5 the effort gap exceeds the model's maximum epistemic value and every tier disengages at t=0, so E1 collapses to a null. V-3 sweeps this parameter across 0.05 to 0.75 and reports the result at every level, which is a stronger answer than recomputing one superseded default.

**V1-2.** Not a criterion recomputation. Not a criterion. V-2b measures the consequence directly: with the symmetrisation removed, random parameterisations produce consensus rather than disagreement, which is exactly what the deviation said at the time.

**V1-3.** Not a criterion recomputation. The original ceiling exceeded uniform entropy, so the assertion it guarded was vacuous, so it could not have failed. Recomputing a vacuous criterion returns 'passes' by construction. Recorded as a criterion that was strengthened rather than weakened.

**V3-2.** The original criterion returned an answer, and the answer was thrown away rather than published, because the criterion needed a precondition on the sign of the exponent and did not have one. This is the cleanest case in the table: a criterion that produced a usable-looking number and was refused.

**V3-4.** Not a criterion recomputation. Changed on a ground documented BEFORE the run. E7's own write-up records that the seeded learner starts at about 95% of oracle mutual information, leaving 5% of the range for a threshold to be resolved in. Both measures are in the committed CSV, so the change is auditable by anyone who wants to.

**V3-5.** The original criterion scores 1.00, and the worst violation of monotonicity is 2.30e-06 nats. The doubt does rise monotonically as transparency falls; what the original criterion punished was ties the construction guarantees. The verdict does not change and the restatement was defensible.

**V4-2.** Under the original criterion E19's POSITIVE CONTROL fails, at an engagement of 0.001, because a reader who has resolved the goal correctly stops paying attention. The decisive cell is untouched: foreign content fails absorption on mass, which is the primary clause and was never changed. So the restatement rescues the control and cannot reach the result.

**V4.5-2.** Not a criterion recomputation. Added before any cell ran, from inspection of the arm definitions, because without them two arms passed signatures for free while having no behaviour at all. The pre-registered primary was not changed and still decides the verdict.

**V4.5-3.** The check fails under both forms, so the addition changed the REASON reported and not the outcome. The pre-registered criterion is retained and still decides the flag. This is the informative case the spec asks to be published: a criterion changed after seeing data, where it made no difference.

**V4.5-5.** Both indices peak at the same overlap, so the added measure changes what the peak MEANS and not where it is. The pre-registered index still decides the outcome string.

**V5-1.** The original clause scores 2.377 against a required 3.0 and FAILS; the restated clause scores 5.984 and passes. So this deviation does decide a verdict, and the original is retained and reported as failing. What the original clause charged as contamination is a legitimate limitation: effort limits how much REAL depth is recoverable, because depth is defined relative to a goal's mode family and a plan cannot be read without partly knowing the goal. The threshold value was not moved, only the quantity it scores.
 *RESULTS_V5.md records the original clause at 1.701 and this recomputation gives a different number. The difference is the averaging: the value in the write-up was produced by the code that existed at the time, and this one is recomputed from the committed cell table with the averaging stated above. Both are well below the required factor and both FAIL, so the conclusion is the same one. But the two numbers are not the same number and it would be wrong to present this as a reproduction of that one.*

**V5-2.** Not a criterion recomputation. The pre-registered measure has no headroom in this design by construction, since depth is built so the goal is exactly as recoverable at every level. The added measure agrees with the pre-registered one on the outcome and disagrees on the reason. All fifteen pre-existing columns came back bit-identical on the re-run.

**VAL-2a.** Not a criterion recomputation. Logged in this pass's own V-2 verdict file rather than here, so it sits with the check that produced it. Named in this table because a validation pass that logs deviations everywhere except in itself is not a validation pass.

---

## V-6: do the versions agree with each other?

Each version of the model has to be able to become the previous one when you turn its new machinery off. Those reductions are already tests; this runs them and, more usefully, finds every place where two different versions measured the same real quantity and puts the two numbers side by side.

Boundary reductions are the checks that each version can become the previous one when its new machinery is switched off. They **all hold**.

| quantity measured twice | in | values | agree |
|---|---|---|---|
| sustained attention on fully foreign content | E19 (V4), E20 (V4.5) | 0.724 vs 0.626 | yes |
| within-reader doubt on GHOST content under a false human label, in nats | E2 (V1), E17 (V3) | 0.09 vs 0.089 | yes |

*sustained attention on fully foreign content:* the nominally identical condition. E20 ran at three times the seeds, which its own deviation V4.5-6 declares, so a gap here is a measurement-precision question rather than a contradiction. It is still the same number twice and it should be quoted as a range

*within-reader doubt on GHOST content under a false human label, in nats:* E17 is E2's dose-response follow-up on the same geometry, so these should be close; a gap would mean one of the two is measuring something else

**Every boundary reduction holds and every quantity measured by two versions agrees within its own spread.**

---

## V-7: a different set of random seeds, and twice the readers

Random draws can flatter a result, and a result measured on too few readers can look solid and then move when you measure it properly. This re-runs the headlines on a completely separate set of random seeds and again at double the number of readers, and reports how far the numbers moved rather than only whether the conclusion survived.

| result | condition | reference | re-measured | moved by | conclusion holds |
|---|---|---|---|---|---|
| label effect | other seed block | 14.86 | 14.419 | 3.0% | yes |
| label effect | double scale | 14.86 | 14.376 | 3.3% | yes |
| interior peak | other seed block | 0.308 | 0.285 | 7.4% | yes |
| interior peak | double scale | 0.308 | 0.314 | 2.1% | yes |
| interior peak | peak location across arms | 0.1 | 0.1 | 0.0% | yes |

**Every headline holds on a disjoint seed block and at double scale, with effect sizes stable to within a quarter of themselves.**

---

## V-8: rebuilt from scratch, from the description alone

The single most legible finding in the project was rewritten from scratch, using only the plain-English description of it and none of the original code, settings or random seeds. If a finding only exists in the code that produced it, it is not a finding about anything. This is the check that tells them apart.

*scripts/independent_two_gates.py imports numpy and the standard library only. It does not import ghostscale, does not read results/, and every parameter in it is declared at the top of the file.*

| | original | independent rebuild |
|---|---|---|
| how much further the reader moves under a false label | 21.8x | 1.4x |
| uptake tracks the reader's own depth estimate | 0.89 | 1 |

**The MECHANISM replicates and the MAGNITUDE does not. In separate code written from the prose alone, a false label still inflates the reader's estimate of how much thinking went into machine-made work, and how far the reader moves still tracks that estimate whichever channel produced it, and the rank correlation across all six conditions is 1.00. But the uptake multiple is 1.4x against the original's 21.8x, a factor of 15 apart and outside the order of magnitude committed before the reimplementation was written.

**The honest reading, and it is the more useful one.** The 22-fold figure is a property of this model's particular dimensions and is not the claim. The claim that survives independent construction is directional: a false provenance label makes a reader take on substantially more from machine-made work than an honest one does, through an inflated estimate of the thinking behind it. Every public-facing use of the multiple has to be stated that way, as a direction with a model-specific size, and the specific number must not be quoted as though it transfers.**

> the reader here also cannot doubt the label it conditioned on, so this multiple is an upper bound for the same reason the original's is. That was not a design choice. It is what the description says the reader does, and reimplementing the description reproduces the limitation along with the result

---

## V-9: one prediction, written down before the experiment exists

Everything in this repository was predicted by people who already knew the theory, and the literature check happened afterwards. That means nothing here is a forward test. This is one: a complete prediction covering which way, how big, and four named ways it could fail, for an experiment that does not exist yet, written down and sealed with a hash before anybody builds it.

**The reader equipped with a hypothesis for what the maker was AVOIDING**

*PRE-REGISTERED. The experiment is not built. Nothing has been run.*

**Why this one.** It is the natural next experiment regardless of validation, and it is the only candidate where the theory makes a prediction that does not follow from anything already measured. Every existing hypothesis space in this project is a space of things a maker might have been TRYING TO DO. Avoidance is a different shape of intention: the maker's purpose is specified by what is absent rather than by what is present, which means the evidence for it is a hole in the distribution rather than a peak in it.

**The setup.** The maker holds an avoidance goal: some region of the feature space it will not enter. Otherwise it behaves exactly as an existing maker does. Two readers see its work. The FIRST holds only the existing hypothesis space, meaning purposes as things being pursued. The SECOND additionally holds avoidance hypotheses, one per region, in the same likelihood family. Both are otherwise identical, including their priors and their random stream.

**The prediction.**

- **primary outcome:** recovered intent on avoidance-driven work, reader two against reader one
- **direction:** reader two recovers the maker's actual constraint and reader one does not, so the accuracy gap is positive
- **magnitude:** reader two's accuracy on avoidance-driven work exceeds reader one's by at least 0.30 in absolute terms, on a four-alternative task where chance is 0.25. Stated as an absolute gap rather than a ratio because reader one is predicted to be near chance and a ratio against chance is unstable.
- **secondary outcome and it is the interesting one:** reader one is predicted to be CONFIDENT and WRONG rather than uncertain: within-reader doubt below 0.5 nats with accuracy at or below chance. Avoidance leaves a hole, the existing hypotheses all have support inside that hole, and the one whose peak sits furthest from it wins by default. That is the same failure mode as the partial-overlap peak, arriving by a different route, and if it holds it is the second independent instance of one mechanism.
- **cost prediction:** reader two pays MORE attention, not less, because an avoidance hypothesis is confirmed by continued absence and absence accumulates slowly. Predicted at least 1.5x the deep looks of reader one.

**Named ways it can fail.** All four written before anything was built.

- **NO_GAP:** reader two does no better. Avoidance is not recoverable from a hole in this geometry, and the framework's claim that intention can be read from what is absent is unsupported in simulation. This is the outcome that costs the most and it is a real possibility: the hole may simply be too weak a signal at this feature count, in which case the result is about the geometry and has to be reported as inconclusive rather than negative.
- **GAP_BUT_READER_ONE_IS_UNCERTAIN:** reader two wins and reader one is appropriately unsure rather than confidently wrong. The primary prediction holds and the interesting secondary one fails, which would separate the two mechanisms this prediction claims are the same.
- **GAP_REVERSES:** reader one does BETTER. The avoidance hypotheses act as an absorbing fallback the way a uniform EXPLORE signature would, and the extra hypotheses hurt. This would be a direct hit on the framework and would be reported as one.
- **READER_TWO_PAYS_LESS:** the cost prediction inverts. An avoidance hypothesis resolves faster than a pursuit hypothesis, which would contradict the metabolic account the whole engagement story rests on.

**What is and is not committed.** COMMITTED: the direction, the 0.30 absolute accuracy gap, the 0.5 nat confidence threshold on reader one, the 1.5x attention ratio, and the four branches above. NOT COMMITTED: the feature count, the number of avoidance regions, the observer count and the seeds, because those are scale decisions and fixing them now without having built anything would be pre-registering guesses as though they were design.

**How it gets scored.** The experiment is built and run once. Its outcome is matched against the branches above by their stated thresholds, and the branch that fires is reported whatever it is. If none fires cleanly, that is recorded as the prediction having been under-specified, which is itself a result about how well this framework can be made to commit in advance.

Locked at `c9d8c782ed4c2ccb`; the lock is intact.

---

## What this pass changed about what may be claimed

1. At least one verdict is a property of the inference shortcut rather than of the model. Those claims carry the failure in the same cell as the claim.
2. Confident belief under a false provenance label is reported as a property of this architecture, because a randomly parameterised model of the same shape produces it too. What the theory keeps is the disagreement.
3. The apparatus's own false-positive rate is 64.0%, and borderline findings are read against that rather than against zero.
4. These deviations do decide a verdict and carry the original outcome beside the claim: V4-2, V5-1.
5. The size of the label effect may not be quoted as though it transfers outside this model. The direction may.

---

## What this pass does not do

- It does not make the work confirmatory-free. Every prediction still came from one prior theory, and no amount of internal checking changes that. The checks bound how much of the agreement is the theory and how much is the apparatus; they do not convert one into the other.
- It does not test anything against people. There is no human data anywhere in this repository and nothing here is evidence about what real readers do.
- It leaves the withheld experiment withheld, its failing test in the suite, and the open residual open.
- The out-of-sample prediction in V-9 is written and not yet built. Until it is, the project has no forward test, and that is the largest single thing it owes.

*Generated from results/validation/ on 2026-07-29 by `scripts/write_validation_md.py`. Every number above is read out of a verdict file; none is typed in.*

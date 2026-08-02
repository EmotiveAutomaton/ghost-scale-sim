# Version 6: the Intent Extraction Limit, implemented

Generated from the verdict files in [results/v6/](../../../results/v6/). Regenerate with `python scripts/write_results_v6.py`. Criteria hash-locked at `d6c5bf4b5729e891` before any cell ran.

---

## What this version was for

Every previous version asked a new question about the world. **This one asks whether the simulation and the theory it claims to implement are the same object.**

Three audit passes had already checked the results. None of them could have found this, because all three took the code's own account of itself as given. Reading the preprint's formal model against the shipped code found **three terms in the equation with no counterpart in the code**, and one of them is not an omission but a disagreement about mechanism.

| term of the Intent Extraction Limit | in V1-V5? |
|---|---|
| the trust amplifier | yes |
| belief movement | yes |
| the value-divergence gate | yes |
| **metabolic reserve** | **no. No depletion state existed anywhere.** |
| **the graded gate** | **no. Replaced by a binary decision.** |
| **trust suppressing the disgust threshold** | **no, and it is a different mechanism from the one the code uses** |

---

## The pass at a glance

| experiment | what it asks | how it came back |
|---|---|---|
| **E35** | Does the damage accumulate in the reader and carry to work it has never seen? | DEPLETION_CARRIES_TO_UNSEEN_HUMAN_WORK *(direction holds on 3 of 3 seed blocks; the pre-registered magnitude threshold on 1 of 3 — see the robustness section)* |
| **E36** | Does the reader recover the maker's method, and does the goal unlock it? | DEPTH_MOVES_PROCESS_UPTAKE / RESOLVING_THE_GOAL_UNLOCKS_THE_PROCESS |
| **E37** | Is the wall a vocabulary deficit or a missing inversion? | THE_WALL_IS_A_DISTINCT_FAILURE |
| **E38** | Does AI literacy stack with art literacy, or replace it? | EXPERTISE_SUBSTITUTES |
| **E39** | Does a reader that can conclude 'no maker' stop cleanly? | TOOL_HYPOTHESIS_REDIRECTS_RATHER_THAN_RELAXES |
| **E40** | How do appeal and endorsement combine, and what if appeal is optimised? | ADDITIVE_FITS / A_THIRD_FAILURE_MODE_OVERENGAGEMENT |
| **E41** | Two mechanisms for the trust exploit. Do they predict the same thing? | COUPLING_PREDICTS_AN_EXPLOIT_THE_RACE_CANNOT |
| **E42** | Is looking deeply the same as being willing to be changed? | ENGAGEMENT_AND_INTEGRATION_DISSOCIATE |
| **E43** | Does the maker lose access to its own reasons as the work deepens? | AUTOMATICITY_HIDES_THE_GOAL_FROM_ITS_OWN_AUTHOR |

---

## E41: the two trust mechanisms disagree, and the code was missing one

The paper says a trusting reader is exploited because trust switches off the alarm that would normally stop it absorbing something. The code says a trusting reader is exploited because the label out-argues the work. Those sound alike and they are not: the first should still happen when the reader is told the TRUTH, and the second cannot. That is the whole experiment.

**Outcome: COUPLING_PREDICTS_AN_EXPLOIT_THE_RACE_CANNOT.** Largest gap in integration on the discriminating cell: **0.440** against a pre-registered 0.2.

| trust | integration, coupled gate | integration, channel race only |
|---|---|---|
| 0.200 | 0.105 | 0.060 |
| 0.400 | 0.178 | 0.060 |
| 0.538 | 0.249 | 0.060 |
| 0.700 | 0.352 | 0.060 |
| 0.900 | 0.500 | 0.060 |

**What this means, stated plainly.** Under the code's mechanism a trusting reader is exploited because the label out-argues the work, so it is *wrong about who made the thing*. Under the preprint's mechanism trust suppresses the threshold that would otherwise refuse the material, so the reader integrates **even when it is told the truth and believes it**. The second is not reachable from the first, and the simulation had only ever implemented the first.

*The values layer rides along: 2 values over 4 goals, non-injective as null N26 requires (True). two different goals can imply the same values, so the gate opens for both. That is 'I disagree with what you were doing but we want the same things', which the code could not previously represent.*

---

## E35: depletion, and the apathy the model could not previously represent

The preprint says the damage accumulates in the reader and ends in apathy. The simulation had no way to represent that at all: every reader arrived fresh. This gives the reader an energy budget that falls when it looks hard and gets nothing, and then measures its engagement with a FIXED human artifact that never changes.

**Outcome: DEPLETION_CARRIES_TO_UNSEEN_HUMAN_WORK.** Engagement with a fixed human artifact the reader has never seen falls by **0.132** across the exposure sequence, monotonically (rank correlation -0.769).

| exposure stream | probe engagement, first | probe engagement, last | reserve, last |
|---|---|---|---|
| intent_empty | 0.142 | 0.009 | 0.499 |
| human | 0.155 | 0.133 | 1.000 |
| mixed | 0.155 | 0.091 | 0.823 |

**Null N22 holds** (True): on a fully resolvable corpus the reserve does not move, 1.000 to 1.000. this is the null the generational experiment never passed. A depletion mechanism that drifts on content it should not touch would produce this experiment's headline for free.

**The design point that decides whether this means anything:** the criterion is the *probe*, which is identical in every arm and at every point in the sequence. A depletion term that only lowered engagement on the content that caused it would be a knob doing what it was pointed at.

---

## E36: what the reader was recovering all along, and why the depth result was null

Every measure of what a reader takes on, in every version of this project, has scored how much of the maker's PURPOSE it got. Depth is built so the purpose is equally readable however deep the work is, so that measure could not move with depth whatever was true. This scores what the reader got of the maker's METHOD, which the reader has been quietly tracking all along and nobody ever read out.

**On the maker's method, depth moves uptake: 0.179 [0.099, 0.267]. On the maker's purpose, measured on the same cells, it does not: -0.028 [-0.116, 0.058].**

That is the whole argument. Depth is *constructed* so the goal is equally recoverable at every level, so the measure every previous version used could not have moved with depth whatever was true of the reader. The experiment was not wrong; it was pointed at the wrong quantity.

| depth | goal accuracy | goal uptake | process recovery | process uptake |
|---|---|---|---|---|
| 1 | 1.000 | 1.366 | 0.265 | -0.034 |
| 2 | 1.000 | 1.382 | 0.388 | 0.202 |
| 3 | 1.000 | 1.338 | 0.435 | 0.145 |

**Null N28 holds** (True): at the shallowest depth there is no process, and recovery carries -0.022 nats of information.

> *at mu = 1 the sub-goal posterior never leaves its uniform prior, so its argmax is a tie broken to index zero, while the true mode is autocorrelated because the chain dwells. Accuracy then measures how often the artifact happened to sit in mode zero and came out BELOW nominal chance, which no amount of information could produce. The information measure is exactly zero on an unmoved posterior whatever the truth was. Declared rather than quietly substituted.*

### The ordering claim: intent as the key that unlocks the method

**The pre-registered test fails and is reported as failing.** Comparing readers who ended up right about the goal against readers who ended up wrong gives a gap of 0.047 against a required 0.15: *PROCESS_IS_INDEPENDENT_OF_GOAL*.

**The temporal test, added afterwards and declared, holds.** Within a single reading, process recovery before the goal settles is 0.050 and after it is 0.130 — a gain of 0.080, interval [0.041, 0.122], over 102 readings: *RESOLVING_THE_GOAL_UNLOCKS_THE_PROCESS*.

The two are not in conflict and the difference is the point. The pre-registered form is a **between-reader** contrast; the claim is a **within-reader, temporal** one. Only the second is what *once you know what someone was for, you can read their actions as being in service of it* actually says. The original criterion is retained and decides nothing.

**Scale invariance** (EXTRACTION_IS_SCALE_INVARIANT): recovery on a quarter-length window scores 0.415 against 0.411 on the whole artifact. The extraction is not tied to the artifact boundary, which is as much of the fractal claim as can be tested without recursion.

---

## E37: legible and empty

The model has been describing unreadable machine content as content written in a vocabulary the reader lacks. The objection is that this is not what it feels like: you can read every word and there is still nothing behind it. This builds content on FAMILIAR features whose maker cannot be inverted from the surface, and asks whether that is a different failure from unfamiliar content.

| content | goal accuracy | uncertainty left | still looking | uptake |
|---|---|---|---|---|
| human | 1.000 | 0.000312 | 0.169 | 1.397 |
| foreign | 0.200 | 1.185 | 0.002 | -0.290 |
| noninvertible | 0.383 | 0.338 | 0.042 | -1.099 |

**Outcome: THE_WALL_IS_A_DISTINCT_FAILURE**, separation 0.886 against a required 0.3. Legible-and-empty: True.

---

## E38: expertise substitutes rather than stacks

If reading generated work needs a different skill rather than more of the same skill, then people who know how these systems work should be able to read them. The question worth asking is what that costs: does the new expertise sit alongside the old one, or eat it?

| reader | content | goal accuracy | uptake | still looking |
|---|---|---|---|---|
| human | human | 1.000 | 1.370 | 0.113 |
| human | machine | 0.320 | -0.288 | 0.760 |
| machine | human | 0.280 | -0.579 | 0 |
| machine | machine | 1.000 | 1.370 | 0.057 |

**Outcome: EXPERTISE_SUBSTITUTES.** The machine-matched reader gains 0.680 on machine content and gives up 0.720 on human content.

---

## E39: the tool hypothesis

Until now the reader could only FAIL to find a maker. It had no way to decide that there was not one. That is what the Ghost Scale is meant to give it -- not distrust of a label, but permission to stop. This gives the reader that hypothesis and asks whether stopping looks different from failing.

| arm | uncertainty left | still looking | invention | uptake |
|---|---|---|---|---|
| no_tool_hypothesis | 1.195 | 0.002 | 0.137 | -0.244 |
| with_tool_hypothesis | 1.407 | 0.010 | 0.085 | -0.405 |
| human_control | 0.041 | 0.170 | 0.833 | 1.618 |

**Outcome: TOOL_HYPOTHESIS_REDIRECTS_RATHER_THAN_RELAXES.** Resolved: False; disengaged: True; not inventing: True.

**Null N27 holds** (True): the extra hypothesis does not absorb human work, which still reads at 1.000. That is the failure version 4 caught with its own fallback hypothesis, checked the same way.

---

## E40: the honeypot, the crowd, and what happens when the honeypot is optimised

The reader in this model decides to look closely on one basis: how much it expects to learn. People do not work that way. Something catches your eye, and someone tells you it is worth your time, and both of those happen before you have learned anything at all. This adds those two channels, and then asks what happens when the eye-catching one is optimised on purpose.

**How the cues combine: ADDITIVE_FITS** (pre-registered: additive). At the corner where the content offers nothing and a cue is present, the additive rule lifts engagement by 0.350 and the multiplicative rule by 0.003.

**The decoupling: A_THIRD_FAILURE_MODE_OVERENGAGEMENT.** With the surface cue maximised on content that has no depth, engagement rises 0.350 above the honest baseline while uptake is -0.281. Pays more, gets less: True.

*not the crash, because the reader is engaged rather than disengaged; not the trust exploit, because no label has lied and the reader is not wrong about provenance. It is a reader correctly reading an artifact shaped to trip its own heuristic for deciding what is worth reading.*

**Null N29 holds** (True): goal accuracy varies by 0 across cue corners, so the cues carry no goal information.

---

## E42: engagement is not integration

I had claimed that choosing to look closely IS the willingness to be vulnerable. The counterexample is a domain where the person doing the deep looking is exactly the one not being changed by it. This model already keeps those two things apart -- attention is one object and permission to integrate is another -- and nobody had ever reported them separately. This does.

| cell | still looking | integration | value divergence | uptake |
|---|---|---|---|---|
| resolving_aligned | 0.111 | 0.994 | 2.17e-07 | 1.410 |
| resolving_divergent | 0.111 | 2.4e-16 | 9.547 | 1.410 |
| sustained_aligned | 0.801 | 0.136 | -2e-12 | -0.196 |
| sustained_divergent | 0.801 | 0.049 | 0.253 | -0.196 |

**Outcome: ENGAGEMENT_AND_INTEGRATION_DISSOCIATE.** 1 of 4 cells combine high engagement with a closed gate: ['sustained_divergent'].

**What this changes.** Willingness to be vulnerable is the **gate**, not the decision to look. A reader can look intently, read the maker accurately, and integrate nothing. The two are driven by different terms and had never been reported apart.

---

## E43: automaticity hides the work from its own author

A novice can tell you exactly which rule they were following, because they are still following it on purpose. A master cannot tell you why the perspective works. The claim is that this is not a fact about personalities but about compression: practice is what makes a decision automatic, and automatic is what makes it unavailable for report.

| depth | the maker's own account | the reader's reading | process recovery |
|---|---|---|---|
| 1 | 0.980 | 1.000 | 0.272 |
| 2 | 0.760 | 1.000 | 0.374 |
| 3 | 0.680 | 1.000 | 0.467 |

**Outcome: AUTOMATICITY_HIDES_THE_GOAL_FROM_ITS_OWN_AUTHOR.** The maker's own account falls by 0.300 across the depth range while the reader moves 0.

*E33 already showed a reader recovering a maker's latent goal while the maker's declared goal collapsed. But there the self-blindness is a MANIPULATED PARAMETER. Here nobody sets it: depth sets it, which is a stronger and different claim.*

---

## Does any of this survive different random seeds?

**17 of 18 outcomes are IDENTICAL across a disjoint seed block. The one that moves is E35.**

*a whole-programme seed offset re-randomises every reader in every cell at once and changes nothing else. The V6 analogue of the validation pass's disjoint seed block.*

| seed block | absolute drop | relative drop | fold reduction | monotonicity | reserve, exposed | reserve, control | verdict |
|---|---|---|---|---|---|---|---|
| 0 | 0.132 | 0.933 | 15.000 | -0.769 | 0.499 | 1.000 | DEPLETION_CARRIES_TO_UNSEEN_HUMAN_WORK |
| 777000 | 0.069 | 0.968 | 31.000 | -0.811 | 0.459 | 1.000 | DEPLETION_DOES_NOT_CARRY |
| 424242 | 0.058 | 0.781 | 4.600 | -0.811 | 0.440 | 1.000 | DEPLETION_DOES_NOT_CARRY |

THE MECHANISM IS STABLE AND THE CRITERION IS NOT. Across three seed blocks the reserve depletes to 0.44-0.50 in the exposed arm and stays at exactly 1.00 in the control, and probe engagement falls monotonically every time (-0.77, -0.81, -0.81, all past the pre-registered -0.70). What moves is the pre-registered ABSOLUTE drop of 0.10, which passes on one block of three -- because baseline probe engagement itself differs about two-fold between blocks, and an absolute threshold cannot be stable on a quantity whose baseline moves that much. The relative drop is 0.78 to 0.97 on every block.

This is the fourth time in this project a criterion has been found unable to do its job, and it is reported rather than repaired: the pre-registered clause still decides, and E35 is reported as PASSING ON ONE SEED BLOCK OF THREE.

**What may be claimed:** that depletion carries to unseen work is supported in DIRECTION and in MECHANISM on every seed block tested, and its pre-registered magnitude threshold is not met on two of three. The direction is the claim. The magnitude is not.

---

## What this version changed about what may be claimed

1. **The simulation was missing a term the theory has.** Metabolic reserve is in the equation with its own symbol and had no counterpart in the code, so the framework's central cultural claim was not untested — it was unrepresentable.
2. **The preprint and the code explain the trust exploit differently, and the difference is testable.** The coupled gate predicts an exploit on a reader that is told the truth and believes it. The channel race cannot produce that.
3. **The depth result was measuring the wrong quantity.** Depth moves what the reader takes of the maker's *method* and provably cannot move what it takes of the maker's *purpose*, because the construction holds the second constant.
4. **Looking deeply and being willing to be changed are separate**, and the model already kept them apart. Vulnerability is the gate.
5. **A cue learned as a predictor of depth, then optimised directly, produces a third failure mode**: the reader pays more and gets less. Not the crash, not the exploit.

## What was deliberately not built

- no creator agent, so Zahavian signalling and the whole reputational-cost security argument remain UNTESTED IN SIMULATION. Named as an open hole, not discovered later.
- no recursive/fractal hierarchy; the scaled version (sub-window recovery) is built instead
- no affective/mirror channel
- no creator-side cognitive-surrender dynamics

*Generated from results/v6/ by `scripts/write_results_v6.py`. Every number above is read out of a verdict file; none is typed in.*

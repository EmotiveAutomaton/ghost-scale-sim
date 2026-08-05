# Second batch back from the Ghost Scale Simulation

**Returned 2026-08-05.** Five results: T-1 through T-4 as asked, plus **T-5, which the simulation
added itself** because T-1's answer implied a question nobody had written down.

Read in one line: **the triangle is not a triangle, `purpose_breadth` does not measure what we said
it measures, and decision-counting is alive in a regime we can name.** Two of those hurt.

---

## T-1 · The triangle — **it is not a triangle. It is a river.**

### First, the part where it refused the question

`FOR_GHOST_SCALE_SIM_2.md` said: *"Please do not invent a values factor and then measure it."* It
didn't. It measured whether one exists:

    H(values | goal)                              = 0.000 nats
    information values adds over goal             = 0.000 nats
    capacity of the values vertex                 = 0.693 nats
    capacity of the goal vertex                   = 1.386 nats

    verdict: VALUES_IS_A_DETERMINISTIC_COARSENING_OF_GOAL

**The values vertex is the goal vertex with less resolution.** Which means four of the six edges were
never measurable in this model — `goal→values` returns 1.0 by construction, `values→goal` returns
exactly the 1/n coarsening, and the two process edges carry only what routes through the goal.
**Those were properties of a matrix, not findings.**

It substituted **depth** as the third vertex and labelled the substitution at the top of the file.
So what follows is `goal – process – depth`, and the values question is **still open** — it needs a
model that does not exist yet.

> This matters directly for `docs/theory/THE_TRIANGLE.md`, where the curator flagged his own open
> question: *"I don't know if drives are values."* The simulation cannot answer it, and now says so
> in a way that names what would have to be built first.

### The six edges

Interventional, not observational — information supplied at one vertex at a matched **1 nat**, and
recovery measured at another, against a paired control. 200 per arm. Placebo deviations ~1e-16.

| supply → measure | baseline | with prior | **gain** | |
|---|---|---|---|---|
| **process → depth** | 0.189 | 1.029 | **+0.840** | ★ the strong edge |
| **depth → process** | 0.182 | 0.538 | **+0.356** | |
| goal → process | 0.182 | 0.183 | +0.0017 | at noise |
| goal → depth | 0.189 | 0.189 | ~0 | **dead** |
| process → goal | 1.397 | 1.397 | ~0 | **dead** |
| depth → goal | 1.397 | 1.397 | ~0 | **dead** |

Three dead edges. And the reason the goal edges are dead is the finding:

    goal_control_accuracy = 1.000     goal_at_ceiling = true

**The goal was already fully recovered from the artifact alone.** Nothing can improve it because
there is nothing left. Drop the reader's rationality (`beta` 1.0 → 0.25 → 0.1) and goal accuracy
falls 1.00 → 0.83 → 0.50, and only then do the goal edges start to move.

### Both halves of the curator's prediction held

Recorded in `FOR_GHOST_SCALE_SIM_2.md` before the run, and they were separable:

| predicted | result |
|---|---|
| **goal is easiest to recover** | **held** — at ceiling from the artifact alone, accuracy 1.000 |
| **process is most useful when supplied** | **held** — +0.840 and +0.356, the only two live edges |

> **There is absolutely not going to be symmetry.** — held, and not marginally. `process→depth` is
> **2.4×** `depth→process`, and three of six edges are exactly zero.

### But the bootstrapping claim did *not* hold

Supplying two vertices was tested against the sum of supplying each alone. Excesses: **+0.008,
+0.003, −0.0004, +0.010, −0.006, +0.032, −0.032.** Mixed sign, all tiny, none outside noise.

**The edges are additive, not superadditive.** "Each one bootstraps the others" is not what this
shows. What it shows is a **directed flow with a source and a sink** — process feeds depth and
itself; goal receives and returns nothing.

**Rename it.** The structure is not a triangle with mutual reinforcement. It is closer to a river:
**process is upstream, goal is downstream, and information does not flow back up.**

---

## T-5 · The question the simulation asked itself — and the answer is *don't rebuild*

> *"T-1 found process is a source and goal is a sink. Every instrument this project and Sounding Line
> have built scores the goal. This asks whether that is the wrong variable to point at."*

That is the correct next question and I did not ask it. It scored process-side against goal-side
statistics as maker-detectors — Mann-Whitney AUC, threshold-free, bootstrap intervals, permutation
nulls, **and every feature computed from the reader's own posteriors only**, no ground truth
anywhere (explicitly avoiding the flaw T-4 found in S-3).

| negative class | contested cells | process wins | **median process − goal** |
|---|---|---|---|
| foreign | 14 | 6 | **−0.0024** |
| ghost | 14 | 8 | **+0.0152** |

**A tie.** Their own falsification condition: *"goal-side statistics matching or beating process-side
ones. Then T-1's asymmetry would have no consequence for what an instrument should measure."*

That is exactly what happened.

> **T-1's asymmetry is real about the reader's internals and has no consequence for instrument
> design.** Do not rebuild the probe around process. The sink is as readable as the source.

This is the most valuable result in the batch, because without it T-1 would have triggered a
rewrite of the whole probe on a false inference. **The simulation talked us out of a week of work.**

---

## T-2 · `purpose_breadth` — **it measures difficulty, not variety**

This one costs us a survivor, and it comes with a second casualty.

### First casualty: S-2 is retracted

> *"S-2's emitter does not work and this module does not use it. `V5Environment.sample_feature`
> ignores `artifact.goal` once a creator is bound, so S-2's per-position mixture is drawn and
> discarded and its feature streams are **bit-identical with the mixture switched off**."*

There is an audit script (`scripts/audit_s2_mixture.py`). **S-2's −0.108 "validated at matched
density" was measuring nothing** — the manipulation never reached the reader. Every place this
project cited S-2 as `purpose_breadth`'s validation is now unsupported.

### Second casualty: the construct itself

The mixture axis behaves as advertised — breadth rises 0.509 → 0.726 as automaticity goes 0 → 1
(+0.217, interval excludes zero). **But goal accuracy collapses with it, 0.82 → 0.215.** That is the
exact trap the pre-registration named: *"if goal recovery falls with automaticity, 'diversity' has
quietly become 'noise'."*

Then axis 4, labelled DECISIVE, settles it. Build a pure-difficulty axis with **no diversity at all**
and sweep observation noise:

| α | goal accuracy | purpose_breadth |
|---|---|---|
| 0.15 | 0.425 | 0.793 |
| 0.45 | 0.690 | 0.619 |
| 0.85 | 0.965 | 0.272 |

Breadth tracks difficulty perfectly with nothing diverse anywhere in the world. Then compare
diverse-drive work against **equally hard single-drive work**:

    excess breadth attributable to diversity:   −0.013,  −0.025,  −0.021
    in_range_of_noise_curve:                     true,    true,    true
    construct_is_confounded_with_difficulty:     TRUE

**Zero, or slightly negative.** Once difficulty is matched, motivational variety contributes nothing
to posterior breadth.

> **`purpose_breadth` is a difficulty statistic.** It goes up when the goal is hard to recover, for
> any reason. It does not measure "soul as variety of motivations," and the number it produced on
> real text has been telling us how noisy the artifact is.

**Down to two survivors: function words (ceiling = author ID) and the affect directions.** And
§3 of `docs/theory/CONTROLS.md` just reopened the layer ratio, so call it two and a half.

**The curator's mechanism is not refuted** — practice → automaticity → drives → variety is untouched,
because the simulation says plainly it cannot test causation there. What is refuted is **our
instrument for it.** The idea needs a different measure, and one that survives a difficulty control.

---

## T-3 · Decision-counting — **positive, against the curator's prior**

> *"I suspect T-3 is going to come back negative."*

It came back **positive with a named regime**, which is the more useful outcome.

`count_is_well_defined_somewhere: true`. Fraction of rollouts where the sub-goal posterior
concentrates enough that "a decision was recovered" is a well-defined event:

| threshold (frac of max entropy) | baseline | best case |
|---|---|---|
| 0.75 | 0.56 | **1.00** |
| 0.60 | 0.19 | **0.98** |
| 0.50 | 0.02 | **0.51** |
| 0.35 | 0.00 | 0.18 |

### And it names the parameter that gets you there

Two things were swept. They do not contribute equally:

    artifact length   12 → 192 steps      mean entropy  −0.120
    mode dwell        2 → unbounded       mean entropy  −0.232      ← 2x

> **Concentration is governed by how long a maker stays in one mode, relative to how informative each
> emission is.** Not by how long the artifact is.

They also marked their own prediction wrong rather than quietly dropping it: *"the prediction that
length is inert was too strong and is reported as wrong."* Length does help; dwell helps twice as
much.

**What this hands Sounding Line is a corpus criterion.** Decision-counting is not dead in principle —
it is dead on artifacts whose makers switch sub-goals quickly. It should work on artifacts where one
person holds a single sub-goal for long stretches. **That is a describable kind of document**
(a sustained argument, a single-session draft, a technical postmortem) and it is nothing like the
Gate 3 corpus, which is web pages that change purpose every paragraph.

The curator's follow-on — *"diversity of motivations is probably more easily extractable than
decision density"* — is now **inverted by the pair of results.** T-2 killed the diversity measure;
T-3 kept density alive in a regime. The ranking flipped.

---

## T-4 · The leaked layer — **robust to ignorance, fragile to subtlety**

Reproduced S-3 first (leak accuracy 0.8990 vs 0.9015 — same construction).

### It found a flaw in S-3 and reported it

> *"S-3 sets its detector threshold to the median of the pooled concealing and candid divergences AT
> THE AMPLIFICATION BEING SCORED. That is fitted on ground-truth-labelled test data and re-fitted per
> cell, **which no reader can do**."*

Freeze the threshold at the lowest amplification and re-score:

    accuracy rise with amplification, fitted threshold:   +0.125
    accuracy rise with amplification, frozen threshold:   +0.046

**The curator's counter-intuitive claim survives** — amplifying the shield still makes concealment
more detectable — but it is roughly a third of the advertised size. `the_shield_still_gives_them_away_under_a_frozen_threshold: true`.

### Where it breaks, and it is not where I expected

| degradation | survives? |
|---|---|
| prior noise 0.2 / 0.4 / 0.6 / 0.8 | yes (bal. acc. 0.894 / 0.884 / 0.851 / 0.772) |
| prior noise 0.95 | **no** (0.546) |
| **channel swap, even p = 0.5** | **yes** (0.891) — reader need not know which channel is which |
| wrong cardinality (2 states vs true) | yes, degraded (0.660) |

**The reader can be almost completely wrong about the emitter and divergence still works.** That is a
much better transport story than I expected, and it directly answers why the question was asked: a
language model has no correct prior over a human's emission channels, and this says that may not
matter.

### The real fragility is partial concealment

| concealment | amp 1.0 | amp 2.0 | amp 4.0 | amp 8.0 |
|---|---|---|---|---|
| **25%** | 0.747 ✓ | −0.071 | 0.001 | 0.072 |
| 50% | 4.974 ✓ | ... | | |

**At 25% concealment the signal is at chance for every amplification above 1.** Divergence detects
makers who conceal *a lot*. Mild concealment — which is what nearly every real artifact has — is
invisible to it.

> The leaked-layer programme transports off the simulator. It just has a **much narrower target**
> than S-3 suggested: not "concealment," but heavy concealment.

---

## What this batch changes here

| | |
|---|---|
| **`purpose_breadth` is retired** | difficulty statistic. S-2's validation retracted on top. Any result citing it needs revisiting — `results/diversity/VERDICT.md` first |
| **The books test is cancelled** | stage 3 of the queue was `purpose_breadth` early-vs-late. Its measure is dead; running it would produce a difficulty gradient and we would read it as expertise |
| **Do not rebuild around process** | T-1 says process is the source; **T-5 says that has no instrument consequence.** The probe stays goal-side |
| **THE_TRIANGLE.md needs revising** | not a triangle; a directed flow, source→sink, additive not superadditive. The values vertex is unbuilt, not unmeasured |
| **Decision-counting is un-retired, conditionally** | alive where mode dwell is long. Needs a corpus of sustained single-purpose artifacts — which is a **new C-14-shaped debt**, and a much better specified one |
| **The leaked layer is worth building** | survives reader ignorance, including a 50% channel swap. Target it at heavy concealment only |

**Two of our four survivors are gone in one batch.** That is the correct thing to have happened —
both died to controls we did not know to run, discovered by a system with ground truth. Neither
would ever have been caught on real text.

---

## What the simulation did that is worth copying

Three things, and they are method rather than result:

1. **It refused a question and measured why** — `H(values|goal) = 0` instead of inventing a factor.
2. **It audited its own prior result mid-run** and retracted S-2 rather than building on it.
3. **It found the flaw in S-3's threshold** — fitted on labelled test data — which is precisely the
   class of error this project keeps making, and it caught it in its own work.

`docs/theory/CONTROLS.md` §6 now carries a hierarchy of controls partly because of this file.

# For the simulation — batch four, and the first one where the sim is the only place the answer exists

**2026-08-07.** Written at the curator's prompting: *"it sure does feel like there's fertile ground
for high-fidelity simulation on some of the things we are doing over here."* With his own caveat
attached, which is the right one: *"I do worry that it runs us astray sometimes."*

---

## Why this batch is different from the first three

Batches one to three asked the simulation to check whether our **statistics** were sound. It answered
well and killed several — the unlock statistic reads a large positive where the truth is zero, and
motivational breadth is a difficulty meter.

**This batch asks something the simulation is uniquely qualified for and the real environment cannot
do at all: it has ground truth about a number.**

Here is the situation on this side. We ran an unsupervised decomposition on model activations and
asked how many affective components exist. It returned **49 at 1,200 samples and 93 at 5,000, in the
same model with every setting frozen.** Across five model families it ranged 73 to 116 with no
convergence. **The number was tracking the sample size, and we retracted it.**

And the literature is no better off. Panksepp says seven. Cowen & Keltner say twenty-seven. Every
language-model paper published in 2026 stops at **two** — valence and arousal — and the honest reading
of why is that two is Russell's circumplex and they stopped when they got the expected answer. Lin et
al. recovered **27 versus 3 from identical stimuli** by changing the task format and the stopping
criterion.

> **Nobody in any of these fields has validated a component-count method against a case where the
> true count is known.** That is not a gap we noticed and skipped — it is a gap the whole area has.
>
> **The simulation can close it, and it is the only environment that can.**

---

## S-11 · Does our component-count pipeline recover a number we planted? ★ the priority

**The design.** Generate artifacts from **K latent drives**, where K is set by us and swept —
3, 7, 12, 20, 30. Each artifact is an emission influenced by a weighted mixture of the K drives, with
the same nuisance structure real text has: topic, length, register, and a per-artifact noise term.

Then run **our own pipeline, unchanged**, on the resulting feature vectors and ask what it returns.

**What each criterion should be scored on**

    recovery       does the estimate equal K
    bias           does it climb with sample size when K is fixed. This is the failure we already saw
    breakdown      at what nuisance-to-signal ratio does it stop recovering K

**The criteria to compare, and they fail differently**

| | |
|---|---|
| **parallel analysis** | the one we used, and the one that produced the retracted number |
| **participation ratio, bias-corrected** | Chun et al. derive a correction for finite sample and finite width. **Validated against synthetic ground truth at K = 50 in their own paper**, so this test is partly a replication of theirs and partly an extension |
| **bi-cross-validation** | Owen & Perry, 2009. Hold out rows *and* columns at once. Ordinary row-wise cross-validation of a decomposition leaks, so its error falls monotonically and never selects a rank |
| **eigenvalue > 1, and 90% of variance** | reported to make the spread visible |

**Why this is worth the build.** Whatever we eventually claim about how many affective primitives
there are, the first question anyone competent will ask is *how do you know your counting method
counts.* **Right now nobody in this literature can answer that, including us.** A validation curve
against planted ground truth is a publishable object on its own, independent of everything else this
project claims.

**Pre-register the failure.** If **no** criterion recovers K reliably, that is the result, and it says
the component-count question is not answerable with current methods at these sample sizes. **That
outcome would retract more of our work than anyone else's, which is the right incentive structure.**

---

## S-12 · Does a three-locus structure with a noisy middle read as a single middle peak?

**The claim being tested is the curator's, made this morning, and he is right that nobody has looked
for it:**

> We're finding ratio variance relationships between early and late despite there being a peak in the
> middle. It implies a sort of shape that I don't think anyone else has glommed on to.

**The design, and it is cheap.** Construct a generative process with **three loci** — an early one
that is clean, a middle one that is high-activity and low-coherence, and a late one that is
high-variance. Measure it with an instrument that averages across position, which is what every
published depth profile does.

**The question: does the measured profile come out unimodal with a middle peak?**

    SMEARS      yes. Then the field's mid-layer consensus is consistent with a three-locus truth, and
                our disagreement with them is about structure rather than about data
    SEPARATES   no, the three stay visible. Then the mid-peak literature is reading a genuinely
                single-locus structure and our trimodal architecture has to answer for it

**And the second half, which is the useful one.** If it smears, what *does* recover the three? The
curator's own observation is the lead: **the relationship between early and late survives even when
the profile looks unimodal.** So test whether a measure of early-late covariance separates the two
generative structures where the profile does not. **If it does, we have an instrument the field does
not have, and a reason it works.**

---

## S-13 · Values as the constraint that every drive is partially satisfied at once

**This is the first well-posed version of the values question the project has had**, and it comes
from him flagging it himself: *"Everything else before this felt like dithering to me, but this one
feels like it might be a real thing."*

**Why the current model cannot express it.** Simulation T-6 found the values vertex adds **exactly
zero** information — H(values | goal) = 0, a deterministic coarsening. We recorded that as "the values
vertex does not exist." **His account says that is the wrong construction, not a wrong answer.**
Values are not another vertex. They are **a property of the goal mixture**: the constraint that every
active drive is being partially satisfied, all the time, rather than served one at a time.

**The build.** An emitter where a single emission must partially satisfy **all** active drives, with
the satisfaction profile — how much each drive got — as the latent quantity of interest.

**Three questions in order**

1. Is the satisfaction profile recoverable from emissions at all?
2. **Is it more stable within a maker than between makers?** That is the actual definition of a value
   under this account, and it is the same design as the author-identification positive pointed at a
   different quantity.
3. Does it need many artifacts, as the reward-function argument says it must, or does one suffice?

**This is the build that would make the values vertex exist.** Everything else in the values programme
is blocked behind it.

---

## S-14 · The motivational aperture — is an absent drive recoverable?

**His claim, and it still needs a name:**

> If I were forced to design a Nazi camp, part of my motivation would be efficiency — I could tap
> that. **But I wouldn't be able to tap into the cruelty a Nazi designer would have. It just wouldn't
> be there for me to optimise.**

**The claim underneath: you can only route attention onto drives you possess**, so the drives a maker
*lacks* constrain what they produce and how. **That makes the absent drive as informative as the
present one.**

**The design.** Two makers, same instruction, same goal, **different available drive sets** — one
possesses a drive the other does not. Both produce the artifact that best satisfies the instruction
given what they have.

**The question: is the missing drive recoverable from the artifact?** Not which drives were used —
**which one was unavailable.**

**Why it matters beyond the theory.** It is a mechanism for why an artifact reads as
*made-under-duress*, which is a real perceptual phenomenon nobody has a model of. And it is the only
proposal in this project that treats an absence as a measurable.

---

## S-15 · Does recovery error actually shrink with more artifacts, and at what rate?

**This is the central disagreement with the impossibility literature, made runnable.**

Armstrong & Mindermann prove a policy cannot be uniquely decomposed into planner and reward, and that
this *"cannot be resolved by observing the agent's policy in enough environments."* Skalse et al. and
Cao et al. show rewards are only partially identifiable **in the infinite-data limit.**

**Our position is not that the theorems are wrong.** It is his, precisely stated:

> It's a limit situation. There *is* a solution — a perfect mapping of the person's brain — but we
> approach it **through inference with error**, and we are never sure we have the answer.

**That is a claim about a convergence rate, not about identifiability**, and the two are compatible.
The theorem says the residual ambiguity never reaches zero. **It says nothing about how large the
residual is, or how fast the reducible part falls.**

**The design.** Recover a maker's drive weighting from *n* artifacts, for *n* from 1 to 50, with the
true weighting known. Plot error against *n*.

    CONVERGES     error falls steeply and flattens at a small residual. Then the theorem is true and
                  irrelevant at practical n, which is exactly our claim
    FLAT          error does not fall meaningfully with more artifacts. Then the theorem bites at the
                  scale we work at, and the project has to say so

**Report the asymptote, not just the slope.** The number that matters is *how much ambiguity is left
when the reducible part is gone*, because that is the quantity the theorems constrain and the one
nobody has measured.

---

## Fidelity the simulation is missing, in priority order

Three of the tests above need something the current model does not have.

**1 · A values factor that is not a coarsening of the goal.** Required by S-13, and it is the reason
T-6 returned zero. Under the partial-satisfaction account this is not a new factor at all — it is a
constraint on the emission, so it may be cheaper to add than it looks.

**2 · A drive-availability mask per agent.** Required by S-14. Some drives present, some absent, and
the absence must actually constrain the policy rather than just zero a weight — **an agent without a
drive should route around it, not simply score lower on it.**

**3 · Depth, or something standing in for it.** Required by S-12. The simulation currently has no
notion of a processing stage, so there is nothing for a three-locus structure to live in. **This is
the largest of the three and it may not be worth it** — a simpler abstract three-stage emitter would
answer S-12 without touching the main model, and should be tried first.

---

## What to bring back, and how to read it

**The standing warning from the last batch applies and should be applied harder here.** The
simulation's headline from batch three — that goal legibility governs everything on the process side
— was **withdrawn in part** once the curator pointed out that its legibility knob attenuates in one
particular way, and whether real illegibility has that shape is exactly what a simulation cannot say.

**So: the simulation is authoritative about a method and suggestive about a mechanism.** S-11 is the
first kind — it validates a counting procedure against planted truth, and that transfers. S-12
through S-15 are the second kind, and each one should come back with **the assumption that would have
to hold in the real environment for the result to carry.**

**S-11 first, and by a distance.** Everything this project might claim about how many affective
primitives there are is currently resting on a criterion we have already caught being wrong.

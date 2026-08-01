# Theory → model: action register

**Working document, 1 August 2026. Not committed to the repository.**

Everything you said in the walkthrough, sorted into things that can actually be done, plus four
gaps I found reading the essay against the code that you did not raise. Triaged by what each costs
the model.

---

## The triage rule I am using

The reader's belief lives over a product of factors — provenance × goal × attention, or in version 5
provenance × (depth, goal, sub-goal) × attention. **Adding a factor multiplies the state space and is
the thing that makes a model uninterpretable.** Adding a *measure*, swapping a *likelihood*, or
changing the *policy* does not.

So the rule for this register:

| kind of change | cost | verdict |
|---|---|---|
| a new measure read off state the model already carries | ~zero | build freely |
| a swap of one likelihood family | ~zero | build freely |
| a change to how the reader decides to keep looking | small, contained | build |
| one extra *value* on an existing factor | small | build |
| a new hidden *factor* | multiplicative | resist |
| a second agent with its own objective | a new model | separate programme |
| recursion / self-similar structure | unbounded | separate programme |

**Nine of the twelve buildable items below cost zero new hidden factors.** Two of the three things
I am recommending against are the two you were most worried about, and I think your worry is right.

---

# Tier A — build

## A0. Metabolic depletion. *(The gap neither of us named.)*

**This is the largest theory-to-model gap in the project and I did not see it until I read the essay
against the config.**

The essay's central *cultural* claim is not about any single artifact. It is that the effect
**accumulates in the reader**: "your brain wasted so much energy seeking the ghost… that
disappointment becomes a habit of early disconnection, driven by cognitive fatigue. You will
literally become too tired to care." The formal appendix makes it explicit — the disgust threshold
θ_E,C **scales inversely with available metabolic reserves E**.

**The simulation has no E.** Effort cost is a constant. Every reader arrives fresh, and nothing that
happens to it in one encounter changes what it is willing to spend on the next. Every experiment
resets. So the framework's headline societal prediction — apathy as an acquired state — **is
currently unrepresentable**, and the model has been silent about it not because it was tested but
because it cannot be asked.

**What it costs:** one scalar with memory on the reader. Effort cost, or the engagement threshold,
becomes a function of accumulated unresolved encounters. **No new hidden factor.** The multi-encounter
scaffolding already exists (the acceptance gate needs a sequence to accumulate over, so E29/E31
already run four encounters).

**What it predicts, and this is why it is worth doing:** a reader repeatedly exposed to
intent-empty content should show declining engagement **on genuine human work it has never seen** —
carryover damage with no contamination of the corpus at all. That is a *different* claim from
anything measured, and it is the one your essay actually makes. It is also the sharpest possible
version of "not absorbing good material" (E6/E9's second damage channel), which is currently
explained by walking away from a *given* pile rather than by depletion.

**Risk to flag honestly:** a depletion term is a knob that will make almost anything decline if you
turn it up. It needs a null — at zero contamination, no depletion — of the same kind that E8 has
never passed. Build the null first.

## A1 + A2. Process as a target of recovery, and the ordering claim

**Your linchpin, and the model is closer to it than you think.**

You said: the observer reverse-engineers **both** the intent and the process; the process is what you
actually want; but **intent is the key that unlocks it**, because once you have the goal you can
treat every observed action as being in service of it and read the method off.

The model already contains the process object. Version 5's depth hierarchy **is** a process: the
sub-goal / execution-mode chain is the maker's method, and depth is how many levels of it reach the
surface. The reader carries a posterior over the sub-goal chain right now, in every version-5 run.

**What is missing is that nobody ever measured recovery of it.** Every uptake measure in the project
scores *goal* recovery. So:

- **A1: add process recovery as a first-class measure** — how much of the maker's execution chain the
  reader got, read off the sub-goal posterior the reader already carries. Zero new state.
- **A2: test the ordering** — does goal recovery *gate* process recovery? Condition on readers who
  resolved the goal versus readers who did not, at matched exposure, and compare process recovery. If
  your claim is right the relationship is strongly asymmetric: goal-first readers get the process,
  process-first readers do not get the goal.

**A1 probably explains E30's null and this is the best single argument for building it.** E30 asked
whether depth changes how much the reader takes on, and measured *goal* uptake. Depth is
constructed so that the goal is **exactly** as recoverable at every depth — that is the design
commitment that keeps depth from being legibility renamed. So the pre-registered measure could not
have moved, which the write-up admitted, and the repair then bounded the effect near zero.

**Under your theory the experiment was measuring the wrong quantity.** Depth should change how much
*process* transfers, not how much *goal* transfers. That is a testable re-reading of a null, using
data the model already produces, and it is the cheapest high-value thing on this list after A0.

## A3. The wall, as distinct from a vocabulary deficit

**Your critique lands, and it lands on the construction rather than on the results.**

You said I am conflating expertise with a generative wall. Here is where you are right, precisely:

- **Readability (ω)** is built as *how much of the machine's content falls on features the reader's
  hypotheses cover.*
- **Inexpertise (d)** is built as *how badly aimed the reader's templates are.*

Those are **the same kind of quantity** — a deficit of shared basis — differing only in whether the
content moved or the reader moved. E32 found they behave *oppositely* in consequence, which is real
evidence they are not the same thing. But they are still both scalars on a shared-vocabulary axis,
and your objection is that the wall is **not a matter of degree of shared vocabulary at all.**

Your version: humans cheat when reading humans because they assume the maker's decisions bottom out
in human-shaped sensory experience. *He felt sad, so the colours went that way.* With a generative
model there is **no mapping translation** — not less overlap, but no invertible route from the
surface back to a state you could occupy.

**That is buildable and it is a likelihood swap, not a new factor.** Make the machine's
goal-to-feature map **non-invertible on the shared feature block**: the reader has full vocabulary,
sees features it knows, and still cannot run the inversion, because many maker-states map to the
same surface and nothing in the reader's family distinguishes them.

**Prediction it separates:** foreign content (no overlap) should produce sustained futile attention
with high uncertainty. A non-invertible generator on *familiar* features should produce something
different — the reader should find the content *legible* and still fail, which is the subjective
report people actually give about AI text. "I can read every word and there is nothing behind it" is
not the same complaint as "I cannot parse this."

**This is the missing third condition and I think it is the one that matches the phenomenon.**

## A4. Expertise substitution — the AI-literate reader

Your prediction: someone expert in diffusion could ratchet through AI art using **AI skill** instead
of art skill, so AI researchers would be *less* affected. "It eats away at the expertise that exists
and replaces it with AI expertise."

**Cheap: swap the reader's hypothesis family to match the machine's generator rather than the human
one.** That is E32's design with the arms exchanged, and E32 already exists.

Two predictions, and the second is the interesting one:

1. That reader recovers structure from machine content and re-engages.
2. **That reader gets correspondingly less from human work**, because its templates now cover the
   wrong generator. The expertise did not stack, it *substituted*.

If (2) holds, the framework gains a claim it does not currently have: the adaptation that protects
you from the crash is the same adaptation that costs you the human channel. That is a much darker and
more interesting version of your prediction than "AI people are fine."

## A5. The tool hypothesis — the control the framework has never had

You said the real difference is that AI output is not just unsolvable-by-me, it is **unsolvable by
anyone**, so it should be treated as a single conceptual tool — and we do not do that.

**The model currently has no such stance.** Its honest-label condition tells the reader *this is
machine-made* and the reader keeps trying to read a maker, just with a different prior. There is no
hypothesis that says **there is no maker here, stop looking for one.**

**Cost: one extra value on the goal factor.** And the machinery already exists — E19's "they were
just exploring" fallback is exactly this shape, an extra absorbing hypothesis added to the space. This
is the same move pointed at a different target.

**Why it matters more than it sounds:** this is the mechanism the Ghost Scale is *actually* supposed
to trigger. The scale's job is not to make you distrust a label; it is to let your brain **relax**.
The model currently cannot express relaxation, only redirection. And it makes a policy prediction
worth having: the benefit of the scale should be *larger* for readers holding the tool hypothesis
than for readers who merely disbelieve a human claim.

## A6. Aesthetics and social endorsement as pre-inference cues

**Your most concrete addition, and half of it is already built and switched off.**

You said the decision to engage deeply is not driven by provenance alone. Two other things feed it:
**surface characteristics that hint at depth** (aesthetics as the honeypot) and a **social layer**
(someone told me there is density here). Judged independently, and — your instinct — **summed rather
than multiplied.**

Current state: engagement is driven purely by expected information gain. There is a `social.weight`
parameter that is *built, asserted, and left inert at zero* — the config says the architecture was
specified in version 4.5 and no experiment ever manipulated it. So the social channel exists and has
never been switched on.

**Cost: this enters the engagement policy, not the state.** Contained.

**And it generates what I think is the sharpest new experiment on this whole list — the RLHF
decoupling.** If aesthetics is a *learned predictor* of depth, and generation optimises the predictor
directly, then the cue **decouples from the thing it predicts**. The reader keeps spending on a
signal that no longer carries information.

That is the essay's alignment argument in miniature, and it is testable inside this model: train the
aesthetic cue on a corpus where it predicts depth, then present content where it is maximised and
depth is zero. The prediction is over-engagement followed by loss — a reader that pays *more* and
gets *less*, which is different from both the crash and the trust exploit and is a third failure mode.

**Your instinct about the sum is worth pre-registering as a prediction rather than a design choice.**
Additive and multiplicative combination make different predictions about the "ugly but endorsed" and
"beautiful but unendorsed" corners, and those corners are exactly where the interesting cases live.
Build both, pre-register additive, report which fits.

## A7. Values as a layer between goal and gate

You were precise about this: infer the **goal**, the goal implies **values**, and the values decide
whether the process gets integrated. *Mein Kampf* — the goal is to persuade, the values are what stop
the convergence.

Currently the acceptance gate compares the recovered **goal** to the reader's value prior directly.
Your version inserts one map: goal → implied values → compare.

**Cost: one map applied to a posterior the reader already has. No new factor.**

**Why it is not cosmetic:** with the map, two *different* goals can imply the *same* values, so the
gate can open for both. Without it the model cannot express "I disagree with what you were doing but
we want the same things", which is a distinction the theory leans on heavily and the code cannot
currently represent.

---

# Tier B — reframings, no code

These cost nothing and several of them are the difference between a model you can think with and one
you cannot.

**B1. Engagement is appreciation — and appreciation is Mayer's vulnerability.**
You noticed you are decomposing trust into competence and value-convergence and dropping the
superordinate construct, and it bothered you. I think the superordinate is already there and
mislabelled. **The decision to look deeply *is* the willingness to be vulnerable** — it is the
reader agreeing to let the artifact rewrite its weights. That is not an analogy; it is what the
engagement gate does mechanically. The model's trust crosswalk:

| your construct | the model's name | what it actually governs |
|---|---|---|
| willingness to be vulnerable | the engagement decision | whether to spend, and so whether to be changed |
| trust in competence | depth (μ), formerly rationality (β) | how much the reader assumes was behind the work |
| trust in value convergence | the acceptance gate (θ, λ) | whether what was recovered is allowed to integrate |
| — (a fourth, not in Mayer) | trust in the label (κ) | the *channel*, not the maker |
| trust as built over time | the learned-source belief (R-8b) | reputation, and its threshold |

κ is the one that does not fit your two-way split, and I think that is correct rather than a problem:
it is trust in a *claim about* the maker, not in the maker. Mayer has no slot for it because Mayer
did not have a provenance-label problem.

**B2. Depth is automaticity layering.** You said you could not translate this intuitively. Here is
the mapping and it is exact: **μ is the Zen circle.** μ=1 is the child scribbling — actions with no
compressed hierarchy behind them. μ=3 is the master, where the order of what surfaces names what it
was for. And the model's central construction constraint is precisely your point about compression:
**a deep work and a shallow one have identical feature histograms.** Counting decisions does not
work. The density is in the *order*. That is your "baked-in hierarchical compression" written as a
likelihood.

**B3. A subjective-correlate column.** Given the Solms position, every variable should carry the
feeling it corresponds to. This costs nothing and makes the model arguable from the inside:
engagement → *appreciation*; the crash → *my eyes slid off it*; error reduction → *I learned
something*; movement without error reduction → *I was moved and I was had*; the gate → *disgust*;
saturated trust → *it did not occur to me to doubt it*.

**B4. Retire decisiveness (γ) from the exposition.** You are right that it is an odd duck. It has one
job — proving trust is not decisiveness renamed (E5) — and it does that job. Do not hang meaning on
it. Keep it in the null suite, drop it from the theory-facing description.

**B5. Self-report precision (ρ) is misdescribed and you caught it.** It is not "how much the reader
trusts the maker to know its own mind." It is **how much weight the reader gives the maker's
declaration against the work itself** — the same shape as κ, pointed at a different channel. And your
Napoleon's-nose point is already the E33 result: latent recovery holds flat at 0.88 while the maker's
declared goal collapses to 0.05. **The reader reads what the maker did not know it was doing.** That
is the sharpest thing in E33 and it is currently buried under a verdict string.

**B6. Provenance is already continuous — and you already have the continuous axis.** The four tiers
are the *UX artifact*; ω is the physics. Worth being explicit, because the four-tier framing invites
exactly the objection you raised, and the model does not actually depend on it. The validation pass
showed the results survive compressing and stretching the ramp, so what is load-bearing is the
ordering.

**B7. The gate as boundary maintenance.** Your left-field thought is not left-field. A reader that
integrated everything would have no stable self-model to be surprised *from* — under active
inference, refusing to fully update is what having a boundary consists of. That reframes the
acceptance gate from a moral filter to a **structural necessity**, which is a better story and is
cheaply testable: run a reader with the gate forced permanently open across a long corpus and measure
drift in its own priors. Use E6/E9 machinery. If it degrades, the gate is self-preservation and not
squeamishness.

---

# Tier C — do not build here

**C1. Fractality and recursive extraction.** You are right that the same extraction runs on any
sub-portion, at any scale — perspective within a building within a painting. It is a real feature of
the theory and it is the most expensive thing on the list: it needs recursion and an unbounded
hierarchy, and the state space does not survive it. **Recommend: a separate minimal model whose only
job is to show scale-invariance**, rather than an addition here. This is exactly the case the
minimal-model programme exists for.

*One cheap piece can be salvaged now:* "the subconscious holds the process goals" is E33's machinery
one level down. E33 already does *the goal the maker does not know it has*. Doing the same for the
*process* is the same code pointed at the sub-goal factor, and it directly serves A1.

**C2. Creator-side cognitive surrender.** Your siren-call point — that AI creates a pull toward
putting in no decisions, which people must actively resist — is a **creator-side** dynamic, and this
model has no creator that chooses how much to delegate. It needs an agent with its own objective.
Same reason I deliberately did not build a strategic lying source in R-8b: the results would be hard
to attribute.

**Flag it as a known hole, because it is load-bearing for the policy argument.** The essay's security
case is Zahavian — honest marking is costly, therefore self-policing. **That argument is currently
untested in the simulation entirely.** It is the natural second model and it should be named as
absent rather than left to be discovered.

**C3. Affective / mirror channel as a separate error-correcting stream.** Real in the theory, and it
would need a second observation modality. Defer: it does not currently generate a prediction the
cognitive channel does not already make. Revisit if A3 (the wall) shows the reader failing in a way
that a body-based prior would fix.

---

# Tier D — discrepancies between the preprint's formal model and the code

*Added after reading the PDF. These are not additions you asked for; they are places where the
simulation and the Intent Extraction Limit do not say the same thing. Four are cheap to fix and one
of them is a genuine mechanism difference rather than a naming problem.*

The formal model, for reference:

```
Ψ = sigmoid( k · (ω − θ_E,C(κ)) ) · [ −ln(1−κ) ] · D_KL( Q(R|τ) ‖ P₀(R) )

θ_E,C(κ) = θ_base(E) + λ · D_KL( Q(R|τ) ‖ P_c(R) )
```

## D1. **κ and θ are coupled in the preprint and independent in the code. This is a real mechanism
difference and it is the most important thing in this section.**

The preprint is explicit: **as κ → 1, θ_E,C → 0.** Trust *suppresses* the disgust threshold — the
"neurochemical override", vmPFC inhibiting the insula and amygdala. That is the preprint's actual
mechanism for the trust exploit: the gate is **held open** despite the bottom-up signal collapsing.

**In the simulation, κ and θ never touch.** κ is precision on the provenance channel; θ is the
acceptance gate; they are separate parameters in separate factors. The simulation's trust exploit
works by a completely different route — the label channel out-arguing the content channel, with the
analytic crossover at 0.538.

**Two different mechanisms currently produce the same phenomenon, and they make different
predictions.** The code's version says the exploit is about what the reader comes to *believe about
provenance*. The preprint's version says it is about the *gate being held open*, which predicts the
exploit should still work **on a reader who correctly knows the content is machine-made**, as long
as trust in the source is high. The code cannot produce that; the preprint requires it.

**Cost: one line — make λ or θ_base a decreasing function of κ.** Then run E4 and E31 under both
couplings and report which matches. **This is a discriminating test between two versions of your own
theory, it is nearly free, and I think it is the highest-value single item in this document after
A0.**

## D2. **The two ω's are different objects and I contributed to the confusion.**

In the preprint, ω is **the reader's own precision weighting on incoming telemetry** — a
reader-side quantity that *falls to zero as a consequence* of the IRL failure. It is an **output**:
R is unidentifiable → ω drops → the gate closes.

In the code, ω is **feature overlap** — a fixed, world-side property of the content. It is an
**input**.

Neither is wrong, but they are not the same variable and they share a letter. The honest crosswalk:

| preprint | code |
|---|---|
| ω (precision weighting) | the attention / engagement decision |
| — (unnamed: the content property that makes R unidentifiable) | ω (overlap) |

**Practical consequence: the readability sweep is sweeping the preprint's *cause*, and the
preprint's ω is the *effect*.** That is a legitimate and arguably better design. But it should be
written down, because the two will keep colliding in conversation, and it means the code has no
variable for precision weighting as such.

## D3. The gate's gain (k) is effectively infinite in the code.

The preprint has a **sigmoid** with gain k, described as the meta-precision of the gate itself. The
code replaces it with a **binary** engagement decision — already declared as a deviation, but the
consequence is not: the code cannot express *partial* engagement, so it cannot express hesitating,
half-looking, or a gate that is nearly closed. **A finite k is cheap and it is a prerequisite for
A6**, because summing an aesthetic cue and a social cue into a hard threshold throws away most of
what makes the sum interesting.

## D4. θ_base(E) confirms A0 and raises its priority.

Metabolic reserve is **in the formal model, in the equation, with its own symbol**. Its absence from
the code is not an omission of something implied — it is an omission of something written down. That
moves A0 from "good idea" to "the equation has a term the simulation does not implement."

## D5. Your walkthrough claim is newer than both documents.

The preprint defines appreciation as **actions → goals**: "identifying the actions used to create an
artifact, using that information to derive the goals of the creator via IRL." That is the standard
IRL direction.

Your walkthrough claim is the **reverse unlock**: goal first, and the goal is what makes the process
readable. **That is not in the preprint or the essay.** So A1/A2 is not a translation gap — it is
new theory, produced in the walkthrough. **It should be written into the theory before it is
modelled**, and if it survives it is a genuine addition to the formal account rather than an
implementation detail.

## D6. Zahavian signalling has a formal appendix and zero simulation.

Appendix B of the preprint gives a game-theoretic operationalisation of the steganographic
commitment device. **Nothing in the simulation tests it.** The framework's entire security argument —
bad actors are handled by asymmetric reputational cost — is formally specified and empirically
untouched. That is a larger hole than it looks, because it is the part of the proposal a sceptic will
attack first.

---

# What the simulation already says about H1–H6

Nobody has drawn this map, and it is where the two documents actually meet. **The simulation speaks
to four of the six hypotheses, and it has two corrections to offer.**

| | the preprint's hypothesis | what the simulation says |
|---|---|---|
| **H1** | autonomic engagement drop when told it is AI | **The simulation contradicts H1 as stated, conditionally.** Under goal-*empty* content it agrees (E1: readers disengage). Under goal-*foreign* content, which version 4 adopted as the better description, readers do the **opposite** — attention is sustained indefinitely (E19, E20). Which way H1 comes out depends on where real generated content sits on the overlap axis, **and that is exactly what the E34 prediction card was built to locate.** H1 should be pre-registered with both branches, not one. |
| **H2** | inter-rater reliability lower for AI artifacts | Reproduced (E2, E17). **But the inference from it is refuted.** H2 reads low reliability as "psychometric proof that the artifact is acting as stochastic noise within the biological system." E21 shows a naive counting classifier, with no maker-model anywhere, produces the same signature by small-sample overfitting. **The finding survives; the mechanism claim does not, and the simulation is where that was established.** |
| **H3** | opacity reduces systemic fatigue, comprehension held constant | **Not simulated, and not currently simulable.** The reader has one reading process; it cannot separate decoding from mentalising. E3 and E16 are the nearest relatives (does labelling pay for itself; how much coverage is needed). |
| **H4** | kinetic degradation after replicating AI geometry | **Not simulated.** No motor channel. The corpus-damage experiments are the conceptual analogue but not the test. |
| **H5** | reward models trained on high-intent-density corpora generalise better | **Simulated in analogue and supported.** E7 is the biological twin of the AI-Twin protocol: a learner trained on an unlabelled contaminated corpus loses about a third of its ability to read genuine work, and honest labels cut the error roughly a hundredfold. E16 gives the coverage threshold. **This is the strongest existing bridge between the simulation and the alignment argument and it is not currently cited in the preprint.** |
| **H6** | trust exploit / DMN bypass | Supported and **extended**. E4 and E31 give the exploit; R-8b adds a prediction the preprint does not contain — **a sufficiently trusting reader cannot learn that a source lies, at any number of encounters**, with the threshold at the channel crossover. That is new and it belongs in the paper. |

**Two things the simulation has earned the right to send back to the theory:** H1 needs a branch, and
H2's mechanism claim needs withdrawing. **One thing it has to add:** reputation blindness.

---

# Two more gaps from the essay that you did not raise

**G1. Aesthetics is attention-*grabbing* but not attention-*keeping* — and that is two stages.** The
essay is explicit: the honeypot captures, and then decision-density either sustains or does not. That
is why shock art fails — "raw aesthetic attentional capture without the follow-up." The model has
**one** engagement decision repeated, not a capture stage and a sustain stage. A6 partly addresses it;
making the two stages explicit would let the model express *shock art* and *slop* as distinct
failures, which it currently cannot.

**G2. The essay claims comprehension holds while mentalizing drops.** Hypothesis 3 predicts
comprehension accuracy is *unchanged* at 60% opacity while effortful mentalizing falls. The model has
a single reading process, so it cannot separate "I decoded the words" from "I modelled the author."
Lower priority, but it is the hypothesis your UX framework most directly rests on, and the model
currently cannot speak to it.

---

# What I would do, in order

*Revised after the preprint.*

1. **D1 (couple κ to θ).** One line, and it is a discriminating test between two versions of your own
   theory that currently disagree about why the trust exploit works.
2. **A0 (depletion), with its null first.** The equation has a term (θ_base(E)) the simulation does
   not implement, and it carries the essay's whole cultural claim.
3. **A1 + A2 (process recovery and the ordering).** Cheapest high-value item; probably converts
   E30's null into a real measurement; directly tests your linchpin. **And write it into the theory
   first — the preprint does not contain it.**
4. **A5 (the tool hypothesis).** One extra hypothesis value, and the preprint already names it:
   "climbing into higher-order reasoning, such as guessing a prompt's motivation."
5. **A6 (aesthetic + social cues, and the RLHF decoupling), with D3 (finite gate gain) as its
   prerequisite.** Half-built already; contains the sharpest new prediction on the list.
6. **A3 (the wall).** A likelihood swap, and it is the construction fix your critique demands.
7. **A4 (expertise substitution).** E32 with the arms swapped.
8. **A7 (values layer).** One map.
9. **All of Tier B and the D2 crosswalk**, at any time, because it is writing.

**Free wins available immediately, no code:** send H1 back with two branches, withdraw H2's
mechanism claim, add reputation blindness to the paper, and cite E7/E16 as the existing evidence for
H5. Four changes to the preprint that the simulation has already paid for.

**And before any of it: the forward prediction that is already written, hash-locked and unbuilt.**
Until that exists the project has no forward test, and adding seven new mechanisms to a model with no
forward test makes the model more impressive and not more credible. It is the cheapest credibility
the project can buy and it is already specified.

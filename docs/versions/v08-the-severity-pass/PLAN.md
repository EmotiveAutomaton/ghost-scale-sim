# Version 8 — a planning document, not a spec

**Nothing is built from this yet.** It is the design conversation written down: what a sweep of the
code, the theory documents and the working record turned up, what I think should be built, and the
decisions that are the author's rather than mine.

A spec follows once the decisions below are made.

---

## Part 0 — The thing that should worry us both

Two rounds have now come back almost entirely positive. Roughly twelve of sixteen new results
confirmed their hypothesis; the exceptions were one clean negative, one retirement, one
pre-registered contrast that still fails, and one honest "no effect either way".

**For exploratory, theory-derived work that is one person's framework written down as code, that
hit rate should not be comfortable.** The README already says the right thing about *why* agreement
is expected — every prediction came from one prior theory, so the simulation reproducing it is the
expected outcome rather than evidence for it. What has not been said is that **nobody has measured
how much of the agreement is the theory and how much is the apparatus**, except once.

### The forking-paths problem, counted rather than described

Across versions 6 and 7 I changed a design or a criterion **after seeing a null**, in seven places:

| experiment | what changed after a null came back |
|---|---|
| E35 depletion | content type changed from low-legibility hierarchical to foreign; criterion changed from absolute to relative |
| E42 vulnerability | regime changed to the one known to sustain attention |
| E45 efficiency | task made harder; held-out goals raised from one to two |
| E36 process | the null's statistic changed from accuracy to information; a second, temporal form of the ordering test added |
| E46 leak | criterion changed from absolute magnitude to ratio; the null restated |
| E38 expertise | machine family changed from degenerate to well-formed |
| C-3 co-location | one artifact per cell rather than per observer |

**Every one of those changes is individually defensible, and each is documented in the code where
it happened.** Several were correcting a genuine measurement error — accuracy on a flat posterior
really is not a chance-level statistic, and a ceiling in both arms really is an absent measurement
rather than a null result.

But the aggregate has a shape, and the shape is: *result comes back flat → find a reason → change
the design → result comes back positive.* That is a garden of forking paths whether or not each
fork was justified, and the fact that I can justify each one is exactly what makes it hard to see.

### The one instrument that would settle it, and it already exists

The validation pass built it and ran it once. Keep the model's *shape*, throw its settings away,
draw new ones at random a few hundred times, and count how often the headline still appears. **For
the label effect that number was 64%** — nearly two thirds of randomly parameterised readers of the
same shape reproduce it. That single number is the most useful thing in this repository, and it is
the reason the label result is now reported as architecture-dependent.

**It has never been run on anything else.** Not on the interior peak, not on the two-dimensions
result, and not on any of the sixteen findings from the last two rounds.

So before anything else is built, I want to know: **for each headline, what fraction of randomly
parameterised models of the same shape produce it?** If E45's efficiency advantage shows up in 60%
of random models it is a property of having a generative model at all, which is interesting but is
not the theory's claim. If it shows up in 3%, the claim is strong and now has a number behind it.

**This is my strongest recommendation and it is cheap.** It reuses machinery that is already
written and it runs against results that already exist.

### The second instrument: severity, not confirmation

A criterion that a false theory would also pass is not evidence. For each surviving result the
question worth asking is: *what would this experiment have shown if the theory were wrong?* In most
cases nobody has said. Naming the answer, per result, would be a page of writing and would sharpen
what may be claimed more than another ten experiments.

---

## Part 1 — What the sweep found: theory terms with no counterpart in the code

Version 6 found three of these in the Intent Extraction Limit. Reading the essay and the preprint
against the source again turned up three more, and they are structural rather than cosmetic.

### 1.1 The reader has no depth of its own — and this is the largest gap left

In the theory, **expertise IS automaticity.** You can see the decisions in a bridge because you have
made those decisions; Teller can see a trick's structure because he has built tricks; the bomb
expert weeps at an explosion a layperson cheers. *"Expertise is simply the cognitive resolution
required to see the load-bearing scaffolding beneath the paint."* The essay is explicit that this is
a hierarchy the reader has built in themselves, by compression, through practice.

In the code, reader expertise is a **template-accuracy** parameter: how badly aimed your goal
signatures are, implemented as a perturbation toward noise. That is a different quantity. It makes
you *wrong*; it does not make you *shallow*.

The consequence: **the model cannot currently express that a shallow reader cannot see a deep
work's depth.** The reader infers the maker's depth, and its ability to do so does not depend on any
depth of its own, because it has none.

**And this may already have shown up as an unexplained result.** The diagnostics pass found depth
recovery *compressed*: a known depth reads back at about a third of its true value, and nobody
could say why. A reader with no hierarchy of its own being able to see only so far up somebody
else's is exactly the shape of that compression.

**What it would predict:** an interaction. Reader depth × maker depth, with a novice systematically
under-reading a master and an expert reading them accurately — and, more interestingly, the reverse
case the essay names, where *the master learns from the student* by seeing decisions the student
could not have known they were making.

### 1.2 Attention costs something. Being changed is free.

In the model, the only non-zero preference is effort: looking closely is expensive, skimming is
cheap. **Integration costs nothing.** A reader that takes on a colossal update pays exactly the same
as one that takes on none.

The theory says the opposite is the main cost. The whole senate metaphor is about *updating ledgers*
being what burns the twenty watts — *"the free energy we try to minimise is simply the effort all
these senators have to spend updating their voting ledgers. The more wrong they were, the more they
have to change."* Surprise is expensive **because rewiring is expensive**.

So the model charges for the search and not for the learning, and the theory charges for the
learning.

**What it would predict, and why it is not cosmetic:** a reader that pays to be changed becomes
conservative in a way this one is not — and **the trust exploit gets worse, not better.** A fooled
reader currently absorbs a large false update for free. Charge for it and the exploit costs the
victim real resources for nothing, which is a sharper version of the harm claim than the one on
record.

### 1.3 Nothing is ever forgotten

No belief in this model decays. The metabolic reserve recovers, but the carried picture of what
people are like does not fade, ever.

That matters most for the newest result. E46 shows drift accumulating monotonically under repeated
rejected exposure — **unbounded, because there is nothing pulling it back.** With forgetting, the
same mechanism gives an *equilibrium*: drift settles where exposure rate meets decay rate.

That is both more realistic and a much better claim. "Propaganda moves you without limit" is weak
and slightly silly. "Propaganda moves you to a level set by how often you meet it against how fast
you forget it" is a real prediction with a policy reading — and it says frequency matters more than
intensity, which is testable and non-obvious.

---

## Part 2 — Edge cases the theory names that the model cannot represent

Not gaps in implementation. Cases the framework explicitly discusses and the current construction
has no way to express.

**The readymade.** Duchamp's *Fountain* is the essay's own hardest case: near-zero fabrication, a
single act of selection, enormous compressed context. The model's depth is *how many levels reach
the surface*, measured over a sequence of observations — so an artifact with almost no observable
extent has almost nothing to read. **The model probably cannot represent high artfulness at low
observable volume**, and that is the case the theory spends the most words defending.

**Distributed authorship.** "Nature" in the essay is the interaction of every creature's decisions,
attributed to a conglomerate. Collaborative work is the same shape. The model has exactly one maker
per artifact and no way to represent intent that is real but has no single owner.

**Attention-grabbing versus attention-keeping.** The essay is precise that aesthetics *captures* and
decision-density *sustains*, and that shock art is capture without follow-through. Version 6 added
the cue channels, but the reader still makes one kind of engagement decision repeatedly. The model
cannot currently distinguish *slop* from *shock art*, and the theory does.

**Reading the tool rather than the maker.** The essay's own suggested escape — "climbing into
higher-order reasoning, such as guessing a prompt's motivation" — is a shift in what the reader
takes the object to be. E39 tried the nearest thing and found it redundant with what the reader
already knows about provenance. The richer version, which is a reader that models *the human who
chose to use a tool*, has not been built.

---

## Part 3 — What I propose building

Ordered by what I would do first, with reasoning. **The author's two named items are in here; the
ordering is a recommendation, not a decision.**

### Tier 1 — before anything else

**S-1. The severity pass.** Random-parameterisation false-positive rates for every headline,
including all sixteen from versions 6 and 7. Reuses existing machinery. **This determines whether
the rest of the programme is measuring the theory or the architecture, and it is cheap.**

### Tier 2 — the two the author named

**F-1. The forward prediction, built and run.**

*What it is, since it has been referenced without being explained.* During the validation pass one
prediction was written down, sealed with a content hash, for an experiment **that did not exist**.
It is the project's only genuine forward test: everything else was predicted by people who already
knew the theory, and the literature check happened afterwards.

The setup: a maker whose purpose is defined by what it **avoids** rather than what it pursues — a
region of the space it will not enter. Two readers see its work. One holds only the ordinary
hypotheses, which are things a maker might be *trying to do*. The other also holds avoidance
hypotheses. The prediction is that the second recovers the constraint and the first does not; and
the interesting half, **the first is predicted to be confidently wrong rather than uncertain**,
because an avoidance leaves a hole, every positive hypothesis has support inside that hole, and the
one whose peak sits furthest from it wins by default.

Four ways it can fail are named in the sealed file, including one that would be a direct hit on the
framework. **It is the cheapest credibility this project can buy and it is already specified.**

**C-1. The creator agent.** A maker that chooses what to make and whether to declare it honestly.
This unlocks the one part of the proposal that has a formal appendix in the preprint and zero lines
of simulation: **the Zahavian security argument** — that honest marking is costly and therefore
self-policing, so bad actors are handled structurally rather than by enforcement.

It also makes reachable, for the first time, the questions that need two agents: whether a labelling
convention is stable when makers can defect, what happens to reputation when detection is
imperfect, and whether the equilibrium depends on the reader's leak (§1.3) — a reader who cannot
fully refuse is a reader a defector can exploit more cheaply.

**This is the largest single build in the project's history and I would not run it against the
current reader without §1.1 and §1.2 settled first**, for the reason in Part 4.

### Tier 3 — the model corrections

**M-1. Reader depth.** §1.1. The largest theory-model gap; may explain an existing unexplained
result; predicts an interaction nothing currently predicts.

**M-2. Integration costs something.** §1.2. Small change, sharpens the harm claim, and moves the
cost to where the theory puts it.

**M-3. Forgetting.** §1.3. Small change, converts an unbounded drift into an equilibrium, and gives
the propaganda result a policy reading.

### Tier 4 — the representational edge cases

**R-1. The readymade.** Can the model express high artfulness at low observable volume? If not,
that is a bounded limitation worth stating rather than a flaw worth hiding.

**R-2. Two-stage attention.** Capture and sustain as separate decisions, so that shock art and slop
become distinguishable.

**R-3. Distributed authorship.** Deferred unless it becomes load-bearing; it needs a different
generative structure and would not reuse much.

---

## Part 4 — Decisions that are the author's, not mine

Kept high level, and I have given my recommendation for each.

**D-1. Ordering.** I recommend severity first, then the forward prediction, then reader depth, then
the creator agent. The author named the creator agent and the forward prediction; the question is
whether severity goes in front of both. **My recommendation is yes, because if the false-positive
rate is high, the creator agent would be a large build on a foundation nobody has checked.**

**D-2. Rerun policy, which is already stated and has a cost worth naming.** The standing instruction
is to re-run everything whenever the model changes. **M-1, M-2 and M-3 each change the reader**, so
each triggers a full re-run of the entire programme. Three separate full re-runs, or one after all
three land, is a real choice: separate runs make each change attributable, and a combined run costs
a third as much and confounds them. **My recommendation is to build all three, then run once with
each switchable, then run the matched pairs that isolate each** — which is what version 6's
switch-everything-off discipline was designed to make possible.

**D-3. Does the leak become the model's standing behaviour?** Currently the gate leak (V7) is off by
default. If §1.3 forgetting is added, the leak becomes much more defensible as a default, because
drift then equilibrates rather than growing without bound. **My recommendation is to make the
decision once, after forgetting exists, rather than now.**

**D-4. Scope of the creator agent.** A maker that chooses *how much to delegate to a tool* is a
different and larger thing than a maker that chooses *whether to declare honestly*. The second is
enough for the Zahavian argument. The first is the cognitive-surrender dynamic the author raised
several rounds ago and set aside. **My recommendation is the second only, with the first named as
out of scope in the spec**, because a two-decision maker makes attribution hard in exactly the way
the repair pass was written against.

**D-5. What "expertise" means going forward.** §1.1 argues the model has the wrong construct. If
reader depth is built, there are then two expertise-like parameters — template accuracy and
hierarchical depth — and the older experiments were all run on the first. **My recommendation is to
keep both, name them differently, and treat the existing results as being about template accuracy
rather than silently reinterpreting them**, which is the same discipline applied when effort was
replaced by depth.

---

## What this document is not

It is not a spec, nothing here is locked, and no criterion in it has been fixed. When the decisions
above are made, SPEC_V8 gets written before any code, as every version has been.

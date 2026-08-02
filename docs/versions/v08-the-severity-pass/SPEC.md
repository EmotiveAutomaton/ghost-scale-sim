# Version 8 — the reader gets a mind of its own

**Written before any V8 code. Not edited afterwards. Deviations are logged in RESULTS_V8.md.**

---

## §0. What this version is for

Every version until now has modelled a reader that **infers** a maker without ever **being** one.
It has no hierarchy of its own, nothing it does costs it anything except looking, and nothing it
learns ever fades. Version 8 gives it all three, and then asks the question those three make
possible for the first time.

It also runs the severity check the whole programme has been missing, builds a maker that can lie,
and takes a serious run at the case the theory spends the most words defending and the model cannot
currently represent at all.

---

## §1. The severity pass, which runs first and gates the rest

Two rounds came back almost entirely positive. **Nobody has measured how much of that agreement is
the theory and how much is the apparatus**, except once: the validation pass found that 64% of
randomly parameterised models of the same shape reproduce the label effect, which is why that result
is now reported as architecture-dependent.

**S-1 runs that check against every headline**, including all sixteen results from versions 6 and 7.
Keep the model's shape, throw its settings away, redraw them a few hundred times, and count how
often the finding survives.

**This runs before anything else is built and its result is reported whatever it is.** If a
headline's false-positive rate is high, that headline is a property of the architecture and the
theory is not entitled to it.

A second half, cheaper and just as useful: **the forking-paths ledger.** For each experiment in the
project, how many designs and how many criteria were tried before the reported one. That number
exists in the commit history and in the module docstrings; it has never been collected.

---

## §2. Three terms the theory has and the code did not

### §2.1 The reader has no depth of its own

In the theory, **expertise is automaticity**: the compressed hierarchy you built by having made
those decisions yourself, which is what lets you see them in someone else's work. In the code,
reader expertise is *template accuracy* — how badly aimed your goal signatures are. That makes a
reader **wrong**; it does not make it **shallow**.

So V8 gives the reader its own depth, and the existing parameter is renamed rather than replaced:

| construct | what it is | what it governs |
|---|---|---|
| **calibration** (formerly reader inexpertise) | how well-aimed the reader's templates are | whether it is right |
| **reader depth** (new) | how many levels of hierarchy the reader has itself | how far up someone else's it can see |

Every V1–V7 result was measured on the first. **They are not silently reinterpreted**, the same
discipline applied when effort was replaced by depth.

*A naming note, per the author's instruction that terms should map onto things that exist rather
than proliferate: both of these are ordinary constructs with ordinary names. Calibration is
calibration. Reader depth is expertise in the sense the essay uses it. The spec uses the plain words
and the code carries both.*

### §2.2 Being changed costs nothing

The only non-zero preference in V1–V7 is effort: looking is expensive, skimming is cheap.
Integration is free. The theory puts the cost on the other side — the senate metaphor is about
*updating ledgers* burning the budget, and surprise is expensive **because rewiring is expensive**.

V8 charges for integration. The prediction that makes this more than tidiness: **the trust exploit
gets worse**, because a fooled reader now pays real resources for a large false update.

### §2.3 Nothing is ever forgotten

No belief in V1–V7 decays. V8 adds forgetting, in the specific shape the author specified:
**associations weaken but do not disappear.** Decay is asymptotic toward a floor, not erasure.

The prediction: E46's unbounded drift becomes an **equilibrium**, set by exposure rate against decay
rate — which says frequency matters more than intensity, and that is a policy claim.

---

## §3. Hypotheses

**H8.1 — a reader can only see as far up a hierarchy as it has built itself.**
Predicted: reader depth × maker depth interact. A shallow reader systematically under-reads a deep
work; a deep reader reads it accurately.
**Fails if** depth recovery is flat in reader depth, which would mean reading depth needs no depth
and the essay's account of expertise is wrong.

**H8.2 — that interaction is what the depth-compression result was.**
The diagnostics pass found a known depth read back at about a third of its true value and nobody
could say why. Predicted: the compression scales with the gap between reader depth and maker depth,
and vanishes when they match.
**Fails if** compression is constant in reader depth, in which case it is a property of the
estimator and the earlier "directions transfer, magnitudes do not" reading stands unexplained.

**H8.3 — reading and making are the same machinery, so appreciation installs capability.**

*This is the author's hypothesis and it is the most consequential thing in this version.*

Under active inference, perception and action are the same computation: both minimise free energy,
one by changing beliefs and one by changing the world. If that is taken seriously, then **a compiled
motor routine and a compiled perceptual schema are the same kind of object** — which would mean the
hierarchy a reader uses to *read* a maker is the hierarchy it would use to *make*.

That is the mechanism underneath "you use your own architecture to simulate theirs". Not an
analogy: literally the same structures, which is why expertise is domain-matched and why you can
only read what you could in principle make.

Predicted: **a reader exposed to work deeper than itself grows its own hierarchy**, and that growth
shows up in what it could produce, not only in what it can recognise. Appreciation is acquisition.
**Fails if** exposure improves recognition without improving production, which would separate the
two hierarchies and refute the identity.

*This is the "art is a virus" claim stated mechanically, and it has never been tested.*

**H8.4 — artfulness is density, not volume.**
The readymade is the theory's hardest case and the model cannot represent it: depth is measured over
a sequence, so an artifact with almost no observable extent carries almost nothing. Scored as
**hierarchy invoked per unit of observable extent**, a single act that requires three levels to
explain is maximally dense.
Predicted, and this is the interesting part: the reading goes **bimodal**. Readers with a matched
hierarchy see enormous depth, readers without see nothing, and there is no middle — which is exactly
what conceptual art does to a room.
**Fails if** the reading is unimodal with a low mean, in which case a readymade is just weak work
and the theory's defence of it does not survive its own model.

**H8.5 — capture and sustain are two decisions, and slop and shock art differ.**
The essay is precise that aesthetics *grabs* and density *keeps*, and that shock art is capture
without follow-through. The model has one engagement decision repeated.
Predicted: split it, and shock art (high capture, no density) separates from slop (low capture, no
density) on the attention trace even though both end with nothing recovered.
**Fails if** the two are indistinguishable, in which case capture buys nothing the model can see.

**H8.6 — honest marking is self-policing.**
The Zahavian argument has a formal appendix in the preprint and zero lines of simulation. A maker
that chooses whether to declare honestly, meeting readers who remember sources, should find honesty
stable where the reputational cost of being caught exceeds the gain from lying.
Predicted: an honest equilibrium exists above some detection rate, and **the leaky gate lowers it** —
a reader who cannot fully refuse is one a defector can exploit more cheaply.
**Fails if** defection dominates at every detection rate, which would be a direct hit on the
proposal's security argument and would be reported as one.

**H8.7 — reading the tool is different from reading no-one.**
E39 gave the reader a "there is no maker here" hypothesis and it bought nothing, because it was
redundant with what the reader already knew about origin. The richer version is a reader that models
**the person who chose to use a tool** — which is a maker, with intent, at one remove.
Predicted: that reader stops cleanly where E39's could not, because it has something to conclude
rather than something to fail at.
**Fails if** it reproduces E39, in which case the affordance genuinely cannot act through the
hypothesis space and must act on the gate.

---

## §4. The forward prediction, and an honest downgrade

The validation pass sealed a prediction for an experiment that did not exist: a maker whose purpose
is defined by what it **avoids**, and two readers, only one of which holds avoidance hypotheses.

**Its epistemic status is downgraded here and the reason is recorded.** The author does not recognise
authoring it; it appears to have come out of an exchange with a different model during an earlier
phase of the work. **A sealed prediction is only worth what the commitment behind it was worth**, and
a commitment nobody remembers making is not a forward test.

So: **the experiment is built and run, because it is a good experiment** — avoidance-defined intent
is a real gap, and its secondary prediction (that the unequipped reader is confidently wrong rather
than uncertain) would be a second independent instance of the interior-peak mechanism. **But it is
not counted as this project's forward test**, and the scoreboard is corrected to say the project
still has none.

*That correction costs the project its only claim to a forward test. It is made anyway, because the
alternative is banking a commitment that was not really made.*

---

## §5. Nulls

N35 — **every V8 addition is off by default**, and with all of them off V8 reproduces V7.
N36 — **forgetting cannot erase.** Decay approaches a floor and never reaches zero.
N37 — **reader depth does not manufacture recovery.** At maker depth 1 there is no hierarchy to see,
so reader depth must not improve anything.
N38 — **integration cost does not become a preference over provenance.** The reader must still be
unable to want work to be human.
N39 — **density does not reward noise.** A short artifact with no hierarchy behind it must score low,
or the readymade measure is a volume discount.
N40 — **the maker's honesty is not free.** A defecting maker that is never detected must do better
than an honest one, or the Zahavian result is an artefact of a rigged payoff.

---

## §6. What is deliberately NOT built

- **Distributed authorship.** The author ruled it out and the reasoning is sound: a conglomerate
  maker with no single owner is not testable here and invites readings the framework should not
  invite.
- **A maker that chooses how much to delegate to a tool.** The cognitive-surrender dynamic. Ruled
  out as a second decision that makes attribution hard, per D-4.
- **Recursion.** Unchanged since V6.

---

## §7. Pre-mortem

1. **The severity pass comes back high and the programme is holed below the waterline.** That is
   the point of running it first. It is reported whatever it says.
2. **Reader depth makes everything better and explains nothing**, because a more capable reader is
   better at every task. Guarded by N37 and by H8.2's requirement that the *interaction*, not the
   main effect, is what carries the claim.
3. **Density rewards short artifacts as such.** Guarded by N39.
4. **The maker's payoff is rigged.** Guarded by N40, which requires that lying pay when it is not
   caught.

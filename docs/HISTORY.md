# How the record got this way

Ten model versions and three audit passes, in one narrative. Every version has a name as well as a number; the name is the theme, the number is the order. Written so that nobody has to read six
write-ups to understand why a number is what it is.

The per-version write-ups are still there and are still the authority on their own version. **They
are history: none of them has been edited after its version shipped**, which is the property that
makes the record checkable. If a number in one of them has since been superseded, this page says so
and points at what superseded it.

---

## The two kinds of pass, and why they are kept apart

**Versions ask new questions about the world.** V1 through V10. Each has a spec written before its
code and a write-up written after its results, and neither is edited afterwards.

**Audit passes ask whether the existing answers can be trusted.** Validation, diagnostics, repair.
None of them asks a new question; all three go back over work already done. Their output documents
are *generated from verdict files*, never hand-written, so an explanation cannot quietly change a
number.

Version 6 is a hybrid and is the only one: it audits the *code against the theory* rather than the
results against themselves, and then builds what was missing.

---

## The versions

### V1 · The Mechanism

The first build. A reader that infers who made a thing and what for, under a cost, with **zero
preference over provenance** so that any effect of a label has to arrive through inference. Three
results came out of it and two of them still stand: readers disengage from purposeless content, and
a false provenance label makes every reader confident and no two of them agree.

Seven deviations logged. The one that matters most: the synthetic content distribution was
*goal-symmetrised*, which turned out later to be the load-bearing decision behind the disagreement
result rather than a tidying-up.

### V2 · The Learner

Added a learner that does not arrive knowing what machine-made content looks like and has to
acquire it from an already-contaminated stream. Found that reader heterogeneity belongs in the
likelihood rather than the prior, and that biased machine content accumulates rather than averaging
out.

### V3 · The Refuted Repair

Written to fix the generational experiment under a diagnosis that the leak was ordinary sampling
noise. **V3's own gate refuted the diagnosis**: the leak did not shrink across a hundredfold more
data. A real estimator bug was found and fixed along the way, cutting the error 171-fold, and the
experiment stayed withheld anyway — it still missed its threshold, narrowly, and narrowly is exactly
the case a no-exceptions rule exists for.

### V4 · Foreign Intent

**The most consequential version.** V1–V3 modelled machine-made content as *goal-empty*: structure
with no purpose behind it, like wood grain. V4 replaced that with *goal-foreign*: a real purpose,
pursued by a real process, expressed in a vocabulary the reader has no entry for — which is a better
description of a generative model, trained on purposeful human output and inheriting its shape.

Under goal-empty, readers disengage. **Under goal-foreign they do the opposite: they keep looking,
keep paying, and never get anywhere.** The failure to read intent survives and gets worse. The
saving of effort does not.

This is also where the feature space had to double, because no disjoint human/foreign partition
existed at the original size. Every claim about the readability axis inherits that decision, and it
is the reason the axis is written as a location to be measured rather than a number to be trusted.

### V4.5 · Three Gates

Added a three-gate reader and promoted the metabolic question from a column to a headline. Four
unwelcome results, including the one that matters most in the whole project: **a naive counting
classifier reproduces the confident-and-contradictory pattern**, so modelling the maker as a mind is
not necessary for it. The framework's claim about its own necessity was withdrawn.

### V5 · Depth Over Effort

Replaced "how hard was the maker trying" with **how many levels of the maker's decision hierarchy
reach the surface** — the Zen master's circle rather than the child's scribble, where the difference
is compressed practice and not effort expended. Built so that a deep work and a shallow one have
*identical feature histograms*: a reader that counts and ignores order cannot tell them apart at
all. Depth lives in the order.

And V5 caught the fast inference shortcut being badly wrong: it returned the shallow answer for
every artifact, confidently, while exact arithmetic on the same observations recovered depth
correctly. That single catch is what motivated the entire validation pass.

### V6 · Code Against Equation

The only version that asks whether the simulation and the theory it claims to implement are the
same object. Reading the preprint's formal model against the shipped code found **three terms with
no counterpart in the code**, one of which is a disagreement about mechanism rather than an
omission. Details in [../RESULTS_V6.md](versions/v06-code-against-equation/RESULTS.md).

### V7 · The Closures

Version 6 ran four experiments it then refused to draw conclusions from, because in each case the
measurement was pointed at the wrong quantity. Version 7 closed all four. The largest was the
two-gates disagreement, which had produced three inconsistent passes: the criterion was scoring
*goal* uptake on a design that holds goal uptake constant. Re-scored on process uptake it came back
at **0.93, interval [0.771, 0.977]**.

The version also answered a direct attack on E21 — *you are using your own architecture to simulate
theirs, so you are cheating the solution space*. E45 asked what the maker-model actually buys and
found two things a counting classifier cannot get at any scale: it needs **4 examples where the
counter needs 512**, and it reads an intention it has never been shown. More training does not close
either gap. [../RESULTS_V7.md](versions/v07-the-closures/RESULTS.md).

### V8 · The Severity Pass

The reader acquires a hierarchy, a cost for being changed, and a memory that fades toward a
permanent residue rather than to zero. A maker that can choose to lie put the security argument in
code for the first time, and found honesty self-policing **only above a detection rate of 0.5**.

But the version is remembered for the severity pass, which asks the question nobody wants to ask
about their own work: keep the model's shape, throw its settings away, redraw at random, and count
how often the finding still appears. **100%, 98% and 0%.** Two of the three headlines are properties
of the architecture rather than evidence for the specific theory. The same pass collected the
forking-paths ledger — seven places across versions 6 and 7 where a design or a criterion changed
after seeing a null. [../RESULTS_V8.md](versions/v08-the-severity-pass/RESULTS.md).

### V9 · Minimal Models

Severity says how much of a result is architectural. It does not say **which part**. So the
complement: keep the settings and strip the *shape* — remove one structural commitment at a time.
**Every surviving finding dies when the reader stops modelling a maker and starts classifying a
surface.** Hierarchy and costly attention turn out to be free. The wall is the only finding that
needs the reader to hold a distribution rather than a best guess, which is the second pass to single
it out as the one genuinely about the theory.

Two experiments rode along, both built from the author's reading of places the published literature
disagreed with the simulation. Both predictions failed, and one of them found the disagreement had
been a comparison error in the first place. The other killed the mechanism by which the Ghost Scale
was supposed to work. [../RESULTS_V9.md](versions/v09-minimal-models/RESULTS.md).

### V10 · Reader As Defence

Every version to nine asked what happens to a reader. This one asked whether **reading intent is
itself a defence**, against a threat that is documented rather than hypothetical: networks
publishing at industrial scale specifically to be absorbed by models rather than read by people.

The answer is yes, on the case that matches the real one. Against content carrying real structure
under a false claim of origin, surface-quality filtering leaves a learner **exactly as damaged as no
filter at all** — which is what E40 predicts, since surface quality is the attacker's objective.
Asking who made this and why cuts the damage 23%, restores the learner's reading of genuine human
work, costs nothing on a clean corpus, and never reads the label. **The Ghost Scale failed twice as
a label makers apply; this is the same idea as a capability readers have, and it needs no social
adoption.**

The version also caught itself. Its most attractive hypothesis — that values ride in on process
through a shut gate — was **withheld because its test arm failed the clean-corpus control**, which
is the null the author had recorded, before the run, as the one he most expected to fail. It failed,
and in failing it disqualified the invalid arm and left the headline standing.

[../RESULTS_V10.md](versions/v10-reader-as-defence/RESULTS.md).

**After this the remaining questions need human subjects or real models, and this apparatus is
neither.**

---

## The audit passes

### Validation — can the recorded answers be trusted?

Nine checks, criteria hash-locked before it started. **Five came back against the work.** Three
matter most: one headline is a property of the model's *architecture* rather than of the theory,
two verdicts were produced by the inference shortcut and do not survive its removal, and the one
result rebuilt independently from its own prose reproduced the mechanism but not the size — fifteen
times smaller.

The single most consequential finding: with the model's settings thrown away and replaced at random,
**100% of random readers of this shape still become confident under a false label.** What a random
reader does *not* produce is the disagreement. The theory is entitled to the second half.

### Diagnostics — can the instruments answer at all?

A different question, and it found four things worth knowing before any number is quoted closely.
Reading the label and reading the work are two competing streams that arrive at every glance and
disagree on a lie, with a crossover computable in closed form. The uptake measure is U-shaped in how
well the reader read the work, because a confidently wrong reader has moved as far as a correct one.
The disagreement figure cannot be read on its own. And five of nine pre-registered criteria are
computed over too few units to separate their own outcomes.

### Repair — what can be fixed, and what does fixing change?

One rule: every change either makes something measurable that was not, or removes something. Four
outcomes. The uptake measure was a *distance*, so being fooled counted as much as being right; split
into a signed measure the headline cell reverses sign. Trust turned out measurable after all — the
earlier verdict had fitted it to the wrong data. A new prediction the fixed-trust model could not
make: **a sufficiently trusting reader cannot learn that a source lies.** And the three headline
criteria that had no error bars now have them.

### The V6 retrofit — does the new machinery change the old answers?

Version 6's additions were demonstrated on new experiments; the retrofit goes back. It is scoped
honestly: experiments with no trust parameter, no depth and no sequence cannot be touched by a trust
coupling, a depth measure or a depletion term, and they are **named as out of scope rather than run
and reported as null**.

Where it does reach, it matters. The two-gates criterion — the project's longest-running open
question, which two solvers disagreed about across two passes — resolves when scored on the quantity
the theory actually names.

---

## Four criteria that could not do their own job

The failure mode this project has most of. Each was caught by a later pass, and each is documented
where it was found rather than quietly fixed.

1. **A rank correlation over six cells** decided the public headline while 2,400 per-reader pairs
   sat unused in the same run. Neither of the two values it produced could have told either way.
2. **A permutation check that could never pass**, because the manipulation it applied inverts the
   effect it was testing for.
3. **A monotonicity criterion punishing ties the construction guarantees.**
4. **An absolute-magnitude threshold on a quantity whose baseline varies two-fold between seed
   blocks** (V6, E35). The mechanism reproduces every time; the threshold passes once in three.

A fifth, caught before it produced anything: **accuracy as a chance-level statistic on a flat
posterior**, which is decided entirely by which value the truth happened to take.

---

## Two experiments removed, one withheld

**Removed as uninterpretable rather than embarrassing:** a test of whether effort gates uptake, and
a test of whether three gates behave differently. Both were run against a construct later found to
be wrong, so reading them requires holding an assumption now known to be false.

**Withheld three times:** the generational experiment. Its own honesty check failed every time.
Its failing test stays in the suite as a visible marker.

**Kept deliberately:** every result that came back against the framework.

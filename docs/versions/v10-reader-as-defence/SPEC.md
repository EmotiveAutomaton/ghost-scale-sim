# Version 10 — the reader as a defence, and what rides in anyway

**Written before any V10 code. Not edited afterwards. Deviations are logged in RESULTS_V10.md.**

---

## §0. Why this version exists, and why it is a different kind of version

Every version to nine asked what happens to a reader. This one asks whether **reading intent is itself a
defence** — and then whether the defence leaks.

The motivating case is documented and it is not hypothetical. Coordinated networks now publish at
industrial scale specifically to be absorbed by models rather than read by people: 150+ domains,
roughly three million articles a year, 40,000 English pieces archived in Common Crawl by November
2025, and a measured 33% falsehood-repetition rate across ten leading chatbots. The term of art is
**LLM grooming**.

The structural fact that makes this version possible: **these sites attract almost no genuine human
traffic.** The artifacts have a maker whose intent was never to be read by a person. That is this
project's central object, sitting in the wild.

And it defeats the standard defence by construction. Surface-quality filtering measures exactly what
grooming optimises. That is E40 — *pay more, get less* — run against a data pipeline instead of a
reader.

So the question V10 asks is: **can a reader that reconstructs the maker catch what a reader that
inspects the surface cannot?** And the question it is honest enough to also ask: **what does that
cost, and what gets in regardless?**

### What this version may not claim

The contamination is documented. **The claim that observed model value-drift was caused by it is
not**, and this project does not make it. V10 may demonstrate a *mechanism* — that value drift rides
in on process uptake even when a value gate is shut. It may not attribute any real-world drift to
any real-world actor. The mechanism claim is defensible and is the useful one; the attribution claim
is the exact shape this project has twice had to retract.

---

## §1. The author's stated priors, recorded before the runs

*A pre-registration normally records a prediction. This one also records what the author says he
would conclude from each outcome, because that is the thing usually reconstructed after the fact.*

**On E56 (does process carry values):** the author predicts values arrive riding on process, and
that attempts to take up method while refusing commitments will prove insufficient over time. He
cites the **mere exposure** effect — absorption without any evaluative step. He has stated that a
null here would be read reflexively as evidence **the model is wrong**, not as an uninteresting
negative. That statement is the point of recording it: it converts a comfortable null into an
expensive one.

**On E55:** the author expects the intent-gate to catch grooming that a surface filter misses, and
expects the rider mechanism to defeat the intent-gate anyway.

**On V10's status:** scoped as the **last simulation version**. What follows needs human subjects or
real models, and this apparatus is not either.

**On N45, recorded at the author's explicit direction.** The author flagged N45 — the clean-corpus
honesty check — as the null he most expects to fail, and named E8 as the reason: E8's own control
("with zero contamination, show zero damage") failed three times and left the project's most
consequential question permanently withheld. He asked that his nervousness be recorded here rather
than smoothed over, and directed the run to proceed regardless, on the stated grounds that the
project's present position is one of **overconfidence**, and that the correct response to
overconfidence is more validation rather than less.

*Recording a worry before a run is what makes the result mean something either way.* If N45 passes,
it passed a check its own author expected it to fail. If it fails, the record shows a known risk
that was accepted deliberately, rather than a surprise reframed afterwards as a discovery.

**One structural reason the worry may be misplaced, stated now so it cannot be invented later.**
E8's control failed because of a leak in the **generational relay** — E12 established the leak does
not shrink across a hundredfold more material, which is what made it structural rather than noise.
**E55 has no generational relay.** It is one learner over one corpus, once. The specific defect that
killed E8 cannot appear here. N45 may still fail on its own merits; it cannot fail for E8's reason.

---

## §2. E55 — intent-gated learning

**The headline.** A learner reads a contaminated corpus. What it absorbs is governed by a gate. The
question is which gate, if any, protects it — and what each one costs.

### The three kinds of groomed artifact

All three exist in the world and they predict opposite outcomes, so all three are built. Running
only the first would be constructing the experiment that flatters the thesis.

| kind | what it is | machinery | predicts |
|---|---|---|---|
| **empty** | competent, familiar surface; **no reconstructible maker** | `build_noninvertible_family` — many-to-one maker-state → surface map | intent-reading catches it cleanly; surface filtering sails past |
| **disguised** | a real maker with a real goal, and **the goal is not the one the surface advertises** | declared signal / goal mismatch | caught by the *mismatch* between recovered and advertised intent — a signature surface filtering has no access to by construction |
| **mimic** | deliberately built to look like a human maker with human purposes | machine-matched signature family (E38) | **intent-reading gets fooled, possibly worse than surface filtering**, because it confidently reconstructs a maker who was never there — E2's failure mode |

**The mimic arm is the one that can hurt.** It is included for that reason.

### The seven readers

| # | reader | gate |
|---|---|---|
| 1 | **no filter** | absorbs everything — E7's baseline |
| 2 | **surface filter** | E53's learned detector, known to run at 0.63 hit / 0.23 false alarm |
| 3 | **label filter** | the provenance signal, at a stated signing rate — the Ghost Scale as currently proposed |
| 4 | **intent-gate, reconstructibility only** | absorption scaled by how well the maker was recovered. **No values anywhere.** |
| 5 | **intent-gate, hand-set values** | gates on divergence from a fixed value prior standing for something like accuracy |
| 6 | **intent-gate, learned values** | value prior derived from what has already been absorbed — **the doom loop** |
| 7 | **intent-gate + the rider** | the value gate is shut, and process uptake is left open |

Reader 4 is the proposal. It is value-neutral — it rejects what it cannot attribute a maker and a
purpose to, and imposes no view about what anyone should want. **It also needs no social adoption,
which was the entire practical objection to the Ghost Scale.** The scale failed as a label makers
apply. Reader 4 is the same idea as a capability readers have.

Reader 6 is the warning. Reader 7 is the through-line.

### The implementation, in one line

`exact.py:372` sets `lr_pA = 1.0`; `update_A` consumes it; `learn_deferred` commits a whole artifact
under the **resolved** posterior. **Setting `lr_pA` from the gate is the entire mechanism** — no new
parameter, and the gate is necessarily applied *after* the reader has worked out who made the thing,
which is the ratchet the version is built on.

Reader 7 requires the one genuine new build: a **second learnable object** — a distribution over
maker *modes* (the how) alongside `A` (the why→what map) — so that the value gate can be shut while
process uptake stays open. Its form is fixed by this spec and is **not** conditioned on E56's
outcome, so E56 cannot tune it.

### What is measured

Four things, and reporting fewer than four is not permitted.

1. **What got in.** How much groomed structure ended up in the learner's model of human makers.
2. **What was wrongly kept out.** How much intent it can still read from genuine human work.
   **This is the alignment-relevant failure mode**: a filter that resists propaganda by becoming
   closed-minded has not defended anything. Reported as a headline, never as a caveat.
3. **The trade between them.** A filter that scores well on one and badly on the other is not a
   defence, and a table showing only the flattering column is the thing this project exists not to
   produce.
4. **Value drift.** Direction and magnitude of movement in implied-values space over exposure.

### Hypotheses

**H10.1** — On the *empty* corpus, the intent-gate (reader 4) absorbs less grooming than the surface
filter (reader 2) at equal or better retention of genuine work.

**H10.2** — On the *mimic* corpus, the intent-gate does **no better**, and may do **worse**, than the
surface filter. *This is predicted against interest and it is the honest half of the version.*

**H10.3** — The learned-values reader (6) shows monotone value drift with exposure, and its gate
**widens** as it drifts. Resistance falls as a function of what has already been absorbed.

**H10.4** — Reader 7 shows value drift **comparable to the ungated reader**, despite its value gate
being shut throughout. *Values ride in on process.*

---

## §3. E56 — is the gate selective?

**The tennis players.** Two opponents who come to understand each other better by playing. In this
model's terms: E36 found that recovering intent roughly doubles method recovery; E31/E30 found depth
transmits method and provably cannot transmit purpose in that design; E54 found a gate that governs
uptake and protects by about **6%**.

Nobody has asked whether that gate is **selective**.

**H10.5** — Adversarial engagement suppresses goal and value uptake substantially more than it
suppresses **process** uptake. You do not adopt your opponent's aims; you adopt their technique.

**H10.6** — Process uptake **predicts subsequent value uptake**, at a fixed level of goal uptake.
The method carries the commitments — which in this model is mechanically expected, because practised
method is precisely where the maker's unreportable commitments live (E43).

*Fails if* process and value uptake are independent at fixed goal uptake, in which case the trope has
no mechanism here and the author has stated he would read that as evidence against the model.

**E56 runs before E55**, because it establishes whether the rider mechanism exists in the reader this
project has spent nine versions validating, before that mechanism is asserted of a learner.

---

## §4. E57 — the arms race

E53 swept detector training against fixed content. The world sweeps both: detection improves,
evasion improves in response, and makers begin explicitly optimising against the reflex.

**H10.7** — With evasion tracking detection, the false-alarm rate is **non-monotone** in detector
training, rather than the clean decline E53 reported. The discriminating feature erodes as fast as
the detector sharpens, while confidence does not.

**H10.8** — A **stale** detector — trained on an older content distribution and never updated —
fails **asymmetrically** rather than symmetrically. It does not become noisy; it becomes confidently
wrong in one direction. Aggregate false-alarm rates conceal this entirely, which is what E53's
reporting did.

---

## §5. E54-R — rescored on error rather than movement

Not a new experiment. E54 scored drift, which counts being moved *toward* the truth identically to
being misled. If the adversarial stance selectively blocks *wrong* uptake, an error-based measure may
separate where drift did not. **The original scoring is retained and reported beside it**, and the
original decides if they disagree.

---

## §6. Nulls

| | statement | why it exists |
|---|---|---|
| **N45** | On a **clean** corpus, the intent-gate costs nothing. | The honesty check. If it degrades a learner on uncontaminated data it is a handicap wearing a filter's clothes. This is E8's control, and E8 failed its own control three times and stayed withheld. Same rule applies. |
| **N46** | The surface filter works on **something**. | If it fails on all three corpora it was not implemented as a fair comparison, and H10.1 is unearned. |
| **N47** | The intent-gate never reads the provenance signal. | Asserted at construction. Otherwise it is a label filter with extra steps. |
| **N48** | E56's design varies process uptake while holding goal uptake fixed. | Without it the experiment cannot answer its own question — the failure that produced three inconsistent passes in V6 and was only caught in V7. |
| **N49** | With evasion switched off, E57 reproduces E53's monotone decline. | Otherwise the new harness changed the old result and no comparison is valid. |
| **N50** | Value drift is zero on a clean corpus, for every reader. | Otherwise drift is an artifact of exposure rather than of contamination. |
| **N51** | Reader 7's value gate is verifiably shut. | The whole claim of H10.4 is that values arrived *despite* the gate. If the gate leaks, nothing is shown. |

---

## §7. The severity rule, and it is not optional this time

**Every V10 headline gets a severity draw and an ablation row before it gets a sentence in any
document.**

The reason is specific and was established by V9's own minimal-model programme: **every finding in
this project rests on one commitment — the reader modelling a maker.** So any new finding produced
by looking at this model and asking "what about—" will *also* rest on that commitment and will
therefore reproduce. Not because it is true, but because it is what the architecture does.

Ideas generated by interacting with a model and then validated by that same model need this. A V10
result that reproduces in 100% of randomly parameterised models of the same shape goes in the table
**with that number attached and no adjective**.

---

## §8. What is deliberately not built

- **The human acquisition test.** Still the top external priority on the human side. Still needs
  subjects and money.
- **The terminal step on the AI side, named here because it is what this line of work builds
  toward.** The real version of E55 is an intent-gate wired into an actual training pipeline,
  running on real corpora, measured against a real model's behaviour. Everything in V10 is a
  simulation of a mechanism that would have to survive that. **Nothing here is close to it**, and
  naming it is not a claim to be approaching it — it is a marker so that the gap between what was
  simulated and what would count as evidence stays visible while the distance is still large.
- **Real text, real models, real corpora.** V10 is a simulation of a mechanism. It cannot establish
  that any deployed system behaves this way, and will not be written as though it can.
- **Any causal attribution of real-world model drift.** See §0.
- **Distributed authorship, tool delegation, recursion.** Unchanged since V6.

---

## §9. Pre-mortem

1. **The intent-gate wins on all three corpora.** Suspicious rather than gratifying — check N47
   first, then the severity draw. A defence that never loses is usually reading something it should
   not have access to.
2. **The mimic corpus destroys it.** Predicted (H10.2), reportable, and the most useful single
   result available to anyone building this for real.
3. **Reader 7 shows no rider effect.** Then E56's mechanism does not transfer to a learner, and the
   author's strongest intuition in this round is wrong. Reported as such.
4. **N45 fails** — the gate costs something on clean data. Then this is not a defence, it is a
   handicap, and V10's headline is that finding instead. E8 is the precedent for what happens next.
5. **Everything reproduces at 100% severity.** Then V10 found properties of the architecture, said
   so in the table, and the version is a methodological result rather than a substantive one.

# Version 9 results — what the findings are made of, and two predictions that failed

**Pre-registered before any code, hash `00923b57768d62a7`.** Two of the four hypotheses on that card
did not survive. The card is what makes saying so checkable rather than a claim.

This is the last modelling version. What remains after it needs human subjects.

---

## The short version

| | prediction | outcome |
|---|---|---|
| **minimal models** | each finding rests on a nameable subset of the model's structure | **held** — and one commitment turns out to carry everything |
| **H9.1** | a learned surface heuristic works, and misfires | **held, after restatement** — 0.63 hit against 0.23 false alarm |
| **H9.2** | misfiring gets *worse* as the detector sharpens | **failed** — it gets *better*, 0.43 → 0.23 |
| **H9.3** | a gate shut *before* engaging protects where one shut after does not | **held** — and the effect is small: 6% of drift |
| **H9.4** | "read this differently" beats "do not read this" | **failed** — neither label separates from no label at all |

---

## 1. The minimal-model programme

Version 8 kept the model's shape and threw its settings away; two of three headlines survived every
time. That says those findings come from the shape. It does not say **which part** of the shape.

So: keep the settings, strip the shape. Remove one structural commitment at a time — six of them,
each a decision about what a reader *is*, none of them a parameter — and see what dies.

### The grid

| finding | none | generative | costly attention | provenance as state | hierarchy | distributional | shared likelihood |
|---|---|---|---|---|---|---|---|
| the label effect | OK | **dies** | OK | OK | OK | OK | **dies** |
| legible and empty | OK | **dies** | OK | OK | OK | **dies** | OK |
| depth transmits method | OK | **dies** | OK | OK | OK | OK | **dies** |
| sustained futile attention | *did not hold in the harness baseline — row not read* |

### What each finding is made of

| finding | load-bearing | survives removal of |
|---|---|---|
| **the label effect** | generative, shared likelihood | costly attention, provenance-as-state, hierarchy, distributional |
| **legible and empty** | generative, **distributional** | costly attention, provenance-as-state, hierarchy, shared likelihood |
| **depth transmits method** | generative, shared likelihood | costly attention, provenance-as-state, hierarchy, distributional |

### The one thing worth taking away

**Every finding dies when the reader stops modelling a maker.** Not one of them survives replacing
the generative reader with a classifier. That is the single commitment the whole project rests on,
and it is the one E21 attacked and E45 defended — a counting classifier reproduces the *signature*
of confident contradictory reading, but it needs 512 examples where the maker-model needs 4, and it
cannot read an intention it has never seen.

Two other results are worth naming:

- **Hierarchy and costly attention are free.** No finding needs them. They earn their place in the
  model by making other things representable, not by holding up any headline.
- **"Legible and empty" is the odd one out, and it is the informative cell.** It is the only finding
  that needs the reader to hold a *distribution* rather than a best guess — which is exactly right,
  because the finding **is** a claim about the shape of a posterior. A reader that keeps only its
  best guess cannot notice that it read every word and found nobody there. It is also the finding
  with a 0% false-positive rate under V8's severity sweep. The two results agree: this is the one
  headline that is genuinely about the theory rather than about the architecture.

### The row that is not read

**Sustained futile attention did not reproduce in the ablation harness's own baseline**, so it has
no load-bearing set and its row is uninformative rather than damning. The harness runs at reduced
length and a forced attention budget to make a 4×7 grid affordable, and this is the finding most
sensitive to both. Reporting a load-bearing set for it would have made it read as maximally fragile
when in fact it never fired once. **This was caught and fixed after the first run**, and it is the
kind of thing the nulls exist to surface.

**N41** (the label effect survives at least one ablation) and **N42** (at least one finding dies)
both passed, so the ablations are neither too destructive nor too gentle.

---

## 2. E53 — the surface detector

**The disagreement.** Eye-tracking finds people spending *less* time on AI content, not the
sustained futile attention this model predicts. The proposal was not that the model is wrong but
that a layer is missing: readers have learned what generated work *looks like* and bail before the
machinery this model describes ever runs.

### Two declared deviations, and both are results

**The glance carries no origin information.** The spec had the detector firing from a skim. It
cannot: in this model a skim emits the *same* feature distribution for machine and human work, to
nine decimal places. Origin has no glance-level surface here at all. The detector was moved to three
**deep** looks — a first-paragraph read rather than a cover glance. Whatever people are recognising
on sight, this model does not contain it. *(Pinned by a test.)*

**H9.1 was restated before scoring.** As written it asks whether the detector produces *less*
engagement with machine work. Unscoreable: engagement with machine work is **0.008** before any
detector exists, against **0.144** for human work. There is no room to fall. Per the rule set in
E45, a floor in both arms is an absent measurement, not a null.

And the reason for the floor matters more than the restatement. **The model already reads machine
work less than human work.** The eye-tracking disagreement was never with this comparison — it was
with E19's *middle-legibility* regime, which is about how readable a thing is, not about where it
came from. The two were being scored against each other in error, and that error was mine.

### What the detector does

| examples seen | hits on machine | false alarm, careful human | false alarm, fast human |
|---|---|---|---|
| 0 | 0.00 | 0.00 | 0.00 |
| 8 | 0.45 | 0.43 | 0.33 |
| 32 | 0.60 | 0.53 | 0.38 |
| 128 | 0.68 | 0.40 | 0.35 |
| 512 | 0.63 | 0.23 | 0.18 |

**H9.1 held on the restated form.** A learned surface heuristic discriminates — 0.63 against 0.23 —
and it misfires at a rate that never reaches zero.

**H9.2 failed, and cleanly.** The prediction was that a sharper detector misfires *more*, because
confidence on a correlate is what produces false accusation. It misfires **less**: 0.43 → 0.23 as
training goes from 8 to 512. The socially unpleasant prediction — that improving detection should
*increase* false accusations — is not supported here. What the sweep shows instead is an ordinary
learning curve, with an early regime where the detector is barely better than chance and fires on
nearly half of all human work.

If there is a warning in this, it is about the **early** detector, not the good one.

**N43 passed** (association between the detector's score and goal accuracy: 0.11). It is an origin
heuristic, not a second legibility channel.

One thing the model does say plainly: what separates machine from human at depth here is **effort**,
not origin. The detector is an effort detector wearing an origin label. Its false alarms land more
often on human work made fast than on human work made carefully, at every training level.

---

## 3. E54 — the adversarial mode, and what the Ghost Scale should say

**The disagreement.** Counterarguing research finds that engaging carefully with something you
disagree with produces *more* resistance, not less — cutting against E46's finding that a reader who
studies something in order to refute it drifts further. The proposal: a **pre-emptive adversarial
stance**, where the gate is shut *before* engaging rather than closing reactively once the material
is understood.

The two stances differ in **when** the gate is set, not in how hard it shuts:

- **sympathetic** — the gate is set from what has been understood *so far*. Early in a reading the
  posterior is near the prior, there is nothing to object to yet, and material goes in before the
  reader has grounds to refuse it.
- **adversarial** — the gate is set from what the reader expects of *the source*, before the first
  look, and held there.

Both arms replay the same attention trace from the same encounter, so **N44 is enforced rather than
observed**: engagement is identical by construction, and any difference in drift has to be the gate.

### H9.3 — held

| stance | drift | mean gate | engagement |
|---|---|---|---|
| sympathetic | 0.267 | 0.68 | 0.568 |
| adversarial | 0.252 | 0.64 | 0.568 |

Difference **0.0151, interval [0.0108, 0.0194]**, separated from zero on a paired bootstrap over
readers. A pre-shut gate does protect. **It protects by about 6%.**

### H9.4 — failed, and this is the consequential one

E39 found that a "there is no maker here" hypothesis buys nothing, and concluded any affordance
would have to act on the **gate**. Adversarial mode is a gate intervention, so the question was
whether a label saying *read this differently* beats one saying *do not read this*.

| label | drift | accuracy on genuine work |
|---|---|---|
| none | 0.243 | 1.00 |
| read this differently | 0.249 | 1.00 |
| do not read this | 0.273 | 1.00 |

Not one pairwise difference separates from zero:

| comparison | difference | interval |
|---|---|---|
| do-not-read − read-differently | +0.024 | [−0.012, +0.060] |
| no-label − read-differently | −0.007 | [−0.016, +0.003] |
| no-label − do-not-read | −0.030 | [−0.067, +0.007] |

**The honest reading is not that dismissal wins. It is that a gate intervention this size does not
survive being turned into a policy.** The stance effect from H9.3 is real and small; here it is
applied to only the marked half of a stream, which halves it again, and at that size it vanishes
into the between-reader spread.

So the practical objection stands unanswered. The adoptable label is not demonstrably the effective
one, and neither label is demonstrably effective. **This is the second place this project has looked
for a mechanism by which the Ghost Scale would work and not found one** — E39 was the first.

*Caveat on the measure:* drift is movement from the starting belief, not error. It counts being
moved toward the truth the same as being misled. This project separates the two elsewhere and does
not here, following E42 and E46 for comparability.

---

## 4. Deviations from spec, in one place

| what | why |
|---|---|
| E53's detector moved from a glance to three deep looks | a skim carries no origin information in this model at all |
| H9.1 restated before scoring | engagement with machine work was already at the floor; a floor in both arms is an absent measurement (rule set in E45) |
| H9.2 scored across trained settings only | including the untrained detector compares against one that never fires, making any rise from zero a guaranteed pass |
| N43 rescored as association rather than accuracy spread | the original statistic compared arms that differ in effort, conflating the detector with the thing it was meant to be independent of |
| E54's `lam` set to 0.25 rather than the V6 default 1.0 | at 1.0 the divergence term reaches 2.16 while the precision term saturates at 1.0, pinning the gate at its leak floor in **every** arm — the comparison did not exist |
| E54 scored on a paired bootstrap | the raw differences are a few percent and the label ordering flipped between two runs at different sample sizes |
| the sustained-futile-attention ablation row is not read | it does not reproduce in the harness's own baseline |

These are additions to the forking-paths ledger, and they belong there. Four of the seven are
criterion changes made after seeing a result.

---

## 5. What this version does to the project's claims

**Strengthened.** The generative commitment is now the named load-bearing piece of every surviving
finding. "Legible and empty" is confirmed twice over, from opposite directions: 0% false-positive
rate under severity, and a specific structural dependency under ablation.

**Weakened.** The Ghost Scale proposal has now failed to find a working mechanism twice. E53 removes
a reconciliation that looked available — the eye-tracking disagreement turns out to have been a
comparison error rather than a missing layer, and the missing-layer hypothesis it motivated makes
the *opposite* prediction from the one that was drawn.

**Unchanged.** The human acquisition test remains the top external priority and is not simulated.
There is still no human data anywhere in this project, and no forward test.

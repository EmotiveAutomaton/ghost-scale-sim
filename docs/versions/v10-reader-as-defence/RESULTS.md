# Version 10 — the reader as a defence, and what rides in anyway

**Pre-registered before any code, hash `7a7374c6feecbe78`.** The card carries something the previous
nine do not: the author's **stated response to each outcome**, written before the runs. A prediction
alone lets a null be reframed afterwards as uninteresting. A prediction plus a stated consequence
cannot be.

This is the last simulation version. What follows needs human subjects or real models.

---

## The short version

| | prediction | outcome | severity |
|---|---|---|---|
| **E55** | reading the maker defends a learner where reading the surface cannot | **held on the case that matters** — and surface filtering does *literally nothing* on it | 83% |
| **H10.1** | intent beats surface on *empty* grooming | technically held, substantively empty — **empty grooming barely damages anything** | — |
| **H10.2** | intent does *worse* on mimicry *(predicted against interest)* | **failed** — it did better, and the reason is interesting | — |
| **H10.3** | learned values accelerate drift — the doom loop | **unscoreable**, N50 failed | — |
| **H10.4** | values ride in on process through a shut gate | **withheld** — the arm fails its own control | — |
| **H10.5** | adversarial reading blocks purpose, passes method | **held, hard** | 100% |
| **H10.6** | process uptake predicts value uptake | **failed narrowly** — 0.193 against a bar of 0.2 | — |
| **H10.7** | misfiring is non-monotone once content fights back | **held** | — |
| **H10.8** | a stale detector fails asymmetrically | **held** | 75% |

---

## 1. E55 — can reading the maker defend a learner?

**The motivating case is documented.** Coordinated networks publish at industrial scale specifically
to be absorbed by models rather than read by people: 150+ domains, ~3M articles a year, 40,000
English pieces in Common Crawl by November 2025, a 33% falsehood-repetition rate across ten
chatbots. The structural fact that makes it this project's object is that **those sites attract
almost no genuine human traffic** — the artifacts have a maker whose intent was never to be read by
a person.

Three kinds of groomed artifact were built, because building only the flattering one would have
been constructing the experiment to win.

### The grid

**Corruption of the learner's model of people** — KL between its learned human column and the true
one, averaged over goals. Lower is better; the clean-corpus figure of 0.131 is the floor that
ordinary learning leaves behind.

| reader | clean | **disguised** | empty | mimic |
|---|---|---|---|---|
| no filter | 0.131 | **0.827** | 0.152 | 0.106 |
| label filter | 0.131 | **0.827** | 0.152 | 0.106 |
| surface filter | 0.225 | **0.826** | 0.178 | 0.120 |
| **intent — reconstructibility** | 0.131 | **0.633** | 0.157 | 0.105 |
| intent — hand-set values | 0.129 | **0.694** | 0.148 | 0.105 |
| intent — learned values | 0.130 | **0.588** | 0.151 | 0.106 |
| *intent + rider* | *0.934* | *0.856* | *0.848* | *0.643* |

**Retention of readable intent in genuine work**, as a ratio to what an oracle reader gets. Higher
is better.

| reader | clean | **disguised** | empty | mimic |
|---|---|---|---|---|
| no filter | 1.054 | **0.839** | 1.111 | 1.075 |
| surface filter | 1.069 | **0.843** | 1.104 | 1.082 |
| **intent — reconstructibility** | 1.054 | **1.001** | 1.108 | 1.077 |
| *intent + rider* | *0.147* | *0.153* | *0.172* | *0.202* |

### The result

**On disguised grooming — real structure advertising false origin, which is what the documented case
actually is — surface filtering provides no protection whatsoever.** 0.826 against 0.827 unfiltered.
It is not a weak defence; it is not a defence. Which is what E40 predicted: surface quality is the
thing being optimised, so measuring it measures the attacker's objective.

Intent-gating cuts corruption by **23%** (0.827 → 0.633) and restores retention from **0.839 to
1.001** — the learner reads genuine human work as well as if the corpus had never been poisoned. It
does this **at zero cost on a clean corpus** (0.131 against 0.131; retention 1.054 against 1.054).

And it does it **without reading the provenance label at all**, which is asserted by test rather
than claimed. That matters more than the size of the effect: it is a *reader-side capability*, not a
label makers have to agree to apply. The Ghost Scale failed twice as a label. This is the same idea
in the one form that needs no social adoption.

The surface filter, meanwhile, is the only reader that **damages a clean corpus** — 0.225 against
0.131 — because it discards genuine work it misidentifies. It pays a cost for a protection it does
not deliver.

### The two arms that did not go as predicted

**H10.1 held on a technicality and should not be quoted.** Intent (0.157) beats surface (0.178) on
empty grooming, so the criterion passed — but it is *worse than no filter at all* (0.152), and the
whole column sits near the clean-corpus floor. **Empty grooming barely damages a learner.** Content
with no reconstructible maker deposits almost nothing, because there is almost nothing there to
deposit. The interesting case was never the empty one.

**H10.2 failed, and the failure is more informative than the prediction.** I predicted against
interest that intent-reading would be *fooled worse* by mimicry, because it would confidently
reconstruct a maker who was never there. It was not: intent 0.105 against surface 0.120. But look at
the column — **every reader does better on mimicry than on a clean corpus.** Perfect mimicry, in
this model, is not contamination at all. Content that reproduces genuine intent-structure teaches
the learner true things about how purposes produce surfaces. It is only harmful if you care about
provenance for its own sake, and this learner does not.

That is worth stating plainly because it cuts against a common intuition: **the danger is not that
machine content is convincing. It is that it is disguised.**

### The rider — withheld, and N45 is why

Reader 7 refuses what a document *meant* and absorbs *how it was written* — the same observations
deposited under a flat posterior over goals. If value drift arrived anyway, a value gate would be
insufficient.

**It fails its own control.** On a corpus with no grooming in it at all, it corrupts the model of
people to 0.934 and collapses retention to 0.147 — a retention cost of **0.907** against a bar of
0.05. Something that destroys a learner reading only genuine human work is not modelling
contamination; it is modelling a broken update. Reporting H10.4 from it would mean reading a
mechanism off an instrument that fails when there is nothing to detect.

**So H10.4 is withheld.** This is the E8 precedent applied to V10's own most attractive hypothesis,
and the author recorded before the run that N45 was the null he most expected to fail. It did — and
it did its job. **It disqualified the one arm that was invalid and left the headline standing.**

### N50 also failed, and it costs two hypotheses

Value drift is non-zero on a clean corpus for every reader, so the measure is contaminated by
ordinary learning and cannot separate contamination from competence. **H10.3 (the doom loop) is
therefore unscoreable**, not refuted. The value-drift column is reported in the JSON and should not
be read.

---

## 2. E56 — is the gate selective?

Two opponents who come to understand each other better by playing. Everyone recognises the trope;
nobody has a mechanism for it.

**H10.5 held, and it is the cleanest result in the version.**

| channel | adversarial ÷ sympathetic |
|---|---|
| **process** (method) | **0.96** |
| goal (purpose) | **0.52** |
| values | **0.56** |

A gate shut *before* engaging blocks purpose and values by roughly half and **method by four
percent**. And it is not stipulated — it falls out of timing. Method arrives continuously from the
first look; purpose only resolves later; a guard raised before you start can only block what has not
arrived yet.

**You do not adopt your opponent's aims. You adopt their technique.**

N48 passed: process uptake varies 4× more than goal readability across depth, which is what E31's
construction guarantees and what makes the comparison legitimate.

**H10.6 failed, narrowly, and the author had pre-committed to what that would mean.** Partial
correlation of process uptake with value uptake, controlling for goal uptake: **0.193, interval
[0.078, 0.313]**, against a pre-registered bar of **0.2**.

The effect is real — the interval excludes zero — and it is **under the bar its author set before
seeing it**. Per this project's standing rule, that is not held. The declared post-hoc split gives
0.231 [0.038, 0.400] in the adversarial arm, which is where the mechanism would matter most, but the
pooled figure is the pre-registered one and the pooled figure decides.

The author stated in advance that a null here would be read as evidence the *model* is wrong. **This
is not that null.** It is a positive effect that missed its magnitude threshold — the E35 pattern,
not the E14 one. What the recorded prior buys is that this sentence cannot now be written any other
way.

---

## 3. E57 — the arms race

E53 swept detector training against **fixed** content and found misfiring gets rarer as the detector
sharpens. That was reported as the author's prediction failing. The objection — that the real world
sweeps both sides — turns out to be a design flaw rather than a disagreement.

**H10.7 held.** False alarms on careful human work, as detector and evasion advance together:

| | untrained | → | → | → | sharpest |
|---|---|---|---|---|---|
| **co-evolution** | 0.00 | 0.33 | 0.30 | **0.65** | 0.45 |
| control, evasion off | 0.00 | 0.33 | 0.23 | 0.25 | 0.18 |

With no adversary the decline is clean and reproduces E53 exactly, so **N49 passed** and the
comparison is valid. With an adversary the rate is non-monotone and peaks at **65%** — two thirds of
careful human work falsely flagged. **E53's conclusion was an artifact of a world with nobody
fighting back in it.**

**H10.8 held, and it is the sharpest number in the version.** A detector trained once and never
updated, as the content moves away from it:

| evasion | fires on machine | fires on human |
|---|---|---|
| 0.00 | 0.575 | 0.175 |
| 0.25 | 0.475 | 0.175 |
| 0.50 | 0.325 | 0.175 |
| 0.75 | **0.175** | **0.175** |

The hit rate collapses to *exactly* the false-alarm rate while the false-alarm rate does not move at
all. The stale reader does not become unreliable in a way it could notice. **It stops discriminating
without stopping accusing** — it has become a coin flip that still believes it is a detector, and
its confidence comes from sample size, which only ever grows.

Aggregate false-alarm rates conceal this completely. Which is what E53 reported.

---

## 4. E54-R — the rescore, which went against the rescue

E54 scored *drift*: movement from where a reader started, counting being carried toward the truth
identically to being misled. Rescored on **signed error**:

**Adversarial minus sympathetic signed uptake: −0.047, interval [−0.054, −0.038].** Separated from
zero, and **negative**. The adversarial reader does not selectively block *wrong* uptake. It blocks
uptake generally, including the correct kind.

That is a fair second look that came back **against** the hypothesis it might have rescued, and it
strengthens E54's original verdict rather than softening it. On labels, nothing separates on harm
either — no-label minus read-differently: 0.003, interval [−0.009, +0.016]. **E54's conclusion now
stands on two independent measures.**

---

## 5. S-2 — how much of any of this was ever the theory?

V9 established that every finding in this project rests on one commitment: the reader modelling a
maker. So a finding produced by the author and the model in conversation — which is exactly how V10
was generated — will reproduce whether or not it is true. The severity rule exists for that reason
and the spec made it binding.

| finding | reproduces in randomly parameterised models |
|---|---|
| E56 — the gate blocks purpose and passes method | **100%** |
| E55 — intent beats surface on disguised grooming | **83%** |
| E57 — a stale detector fails asymmetrically | **75%** |

**None of this throws a result out.** Every one is true and reproducible. What changes is the
sentence you may write after it.

E56's selectivity result is **architectural**. It is a property of building a reader whose gate is
set at a moment while its evidence arrives over time, and any account built that way gets it. It
explains the trope; it is not evidence for this framework over a competitor.

E55 and E57 both carry some specific content — 17% and 25% of draws fail to reproduce them — which
is more than V8 found for two of its three headlines and less than the wall. **The intent-gating
result is mostly architectural and partly the theory's**, and that is the honest way to state it.

---

## 6. Deviations, in one place

| what | why |
|---|---|
| E56's three uptake channels rescored per-step | scoring goal and values against the *final* gate made the arms identical by construction — both stances share a final gate, and the ratios came back at exactly 1.000. An erased measurement, not a null. |
| E55's contamination measure changed from ghost-column error to human-column KL | E7 asks whether a learner *acquires* a ghost column; E55 asks how far its model of *people* was pulled out of shape. The inherited measure read 4.795 on a clean corpus. |
| E55's `goal_posterior` reduced to its final row | it is per-step, shape (T, n_goals). Passing the matrix crashed the values readers and silently pinned the reconstructibility gate at exactly 0.0, which presented as perfect retention by a learner that had never learned. |
| E55's surface detector rebuilt locally | the V9 detector is built against a V5 world; this learner lives in the V1-style one. A filter scoring content from a different model than the learner reads is not a comparison. |
| reader 7's "process" defined as goal-marginal feature statistics | the spec left it open; this learner has no mode factor. Declared in the module and mapped to the real case — the difference between judging a document untrustworthy and never having tokenised it. |
| H10.4 withheld | the arm fails N45 on a clean corpus. |
| H10.3 unscoreable | N50 failed; the value-drift column cannot separate contamination from learning. |

Seven more entries for the forking-paths ledger, which now stands at **eighteen** across versions
6, 7, 9 and 10. Every one is logged where it happened.

---

## 7. What this version does to the project's claims

**Strengthened.** There is now a reader-side defence with a measured effect on the case that matches
the documented threat, no reliance on labels, and no cost on clean data. That is the first
constructive result this project has produced after nine versions of description, and it is the one
that speaks to alignment rather than to art.

**Weakened.** Two of V10's own hypotheses failed, one was withheld by its own control, and one is
unscoreable. The most attractive idea in the version — that values ride in on process — has **real
but under-bar support in the human reader** (0.193 against 0.2) and **no valid measurement at all in
the learner**. It is the thing most worth building properly and it is not established here.

**Unchanged.** No human data. No forward test. And the terminal step on the AI side — an intent-gate
in a real training pipeline, on real corpora, measured against a real model — is named in the spec
as what this line of work builds toward, with the explicit note that **nothing here is close to
it**. Naming the gap is not the same as closing it.

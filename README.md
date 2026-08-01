# Ghost Scale Simulation

![The Ghost Scale, drawn by a human and rendered by a machine](figures/ghost_scale_pair.png)

*Left: 100% intent. Right: 60%. Same figure, same information.*

A working model of how people work out what someone was trying to do when they made something.

> **◐ Curator — 60% down to [What this is](#what-this-is). ○ Ghost — 5% after that.**

---

## Where to start

| you want | read | how long |
|---|---|---|
| **the whole argument, in pictures** | **[WALKTHROUGH.md](WALKTHROUGH.md)** | 5 minutes |
| every question and its current answer | [FINDINGS.md](FINDINGS.md) | 15 minutes |
| the theory this implements, and the vocabulary | [docs/theory/](docs/theory/) | 10 minutes |
| how the record got this way — six versions, four audit passes | [docs/HISTORY.md](docs/HISTORY.md) | 15 minutes |
| the limits on any specific number | [DIAGNOSTICS.md](DIAGNOSTICS.md) | read before quoting |
| everything else | [docs/README.md](docs/README.md) — the map | |

The rest of this page is the public framing: what was asked, what came back, and what not to
believe about it.

---

## Contents

**[The results table](#what-was-tested-and-what-came-back)** — start here.

| | |
|---|---|
| [What this is](#what-this-is) | the model in a paragraph, and the five audit passes |
| [What was tested, and what came back](#what-was-tested-and-what-came-back) | **every experiment and its current answer** |
| [A claim this repository withdrew](#a-claim-this-repository-withdrew) | and what later work established in its place |
| [The distinction everyone gets wrong](#the-distinction-everyone-gets-wrong) | four things that get confused constantly |
| [How this was kept honest](#how-this-was-kept-honest) | pre-registration, nulls, and what the checks found |
| [Scope and limits](#scope-and-limits) | read before quoting anything |
| [Install and run](#install-and-run) | |
| [How the model works](#how-the-model-works) | the one empirical commitment, and the two solvers |
| [Relationship to existing work](#relationship-to-existing-work) | the honest crosswalk |
| [Repository layout](#repository-layout) | |

---

## What was tested, and what came back

**Numbered in the order they were run. Listed in the order they matter to the theory.** Those are
two different orderings and conflating them turns a results table into a diary. A low number means
early, not important.

Full provenance for any row: [FINDINGS.md](FINDINGS.md). The visual version:
[WALKTHROUGH.md](WALKTHROUGH.md).

### The load-bearing results

If these are wrong, the argument is wrong.

| # | The question | Where it stands |
|---|---|---|
| E2 | What does a false claim of authorship do to a reader? | **Every reader becomes certain and no two agree.** Told the truth about the same object they are appropriately unsure. On a signed measure the lie carries readers **away** from the maker's intent — four times further wrong than the truth carries them right. |
| E36 | Does working out what someone was for unlock how they did it? | **Yes.** Within a single reading, a reader picks up 2.6× more of the maker's method *after* it settles on what the work was for. Intent is the key; the method is what it opens. |
| E31 · E30 | Does compressed intent transmit? | **Yes — as method.** Depth moves how much of the maker's *process* transfers (ρ 0.93) and provably cannot move how much of the *purpose* does, because the construction holds the purpose equally readable at every depth. Five versions measured the purpose and found nothing. |
| E45 | What does modelling a maker actually buy? | **Everything, on the axes E21 never tested.** A reader that simulates needs **4 examples** where a counting reader needs **512** — and reads an intent it has *never seen* at 0.77 where the counter sits near chance and more data does not help. |
| E20 | Where on the readability axis does appreciation break? | **In the middle, around a tenth readable** — not at the empty end. Enough familiar structure to make an explanation seem available, not enough to make it right. |
| E19 | Do readers disengage from intentless work, or keep paying? | **They keep paying.** Structure the reader cannot parse holds attention indefinitely, because every look keeps promising an answer that never arrives. This inverted the framework's own prediction. |
| D-1 | What is the trust exploit made of? | Two streams of evidence about origin arrive every glance and, on a lie, disagree. Which wins is an inequality with a crossover at **0.54**. Every claim of the form *a label does X* is really *a label trusted above 0.54 does X*. |
| E41 | Does trust exploit you by fooling you, or by lowering your guard? | **Different mechanisms, and the published theory uses the one the code never had.** The paper's version predicts a reader *told the truth, believing it, and absorbing the work anyway*. The code's version structurally cannot produce that. |
| E46 | Can you look at something, reject it, and be unchanged? | **No, and this is why propaganda works.** A gate that shuts completely is a missing term — deciding you disagree means working out what was said, and that means partly running it. Rejected material still drifts a reader, and **the reader who studies it carefully to refute it drifts 7× more than the one who skims.** |
| R-8b | Can a reader learn that a source lies? | **Not if it trusts labels enough** — at any number of encounters. Detecting a lie means noticing label and work disagree, and past the crossover the label wins before the disagreement registers. |

### The generative crash, and what kind of failure it is

| # | The question | Where it stands |
|---|---|---|
| E37 | Is the wall a missing vocabulary, or a missing inversion? | **A missing inversion.** Familiar material whose maker cannot be reconstructed produces **legible and empty** — the complaint people actually make, and not "I cannot parse this". |
| E32 | Is unreadable content the same as an unskilled reader? | **Opposites.** At an identical deficit the unskilled reader quits and feels settled; the expert facing unreadable content keeps working and stays lost. **The second dimension is whether you can tell you are failing.** |
| E1 | Do readers give up on work made with no intent behind it? | Yes, from plain cost-benefit — no built-in dislike of machines. *Holds for intent-**empty** content; under intent-**foreign**, the better description, readers do the opposite (E19).* An unexpected rider: the Ghost Scale tier meant to say *don't spend effort here* is the one readers spend most on. **The model and the design disagree, and the design is probably right.** |
| E10 | Does a reader's own expertise cap what can be recovered? | **Yes**, on a corpus with *zero* machine content anywhere. |
| E40 | What happens when the signal of depth is optimised directly? | **Readers pay more and get less** — a third failure mode. Not the crash (engaged), not the exploit (nobody lied). A reader correctly reading something built to trip its own heuristic. |
| E38 | Does understanding the machine protect you? | **Yes, by swapping a skill out.** A machine-tuned reader reads machine work perfectly and loses half its accuracy on human work. A crossover, not an upgrade. |
| E39 | Can a reader be given permission to stop? | **Not by a hypothesis about the maker** — that is redundant with what it already knows about origin. **If the Ghost Scale is to let a brain stand down, it has to act on the gate.** |

### What a contaminated corpus does over time

| # | The question | Where it stands |
|---|---|---|
| E6 · E9 | Are there two kinds of damage? | **Yes, separable.** Absorbing bad material scales with how much there is. *Failing to absorb good material does not* — fully present at zero contamination, because it is driven by walking away. |
| E7 | Can a reader learn to spot intentless work unaided? | **No.** It folds machine structure into its picture of people and loses about a third of its ability to read genuine work. Honest labels: ~6× faster. |
| E16 · E47 | How much labelling is enough? | **About a third — but only for readers who know the convention exists.** Naive readers need three-quarters and never fully get there. **The figure survives the trust-mechanism question** (E47): it does not move under either account. |
| E35 | Does the damage accumulate in the reader? | **Yes.** A reader worn down by content that holds attention and gives nothing engages far less with *a fixed human artifact it has never seen* — a 15-fold fall, monotone (ρ −0.98). |

### Reading the maker

| # | The question | Where it stands |
|---|---|---|
| E17 | Does invention scale with how little intent survives? | **Yes, graded.** State as an ordering: four tiers give three steps. |
| A1 · A2 | Is mislabelling symmetric? How miscalibrated does it make you? | **No, sharply.** Human work called machine-made is still read *accurately* — what is lost is willingness to look. Machine work called human hits 99.5% of the disagreement ceiling, with all four thousand readers in the top confidence band while performing at chance. |
| E4 | Where does invented meaning start? | A switch, not a slope: below ~one-fifth trust it largely stops. And the half that gets quoted less — even a fully sceptical reader with no label invents about one time in five. |
| E33 | Can a reader know a maker better than the maker knows themselves? | **Yes**, and the margin grows as the self-account degrades. Work by a maker driven by something it cannot see **is** measurably marked, and **no reader can tell**. The mark exists and is unreadable. |
| E43 | Does automaticity hide the work from its own author? | **Yes.** The more practised the work, the less reliably its maker can name its purpose, while the reader is unaffected. Nobody sets that — depth sets it. |
| E42 | Is appreciation the same as being willing to be changed? | **No.** A reader can study something closely, read its maker accurately, and let none of it in. |
| E3 | Does the labelling scheme pay for itself? | **Roughly halves wasted effort**, at 1–3 points of accuracy on genuinely human work. |
| E5 | Is trust in provenance just decisiveness renamed? | **No.** Only trust produces the human/machine gap. |

### The results that came back against the framework

The section a stranger should use to decide whether to trust the rest.

| # | The claim | What happened |
|---|---|---|
| E21 | Modelling the maker as a mind is required for confident invention | **False, and the framework withdrew it.** A counting classifier reproduces the signature through small-sample overfitting. *E45 later established what the maker-model does buy — evidence efficiency and reading the unseen — so the withdrawal stands and its scope is now narrower than it has been read.* |
| E15 | The competence collapse is a cliff | **A knee. The author's own claim, tested knowing it could only survive or weaken.** Width unchanged across 16× the data. Belief accuracy breaks down well before choice accuracy. |
| E14 | Readers were quitting before they worked it out | **Mine, and it died.** Forcing them to keep looking made it worse. |
| E29 | A spike in value divergence separates the gates | **The pre-registration's own prediction, and it died.** Low assumed rationality spikes too. |
| N21 | Depth is not effort | **Still open, and reported as failing.** The pre-registered contrast returns *effort can manufacture depth* under exact inference (ratio 1.26 against a bar of 3.0). On what actually transfers, depth dominates effort **97-fold** — so the *estimate* of depth is contaminated by effort while the *transfer* is not. Both reported; the original decides. |
| E20 | The collapse and the invention peak occupy one band | **Retired.** The peak is unchanged and is not in question. The co-location held under a superseded solver: exact arithmetic leaves the reader *less* uncertain at partial overlap, so the conjunction stops holding. **Removed from this README and the prediction card.** |

### The generational question, and the instrument findings

*Not results against the framework — findings about what this apparatus can and cannot measure.*

| # | The question | Where it stands |
|---|---|---|
| E8 | Does damage compound across generations? | **Withheld three times.** Its honesty check — *with zero contamination, show zero damage* — failed every time. **This is not "we found no effect"; it is "we could not measure".** The failing test stays in the suite as a visible marker. |
| E12 | Is the generational leak sampling noise? | **No** — it does not shrink across a hundredfold more material. A finding about an edge case of the instrument rather than about the theory: it tells you the relay is lossy for a structural reason, which is why E8 stays withheld. |
| E18 | Was passing one reader's estimate forward the whole problem? | **No.** Fixing that channel left the damage where it was. A second contributor exists and has not been found. |
| E13 | Are the freeze and the leak two different defects? | **One shared axis.** Notable for how it was scored: the criterion produced a usable-looking number and was *thrown away*, because it lacked a precondition it needed. |
| E11 | Is belief distance a poor proxy for harm? | **Better than predicted** — it is a decent proxy, and the two measures together explain more than either alone. A measurement result, not a hit on the theory. |
| E28 · E29 | The retired rationality construct | Kept as records of a mis-specified construct. Superseded by E41 and E42, which measure the gate directly. |
| E34 | Where does real generated content sit on the readability axis? | **Not answerable in simulation, and that is the point.** Written as a prediction card a human study can use. |


---

### Where the literature landed afterwards

A retrospective search, run *after* the simulations, to check whether any of this was already known.
**It informed no design.** It is a coherence check, not evidence.

- **The label effect (E2) is well replicated.** Several studies hold the artwork constant, change
  only the stated author, and find the same collapse in appreciation — mediated by mind-perception,
  which is the mechanism this model proposes.
- **The second kind of corpus damage (E6/E9) has strong support.** Cognitive offloading reduces
  engagement including self-monitoring; skipping effort impairs skill acquisition; users perform
  worse than never-users once the tools are removed. Almost nothing on the first kind.
- **The coverage result (E16) is complicated by the implied truth effect.** Warning-labelling *some*
  false headlines makes the unlabelled ones look truer, replicated for AI content as an implied
  authenticity effect. Same inference, opposite valence.
- **The knee (E15) has one direct hit.** Experts rating AI safety responses agreed so little that
  roughly nine-tenths of the variance in a label reflected the rater rather than the response.
- **Sustained attention (E19) has one suggestive hit.** Eye-tracking finds AI-labelled artworks
  produce more dispersed gaze. Dispersed is not disengaged — it is searching without settling.
- **The asymmetry (A1) holds in direction and reverses in consequence.** Expert artists detect AI
  images well but produce more false accusations than automated tools, and false accusation is
  socially costly. We measure damage to understanding; the world measures damage to people.
- **Nobody has asked E21's question.** Our negative is the only data point that exists.

Every number above traces to a committed CSV in [results/](results/). The visual version of this
table is [WALKTHROUGH.md](WALKTHROUGH.md); the per-row provenance is [FINDINGS.md](FINDINGS.md).

---

---

> ## ○ Ghost text — 5%
> **Everything from here down is generated, human-curated, not word-by-word written.**
> Checkable, not composed. Skim it, or hand it to a model.
> *Prompt: an extended working session; the author set every question and decision.*

---

## What this is

When you look at a piece of work you run a guess about the person behind it. Why this word, this
shot, this colour. That guess is what this project models: how it is formed, what it costs to keep
running, and what happens to it when there was no person behind the work at all.

It is a simulation of a proposed mechanism. There are no human subjects anywhere in it and nothing
here is evidence about what real people do. It is the companion code to *Art as an Algorithmic
Virus* (Zenodo DOI [`10.5281/zenodo.19407789`](https://doi.org/10.5281/zenodo.19407789)).

**All of this is exploratory and theory-derived.** Every prediction came from one prior theory. The
simulations write that theory down as code and check whether its parts fit together, so agreement
between the model and the theory is the expected outcome rather than evidence for it. That framing
is stated once, here, and then not relitigated. It is the accurate description of the work.

**A validation pass has been run over the whole body of results**, at 60 simulated readers and 16
random seeds per cell with 600 random-model draws, against criteria that were fixed and hash-locked
before it started. Five of its nine checks came back against the work. Three of those matter most,
and they are stated here rather than buried: one headline is a property of the model's
*architecture* rather than of the theory, two verdicts were produced by a shortcut in the
arithmetic and do not survive its removal, and the one result rebuilt independently from its own
description reproduced the mechanism but not the size of the effect. Details in
[VALIDATION.md](VALIDATION.md), and each affected row of the table below carries its own status.

**A second pass then checked the instruments themselves**, which is a different question: not whether
the recorded answers can be trusted, but whether the measurements can answer at all. It found four
things worth knowing before any of this is read closely. **Trust in the label cannot be measured** at
all: no data this model generates locates it, so it is a modelling choice rather than a quantity.
**Reading the label and reading the work are two competing streams of evidence** that arrive at every
glance and disagree on a lie, and which one wins is decided by an inequality with a crossover at trust
0.54, so the trust exploit is a claim about labels trusted above that rather than about labels.
**The amount a reader takes on is U-shaped in how well it read the work**, because a confidently
wrong reader has moved as far from its starting point as a correct one, which is a better explanation
of one flat result than the one on record. And **the disagreement figure cannot be read on its own**,
because confident readers who differ and unsure readers who are guessing produce the same number. All
of it is in [DIAGNOSTICS.md](DIAGNOSTICS.md).

**A third pass then repaired what could be repaired**, under one rule: every change either makes
something measurable that was not, or removes something. Four things came out of it. **The measure
of "how much a reader takes on" was a distance, so being fooled counted as much as being right**;
split into a signed measure, the false-label cell reads as strongly *negative*, which sharpens the
headline rather than softening it. **Trust is measurable after all**, over the lower half of its
range, and the earlier verdict was wrong because it was fitted to the wrong data. **A sufficiently
trusting reader can never learn that a source lies**, at any number of encounters, which is a
prediction the fixed-trust model could not make. And the three headline criteria that had no error
bars now have them: two hold, one bounds its effect near zero. All of it is in
[REPAIR.md](REPAIR.md).

**A fourth pass then checked the code against the theory it implements**, which is a different
question again and the first one nobody had asked: the three earlier passes all took the code's own
account of itself as given. Reading the published equation against the shipped code found **three
terms with no counterpart in the code**. Two were omissions — there was no way for a reader to get
tired, and no way for it to be partly engaged. The third was not: **the paper and the code explain
the trust exploit by different mechanisms**, and the paper's version predicts something the code
structurally cannot produce — a reader that is told the truth, believes it, and absorbs the work
anyway. That version also settled the project's longest-running open question, by noticing that its
criterion had been pointed at the one quantity the design holds constant. All of it is in
[RESULTS_V6.md](RESULTS_V6.md).

**A fifth pass closed what the fourth would not draw, and went back at the withdrawn claim.** Four
results had been held out of the visual walkthrough because each carried an open question. Three are
now settled and one is **retired**: the claim that the collapse and the invention peak occupy one
band does not survive exact arithmetic, and it has been removed from this page and from the
prediction card. And the experiment that made this project withdraw its central claim was attacked
on the axis it had never been tested on. It asked whether a reader *without* a model of the maker
can produce the confident, contradictory signature — it can. It never asked what the maker-model
buys. **It buys almost everything: a reader that simulates needs four examples where a counting
reader needs five hundred and twelve, and reads an intent it has never encountered where the counter
sits at chance no matter how much data it is given.** The withdrawal stands; its scope was much
narrower than it has been read. All of it is in [RESULTS_V7.md](RESULTS_V7.md).

## A claim this repository withdrew

The framework used to say that producing confident, mutually contradictory readings of empty
content requires a reader that models the maker as a mind. **That is false, and the experiment that
killed it is in here.**

A naive Bayes classifier, trained by counting on 200 examples, never representing a creator or a
purpose or an intention, reproduces the effect: within-reader certainty 0.126 and between-reader
disagreement 1.379, against the full model's 0.108 and 1.377. The mechanism turns out to be
finite-sample overfitting, not theory of mind. (E21)

Two results survive that comparison, and no baseline reproduces either. One is the label-induced
switch: the same content read as uncertain (1.335 nats of doubt) or certain (0.108) depending only
on what the label says. The other is sustained futile attention on content whose structure is real
but foreign, which only a reader that keeps *expecting* to learn something can produce.

Those are the two the framework actually uses. The withdrawn claim is marked at every point it was
made.

## The distinction everyone gets wrong

Four different things get confused constantly. They are not the same and they make different
predictions. *(In the code these are `goal_empty`, `goal_foreign`, the value gate, and the
non-invertible family; the theory's words are used here.)*

- **Intent-empty is wood grain.** Structure with nothing behind it, because nothing was deciding.
  Versions 1 to 3 modelled machine-made content this way.
- **Intent-foreign is a page in a script you cannot read.** Real intent, pursued by a real process,
  expressed in a vocabulary your reading apparatus has no entry for. Version 4 replaced
  intent-empty with this, because it is a better description of a generative model — trained on
  purposeful human output and inheriting its shape.
- **Intent-unrecoverable is every word familiar and nobody there.** The vocabulary is yours; the
  route back from the surface to a state you could occupy does not exist, because many maker-states
  produce the same surface. Version 6 added this, and it is the one that matches what people
  actually report about generated text.
- **Value divergence is a person who wants something you do not want.** A fourth thing again, not
  what any of the above describes, and the model keeps it in its own parameter.

None of the four is "the intent is repugnant". That is a fifth thing and the model does not contain
it.

**The switch from intent-empty to intent-foreign inverted a headline result.** Under intent-empty,
readers disengage: nothing is identifiable, so paying attention stops being worth it. Under
intent-foreign they do the opposite. They keep looking, keep paying, and never get anywhere. The
failure to read intent survives and gets worse. The saving of effort does not.

Both cannot be right, and which one holds depends on a fact about real machine-made content that a
simulation cannot settle: **how much human-shaped structure it actually carries.** That question is
written up as [E34](results/e34_prediction_card.csv), a prediction card a human study can use to
locate real content on the axis rather than argue about it.

---

## What was removed

Two experiments were run against a version of the model that was later found to be wrong, and are
**uninterpretable rather than embarrassing.** Reading them requires holding an assumption we now
know to be false, which makes the numbers meaningless rather than inconvenient.

- **A test of whether "how hard the maker was trying" gates uptake.** The construct was wrong. What
  gates uptake is not effort but the depth of thinking behind the work, which is a different
  quantity: a fully committed shallow effort and a master's offhand sketch sit on opposite corners.
  Superseded by the depth test in the table.
- **A test of whether three separate gates behave differently.** One of the three gates was the
  mis-specified effort construct, so the design was testing something that does not exist. Re-run
  correctly, with two gates, as the same-mechanism test in the table.

**Kept deliberately:** every result that came back against the framework. The counting-classifier
result that withdrew a claim, the knee that weakened a stronger framing, the prediction that missed
once all four variants were run rather than only the flattering one. Those are unflattering and
interpretable, which is the distinction being drawn.

## What died

Seven ideas were tested and killed, and the list is in the table above under
[the results that came back against the framework](#the-results-that-came-back-against-the-framework).
Two of them were the author's own and one was the framework's claim about its own necessity.

The fourth died to a test the author approved knowing it had two possible outcomes: his claim
survives, or his claim weakens. There was no version where it got stronger.

### How the scoreboard is counted

"Held" means the prediction was written down before the run, in a spec, a pre-registration file or
a signed-off decisions document, and the measured outcome met the criterion as stated. Not "broadly
went the right way". Anything that needed the criterion softened, the framing widened or the
outcome reinterpreted is not counted as held.

| outcome | count | which |
|---|---|---|
| held | 15 | E1, E2, E3, E4, E5, E6b, E7, E10, E16, E17, E18, E19, E20, E31, E32 |
| held in part | 6 | E6, E9, E21, E28, E29, E33 |
| did not hold | 5 | E11, E12, E14, E15, E30 |
| classification undefined | 1 | E13 |
| withheld, never passed its own control | 1 | E8 |
| not answerable in simulation | 1 | E34 |

The validation pass moves three of these. E19 and E31 held under the approximate solver and do not
hold under the exact one, and N21 reverses. The counts above are the record as it was made; the
column in the table is the record as it stands.

---

## How this was kept honest

- Readers in the model have **zero preference over provenance**. They cannot want work to be human.
  Any effect of a provenance signal has to come through inference, never through wishing. Asserted
  at every construction.
- Every headline effect has a matching **null condition** that has to come out null. There are
  twenty-nine of them, in `tests/test_nulls*.py`.
- Acceptance criteria are **pre-registered as executable code** and content-hash locked before any
  run. The written criterion and the applied criterion are the same object, so they cannot drift
  apart.
- **Every deviation is logged**, including two where a criterion was restated after seeing a
  measurement. In both cases the original criterion is retained, still computed, and reported as
  failing. [VALIDATION.md](VALIDATION.md) recomputes every one of them that the committed data
  supports and reports which verdicts would change.
- Some checks exist because the failure they guard against **actually happened** during a build and
  was caught by an assertion rather than by a result. Those are marked as such in the code.
- **The validation pass could return the unwelcome answer, and did.** Its criteria were fixed and
  hash-locked before it ran, its own restated criterion is logged in its own verdict file, and
  five of its nine checks came back against the work. Four came back clean: the robustness sweep,
  the cross-version consistency check, the seed-and-scale check, and the forward prediction, which
  is locked and not yet testable either way.

## Scope and limits

Stated generously, because the point of a limits section is to be the thing a sceptical reader
reaches for and finds already there.

- **A simulation of a mechanism, not a study of people.** No human data. Nothing here is evidence
  about what real readers do.
- **The specific numbers are properties of this model's dimensions and do not transfer. The shapes
  are the claims.** The independent reimplementation makes this concrete: rebuilt from the prose
  alone, the label effect points the same way and comes out fifteen times smaller. Quote directions
  and orderings. Do not quote multiples.
- **One headline is architecture-dependent.** Inside this model's own two long-standing design
  decisions, 64% of randomly parameterised versions reproduce the label effect, and the effect this
  model produces sits near the middle of what random ones produce. The contribution there is the
  architecture and its constructions, not the settings, and it should be read that way.
- **The central assumption is that Ghost Scale opacity can be read directly as the fraction of a
  maker's intent that survives into the work.** That is a modelling choice, it is stated in the
  open below rather than buried, and it is the line to argue with.
- **The readability axis is downstream of a design decision.** The human and machine feature blocks
  were made disjoint because no such split existed at the original feature count, so the space was
  doubled. Every claim about that axis inherits it. A different split would keep the shape of the
  claim, an interior maximum, and would move the specific location. That is why the prediction card
  is written as a location to be measured rather than a number to be trusted.
- **The labelling-coverage figure is a lower bound.** The convention-aware reader is handed the
  true coverage, which is the most generous assumption available, so a real reader can only do
  worse.
- **The trust-exploit multiple is an upper bound.** The reader in this build cannot doubt the label
  it conditioned on, so a reader who could would be harder to fool.
- **The Ghost Scale's own attention gradient is non-monotone, and in the wrong place.** The tier
  meant to signal "do not spend effort here" is rendered as the visually loudest element on the
  page, while the Curator tier's reduced contrast makes it genuinely easy to skip, and E1 finds
  Curator the most expensive tier for readers. The model and the design disagree, and the design is
  probably right about human behaviour. The 5% tier is close to a logical impossibility besides:
  prompting alone constitutes more selection than 5% implies. **This repository's own charting code
  quietly declined to use the published opacity ramp and substituted a monotone one, which is the
  argument having been made in code before it was made in prose.**
- **Version 5's construct corrections are theoretically motivated and largely untested.** They are
  more defensible than what they replace, which is not the same as being right.
- **A literature search has been run and it is retrospective.** It happened after the simulations,
  it is reported in its own column, and it informed no design. It is a coherence check. The project
  has exactly one forward test, it is written down in
  [VALIDATION.md](VALIDATION.md) as a hash-locked prediction, and the experiment it predicts has
  not been built.

## Install and run

```bash
git clone https://github.com/EmotiveAutomaton/ghost-scale-sim
cd ghost-scale-sim

# with uv (preferred) or pip
uv sync

# run everything, full scale, parallel
python run_all.py

# one experiment (writes results/eN_*.csv and figures/eN_*.png)
python -m ghostscale.experiments.e1_crash

# the validation pass (writes results/validation/, then VALIDATION.md)
python run_validation.py
python scripts/write_validation_md.py

# the diagnostics pass on the instruments (writes results/diagnostics/, then DIAGNOSTICS.md)
python run_diagnostics.py
python scripts/write_diagnostics_md.py

# the repair pass (writes results/repair/, then REPAIR.md)
python run_repair.py
python scripts/write_repair_md.py

# version 6 (writes results/v6/, then RESULTS_V6.md)
python run_v6.py
python scripts/write_results_v6.py

# version 7 (writes results/v7/, then RESULTS_V7.md)
python run_v7.py
python scripts/write_results_v7.py

# redraw every chart from the committed CSVs, without re-running anything
python scripts/make_walkthrough_plates.py
python scripts/rebuild_figures.py
python scripts/make_social_figures.py
python scripts/make_diagnostic_figures.py

# tests
pytest -q
```

## How the model works

### The one empirical commitment: opacity means recoverable intent

The four published Ghost Scale tiers, CREATOR / POLISHED / CURATOR / GHOST, are drawn at 100%, 95%,
60% and 5% opacity. The model reads that opacity directly as the fraction of the maker's intent
that survives into the work:

```
alpha = {CREATOR: 1.00, POLISHED: 0.95, CURATOR: 0.60, GHOST: 0.05}
A[0][:, tier, goal, DEEP] = alpha[tier] · sig[goal] + (1 - alpha[tier]) · noise_free_synth
```

This is the load-bearing assumption of the whole framework, so it is in the open rather than buried
in an implementation file. If you think opacity and recoverable intent are not the same quantity,
this is the line to argue with. The validation pass shows the label result survives compressing and
stretching this ramp, so what matters is the ordering rather than the exact published values.

### Machine-made content is structured, and that is the point

`noise_free_synth` is a structured, non-uniform, goal-independent distribution over features, with
entropy well below uniform, asserted in code. Machine-made artifacts in this model are not noise.
They are richly patterned. What they lack is any dependence of that pattern on a goal.

That distinction matters because the obvious strawman, "machine-made content is random noise",
would produce the same crash for the wrong reason. Null N6 exists to separate them. Both are low
information. Only the strawman is high entropy.

**And validation found the sharper version of this.** What the disagreement result actually depends
on is not that the distribution is structured but that it is *goal-symmetric*, sitting equidistant
from every goal the reader holds. Switch the symmetrisation off and the readers still become
confident, but they all become confident about the *same* goal, which is shared error rather than
invention. That was written down as a design decision in version 1 and the pass confirms it is
load-bearing.

### Depth is in the order, not in the histogram

Version 5 models how much *thinking* sits behind a work as the number of levels of the maker's
decision hierarchy that reach the surface. It is built so that a deep work and a shallow one have
**identical** feature histograms, to machine precision: a reader that counts features and ignores
their order cannot tell them apart at all. What distinguishes them is the order. A deep work moves
through stages, and a deeper one moves through them in an order that names what it is for.

That constraint is asserted at every build, and it is what stops "depth" from being a second name
for "legibility", which the model already measures elsewhere.

### What the learner already knows, and why that is a claim

The learner in version 2 does not arrive holding a correct model of what machine-made content looks
like. It has to acquire one from an unlabelled, already-contaminated corpus. What it starts with is
a deliberate theoretical position: it knows the shared goal-to-feature family, and it does not know
how provenance modulates it. Readers share a likelihood family because they share a body plan. What
they do not share, and have to learn, is which sources carry intent.

It is also forced. A genuinely uninformative prior is not a slow learner, it is an unidentifiable
one: information between features and goal sits at exactly 0.0000 nats after 400 artifacts and all
four learned columns stay bit-identical. That measurement is kept as a live test, so if it ever
stops being true the decision gets revisited instead of inherited.

### Two solvers, and why that matters

The observer's beliefs can be updated two ways. `inference.exact: false` uses pymdp's variational
solver, which keeps beliefs about different unknowns separately and updates each using an average
over the others. Every committed number in `results/` was produced that way. `inference.exact: true`
uses [`ghostscale/exact.py`](ghostscale/exact.py), which carries the belief over every combination
of unknowns at once with no independence assumption anywhere.

The shortcut is fast and it is known to have been badly wrong once. Version 5 caught it returning
the shallow answer for every artifact, confidently, while exact arithmetic on the same observations
recovered depth correctly. Switching solvers is one config flag and every experiment runs unchanged
under both, which is what made the validation pass a substitution rather than a rewrite.

### The Ψ analogue is a reimplementation, not a port

`metrics.psi_analogue` is a discrete stand-in for the closed-form Ψ of the preprint:

```
psi = [engaged] · (-ln(1 - κ)) · KL( Q(goal | τ) ‖ P0(goal) )
```

The sigmoid gate of the closed form is replaced by the binary engagement decision. This
reimplements the intuition, which is engagement-gated, trust-weighted surprise about the goal. It is
not a port of the equation and no equivalence is claimed.

## Relationship to existing work

This model fills a named gap in the active-inference literature on Theory of Mind. The honest
mapping:

| this model | corresponding construct | source |
|---|---|---|
| `creator_goal` as a hidden state the reader infers | opponent preference parameters θ_j inferred online | Albarracin et al. 2026, arXiv 2602.20936 |
| the reader's model of the maker | structurally matched other-model M_other | Albarracin et al. 2026 |
| κ as precision on the provenance channel | reliability-gating *r* between learned and static ToM prediction | Albarracin et al. 2026, Eq. 15 |
| engagement policy over attention | epistemic-value term in social EFE | Pitliya et al. 2025, arXiv 2508.00401 |
| Ghost Scale tiers (alpha as opacity) | no existing analogue | |
| a generator with no preferences at all | no existing analogue | |

Two things this model adds, stated narrowly because the value of the contribution is that it fills a
named gap, and that argument gets weaker if it claims more ground than it holds.

1. **What sets the weight.** Albarracin et al. fix the empathy parameter λ from outside the model
   and name "what sets λ" as their central open question. κ, gated on provenance and on value
   divergence, is a candidate answer: the weight you put on another agent's inferred preferences
   should scale with the evidence that those preferences exist at all.
2. **What happens when the other has no preferences.** Existing work assumes the observed other is
   an agent with preferences. This model asks what happens when it is not, and finds the inference
   does not fail gracefully. Under high trust it fabricates. (E2, E4)

## Repository layout

```
README.md                     this page
FINDINGS.md                   every question and its CURRENT answer; the one page to read
docs/HISTORY.md               how the record got that way, six versions and four passes
VALIDATION.md                 generated from results/validation/, never hand-written
DIAGNOSTICS.md                generated from results/diagnostics/, never hand-written
REPAIR.md                     generated from results/repair/, never hand-written
RESULTS_V6.md                 generated from results/v6/, never hand-written
RESULTS_V7.md                 generated from results/v7/, never hand-written
run_all.py                    the experiment programme
run_validation.py             the validation pass, V-1 through V-9
run_diagnostics.py            the diagnostics pass on the instruments, P-1, P-2 and D-1 to D-6
run_repair.py                 the repair pass, R-1 through R-13
run_v6.py                     version 6, E35 through E43, plus the retrofit
run_v7.py                     version 7, the four closures and E45 through E47

ghostscale/                   generative_model, creators, environment, observer, learning, metrics
ghostscale/exact.py           exact joint inference; the solver the validation pass substitutes in
ghostscale/fitting.py         parameter estimation by exact likelihood, for the three parameters
                              that are not hidden states and so have no posterior to read
ghostscale/v4_model.py        hypothesis-space overlap; goal-foreign content
ghostscale/v4_5_model.py      the three-gate observer
ghostscale/v5_model.py        model depth as a hierarchy the reader infers
ghostscale/latent_goal.py     the goal a maker does not know it has
ghostscale/experiments/       e1 through e34, each runnable standalone
ghostscale/prereg_*.py        acceptance criteria as executable, hash-locked code
ghostscale/validation/        the nine checks, plus their own hash-locked criteria
ghostscale/diagnostics/       the eight instrument checks, plus their own separate lock
ghostscale/repair/            the repair pass, plus a third separate lock
ghostscale/v6_model.py        the Intent Extraction Limit: depletion, the graded gate, the
                              trust-to-threshold coupling, process recovery
ghostscale/v6/                version 6's experiments, plus a fourth separate lock

tests/                        model invariants, the null suite (N1 to N21), exact-inference tests
config/default.yaml           every parameter, for every version, plus the solver switch

WALKTHROUGH.md                seventeen plates, in the order that makes the argument
docs/theory/                  the theory this implements, and the code-to-theory vocabulary
docs/HISTORY.md               six versions and four audit passes, as one narrative
docs/specs/                   the build spec each version was written against
docs/writeups/                RESULTS_V1 through RESULTS_V5: the full record, with every deviation
docs/decisions/               design decisions signed off before each build
docs/EXPERIMENTS.md           the original plain-language table, superseded by FINDINGS.md

results/                      summary CSVs and JSON verdicts (committed)
results/validation/           one verdict file per check, plus the side-by-side tables
results/diagnostics/          one verdict file per instrument check, plus the recovery sweeps
results/repair/               one verdict file per repair item, plus the matched-pair sweep in
                              which every reachable experiment was run under both code paths
results/diagnostics/          labelled diagnostic runs, kept separate so they cannot be mistaken
                              for reportable output
figures/walkthrough/          the seventeen plates WALKTHROUGH.md is built from
figures/archive/              the per-experiment research charts from versions 1 to 5
figures/social/               the five distribution slides, the PDF, the preview image
figures/diagnostics/          the recovery panels, the difficulty axis, the uptake curve
notebooks/walkthrough.ipynb   runs E1 and E2 end to end, narrated
scripts/                      chart rebuilders, the version-specific runners, the VALIDATION.md
                              generator, and the independent reimplementation of the two-gates
                              result
```

Raw per-reader CSVs are not committed, because `e4_raw.csv` alone is 16 MB. Everything a number in
this README or a chart in `figures/` depends on is committed. Regenerate the raw files with
`python run_all.py`.

Several defaults were recalibrated on contact with the implementation. Each is documented under
"Deviations" in the matching write-up, with the evidence that motivated it, so any of them can be
argued with, and [VALIDATION.md](VALIDATION.md) recomputes the originals. The load-bearing
constraints have not changed: zero reader preference over provenance, structured rather than
uniform machine-made content, reader heterogeneity, the full null suite, and the honest crosswalk
above.

## Links

- Preprint: *Art as an Algorithmic Virus*, [`10.5281/zenodo.19407789`](https://doi.org/10.5281/zenodo.19407789)
- Plain-language essay: <https://abrahamhaskins.org/art>
- Ghost Scale Figma kit: <https://www.figma.com/community/file/1624141586132218953>

## License and citation

Code is MIT. Prose, figures and data are CC BY 4.0. See [LICENSE](LICENSE).

To cite this repository, see [CITATION.cff](CITATION.cff), or cite the preprint directly.

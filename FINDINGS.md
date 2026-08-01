# Every question this project asked, and where its answer stands today

**This is the one page to read if you want the current state.** It is a synthesis across six model
versions and four audit passes, so unlike the generated pass documents it is written by hand. Every
number in it traces to a verdict file named in the row, and where a number has been superseded the
current one leads and the earlier one is named.

If you want *how the record got this way*, read [docs/HISTORY.md](docs/HISTORY.md).
If you want a specific number's full provenance, each row names the document that owns it.

---

## The shape of the whole thing, in one paragraph

A reader looks at something made and tries to work out what the maker was for. This project models
that as an inference under a metabolic budget: reading intent costs something, the reader decides
each moment whether another look is worth it, and the interesting failures are failures of that
cost-benefit calculation rather than failures of perception. **The reader has no preference over
provenance** — it cannot want the work to be human — so every effect of a provenance label has to
arrive through inference or not at all. That constraint is asserted at every construction and it is
the single thing that keeps the results from being circular.

---

## What the project is confident about

**1. A false provenance label does not merely move a reader — it moves them away from the truth.**
Machine-made work labelled honestly leaves a reader roughly where it started. The same work passed
off as human moves the reader a long way, and *in the wrong direction*: it scores strongly negative
on a measure of how much closer to the answer the reader got, while honest human work scores
strongly positive. Opposite signs, not different magnitudes. **This is the project's central
result.** It survives exact arithmetic, a disjoint seed block, double scale, and 22 of 27 robustness
cells. (REPAIR.md, VALIDATION.md)

*What it does not survive: being quoted as a size.* Rebuilt from scratch by someone reading only
the prose, the effect points the same way and comes out fifteen times smaller. Quote the direction.

**2. Confident invention peaks in the middle of the readability axis, not at the empty end.**
Content that is *about a tenth* readable produces the most confident, most mutually contradictory
readings — enough familiar structure to make an explanation seem available, not enough to make it
right. This is the most robust result in the project: same location under exact inference, in all
seventeen robustness cells, on a disjoint seed block, at double scale, and in 100% of bootstrap
draws. (RESULTS_V4_5, VALIDATION.md, REPAIR.md)

**3. Unreadable content and an unskilled reader are opposites, not degrees of the same thing.**
At an identical information deficit the unskilled reader quits almost immediately and feels
settled; the expert facing unreadable content keeps working and stays lost. The second dimension is
**whether you can tell you are failing** — a badly aimed template fails silently, out-of-range
content fails loudly. (RESULTS_V5, VALIDATION.md)

**4. Readers do not disengage from machine-made content. They keep paying.**
This inverted the framework's original prediction and the inversion is the finding. Content with
real structure the reader cannot parse holds attention indefinitely, because every look keeps
promising an answer that never arrives. (RESULTS_V4, REPAIR.md)

**5. A sufficiently trusting reader cannot learn that a source lies — at any number of
encounters.** Not slow learning: learning that cannot start. Noticing a lie means noticing that the
label and the work disagree, and above a computable threshold the label has already won that
argument before the disagreement can register. (REPAIR.md)

---

## What the project withdrew

**Modelling the maker as a mind is not necessary for confident, mutually contradictory readings of
empty content.** A naive counting classifier — trained by tallying 200 examples, never representing
a creator or a purpose — reproduces the pattern, through nothing more than small-sample
overfitting. The framework used to claim this required theory of mind. It does not, and the
experiment that killed the claim is in this repository. (E21, RESULTS_V4_5)

What the counting classifier *cannot* do is respond to a label, or keep paying attention to
something it cannot resolve. Those two need the full machinery, and those are the two the framework
now rests on.

---

## Every experiment

Grouped by what it is about. **"Where it stands"** is the current reading; **"was"** names the
earlier reading where it changed.

### The metabolic core — does reading intent cost something, and do readers stop paying?

| # | The question | Where it stands |
|---|---|---|
| E1 | Does a reader give up on something made with no purpose behind it? | **Yes**, and out of ordinary cost-benefit reasoning with no built-in dislike of machines. An unexpected finding rides along: the Ghost Scale tier meant to say *don't spend effort here* is the one readers spend most on. The model and the published design disagree, and the design is probably right about people. |
| E3 | Does the labelling scheme pay for itself? | **Roughly halves wasted effort**, at a cost of one to three points of accuracy on genuinely human work, because a label-aware reader occasionally walks away from something real. |
| E5 | Is trust in provenance just general decisiveness renamed? | **No.** Trust changes the *gap* between how a reader treats human and machine work. Decisiveness only moves the overall level and never produces the gap. |
| E14 | Were readers quitting before they worked it out? | **No — my hypothesis, and it died.** Forcing them to keep looking made things worse. |

### The label — what happens when you lie about who made something

| # | The question | Where it stands |
|---|---|---|
| E2 | If you lie about who made something, what happens to the reader? | **Every reader becomes confident and no two agree.** Told the truth about the same object they become appropriately unsure. Four qualifications, all real: the *confident* half is architectural (a randomly parameterised reader of this shape does it too), it is a claim about readers who trust labels above a threshold rather than about labels in general, the disagreement figure cannot be quoted alone, and on a signed measure the lie moves readers *away* from the answer. |
| E17 | Does invention scale with how hollow the content is? | **Yes, graded.** Doubt under a human claim rises as transmitted intent falls. State it as an ordering, not a slope: four tiers give three steps and two of them sit within 5% of each other. |
| E4 | Where exactly does invented meaning happen? | **A switch, not a slope** — below roughly one-fifth trust, invention largely stops. And the half that gets quoted less: even a fully sceptical reader with no label at all invents about one time in five. *Caveat: this experiment reports trust multiplied by a belief distance, and the trust factor alone varies forty-fold across the sweep.* |
| A1 | Is mislabelling symmetric? | **No, and the asymmetry is sharp.** Human work called machine-made is still read *accurately* — what is lost is the willingness to look at all. Machine work called human produces disagreement at 99.5% of the theoretical ceiling. One costs attention, the other costs the model. |
| A2 | How miscalibrated does a false label make you? | **Unanimously.** All four thousand readers landed in the highest confidence band while performing at chance. Not a bad tail. |

### The corpus — what a contaminated stream does to a reader over time

| # | The question | Where it stands |
|---|---|---|
| E6 / E6b | Are there two different kinds of damage? | **Yes, separable.** Absorbing bad material scales with how much there is. *Failing to absorb good material does not* — it is fully present at zero contamination, because it is driven by walking away rather than by what is in the pile. This has the strongest independent support of anything in the project. |
| E7 | Can you learn to spot hollow content without being told? | **No.** The learner folds machine structure into its picture of humans and loses about a third of its ability to read genuine work. With honest labels it builds a clean picture roughly six times faster. **Now validated under exact inference**, which it could not be until the repair pass built an exact learning path. |
| E9 | Poisoning versus starvation | **Separable**, and the starvation signature is present at zero contamination. Also now validated. |
| E16 | How much labelling is enough? | **About a third** — but only for readers who know the convention exists. Readers who do not need three-quarters and never build a reliable picture at any coverage. **A lower bound by construction:** the convention-aware reader is handed the true coverage, the most generous assumption available. |

### The generational question — and the thing that could never be measured

| # | The question | Where it stands |
|---|---|---|
| E8 | Does damage compound across generations? | **Withheld three times. Still withheld.** Its own honesty check — *at zero contamination, show zero damage* — failed every time. This is not "we found no effect"; it is "we could not measure". The failing test stays in the suite as a visible marker so any future fix has to switch it off deliberately. |
| E12 | Is the leak just sampling noise? | **The framework's own claim, and it died.** The leak does not shrink across a hundredfold more material, and under the repaired model it slightly grows. |
| E18 | Was passing one reader's estimate forward the whole problem? | **No.** Fixing that channel left the damage where it was. A second contributor exists and has not been found. |
| E13 | Are the freeze and the leak two different defects? | **One shared axis.** Notable for how it was scored: the criterion written to classify it produced a usable-looking number and was *thrown away*, because it needed a precondition it did not have. |

### The reader's own limits

| # | The question | Where it stands |
|---|---|---|
| E10 | Does a reader's own skill cap what can be extracted? | **Yes**, measured on a corpus with *zero* machine content anywhere — hold the material perfectly constant, vary only the reader, and extraction collapses. |
| E15 | Is that collapse a cliff or a knee? | **A knee — and this was the author's own claim, tested knowing it could only survive or weaken.** A real cliff sharpens as you add evidence; this one did not budge across sixteen times the data. The sharper secondary finding: belief accuracy breaks down well before choice accuracy does. A rater's internal picture rots while their picks stay right. |
| E11 | Is belief distance a poor proxy for actual harm? | **No — the prediction failed and the truth is more useful.** It is a decent proxy. |

### The readability axis — the strongest part of the project

| # | The question | Where it stands |
|---|---|---|
| E19 | Do readers disengage from machine content, or keep paying? | **They keep paying.** The generous-fallback question attached to this has the most convoluted history in the record: the original finding was reduced to inconclusive by the validation pass, then restored by the repair pass once the reason was traced — the experiment's own control required the reader to keep paying attention *after* correctly resolving the goal, which a rational reader never does. **Restored on stronger footing than it originally had.** |
| E20 | Where along the readability axis does it break? | **In the middle, at about a tenth readable.** The single most robust result here. *One thing did move:* under the repaired model the reader at that point ends up *less* uncertain, so the "crash" signature no longer co-locates with the invention peak. The peak is unchanged; the co-location claim needs revising. |
| E32 | Is unreadable content the same as an unskilled reader? | **No, opposites.** See finding 3 above. Survives exact inference on all five measures. |
| E21 | Is a model of the maker's mind necessary? | **Partly, and the unwelcome half comes first.** See "What the project withdrew". |

### Depth — how much thinking sits behind the work

*The weakest area for five versions, and the one version 6 changed most.*

| # | The question | Where it stands |
|---|---|---|
| E30 | Does how much thought went in change how much you take away? | **Yes, on the maker's *method*. It provably cannot on the maker's *purpose*, and that is what was being measured.** Depth is built so the purpose is equally readable at every level — that is the commitment that stops "depth" being "legibility" renamed — so the pre-registered measure could not have moved whatever was true. Scored on how much of the maker's execution chain the reader recovered, depth moves it and the interval excludes zero. **Was: an unexplained null, later bounded near zero.** (RESULTS_V6) |
| E31 | Are the collapse and the trust exploit the same mechanism? | **Yes, and this is now settled after being the project's longest-running open question.** The mechanism — a dishonest label inflating the reader's estimate of the thinking behind the work — held under everything. The *common-path* claim did not: it scored 0.89 under the approximate solver and 0.60 under exact arithmetic, twice, by two independent routes. Re-scored on process uptake it comes back at **0.83**, above the original bar. The criterion was pointed at the one quantity the design holds flat. **Was: unresolved, with two solvers disagreeing.** (RESULTS_V6) |
| N21 | Is "depth" just "effort" wearing a hat? | **The readability of depth is established; the dissociation is a construction commitment.** With the effort axis pinned so no "offhand but deep" corner exists, depth still separates. But the effort parameter was rebuilt to make that corner representable *before* it was measured, and the pass condition was rewritten after the first version failed. Both are logged and the original is retained and reported as failing. |
| E28 / E29 | The retired rationality construct, and gate dissociation | **Kept as records of a mis-specified construct**, not extended. Superseded by E41 and E42, which measure the gate directly. |
| E33 | Can a reader know a maker better than the maker knows themselves? | **Yes**, and the margin grows as the maker's self-account degrades. *Scope: the reader is told how unreliable the report is.* The self-blindness half splits: work by a maker driven by something it cannot see **is** measurably marked on the object, and **no reader in this model can tell** — the readings differ in the fourth decimal place. The mark exists and is unreadable. |

### Version 6 — checking the code against the equation rather than against its own results

*Three audit passes had checked the results. None could have found these, because all three took
the code's own account of itself as given.*

| # | The question | Where it stands |
|---|---|---|
| E41 | The paper and the code explain the trust exploit differently. Do they predict the same thing? | **No, and the difference is large.** The paper says trust switches off the alarm that would stop you absorbing something; the code says the label out-argues the work. The first predicts an exploit on a reader **told the truth and believing it** — which the code structurally cannot produce. Run on the original label cells: a reader told the truth integrates almost nothing under the code's mechanism and about half under the paper's. |
| E35 | Does the damage accumulate in the reader and carry to work it has never seen? | **Direction yes, magnitude unstable.** A reader worn down by content that holds attention and gives nothing back engages far less with a *fixed human artifact it has never encountered*. The mechanism reproduces on all three seed blocks tested; the pre-registered magnitude threshold is met on one of three, because it is an absolute threshold on a quantity whose baseline varies two-fold. **The direction is the claim.** |
| E36 | Does the reader recover the maker's method, and does knowing the goal unlock it? | **Both yes**, and the second only in its temporal form. Between readers, getting the goal right does not predict getting the method. Within a single reading, process recovery roughly doubles *after* the goal settles. The pre-registered between-reader test is reported as failing. |
| E37 | Is the wall in front of generated content a vocabulary deficit, or a missing inversion? | **A missing inversion, and it is a distinct failure.** Content on *familiar* features whose maker cannot be reconstructed produces a signature neither existing condition does: **legible and empty**. Which is the complaint people actually make, and is not "I cannot parse this". |
| E38 | Does AI literacy stack with art literacy, or replace it? | **It replaces it.** A reader whose expectations match the machine reads machine work perfectly and loses half its accuracy on human work. A crossover, not an upgrade. |
| E39 | Does a reader that can conclude "there is no maker here" stop cleanly? | **No — it redirects rather than relaxes.** It cuts invention by about 38% and produces no resolution, and the reason is structural: a no-maker hypothesis about the *goal* is redundant with what the reader already knows about *provenance*. If the Ghost Scale is meant to let a brain relax, the relaxation cannot come from a hypothesis about the maker. |
| E40 | How do surface appeal and social endorsement combine, and what happens when appeal is optimised? | **Additively**, as predicted. And optimising the surface cue directly produces a **third failure mode**: the reader pays more and gets less. Not the crash, not the exploit. |
| E42 | Is looking deeply the same as being willing to be changed? | **No, and the model already kept them apart.** There is a stable regime with high engagement and a closed gate: a reader that looks intently, reads the maker accurately, and integrates nothing. |
| E43 | Does the maker lose access to its own reasons as the work deepens? | **Yes.** Practice compresses decisions, and compression is what makes a decision unavailable for report — while the reader is unaffected. |

### Version 9 — what each finding is made of, and two attempts to reconcile the literature

*The complement of the severity pass. Severity asks how much of a result is architectural; this asks
**which part** of the architecture. Keep the settings, strip the shape, remove one structural
commitment at a time.*

| # | The question | Where it stands |
|---|---|---|
| MIN | Which structural commitment is each finding actually made of? | **All of them rest on one.** No surviving finding outlives replacing the maker-modelling reader with a surface classifier. Hierarchy and costly attention are free — no finding needs them. And **the wall is the only finding that needs the reader to hold a distribution** rather than a best guess, which is exactly right, because the finding *is* a claim about the shape of a posterior. |
| E53 | Have readers learned a surface signature of generated work that fires before intent-reading? | **Yes, and the prediction drawn from it was backwards.** A learned heuristic discriminates (0.63 against 0.23) and never stops misfiring. But sharpening it makes misfiring **rarer**, not commoner. The eye-tracking result this was built to reconcile turns out not to have been in conflict: the model already reads machine work less than human work, and the two were being scored against each other in error. |
| E54 | Is there a stance where the gate shuts *before* engaging — and is that what the Ghost Scale should trigger? | **The stance is real; the affordance is not.** A pre-shut gate protects where a reactively-shut one does not (+0.015, interval [0.011, 0.019]) at engagement matched by construction. It protects by about 6%, and when applied to only the marked half of a stream **neither label separates from no label at all.** |

### Not answerable in simulation

| # | The question | Where it stands |
|---|---|---|
| E34 | Where does real machine-made content sit on the readability axis? | **A prediction card, not a result.** Everything about disengagement depends on how much human-shaped structure real generated output actually carries, and a simulation cannot settle that. Written as a set of signatures a human study can use to locate real content rather than an argument to have. |

---

## Where the theory should be updated

**The security argument's mechanism.** The framework reaches self-policing disclosure through
Zahavian signalling — honesty is stable because the honest signal is *wasteful*. Signalling theory
has since moved to a **trade-off** account: honesty holds where deception is costly relative to what
it gains. E51 produced a detection-rate threshold, which is a trade-off result, without being aimed
at one. **The conclusion survives; the stated justification should be updated.** See
[EVIDENCE.md](EVIDENCE.md).

---

## What is still open

1. **N21 remains split.** The pre-registered depth-versus-effort contrast fails under exact
   inference; on what actually transfers, depth dominates effort ninety-fold. The reader's
   *estimate* of depth is contaminated by effort and the *transfer* is not, and no single number
   states that.
2. **The project has no forward test.** There was one sealed prediction; its status was withdrawn
   in version 8 because the author does not recognise authoring it, and a commitment nobody
   remembers making is not a forward test. The experiment ran anyway (E52) and its primary held.
   Everything here is still a theory checked against itself, and **this is the largest single thing
   the project owes.**
3. **No human data anywhere.** The sharpest available test is the acquisition test — expose a
   reader to work above its level and measure what it can then *produce*. It needs subjects and
   money and is named as the top external priority, deliberately not simulated.
4. **The Ghost Scale has no demonstrated mechanism.** Two attempts, two failures: E39 found that a
   hypothesis about the maker buys nothing, and E54 found that the gate intervention it pointed to
   is real but too small to carry a label. The proposal is not refuted; it is unsupported.
5. **Three experiments still cannot run under exact inference** — a known, bounded gap.
6. **The headline effect's size depends on which content model you believe**, and the largest
   numbers come from the description the project itself replaced on theoretical grounds.

*The crash/peak co-location claim has been retired rather than revised: it does not survive exact
inference and has been removed from the README and the prediction card.*

---

## How to read any number here

- **A simulation of a mechanism, not a study of people.** No human data anywhere.
- **The shapes are the claims; the specific numbers are properties of this model's dimensions.**
  Quote directions and orderings. Do not quote multiples.
- **Every criterion was written down before its run**, as executable code, content-hash locked.
  Where one was changed afterwards it is logged, the original is retained and still computed, and
  it is reported as failing if it fails.
- **Four separate criteria have now been found unable to do their own job**, each caught by a later
  pass. That is the failure mode this project has most of, and each instance is documented where it
  was found.

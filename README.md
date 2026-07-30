# Ghost Scale Simulation

A working model of how people work out what someone was trying to do when they made something.

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

Three different things get confused constantly. They are not the same and they make different
predictions.

- **Goal-empty is wood grain.** Structure with no purpose behind it, because no purpose was
  involved. Versions 1 to 3 modelled machine-made content this way.
- **Goal-foreign is a page in a script you cannot read.** A real purpose, pursued by a real
  process, expressed in a vocabulary your reading apparatus has no entry for. Version 4 replaced
  goal-empty with this, because it is a better description of a generative model, which is trained
  on purposeful human output and inherits its shape.
- **Value divergence is a person who wants something you do not want.** That is a third thing
  again, it is not what either of the above describes, and the model keeps it in its own parameter.

None of the three is "the goal is repugnant". That is a fourth thing and the model does not contain
it.

**The switch from goal-empty to goal-foreign inverted a headline result.** Under goal-empty, readers
disengage from machine-made content: nothing is identifiable, so paying attention stops being worth
it. Under goal-foreign they do the opposite. They keep looking, keep paying, and never get
anywhere. The failure to read intent survives and gets worse. The saving of effort does not.

Both cannot be right, and which one holds depends on a fact about real machine-made content that a
simulation cannot settle: **how much human-shaped structure it actually carries.** That question is
written up as [E34](results/e34_prediction_card.csv), a prediction card a human study can use to
locate real content on the axis rather than argue about it.

---

## What was tested, and what came back

One table. Plain-English questions, plain-English answers, numbers inline. There is no second
technical version of it, deliberately: a technical table and a plain-English table drift apart
within three revisions, and that is the same instrument-versus-claim failure this repository has
caught seven times.

**The "found afterward" column** lists published work located *after* the simulations ran, in a
literature search done specifically to check whether any of this was already known. Nothing in that
column informed any design. It is a coherence check, not evidence.

**The "validation" column** records what the pass in [VALIDATION.md](VALIDATION.md) did to the row.
"Not individually checked" means exactly that: the pass covered five headline families and the
checks that could be run across all of them, and it did not re-run everything. A blank would read
as "fine" to anyone who did not look.

| The question | What came back | Found afterward | Validation |
|---|---|---|---|
| **Does a viewer give up on something made without a purpose behind it?** | Yes, and it falls out of ordinary cost-benefit reasoning. No special mechanism for disliking machines was needed, and none was built in. | — | Superseded. This holds under goal-empty content and inverts under goal-foreign, which is the better model. See the row on disengagement below. |
| **If you lie about who made something, what happens to the viewer?** | Every viewer becomes confident. No two agree on anything. Told the truth about the same object, they become appropriately unsure instead. | Well replicated. Several studies hold the artwork constant and change only the stated author, and find the same collapse in appreciation. Mind-perception mediates it. | **Survives exact inference** to four decimal places, and survives a different seed block and double scale. **But the confident half is architectural:** 64% of randomly parameterised models of this shape reproduce it, and the disagreement half requires a specific design decision. See below. |
| **Does the labelling scheme pay for itself?** | Roughly halves wasted effort. Costs one to three points of accuracy on genuinely human work, because a label-aware viewer occasionally walks away from something real. | — | Not individually checked. |
| **Where exactly does invented meaning happen?** | Trust has a sharp switch, not a slope: below roughly one-fifth trust in the label, invention stops. Even a fully sceptical viewer with no label at all invents about one time in five. | — | Not individually checked. Trust is one of the parameters swept in the robustness matrix, and the label result holds across it. |
| **Is trust in provenance just general decisiveness renamed?** | No. Trust changes the *gap* between how a viewer treats human and machine work. Decisiveness only moves the overall level and never produces the gap. | — | Not individually checked. |
| **Can you learn to spot hollow content without being told?** | No. The learner folds machine structure into its picture of what humans are like, and loses about a third of its ability to read genuine work. With honest labels it builds a clean picture quickly. | Model collapse is established in machine learning. No human equivalent has been measured. | Not individually checked. The exact-inference solver does not support the learning path, so this one could not be re-run under it, and that gap is real. |
| **Are there two different kinds of damage?** | Yes, separable. Absorbing bad material scales with how much of it there is. Not absorbing good material does not. It is fully present even at zero contamination, because it is driven by walking away rather than by what is in the pile. | **Strong support for the second kind.** Cognitive offloading reduces engagement including self-monitoring; skipping effort impairs skill acquisition; and users perform worse than never-users once the tools are removed. Almost nothing on the first kind. | Not individually checked. |
| **Does a viewer's own skill cap what can be extracted?** | Yes, on a corpus with zero machine content anywhere. Hold the material perfectly constant, vary only the reader, and extraction collapses. | Expertise is known to moderate aesthetic processing broadly, and artists' eye movements are measurably less driven by surface features than novices'. The threshold shape is untested. | Not individually checked, but it shares its geometry with the two-dimensions result below, which does survive exact inference. |
| **Is that collapse a cliff or a knee?** | A knee. A real cliff sharpens as you add evidence; this one did not budge across sixteen times the data. **And belief accuracy breaks down well before choice accuracy does:** a rater's internal picture rots while their picks stay right. | **One direct hit.** Experts rating AI safety responses agreed so little that roughly nine-tenths of the variance in a label reflected the rater rather than the response. Reward models trained on that learn rater habits. | Not individually checked. |
| **How much labelling is enough?** | About a third of machine content, **but only for viewers who know the labelling convention exists.** Viewers who do not know need three-quarters, and never build a reliable picture at any coverage. | **Directly relevant, and it complicates us.** The implied truth effect: warning-labelling *some* false headlines makes the unlabelled ones look truer. Replicated for AI content as an implied authenticity effect. Same inference, opposite valence. The literature calls coverage the key variable and has never produced a threshold. | Not individually checked. The number is a lower bound by construction: the convention-aware reader is handed the true coverage, which is the most generous assumption available. |
| **Does invention scale with how hollow something is?** | Yes, smoothly. Telling the truth about hollow content converts near-certainty into honest uncertainty. | — | Not individually checked, but its GHOST cell agrees with the label-effect result to four decimal places across two versions. |
| **Is mislabelling symmetric?** | No. Same confidence either way, but the disagreement differs enormously. Human work called machine-made is still read correctly. Machine work called human produces maximum disagreement. | **Direction holds, consequences reverse.** Expert artists detect AI images well but produce more false accusations than automated tools, and false accusation is socially costly. We measure damage to understanding; the world measures damage to people. | Not individually checked. |
| **How miscalibrated does a false label make you?** | Every one of four thousand viewers landed in the highest confidence band while performing at chance. Not a bad tail. Unanimous near-certainty about nothing. | — | Not individually checked. It is a restatement of the label-effect row, so it inherits that row's architecture-dependence. |
| **Does the collapse survive a more generous set of explanations?** | Yes. Adding "they were just exploring" as an available explanation absorbed exploratory *human* work convincingly and did nothing at all for machine work. It was chosen *less often than random guessing*. | — | **Does not survive exact inference.** The experiment's own positive control fails once the shortcut is removed, which makes the verdict inconclusive rather than reversed. This is the most damaging single finding of the pass and it is the reason the row above it is now the load-bearing one. |
| **Do viewers actually disengage from machine content, or keep paying?** | **They keep paying.** Content with real structure the viewer cannot parse holds attention indefinitely, because every look keeps promising an answer that never arrives. This inverts the earlier prediction. | **One suggestive hit.** Eye-tracking found AI-labelled artworks produce more dispersed gaze. Dispersed is not disengaged, it is searching without settling. | The measurement survives exact inference (attention 0.683 approximate against 0.682 exact). The *verdict* it was reported inside does not; see the row above. |
| **Where along the readability axis does it break?** | In the middle, not at the empty end. Invention peaks where the content is about a tenth readable: enough familiar structure to make an explanation seem available, not enough to make it right. **The collapse and the invention peak occupy the same narrow band**, which the framework had always treated as two separate phenomena. | — | **The strongest result in the project after validation.** The peak sits at the same place under exact inference, in all seventeen cells of the robustness sweep, on a disjoint seed block, and at double scale. Its axis is downstream of a design decision rather than a measurement, and that is stated in full in [VALIDATION.md](VALIDATION.md). |
| **Is a model of the maker's mind actually necessary?** | **Partly, and the unwelcome half comes first.** A simple counting classifier that never represents a maker at all reproduces the confident-and-inconsistent pattern, through nothing more than small-sample overfitting. What it *cannot* do is respond to a label, or keep paying attention to something it cannot resolve. Both of those need the full machinery. | Nobody has asked this question. Our negative is the only data point that exists. | Not individually checked. The random-model result strengthens the unwelcome half: it is not only a counting classifier that does this, it is a randomly parameterised reader of this shape. |
| **Are the collapse and the trust exploit the same mechanism?** | **Yes**, as originally reported. The same machine-made object, labelled honestly, reads as shallow and moves the viewer barely at all. Passed off as human, it reads as deeper and moves them twenty-two times as far. | — | **The effect survives and gets larger. The explanation does not.** Under exact inference the dishonest label still inflates the reader's estimate of the thinking behind the work, four times more strongly than reported. But "how far the reader moves tracks that estimate whichever channel produced it" falls from 0.886 to 0.600 and misses its pre-registered bar. And rebuilt independently from the prose alone, the direction holds while the multiple comes out fifteen times smaller. **The direction is the claim. The number is not.** |
| **Is unreadable content the same as an unskilled reader?** | No, and they are opposites. At an identical information deficit, the unskilled reader quits almost immediately and feels reasonably settled; the expert facing unreadable content keeps working and stays lost. **The second dimension is whether you can tell you are failing.** A badly aimed template fails silently, out-of-range content fails loudly. **And the unskilled reader of human work is substantially more accurate than the expert reader of machine work.** | — | **Survives exact inference** on all five measures and on the verdict. |
| **Can a viewer know a maker better than the maker knows themselves?** | Yes, and the margin grows the more wrong the maker's self-account is. The viewer's accuracy stays flat as the maker's self-report degrades to nothing. *Scope: the viewer is told how unreliable the report is, so this is a calibrated reader discounting a known-bad source.* | — | Not individually checked. |
| **Does self-blindness leave a mark on the work?** | **Yes on the object, no to the viewer.** Work made by a maker driven by something they cannot see is measurably marked; work by a liar, or by a system with no self-model, is not. But no viewer in this model can tell the three apart, because the readings differ in the fourth decimal place. **The mark exists and is unreadable**, which is a different result from there being no mark, and only measuring the object directly could distinguish them. | — | Not individually checked. |
| **Does how much thought went in change how much you take away?** | **Inconclusive, and the construction is at fault.** Depth was visible and absorption was flat, but the measure written down in advance could not have moved, because depth was deliberately built so the goal is equally readable at every level. Two of the three depth levels also turned out indistinguishable. **What did show up unpredicted: depth drove attention roughly six-fold and absorption not at all.** | — | Not individually re-run. Its dissociation is reported as a construction commitment rather than a finding; see the next row. |
| **Is "depth" just "effort" wearing a hat?** | No. Depth reading tracks depth about six times more than it tracks effort, and effort cannot make a viewer see depth that is not there. | — | **Does not survive exact inference:** the null returns the opposite verdict, that effort *can* manufacture depth. Separately, the dissociation was made representable by rebuilding the effort parameter before it was measured, so it is a construction commitment. What does survive is narrower and still worth having: with the effort axis pinned so no "offhand but deep" corner exists, depth still separates by 0.91, so the depth estimator is reading structure in the work rather than the effort setting renamed. |
| **↳ Two experiments were removed from this table**, because they were run against a version of the model later found to be wrong. They are uninterpretable rather than embarrassing. | See ["What was removed"](#what-was-removed) below. | — | — |

Every number above traces to a committed CSV in [results/](results/) and a chart in
[figures/](figures/).

---

## What was not reported

**One experiment was withheld three times and is not in the table.** It asked whether damage
compounds across generations, as people who learned from contaminated material become the source
for the next round. Its own honesty check, *with zero contamination, show zero damage*, failed
every time. The relay leaks, so real generational decay cannot be told apart from the instrument's
own noise.

The failing test is kept in the suite as a visible `xfail(strict)` marker rather than deleted, so
any future fix has to switch it off deliberately. The drift turned out to be a bookkeeping error in
this repository's own code: evidence about each artifact was filed at every timestep using the
belief held at that moment, so the first observation of every artifact was recorded before the
reader had worked out what it was looking at. Fixing that cut the error 171-fold. It still misses
the threshold, narrowly: 0.00116 against a ceiling of 0.001. Narrowly is exactly the case a
no-exceptions rule exists for. (E8, E18)

**This is not "we found no effect."** It is "we could not measure." The four diagnostic runs that
chased the leak are part of that investigation and are not standalone results.

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

The section a stranger should use to decide whether to trust everything above it. Seven ideas were
tested and killed. Two of them were the author's own, and one was the framework's claim about its
own necessity.

| Whose | The claim | What killed it |
|---|---|---|
| The framework's | The leak across generations is ordinary sampling noise | It did not shrink across a hundred times more data (E12) |
| The framework's | Passing one reader's estimate forward was the whole problem | Fixing that channel left the damage where it was (E12) |
| Mine | Viewers were quitting before they worked it out | Forcing them to keep looking made it worse (E14) |
| **The author's** | The competence threshold is a cliff | Its width did not change across sixteen times the evidence (E15) |
| Mine | Zero effort and "just exploring" are the same hypothesis | They give different answers, for a structural reason (E19) |
| The framework's | Modelling another mind is required for confident disagreement | A counting classifier does it (E21) |
| The pre-registration's | A spike in value-divergence separates the gates | Low effort spikes too, for unrelated reasons (E29) |

The fourth one died to a test the author approved knowing it had two possible outcomes: his claim
survives, or his claim weakens. There was no version where it got stronger.

Two more failed to replicate rather than being killed outright: distance between belief
distributions turned out to be a decent proxy for harm rather than a poor one (E11), and inexpert
readers did not produce a more flattering downstream agent (E10).

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

# redraw every chart from the committed CSVs, without re-running anything
python scripts/rebuild_figures.py
python scripts/make_social_figures.py

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
VALIDATION.md                 generated from results/validation/, never hand-written
run_all.py                    the experiment programme
run_validation.py             the validation pass, V-1 through V-9

ghostscale/                   generative_model, creators, environment, observer, learning, metrics
ghostscale/exact.py           exact joint inference; the solver the validation pass substitutes in
ghostscale/v4_model.py        hypothesis-space overlap; goal-foreign content
ghostscale/v4_5_model.py      the three-gate observer
ghostscale/v5_model.py        model depth as a hierarchy the reader infers
ghostscale/latent_goal.py     the goal a maker does not know it has
ghostscale/experiments/       e1 through e34, each runnable standalone
ghostscale/prereg_*.py        acceptance criteria as executable, hash-locked code
ghostscale/validation/        the nine checks, plus their own hash-locked criteria

tests/                        model invariants, the null suite (N1 to N21), exact-inference tests
config/default.yaml           every parameter, for every version, plus the solver switch

docs/EXPERIMENTS.md           the consolidated plain-language table this README's version came from
docs/specs/                   the build spec each version was written against, and this pass's spec
docs/writeups/                RESULTS_V1 through RESULTS_V5: the full record, with every deviation
docs/decisions/               design decisions signed off before each build

results/                      summary CSVs and JSON verdicts (committed)
results/validation/           one verdict file per check, plus the side-by-side tables
results/diagnostics/          labelled diagnostic runs, kept separate so they cannot be mistaken
                              for reportable output
figures/                      every research chart (committed)
figures/social/               the five distribution slides, the PDF, the preview image
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

# Ghost Scale Simulation

A simulation of how people work out what someone was trying to do when they made something, what
breaks when nothing was trying to do anything, and whether a label saying "this was generated"
helps.

## What this is

When you look at a piece of work, you run a guess about the person behind it. Why did they choose
this word, this shot, this colour. That guess is the thing this project models. It's expensive to
run, so you only keep running it while you're still learning something.

This is a simulation of a proposed mechanism. It is not a study of human subjects, and nothing here
is evidence about what real people do. It's the companion code to *Art as an Algorithmic Virus*
(Zenodo DOI [`10.5281/zenodo.19407789`](https://doi.org/10.5281/zenodo.19407789)).

## A claim this repository withdrew

The framework used to say that producing confident, mutually contradictory readings of empty
content requires a reader who models the maker as a mind. **That is false, and the experiment
that killed it is in here.**

A naive Bayes classifier, trained by counting on 200 examples, never representing a creator or a
purpose or an intention, reproduces the effect: within-reader certainty 0.126 and between-reader
disagreement 1.379, against the full model's 0.108 and 1.377. The mechanism turns out to be
finite-sample overfitting, not theory of mind. (E21)

Two results survive that comparison, and no baseline reproduces either:

- **the label-induced switch** — the same content read as uncertain (1.335 nats of doubt) or
  certain (0.108) depending only on what the label says. The counting classifier ignores labels
  and is confidently wrong either way.
- **sustained futile attention** on content whose structure is real but foreign, which only a
  reader that keeps *expecting* to learn something can produce.

Those are the two the framework actually uses. The withdrawn claim is marked at every point it
was made.

## What is being modelled, and the distinction everyone gets wrong

Three different things get confused with each other constantly. They are not the same and they
make different predictions.

- **Goal-empty** is wood grain. Structure with no purpose behind it, because no purpose was
  involved. Versions 1 to 3 modelled generated content this way.
- **Goal-foreign** is a page of a script you cannot read. A real purpose, pursued by a real
  process, expressed in a vocabulary your reading apparatus has no entry for. Version 4 replaced
  goal-empty with this, because it is a better description of a generative model, which is
  trained on purposeful human output and inherits its shape.
- **Value divergence** is a person who wants something you don't want. That is a third thing
  again, it is not what either of the above describes, and the model keeps it in a separate
  parameter.

**The switch from goal-empty to goal-foreign inverted a headline result.** Under goal-empty,
readers disengage from generated content: nothing is identifiable, so paying attention stops
being worth it. Under goal-foreign they do the opposite. They keep looking, keep paying, and
never get anywhere. The failure to read intent survives and gets worse. The saving of effort
does not.

Both cannot be right, and which one holds depends on a fact about real generated content that a
simulation cannot settle: **how much human-shaped structure it actually carries.** That question
is written up as [E34](results/e34_prediction_card.csv), as a prediction card a human study can
use to locate real content on the axis rather than argue about it.

## What was found

- **Whoever reads the work sets the ceiling on it.** Holding the work perfectly constant and
  varying only the reader's skill, recoverable intent falls from 3.23 to 2.33 and accuracy from
  0.999 to 0.239. No generated content was involved at any point. (E10, E15)
- **Belief breaks before choice does.** Two measures of what the reader believes degrade at
  reader-skill 0.70, while its ability to pick the right answer holds until 0.878. Measure only
  the answer and you see the damage late. (E15)
- **A false label produces confident invention. An honest one produces honest doubt.** Told
  machine work is human, readers stay certain (0.09 nats of doubt) and disagree completely with
  each other (1.38, against a 1.386 ceiling). Told the truth about the same work, the
  disagreement is identical and the doubt rises to 1.29. (E2, E17)
- **Mislabelling is asymmetric, and the asymmetry has a direction for policy.** Human work
  falsely labelled AI is still read accurately and readers agree about it. AI work falsely
  labelled human produces confident invention at ceiling disagreement. Mislabelling human work
  costs engagement; mislabelling machine work costs the model. (A1)
- **The crash and the confident invention are the same narrow band.** Sweeping how much
  human-shaped structure generated content carries, confident fabrication peaks at 10% overlap
  and the disengagement-without-resolution signature is true at exactly that one point. Attention
  stops being sustained below about 4% overlap. Two phenomena the framework had always treated
  separately turn out to be one region of one parameter. (E20)
- **Being out of your depth and reading something foreign are not the same failure.** Matched so
  the reader can extract exactly the same amount of information either way, they differ on every
  measure. An unskilled reader disengages almost immediately (0.001) and is confidently wrong; a
  competent reader facing foreign content stays engaged (0.611) and stays uncertain. **A
  mis-aimed template fails silently. An out-of-range one fails loudly.** (E32)
- **A dishonest label and genuine depth move the reader by the same mechanism, in opposite
  directions.** How far a reader shifts its beliefs tracks how much thought it thinks went in,
  regardless of whether content or a label put that idea there (rho 0.886). Machine content
  labelled honestly reads as shallow and moves nobody; the same content passed off as human
  reads as deeper and moves them a lot. (E31)
- **Unlabelled generated content gets absorbed into the reader's model of human work.** A learner
  with no labels loses a third of its ability to read intent from genuine human work. Honest
  labels cut that error about a hundredfold. (E7)
- **Labelling works, but only for readers who know the rule exists.** A reader who knows
  generated work gets disclosed needs 31% coverage. A reader who doesn't know needs 74%. You need
  the disclosure and the convention. (E16)
- **A reader can know you better than you know yourself.** When a maker's account of its own
  purpose is wrong, the reader recovers the real one from the work anyway. (E33)

Every number traces to a committed CSV in [results/](results/) and a chart in
[figures/](figures/).

## What died

The section a stranger should use to decide whether to trust everything above it. Nine
hypotheses were tested and failed. Two of them were the author's own, and one was the
framework's claim about its own necessity.

| what was believed | what killed it |
|---|---|
| Reading intent from a mind-model is *required* to produce confident contradictory readings | a counting classifier does it (E21) |
| Readers disengage from generated content | true only if it's goal-empty; goal-foreign inverts it (E19, E20) |
| The recursive-training drift is a shortage of data | across a 100-fold range of sample size it doesn't shrink (E12) |
| The drift comes from seeding on a single reader | averaging over readers doesn't remove it (E12) |
| The drift comes from readers disengaging | forcing attention doesn't close it (E14) |
| The drift comes from readers being unskilled | it persists at zero inexpertise (E14) |
| Distance between belief distributions is a poor proxy for harm | it explains 66% of the variance in actual harm (E11) |
| Reader competence fails at a cliff | it's a knee; the transition doesn't sharpen with evidence (E15) |
| Inexpert readers produce a more flattering downstream agent | did not replicate (E10) |

And one experiment has never been reportable. **E8** asks what happens to chains that train on
their own output. It has failed its acceptance test in all three versions it was built for, so
it is not reported. The failing test stays in the suite as a visible `xfail(strict)` rather than
being deleted, so if a future change ever fixes it the suite forces the marker off. The drift
turned out to be a bookkeeping error in this repository's own code — evidence about each artifact
was filed at every timestep using the belief held at that moment, so the first observation of
every artifact was recorded before the reader had worked out what it was looking at. Fixing it
cut the error 171-fold. It still misses the threshold, narrowly: 0.00116 against a ceiling of
0.001. Narrowly is exactly the case a no-exceptions rule exists for. (E18)

### How the scoreboard is counted

"Held" means the prediction was written down before the run, in a spec, a pre-registration file
or a signed-off decisions document, and the measured outcome met the criterion as stated. Not
"broadly went the right way." Anything that needed the criterion softened, the framing widened or
the outcome reinterpreted is not counted as held.

| outcome | count | which |
|---|---|---|
| held | 15 | E1, E2, E3, E4, E5, E6b, E7, E10, E16, E17, E18, E19, E20, E31, E32 |
| held in part | 6 | E6, E9, E21, E28, E29, E33 |
| did not hold | 5 | E11, E12, E14, E15, E30 |
| classification undefined | 1 | E13 |
| withheld, never passed its own control | 1 | E8 |
| not answerable in simulation | 1 | E34 |

## The experiments

One table. Plain-language questions, plain-language answers, numbers inline. There is no second
technical version of this table, deliberately: a technical table and a plain-English table drift
apart within three revisions, and that is the same instrument-versus-claim failure this
repository has caught six times.

| ID | what it asked | what it found | what it did *not* find |
|---|---|---|---|
| E1 | Do readers stop looking closely at work with no intent behind it? | yes, and Curator costs the most deep looks | — |
| E2 | Does a false "human-made" label produce confident disagreement? | yes: doubt 0.09, disagreement 1.38 of a possible 1.386 | — |
| E3 | Does a provenance label save effort without costing accuracy? | yes, across the effort-cost range | — |
| E4 | Where do readers confidently invent a purpose? | high trust plus a dishonest label | — |
| E5 | Is trust in the label a different knob from general decisiveness? | yes; they act at different points and leave different traces | — |
| E6 | What does generated content do to a system's read on what people want? | degrades it as contamination rises | not uniformly; the effect depends on the signal |
| E6b | Does one-sided generated content average out, or pile up? | it piles up, linearly in sample size | — |
| E7 | Can a reader learn which sources are hollow with no labels? | yes, but it loses a third of its read on human work first | that trusting an honest label is harmful — it isn't |
| E8 | What happens to chains that train on their own output? | **never passed its own control; withheld** | — |
| E9 | Do misleading and missing data damage a reader differently? | yes, they dissociate | not on every measure |
| E10 | Does the reader's own skill cap what can be recovered? | yes: 3.23 down to 2.33 with the work held constant | that inexpert readers flatter the downstream agent |
| E11 | Are we measuring harm, or just measuring distance? | distance is a decent proxy; it explains 66% of variance | the predicted dissociation |
| E12 | Is the drift just a shortage of data? | no: exponent −0.017 against a predicted −1 | — |
| E13 | Are "too little data" and "the wrong data" the same problem? | classification undefined | — |
| E14 | Does forcing readers to pay attention close the gap? | no, at any level of reader skill | — |
| E15 | Does skill run out gradually or fall off a cliff? | a knee; belief breaks at 0.70, choice at 0.878 | width does not narrow with evidence |
| E16 | How much generated content has to be labelled? | 31% if readers know the rule, 74% if they don't | — |
| E17 | Is confident invention graded by opacity, or all-or-nothing? | graded; 60%-intent work lands between the extremes | — |
| E18 | Does fixing the bookkeeping remove the drift? | 171-fold reduction, still misses E8's threshold | — |
| E19 | Does the failure survive the most generous excuse for the maker? | yes; "they were just exploring" gets 0.204 belief against 0.200 for no idea at all | — |
| E20 | Where does content stop holding attention? | below ~4% overlap; fabrication peaks at 10%, and so does the crash | — |
| E21 | Is a model of the maker's mind necessary? | **no.** A counting classifier reproduces the dissociation | it does not reproduce label induction or futile attention |
| E28 | Can a reader tell how hard the maker was trying? | yes (rho 1.000), and it moves them less when the answer is "not much" | legibility holds only over the upper half of the range |
| E29 | Do the three proposed gates come apart? | partly; the two hardest to tell apart are told apart on behaviour | three of four predicted signatures missed |
| E30 | Does how much *thought* went in change how much readers take on? | **no.** Depth is readable and inert | the two deepest levels are not distinguishable from each other |
| E31 | Are the crash and the trust exploit one mechanism? | yes: what a reader takes on tracks estimated depth whichever channel moved it (rho 0.886) | — |
| E32 | Are foreign content and an unskilled reader the same failure? | no; they differ on all five measures | — |
| E33 | Can a reader recover a purpose the maker doesn't know it has? | yes, and it beats the maker's own account | the reader cannot detect *that* the maker was self-blind |
| E34 | Where does real generated content sit on the overlap axis? | **not answerable in simulation.** Shipped as a prediction card for a human study | — |
| N21 | Is "depth" just "effort" renamed? | no: 5.98× separation | — |

Full write-ups, with every deviation and non-replication, are in [RESULTS.md](RESULTS.md)
(version 1), [RESULTS_V2.md](RESULTS_V2.md), [RESULTS_V3.md](RESULTS_V3.md),
[RESULTS_V4.md](RESULTS_V4.md), [RESULTS_V4_5.md](RESULTS_V4_5.md) and
[RESULTS_V5.md](RESULTS_V5.md). Design decisions signed off before each build are in
[DECISIONS_V2.md](DECISIONS_V2.md) and [DECISIONS_V3.md](DECISIONS_V3.md).

## How this was kept honest

- Readers in the model have **zero preference over provenance**. They can't want work to be
  human. Any effect of a provenance signal has to come through inference, never through wishing.
  Asserted at every construction.
- Every headline effect has a matching **null condition** that has to come out null. There are
  now twenty-nine of them, in `tests/test_nulls*.py`.
- Acceptance criteria are **pre-registered as executable code** and content-hash locked before
  any run. The written criterion and the applied criterion are the same object, so they cannot
  drift apart.
- **Every deviation is logged**, including two in version 5 where a criterion was restated after
  seeing a measurement. In both cases the original criterion is retained, still computed, and
  reported as failing.
- Some checks exist because the failure they guard against **actually happened** during a build
  and was caught by an assertion rather than by a result. Those are marked as such in the code.

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

# redraw every chart from the committed CSVs, without re-running anything
python rebuild_figures.py

# tests
pytest -q
```

## Scope and limits

- A simulation of a mechanism, not a study of people. No human data.
- The central assumption, that Ghost Scale opacity can be read directly as the fraction of a
  maker's intent that survives into the work, is a modelling choice. It is stated in the open
  below rather than buried, because it is the line to argue with.
- Version 5's construct corrections are **theoretically motivated and largely untested**. They
  are more defensible than what they replace, which is not the same as being right.
- **The Ghost Scale's own attention gradient is non-monotone, and in the wrong place.** The tier
  meant to signal "do not spend effort here" is rendered as the visually loudest element on the
  page, while the Curator tier's reduced contrast makes it genuinely easy to skip — and E1 finds
  Curator the most expensive tier for readers. The model and the design disagree, and the design
  is probably right about human behaviour. The 5% tier is close to a logical impossibility
  besides: prompting alone constitutes more selection than 5% implies. This repository's own
  charting code quietly declined to use the published opacity ramp, because it isn't legible.
- No literature search has been run against any of these predictions. They were derived from
  theory and tested in simulation. Whether they are already known is an open and unexamined
  question.

## How the model works

### The one empirical commitment: opacity means recoverable intent

The four published Ghost Scale tiers, CREATOR / POLISHED / CURATOR / GHOST, are drawn at 100%,
95%, 60% and 5% opacity. The model reads that opacity directly as the fraction of the maker's
intent that survives into the work:

```
alpha = {CREATOR: 1.00, POLISHED: 0.95, CURATOR: 0.60, GHOST: 0.05}
A[0][:, tier, goal, DEEP] = alpha[tier] · sig[goal] + (1 - alpha[tier]) · noise_free_synth
```

This is the load-bearing assumption of the whole framework, so it's in the open rather than
buried in an implementation file. If you think opacity and recoverable intent aren't the same
quantity, this is the line to argue with.

### Generated content is structured, and that's the point

`noise_free_synth` is a structured, non-uniform, goal-independent distribution over features,
with entropy well below uniform, asserted in code. Generated artifacts in this model aren't
noise. They are richly patterned. What they lack is any dependence of that pattern on a goal.

That distinction matters because the obvious strawman, "generated content is random noise,"
would produce the same crash for the wrong reason. Null N6 exists to separate them. Both are low
information. Only the strawman is high entropy.

### Depth is in the order, not in the histogram

Version 5 models how much *thinking* sits behind a work as the number of levels of the maker's
decision hierarchy that reach the surface. It is built so that a deep work and a shallow one have
**identical** feature histograms, to machine precision — a reader who counts features and ignores
their order cannot tell them apart at all. What distinguishes them is the order: a deep work moves
through stages, and a deeper one moves through them in an order that names what it is for.

That constraint is asserted at every build, and it is what stops "depth" from being a second name
for "legibility", which the model already measures elsewhere.

### What the learner already knows, and why that's a claim

The learner in version 2 doesn't arrive holding a correct model of what generated content looks
like. It has to acquire one from an unlabelled, already-contaminated corpus. What it starts with
is a deliberate theoretical position: it knows the shared goal-to-feature family, and it doesn't
know how provenance modulates it. Readers share a likelihood family because they share a body
plan. What they don't share, and have to learn, is which sources carry intent.

It's also forced. A genuinely uninformative prior isn't a slow learner, it's an unidentifiable
one: information between features and goal sits at exactly 0.0000 nats after 400 artifacts and
all four learned columns stay bit-identical. That measurement is kept as a live test, so if it
ever stops being true the decision gets revisited instead of inherited.

## The Ψ analogue is a reimplementation, not a port

`metrics.psi_analogue` is a discrete stand-in for the closed-form Ψ of the preprint:

```
psi = [engaged] · (-ln(1 - κ)) · KL( Q(goal | τ) ‖ P0(goal) )
```

The sigmoid gate of the closed form is replaced by the binary engagement decision. This
reimplements the intuition, which is engagement-gated, trust-weighted surprise about the goal.
It isn't a port of the equation, and no equivalence is claimed.

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

Two things this model adds, stated narrowly because the value of the contribution is that it
fills a named gap, and that argument gets weaker if it claims more ground than it holds.

1. **What sets the weight.** Albarracin et al. fix the empathy parameter λ from outside the model
   and name "what sets λ" as their central open question. κ, gated on provenance and on value
   divergence, is a candidate answer: the weight you put on another agent's inferred preferences
   should scale with the evidence that those preferences exist at all.
2. **What happens when the other has no preferences.** Existing work assumes the observed other
   is an agent with preferences. This model asks what happens when it isn't, and finds the
   inference doesn't fail gracefully. Under high trust it fabricates. (E2, E4)

## Repository layout

```
ghostscale/               generative_model, creators, environment, observer, learning, metrics
ghostscale/v4_model.py    hypothesis-space overlap; goal-foreign content
ghostscale/v4_5_model.py  the three-gate observer
ghostscale/v5_model.py    model depth as a hierarchy the reader infers
ghostscale/latent_goal.py the goal a maker doesn't know it has
ghostscale/experiments/   e1 through e34, each runnable standalone
ghostscale/prereg_*.py    acceptance criteria as executable, hash-locked code
tests/                    model invariants and the null suite (N1 through N21)
config/default.yaml       every parameter, for every version
notebooks/walkthrough.ipynb   runs E1 and E2 end to end, narrated
rebuild_figures.py        redraws every chart from committed CSVs, no re-running
results/                  summary CSVs and JSON verdicts (committed)
figures/                  every chart (committed)
docs/                     the build specs each version was written against
```

Raw per-reader CSVs are not committed, because `e4_raw.csv` alone is 16 MB. Everything a number
in this README or a chart in `figures/` depends on is committed. Regenerate the raw files with
`python run_all.py`.

Several defaults were recalibrated on contact with the implementation. Each is documented under
"Deviations" in the matching results file, with the evidence that motivated it, so any of them
can be argued with. The load-bearing constraints haven't changed: zero reader preference over
provenance, structured rather than uniform generated content, reader heterogeneity, the full
null suite, and the honest crosswalk above.

## Links

- Preprint: *Art as an Algorithmic Virus*, [`10.5281/zenodo.19407789`](https://doi.org/10.5281/zenodo.19407789)
- Plain-language essay: <https://abrahamhaskins.org/art>
- Ghost Scale Figma kit: <https://www.figma.com/community/file/1624141586132218953>

## License and citation

Code is MIT. Prose, figures and data are CC BY 4.0. See [LICENSE](LICENSE).

To cite this repository, see [CITATION.cff](CITATION.cff), or cite the preprint directly.

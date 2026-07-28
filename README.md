# Ghost Scale Simulation

A simulation of how people work out what someone was trying to do when they made something, what
breaks when nothing was trying to do anything, and whether a label saying "this was generated"
helps.

## What this is

When you look at a piece of work, you run a guess about the person behind it. Why did they choose
this word, this shot, this colour. That guess is the thing this project models. It's expensive to
run, so you only keep running it while you're still learning something.

Generated content breaks the guess in a specific way. It isn't noise. It's highly structured. What
it lacks is any connection between that structure and a purpose, because no purpose produced it.
The model asks two questions about that. First, what happens to a reader who keeps trying anyway.
Second, what happens if you tell the reader the truth about where the work came from, or lie to
them about it.

This is a simulation of a proposed mechanism. It is not a study of human subjects, and nothing here
is evidence about what real people do. It's the companion code to *Art as an Algorithmic Virus*
(Zenodo DOI [`10.5281/zenodo.19407789`](https://doi.org/10.5281/zenodo.19407789)).

## What was found

- **Whoever reads the work sets the ceiling on it.** Holding the work perfectly constant and
  varying only the reader's skill, recoverable intent falls from 3.23 to 2.33 and accuracy from
  0.999 to 0.239. Not one piece of generated content was involved. (E10, E15)
- **Belief breaks before choice does.** Two independent measures of what the reader believes
  degrade at reader-skill 0.70. Its ability to pick the right answer holds until 0.878. If you only
  measure which answer gets picked, you see the damage late. (E15)
- **A false label produces confident invention. An honest one produces honest doubt.** Told machine
  work is human, readers stay certain (0.09 nats of doubt) and disagree completely with each other
  (1.38 nats, against a 1.386 ceiling). Told the truth about the same work, they arrive at the same
  disagreement while their doubt rises to 1.29. Same content, same disagreement, opposite
  confidence. (E2, E17)
- **Partial hollowness does partial damage.** Work carrying 60% of its maker's intent lands
  between the two extremes on every measure, so this behaves like a dial rather than a switch.
  (E17)
- **Unlabelled generated content gets absorbed into the reader's model of human work.** A learner
  with no labels loses a third of its ability to read intent from genuine human work, and never
  works out what hollow content looks like. Honest labels cut that error about a hundredfold.
  (E7)
- **Labelling works, but only for readers who know the rule exists.** A reader who knows that
  generated work gets disclosed needs 31% coverage. A reader who doesn't know needs 74%. You need
  both the disclosure and the convention. (E16)
- **The obvious deflationary explanation was tested and didn't hold.** Maybe readers only fail on
  machine work because we gave them too few options. So we added the most generous one the theory
  allows, "they were just exploring", and checked it works: given human work made while dabbling,
  91% of readers picked it. Given machine work they gave it 0.204 of their belief, against 0.200
  for a reader with no idea at all. (E19)

Every number above traces to a committed CSV in [results/](results/) and a chart in
[figures/](figures/).

## What did not work

**One experiment has never been reportable.** E8 asks what happens to chains that train on their
own output. It has failed its acceptance test in all three versions, so it isn't reported. The
failing test is kept in the suite as a visible `xfail(strict)` rather than deleted, so if a future
change ever fixes it the suite forces the marker off. Version 3 was built specifically to repair
E8. Version 3's own pre-registered gate refuted version 3's diagnosis: across a hundredfold range
of sample size, the drift doesn't shrink with data (log-log exponent -0.017, t = -0.28), against a
predicted -1.

**Five explanations died on the way.** Too little data (E12). Single-reader seeding (E12). Readers
disengaging (E14). Readers being unskilled (E14 at zero inexpertise). A rounding error in the
seeding path (E18). Each was tested and each was wrong.

**What survived is smaller and duller than what we set out to prove.** The drift turned out to be a
bookkeeping error in our own code. The learner filed evidence about each artifact at every timestep
using the belief it held at that moment, so the first observation of every artifact got recorded
before the reader had worked out what it was looking at. A fixed fraction of all evidence, about
one over the number of inference steps, landed in the wrong bucket. Fixing the bookkeeping cuts the
learned-model error 171-fold. It still misses E8's acceptance threshold, narrowly: 0.00116 against
a ceiling of 0.001. Narrowly is exactly the case a no-exceptions rule exists for, so E8 stays
withheld. (E18)

**Two predictions from earlier versions didn't replicate**, and the write-ups say so where the
original claims were made: inexpert readers don't produce a more flattering downstream agent
(E10), and trusting the label isn't harmful when the label is honest (E7).

**And one headline result did not carry over when the model of machine content changed.** V1
through V3 treated machine content as having no purpose behind it at all, and found readers
disengage from it. V4 treats it as having a purpose the reader has no vocabulary for, which is
a better description of a generative model, and finds the opposite: readers keep looking, keep
paying, and never get anywhere. The failure to read intent survives and gets worse. The saving
of effort does not. Both can't be right, and which one is depends on a fact about real
generated content that a simulation cannot settle. It's written up in
[RESULTS_V4.md](RESULTS_V4.md).

### How the scoreboard is counted

"Held" below means one specific thing: the prediction was written down before the run, in a spec,
a pre-registration file, or a signed-off decisions document, and the measured outcome met the
acceptance criterion as stated. Not "broadly went the right way." Anything that needed the
criterion softened, the framing widened, or the outcome reinterpreted is not counted as held.

Nineteen primary predictions, one per experiment:

| outcome | count | which |
|---|---|---|
| **held** | 11 | E1, E2, E3, E4, E5, E6b, E7, E10, E16, E17, E18 |
| **held in part** | 2 | E6, E9 |
| **did not hold** | 4 | E11, E12, E14, E15 |
| **classification undefined** | 1 | E13 |
| **withheld, never passed its own control** | 1 | E8 |

The four that didn't hold are worth reading for. E11 predicted that distance between numbers is a
poor proxy for harm, and it turned out to be a decent one (it explains 66% of the variance). E15
predicted a cliff and found a knee, and the arm that settled it was a width test the first design
didn't include.

## How this was kept honest

- Readers in the model have **zero preference over provenance**. They can't want work to be human.
  Any effect of a provenance signal has to come through inference, never through wishing.
- Every headline effect has a matching **null condition** that has to come out null. Fifteen of
  them, in `tests/test_nulls*.py`.
- Acceptance criteria are **pre-registered as executable code** and content-hash locked before any
  run, in [`ghostscale/prereg_v3.py`](ghostscale/prereg_v3.py). The written criterion and the
  applied criterion can't drift apart, because they're the same object.
- **Deviations are logged**, including the two places a criterion was changed after seeing data.
  Both are marked as deviations rather than presented as the original plan.

**The most useful thing here is a habit, not a result.** Three separate times, a coherent and
plausible story was one step from being written down, and what stopped it was checking the
instrument instead of the phenomenon. A stale file silently substituting its own parameter. A test
that checked a weaker claim than the hypothesis it was gating. A statistic read at the wrong
timestep, which understated the effect fiftyfold and nearly produced a confident "mechanism
refuted." Two of the three produced not just plausible numbers but plausible explanations for
them. All three are written up in [RESULTS_V3.md](RESULTS_V3.md).

## Scope and limits

Read this before quoting anything.

- **This is a simulation, not a study.** No human subjects. It shows that a proposed mechanism
  produces a particular pattern, which is not the same as showing the mechanism operates in people.
- **The shape is the claim. The numbers are not portable.** Figures like the 0.878 midpoint depend
  on this model's four goals and its feature vocabulary. Change either and the number moves. What
  should survive is the ordering: belief before choice, graded rather than binary, reader-limited
  rather than material-limited.
- **The 31% labelling figure is a floor, not a target.** The reader that achieves it is handed the
  true coverage rate, which is the most generous assumption available. A reader with a wrong belief
  about how much gets labelled can only do worse.
- **The link between generated content and RLHF is an analogy, not a measurement.** This model's
  "choice" is picking one of four goals. Nothing here was run against a language model.
- **E8 is withheld.** Any claim about recursive collapse in this repository is a claim about a
  chain we couldn't get to pass its own control.

## Install and run

Requires Python 3.10 or newer. Uses the legacy NumPy interface of pymdp throughout
(`from pymdp.legacy.agent import Agent`). pymdp 1.x is JAX-first at the top level, and the legacy
object-array API is the right target for running hundreds of small independent agents in a loop.

```bash
# with uv (preferred) or pip
uv pip install -e .          # or:  pip install -e .

# run everything, full scale, parallel
python run_all.py            # or: make all
python run_all.py --quick    # fast smoke scale for a laptop

# one experiment (writes results/eN_*.csv and figures/eN_*.png)
python -m ghostscale.experiments.e2_variance --workers 8

# redraw every chart from the committed CSVs, without re-running anything
python rebuild_figures.py

# tests
pytest -q                    # or: make test
```

Everything is seeded, one generator per reader, with seeds recorded in every CSV. Same seed gives
the same CSV, and an invariant test checks it. Every parameter lives in
[`config/default.yaml`](config/default.yaml) rather than in code.

## The experiments

| ID | question | status | outputs |
|---|---|---|---|
| E1 | Do readers stop looking closely at work with no intent behind it? | held | [chart](figures/e1_crash.png), `e1_*.csv` |
| E2 | Does a false "human-made" label produce confident disagreement? | held | [chart](figures/e2_variance.png), `e2_*.csv` |
| E3 | Does a provenance label save effort without costing accuracy? | held | [chart](figures/e3_titration_accuracy.png), `e3_*.csv` |
| E4 | Where is the region where readers confidently invent a purpose? | held | [chart](figures/e4_trust_exploit_rate.png), `e4_*.csv` |
| E5 | Is trust in the label a different knob from general decisiveness? | held | [chart](figures/e5_precision_baseline.png), `e5_*.csv` |
| E6 | What does generated content do to a system's read on what people want? | held in part | [chart](figures/e6_corpus_corruption.png), `e6_*.csv` |
| E6b | Does generated content that leans one way average out, or pile up? | held | [chart](figures/e6b_corpus_biased.png), `e6b_*.csv` |
| E7 | Can a reader learn which sources are hollow with no labels at all? | held | [chart](figures/e7_learn_ghost.png), `e7_*.csv` |
| E8 | What happens to chains that train on their own output? | **withheld** | [chart](figures/e8_recursive.png), `e8_*.csv` |
| E9 | Do misleading data and missing data damage a reader differently? | held in part | [chart](figures/e9_poison_starve.png), `e9_*.csv` |
| E10 | Does the reader's own skill cap what can be recovered? | held | [chart](figures/e10_expertise_gradient.png), `e10_*.csv` |
| E11 | Are we measuring harm, or just measuring distance? | did not hold | [chart](figures/e11_regret_vs_kl.png), `e11_*.csv` |
| E12 | Is the drift just a shortage of data? | did not hold | [chart](figures/e12_leak_convergence.png), `e12_*.csv` |
| E13 | Are "too little data" and "the wrong data" the same problem? | undefined | [chart](figures/e13_shared_signature.png), `e13_*.csv` |
| E14 | Does forcing readers to pay attention close the gap? | did not hold | [chart](figures/e14_engagement_floor.png), `e14_*.csv` |
| E15 | Does skill run out gradually or fall off a cliff? | did not hold | [chart](figures/e15_competence_cliff.png), `e15_*.csv` |
| E16 | How much generated content has to be labelled? | held | [chart](figures/e16_label_coverage.png), `e16_*.csv` |
| E17 | Is confident invention graded by opacity, or all-or-nothing? | held | [chart](figures/e17_tier_dose_response.png), `e17_*.csv` |
| E18 | Does fixing the bookkeeping remove the drift? | held | [chart](figures/e18_deferred_estimator.png), `e18_*.csv` |
| E19 | Does the failure survive giving readers the most generous possible excuse? | held | [chart](figures/e19_explore.png), `e19_*.csv` |

Full write-ups, including every deviation and every non-replication, are in
[RESULTS.md](RESULTS.md) (version 1), [RESULTS_V2.md](RESULTS_V2.md),
[RESULTS_V3.md](RESULTS_V3.md) and [RESULTS_V4.md](RESULTS_V4.md). Design decisions signed off
before each build are in [DECISIONS_V2.md](DECISIONS_V2.md) and
[DECISIONS_V3.md](DECISIONS_V3.md).

Version 4 is in progress and only its first stage is done. It changes what the model says
machine content *is*, from "made with no purpose" to "made with a purpose you have no
vocabulary for", and then checks whether the earlier results survive. E19 is stage 1 of 8;
stages 3 onward are explicitly optional.

## How the model works

Three sections for readers who want the mechanism rather than the results.

### The one empirical commitment: opacity means recoverable intent

The four published Ghost Scale tiers, CREATOR / POLISHED / CURATOR / GHOST, are drawn at 100%,
95%, 60% and 5% opacity. The model reads that opacity directly as the fraction of the maker's
intent that survives into the work:

```
alpha = {CREATOR: 1.00, POLISHED: 0.95, CURATOR: 0.60, GHOST: 0.05}
A[0][:, tier, goal, DEEP] = alpha[tier] · sig[goal] + (1 - alpha[tier]) · noise_free_synth
```

This is the load-bearing assumption of the whole framework, so it's stated here in the open rather
than buried in an implementation file. If you think opacity and recoverable intent aren't the same
quantity, this is the line to argue with.

### Generated content is structured, and that's the point

`noise_free_synth` is a structured, non-uniform, goal-independent distribution over features, with
entropy well below uniform, asserted in code. Generated artifacts in this model aren't noise. They
are richly patterned. What they lack is any dependence of that pattern on a goal. Information
between features and the goal goes to zero while feature entropy stays low.

That distinction matters because the obvious strawman, "generated content is random noise," would
produce the same crash for the wrong reason. Null N6 exists to separate them. Both are low
information. Only the strawman is high entropy.

### What the learner already knows, and why that's a claim

The learner in version 2 doesn't arrive holding a correct model of what generated content looks
like. It has to acquire one from an unlabelled, already-contaminated corpus. What it does start
with is a deliberate theoretical position:

> The learner knows the shared goal-to-feature family. It doesn't know how provenance modulates
> it. Its prior treats every source as human-like, the way a naive reader assumes everything they
> read was meant, and what it has to learn is which sources are hollow.

Readers in this model share a likelihood family because they share a body plan. What they don't
share, and have to learn, is which sources carry intent. Granting the learner the family isn't a
concession that makes the problem easier by fiat. It's where the theory says the shared part of
perception ends.

It's also forced. A genuinely uninformative prior, uniform over features in every column, isn't a
slow learner. It's an unidentifiable one. Learning attributes each observed feature to the goal
the reader currently believes in, and under a uniform prior that belief never moves, so every
observation deposits an equal share into all four goal columns and they converge to the same
thing. Measured, with engagement forced so disengagement can't be the explanation: information
between features and goal sits at exactly 0.0000 nats after 400 artifacts, and all four learned
columns stay bit-identical. The provenance-uninformative seeding reaches 1.025 nats against an
oracle ceiling of 1.089. That measurement is kept as a live test
(`tests/test_nulls_v2.py::test_D1_uniform_prior_is_unidentifiable`), so if it ever stops being true
the decision gets revisited instead of inherited.

This scopes what E7 is asking. Not "can intent-reading be learned from scratch?" but "can you learn
which sources are hollow, from content alone, with no labels?"

## The Ψ analogue is a reimplementation, not a port

`metrics.psi_analogue` is a discrete stand-in for the closed-form Ψ of the preprint:

```
psi = [engaged] · (-ln(1 - κ)) · KL( Q(goal | τ) ‖ P0(goal) )
```

The sigmoid gate of the closed form is replaced by the binary engagement decision. This
reimplements the intuition, which is engagement-gated, trust-weighted surprise about the goal. It
isn't a port of the equation, and no equivalence is claimed.

## Relationship to existing work

This model fills a named gap in the active-inference literature on Theory of Mind. The honest
mapping:

| this model | corresponding construct | source |
|---|---|---|
| `creator_goal` as a hidden state the reader infers | opponent preference parameters θ_j inferred online | Albarracin et al. 2026, arXiv 2602.20936 |
| the reader's model of the maker (`sig`, the creator POMDP) | structurally matched other-model M_other | Albarracin et al. 2026 |
| κ as precision on the provenance channel `A[1]` | reliability-gating *r* between learned and static ToM prediction | Albarracin et al. 2026, Eq. 15 |
| engagement policy over attention (DEEP/SKIM) | epistemic-value term in social EFE | Pitliya et al. 2025, arXiv 2508.00401 |
| Ghost Scale tiers (alpha as opacity) | no existing analogue | |
| a generator with no preferences at all | no existing analogue | |

Two things this model adds. They're stated narrowly, because the value of the contribution is that
it fills a named gap, and that argument gets weaker if it claims more ground than it holds.

1. **What sets the weight.** Albarracin et al. fix the empathy parameter λ from outside the model
   and name "what sets λ" as their central open question. They observe that accurate other-modeling
   with low λ produces better exploitation rather than cooperation, and that optimising precision
   alone is motivationally neutral. κ, gated on provenance and on value divergence, is a candidate
   answer: the weight you put on another agent's inferred preferences should scale with the
   evidence that those preferences exist at all.
2. **What happens when the other has no preferences.** Existing work assumes the observed other is
   an agent with preferences. This model asks what happens when it isn't, and finds the inference
   doesn't fail gracefully. Under high trust it fabricates (E2, E4).

## Repository layout

```
ghostscale/               generative_model, creators, environment, observer, learning, probes, metrics
ghostscale/experiments/   e1 through e18, each runnable standalone
ghostscale/prereg_v3.py   acceptance criteria as executable, hash-locked code
tests/                    model invariants and the null suite (N1 through N15)
config/default.yaml       every parameter, for every version
notebooks/walkthrough.ipynb   runs E1 and E2 end to end, narrated
rebuild_figures.py        redraws every chart from committed CSVs, no re-running
results/                  summary CSVs and JSON verdicts (committed)
figures/                  every chart (committed)
docs/                     the build specs each version was written against
```

Raw per-reader CSVs are not committed, because `e4_raw.csv` alone is 16 MB. Everything a number in
this README or a chart in `figures/` depends on is committed. Regenerate the raw files with
`python run_all.py` if you need them.

Several defaults were recalibrated on contact with the implementation, for instance the effort cost
and the construction of the synthetic likelihood. Each one is documented under "Deviations from
spec" in the matching results file, with the evidence that motivated it, so any of them can be
argued with. The load-bearing constraints haven't changed: zero reader preference over provenance,
structured rather than uniform generated content, reader heterogeneity, the full null suite, and
the honest crosswalk above.

## Links

- Preprint: *Art as an Algorithmic Virus*, [`10.5281/zenodo.19407789`](https://doi.org/10.5281/zenodo.19407789)
- Plain-language essay: <https://abrahamhaskins.org/art>
- Ghost Scale Figma kit: <https://www.figma.com/community/file/1624141586132218953>

## License and citation

Code is MIT. Prose, figures and data are CC BY 4.0. See [LICENSE](LICENSE).

To cite this repository, see [CITATION.cff](CITATION.cff), or cite the preprint directly.

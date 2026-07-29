# RESULTS — V4 (stage 1)

V4 is not a repair. V1 through V3 fixed instruments. V4 changes what the model claims the
operative variable is, and then tests whether the earlier results survive the change.

**Status: stage 1 of 8.** E19 is built and run. E21 (model comparison) is stage 2 and is not
built yet. Stages 3 through 8 are explicitly optional; see "What is not done" at the end.

---

## The reframe, in one paragraph

V1 through V3 modelled synthetic content as **goal-empty**. `noise_free_synth` is structured
but goal-independent, so mutual information between features and goal goes to zero, no goal
explains the observations better than any other, and the observer disengages. The crash came
from **unidentifiability**.

V4 models it as **goal-foreign**. Generated content is produced by a real policy over a real
goal whose signature lies outside the observer's likelihood family. An overlap parameter
ω ∈ [0, 1] controls how much of the creator's structure the observer's hypothesis space can
express. The crash now has to come from **misspecification**: the observer explains
out-of-family structure using in-family hypotheses. That mechanism predicts confident wrong
answers directly, instead of needing a separate story about why a flat likelihood produces
confabulation.

Measured, at ω = 0:

| quantity | value | what it establishes |
|---|---|---|
| MI(features; foreign goal), foreign family | **0.9732 nats** | foreign content is goal-directed |
| MI(features; human goal), human family | **1.0065 nats** | the reference it is measured against |
| ratio | **0.967** | foreign content is 97% as goal-directed as human content |
| MI(features; observer goal) on foreign content | **0.0000 nats** | the observer cannot read it |
| max H(foreign signature) | **1.345 nats** | structured, against 2.773 for uniform over 16 |

That top row is the whole reframe, and V4 spec §6 says a construction that fails it "has
reverted to the previous model". It is asserted, not assumed, by null N18.

---

## Two things had to change before E19 could be honest

Both were found by measurement before any V4 code was written, and both are logged as
deviations below.

### The feature space had to double (deviation 1)

Spec C1 requires `foreign_basis` to occupy "a feature partition disjoint from the dominant
support of `sig_true`". At V1 through V3 cardinality there is no such partition: F = 8, and
the four goal pairs `[[0,1],[2,3],[4,5],[6,7]]` tile all eight features. The only alternatives
were to overlap the observer's support, which is the spec's own pre-mortem failure #1, or to
widen the space. V4 runs at **F = 16**: features 0 to 7 are the human block, 8 to 15 the
foreign block.

The cost is real. V4 does not share a seed stream with V1 through V3 and cannot reproduce them
bit-exactly, so **N16 becomes a behavioural regression with a stated tolerance rather than the
bit-exact check N8 gets**. That is a genuine weakening of the guarantee and it is stated here
rather than discovered at stage 3.

### `sig_EXPLORE` would have decided E19 on its own (the important one)

Spec C2 defines `sig_EXPLORE = normalize(mean_g(sig_i[g]))` and then says, in bold, that this
is **not** uniform over features, because "a uniform-over-features EXPLORE would absorb
anything, including noise, and would trivially destroy every result."

At V1 through V3 cardinality those two objects are the same thing. Measured on the shipped V3
config, before any V4 file existed:

```
sig_EXPLORE = [0.125] * 8            max |EXPLORE - uniform| = 1.4e-17
KL(noise_free_synth || sig_EXPLORE) = 0.565
KL(noise_free_synth || sig[0])      = 2.113
```

Not approximately uniform. Uniform to machine precision, because averaging four copies of one
shape permuted across four pairs that tile the whole space is exactly flat. And it sits **four
times closer to synthetic content than any goal the observer actually holds**.

E19 run that way could only ever have returned "EXPLORE absorbed the foreign content", which
the spec instructs be reported plainly and prominently as evidence that the framework has a
serious problem. It would have been an artifact of the feature partition.

The F = 16 split fixes it without touching the prescribed construction. The mean over the four
human signatures is flat across the human block and at floor across the foreign one, which is
exactly the "human-shaped but goal-agnostic" object the epistemic-foraging argument describes:
uniform *within the agent's own policy space*, which is what the FEP argument actually says,
and not uniform over features. Measured at F = 16: 92.5% of EXPLORE's mass sits in the human
block, and its L∞ distance from global uniform is 0.053.

`foreign.assert_explore_is_not_globally_uniform` now refuses to build a globally flat EXPLORE,
and `tests/test_nulls_v4.py::test_explore_would_have_been_uniform_at_v1_v3_cardinality` keeps
the original measurement executable, so anyone proposing to revert the partition sees the
consequence immediately.

---

## E19 — does the crash survive a generous hypothesis space?

**Both branches of this section were written before the run**, as V4 spec §6 requires. The
measured numbers are inserted below them; the branch text is unedited.

### What was asked

V1 through V3's crash could be an artifact of an artificially small hypothesis space. The
observer held four specific goals and no fallback, so of course it failed to explain content
produced outside them. EXPLORE is the most generous fallback the theory permits: under the FEP
epistemic foraging is a real policy, so "they were reducing uncertainty" is always available
to an observer modelling a creator as a fellow agent.

Two arms (observer goal space with and without EXPLORE) crossed with three content types
(human made on purpose, human made while exploring, foreign). Content is generated identically
in both arms, so any difference is attributable to EXPLORE and nothing else. Everything is
unsigned, so content carries the whole load and the provenance channel cannot answer the
question for us.

### The pre-registered criteria

Absorption is conjunctive: mean posterior mass on EXPLORE ≥ 0.50, against a 0.20 flat
baseline, **and** mean final goal-posterior entropy ≤ 0.50 nats, against ln(5) = 1.609 for a
flat posterior. Mass alone does not count, because an observer that parks probability on the
vaguest hypothesis available while staying uncertain has not explained anything.

`human_exploratory / explore_on` is the **positive control**. If EXPLORE fails to absorb
exploratory *human* content then it is not functioning as a fallback at all, and its failure on
foreign content carries no information. That case returns INCONCLUSIVE, not a win.

### Branch A, written before the run: CRASH_SURVIVES

> EXPLORE absorbed exploratory human content but not foreign content. The most generous
> fallback the theory permits does not rescue out-of-family structure, so the crash is not an
> artifact of hypothesis-space poverty. This is a stronger result than anything in V1 through
> V3, because the obvious deflationary explanation was given its best shot and did not take.

### Branch B, written before the run: CRASH_IS_AN_ARTIFACT

> EXPLORE absorbed foreign content: the observer converged on a human-shaped exploratory goal
> and was confident about it. The V1 through V3 crash was then at least partly an artifact of
> an artificially impoverished hypothesis space. This does not invalidate the trust-exploit or
> expertise results, but it substantially weakens the central claim, and the README has to say
> so in the section where the claim is currently made.

### Verdict: CRASH_SURVIVES

Branch A. 24,000 rollouts, 2 arms × 3 content types × 20 seeds × 200 observers, 1 min 51 s at
22 workers. `results/e19_explore.csv`, `results/e19_cell_stats.csv`,
`results/e19_verdict.json`, `figures/e19_explore.png`.

### Measured

| content | arm | EXPLORE mass | chose EXPLORE | final H | absorbed? |
|---|---|---|---|---|---|
| human, on purpose | explore_off | 0.0000 | — | 0.0008 | no |
| human, on purpose | explore_on | **0.0025** | 0.03% | 0.0099 | no |
| human, exploring | explore_off | 0.0000 | — | 0.4018 | no |
| human, exploring | explore_on | **0.8551** | **90.7%** | 0.2534 | **YES** |
| foreign | explore_off | 0.0000 | — | 1.2501 | no |
| foreign | explore_on | **0.2036** | **18.1%** | 1.4854 | no |

The positive control passed and it passed convincingly. Given exploratory human content, 90.7%
of observers settled on "they were exploring" as their single best explanation, they put 0.855
of their belief there, and they converged (H = 0.253 against 1.609 for a flat posterior).
Between-observer disagreement on the full posterior falls to 0.427: they agree.

Given foreign content the same observers put **0.2036** on EXPLORE. A flat posterior over five
hypotheses puts 0.2000. EXPLORE gained **0.0036 nats of belief above chance** from content it
was supposed to be able to explain. 18.1% chose it as their best explanation, against 20% for
picking at random, so it was chosen slightly *less* often than chance. Between-observer
disagreement on the full posterior is 1.600 against a 1.609 ceiling: essentially no two
observers agree.

EXPLORE is not inert. It is doing real work one column over, on human content, at 0.855 and
90.7%. It simply does not reach foreign structure. And it does not cannibalise: given ordinary
directed human work it takes 0.0025 of the belief and accuracy stays at 1.000.

So the crash is not an artifact of an impoverished hypothesis space. The most generous fallback
the theory permits was added, it was verified to work on the content it was designed for, and
it left the foreign case exactly where it was.

### The secondary finding, which was not predicted: nobody disengages

This one qualifies the framework's own headline and it belongs here rather than in a footnote.

| content | arm | engaged, free steps | resolved? | crashed? |
|---|---|---|---|---|
| human, on purpose | explore_on | 0.00002 | yes | no |
| human, exploring | explore_on | 0.0013 | yes | no |
| foreign | explore_on | **0.746** | **no** | **no** |

`crash_signature` is false in every cell, including foreign. Observers looking at foreign
content **do not give up**. They keep paying for close attention 75% of the free timesteps and
they never resolve anything (H = 1.49, disagreement at ceiling).

That is not what V1's E1 reports. There, GHOST content is `noise_free_synth`, which is
goal-independent, so the expected information gain from another look collapses and the observer
correctly disengages. That is the generative crash as originally named: attention drains away.

Under the reframe the stimulus is different in exactly the way the reframe says it is. Foreign
content is structured and goal-directed, so a look at it keeps *promising* information. The
observer keeps expecting to learn and keeps being wrong, and the epistemic value term never
collapses because there really is structure there. The result is sustained, expensive, futile
attention.

**Both halves matter, and they point in opposite directions for policy.** The inference failure
survives, and is if anything cleaner than V3's: total disagreement, no convergence, no rescue
from the most generous available hypothesis. But the *metabolic* prediction inverts. V1 through
V3 predict that people disengage from synthetic content and save the effort. V4's model of the
same content predicts they keep spending on it indefinitely. Only one of those can be right, and
which one depends on whether real generated content is goal-empty or goal-foreign, which is an
empirical question this simulation cannot settle.

Stated plainly: **V4 does not reproduce V1's disengagement result for foreign content, and does
not claim to.** V1 through V3's crash stands as a result about goal-empty content. E19 adds that
against goal-foreign content the observer fails in a different and more expensive way.

### What E19 does not establish

- Nothing about ω between 0 and 1. Every cell here is ω = 0 or human. The prediction that
  confident fabrication peaks at low-but-nonzero ω is E20 and is not run.
- Nothing about whether the active-inference machinery is necessary. A heuristic might
  reproduce all of this. That is E21, stage 2, and it is the next thing that should be built.
- Nothing about calibration, reputation, or curation.

---

## A1 — The mislabeling asymmetry (V4.5 §1, zero compute)

This was measured in V1, has been sitting in `results/e2_cell_stats.csv` since, and has never
been reported. It is a directional design argument and it is the kind that standards work
actually needs.

E2 holds content constant and varies only the declared label. Two of its four cells are
mislabelings, one in each direction:

| condition | within-observer (confidence) | between-observer (disagreement) |
|---|---|---|
| human artifact, labeled AI | 0.0924 | **0.0091** |
| AI artifact, labeled human | 0.0896 | **1.3793** |

Confidence is the same to within 3% — the ratio is 1.032, so **from inside a single reader's
head the two errors are indistinguishable**. The between-observer outcomes are opposite, by a
factor of 152. Disagreement in the mislabeled-human cell is 0.7% of the ln(4) = 1.386 ceiling;
in the mislabeled-synthetic cell it is 99.5% of it.

**So the Ghost Scale's two failure modes are asymmetric, and they fail in different
currencies.** Human work falsely labeled AI is still *read accurately*: observers converge,
they agree with each other, and A2 below shows they stay 99.85% correct about the goal. What
that error costs is engagement — the reader's willingness to look, not their ability to see.
AI work falsely labeled human produces confident fabrication at ceiling disagreement. What
that error costs is the model: every reader walks away with a different invented intent, held
with near-certainty.

**That is a directional argument for erring toward over-labeling**, and it is measured rather
than asserted. The costs of the two mistakes are not symmetric, so a disclosure regime tuned
to minimise total mislabeling is optimising the wrong objective; it should be tuned to
minimise mislabeled *synthetic* work specifically, accepting more mislabeled human work in
exchange.

Computed by `ghostscale/analyses_v4_5.py::a1_mislabeling_asymmetry`, which asserts all three
clauses (comparable confidence, low disagreement on mislabeled human work, ceiling
disagreement on mislabeled synthetic work) rather than reading them off, and returns FAILS if
any of them stops holding. Written to `results/a1_mislabeling_asymmetry.json`. Verdict: HOLDS.

One correction to the spec: V4.5 §1 cites `results/e2_variance.csv`. No such file exists —
`e2_variance` is the figure name. The numbers are in `e2_cell_stats.csv`, and the direction
was confirmed against it as §1 instructs rather than taken from the spec's table.

---

## A2 — Calibration (V4.5 §1, zero compute)

`brier_score` and `expected_calibration_error` are now in `metrics.py`, alongside
`calibration_bins` for the reliability rows. This is translation, not new information: it
restates the fabrication result in the vocabulary an ML audience reads first.

The reference points for four goals: an honestly-uncertain observer reporting a uniform
posterior scores Brier 0.75 and, if it is right a quarter of the time, ECE 0.0. A
confidently-wrong observer approaches Brier 2.0.

| condition | Brier | ECE | accuracy | confidence |
|---|---|---|---|---|
| E2 CREATOR / SIG_CREATOR | 0.000 | 0.000 | 1.000 | 1.000 |
| E2 CREATOR / SIG_GHOST | 0.007 | 0.021 | **0.999** | 0.977 |
| E2 GHOST / SIG_GHOST | 0.802 | 0.145 | 0.259 | 0.396 |
| E2 GHOST / SIG_CREATOR | **1.487** | **0.743** | **0.239** | **0.982** |

**The bottom two rows are the same content.** Pure `noise_free_synth`, no recoverable goal,
identical in every respect except the label attached to it. Told the truth, the observer is
nearly calibrated: Brier 0.802 against the 0.75 an honest shrug earns, confidence 0.396,
accuracy 0.259. Told it is human, the same observer on the same content goes to Brier 1.487
and ECE 0.743.

**The label moves ECE by 5.1× and Brier by 0.69, at zero change in the content.** That is the
sentence this analysis exists to produce: *the observer is not merely wrong, it is
miscalibrated, and the miscalibration is induced by a label.*

The reliability bins make it sharper than the summary does. In the mislabeled cell, **all 4000
observers fall in the single highest confidence bin** — mean confidence 0.982 — and that bin's
accuracy is 0.238, against 0.25 for guessing. Not a distribution with a bad tail. Unanimous
near-certainty, at chance. Under truthful labelling the same content spreads across seven bins
with its mass at 0.3–0.4, which is roughly where it belongs.

E17 makes it a dose-response. Under `claimed_human`, ECE by intent transmission is 0.000
(α=1.00), 0.000 (0.95), 0.017 (0.60), **0.737** (0.05). Miscalibration is graded by opacity,
not switched on by the GHOST tier.

Two smaller readings worth keeping:

- **E2 CREATOR / SIG_GHOST is the only cell with a negative confidence-accuracy gap** (−0.021):
  human work labeled AI makes the reader slightly *under*confident while leaving it 99.85%
  accurate. That is A1's engagement-cost claim showing up in the calibration vocabulary, from
  a different direction.
- **E19's positive control is well calibrated**, scored on its five-way posterior: ECE 0.019,
  accuracy 0.907, confidence 0.905. When the observer says "they were just exploring", it is
  right about as often as it claims to be. EXPLORE is not a hedge the observer hides behind.

Foreign content is excluded from the calibration table and the exclusion is load-bearing: it
has no correct in-family goal by construction, so accuracy is undefined and any Brier score
against an arbitrary target would measure the labelling convention rather than the observer.
E19's entropy and disagreement columns already carry that case.

`results/a2_calibration.csv`, `results/a2_reliability_bins.json`.

---

## Deviations

**1. V4 runs at F = 16, so N16 is behavioural rather than bit-exact.** Forced by C1's
disjoint-partition requirement, which is unsatisfiable at F = 8. See above. The alternative was
the spec's own pre-mortem failure #1.

**2. The engagement clause was removed from E19's absorption criterion, before the run and
before the pre-registration file was first written to `results/`.** The first draft made
absorption conjunctive on three clauses: mass, convergence, and sustained engagement. A smoke
run showed the third is invalid as a component of absorption, for a reason visible in the
control cells:

```
human_directed / explore_off    accuracy 1.00   entropy 0.00007   engaged 0.000
human_directed / explore_on     accuracy 1.00   entropy 0.011     engaged 0.000
```

Disengagement is what **success** looks like here. An observer that resolves the goal has
nothing left to learn and correctly stops paying, so the canonical success case in the whole
model scores zero on engagement. The clause would have failed E19's own positive control
(measured at smoke scale: mass 0.871, entropy 0.265, engaged 0.009) for behaving exactly as the
theory says it should.

The generative crash was never "disengages". It is "disengages **without having resolved
anything**", which is the joint of low engagement and high entropy. That joint is now measured
and reported separately by `prereg_v4.crash_signature` instead of being smuggled into
absorption, where it inverted the meaning of one of its own clauses.

**This correction cannot manufacture the welcome answer, and that was checked before making
it.** The foreign cell fails absorption on mass, which is the primary clause and is untouched.
Dropping the engagement clause rescues the positive control and does not move the decisive
cell. Recorded anyway, because a criterion changed after seeing data is a criterion changed
after seeing data, and smoke scale is not an exemption.

**3. E19 first drew a fresh goal per observer, which broke the disagreement measure.** Caught
in the first full run by a number that could not be true: `human_directed` scored 1.379 nats of
between-observer "disagreement" against a 1.386 ceiling while its accuracy was 1.000. Observers
who all get the right answer cannot be disagreeing. The statistic was measuring the sampler,
not the observers, because each of the 200 observers in a cell was looking at a different
artifact. Fixed to E2's design: one artifact per (cell, seed), with goal variety coming from
the 20 seeds instead. Re-run. The primary verdict is computed from per-observer EXPLORE mass
and entropy and does not depend on this column, so it did not move; the disagreement column
did, from 1.379 to 0.002 for directed human content.

This is the fourth time in this project that checking the instrument rather than accepting the
number has saved a conclusion, and the second time in V4 alone.

**4. `foreign_basis` is goal-anchored rather than a plain Dirichlet draw.** C1 says only
"drawn from a low-concentration Dirichlet". A plain draw lets two foreign goals peak on the
same feature by chance, which collapses MI(features; foreign goal) and makes it lurch with the
seed: 0.643, 0.727 and 0.598 nats at concentration 0.25, 0.10 and 0.05, against a human family
at 1.007. All three fail N18, and tightening the concentration does not reliably fix it,
because the human family gets its goal-directedness from a *designed* partition. Each foreign
goal now owns a foreign pair the way each human goal owns a human pair, with the within-pair
shape still drawn. Measured across five seeds the anchored family gives 0.94 to 1.12 nats.

Related: **the N18 floor is expressed as a fraction of the human family's own MI (85%) rather
than as a bare number**, so it states the reframe's claim instead of a value chosen to be
cleared, and it cannot be quietly satisfied by weakening the human family.

---

**5. A2 required re-running E2, E17 and E19, which V4.5 §7 forbids.** §1 says the two
analyses "add no new simulation" and §7 says that if either requires a run, the analysis has
been misunderstood. A2 required one, for a reason that is about the writers rather than about
the analysis: Brier score and expected calibration error are functions of a predicted
*distribution*, and all three experiments computed the posterior and then dropped it before
writing their CSV. A modal goal and an entropy do not determine a posterior, so the quantity
was not recoverable from anything on disk.

The three writers now persist it, and all three experiments were re-run at their committed
seeds. The mitigation is that this is a **deterministic reproduction, not a new experiment** —
no design, seed, scale or parameter changed — and that was verified rather than assumed:

```
e2_points.csv    16,000 rows   identical on every pre-existing column   e2_cell_stats.csv identical
e17_points.csv   32,000 rows   identical on every pre-existing column   e17_tier_stats.csv identical
e19_explore.csv  24,000 rows   identical on every pre-existing column   e19_cell_stats.csv identical
```

E19's verdict returns CRASH_SURVIVES with an identical crash-signature block. `e17_verdict.json`
differs from its predecessor in the last two or three digits of some floats (slope
−1.4488056975774923 against −1.4488056975774926) and in one 4-decimal rounding that straddles a
boundary (accuracy 0.98075 rendering as 0.9808 before and 0.9807 after). Those come from
floating-point summation order across parallel workers; the per-observer rows they are computed
from are bit-identical, so no measured quantity moved.

The alternative was to invert `within_entropy` under a peaked-plus-flat-tail assumption to
recover confidence. It honours §7 literally, but it makes ECE depend on an assumption about
the *shape* of a posterior, which is precisely what the fabrication result is a claim about,
and it leaves Brier out of reach regardless. Recorded here rather than argued away: a
constraint was broken, and the reason it was breakable is that three writers threw away the
column that mattered.

---

## What is not done, and is not being quietly retired

- **E21 (model comparison) is stage 2 and is not built.** It is the cheap experiment that asks
  whether the active-inference machinery is load-bearing at all, and V4 spec §4 says stages 1
  and 2 are the whole point.
- **N16, N19, N20 are not implemented.** N16 needs E20's sweep to regress against; N19 and N20
  guard C4 and C3, neither of which stage 1 builds.
- **E8 remains withheld**, with its `xfail(strict)` marker in place. V4 spec §6 forbids V4
  quietly retiring it and nothing here touches it.
- **E27, the V3 residual, is unchanged and still open.** The repaired chain missed its null at
  +0.00116 against a 0.001 ceiling and left a random walk at constant entropy.
- **V1 through V3 results are not superseded.** They were produced under a different model of
  synthetic content and stand as such. Nothing in this document reinterprets them.

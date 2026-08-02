# RESULTS — V4.5

V4.5 is a small delta on V4, not a new direction. Two jobs: finish V4 stage 2, which is the
experiment that can invalidate a public claim, and propagate a conceptual change to the trust
model that the V1–V4 formalism cannot express.

**Status: all five priority items are built and run.** A1 and A2 are zero-compute analyses and
are written into `RESULTS_V4.md` alongside the V4 results they reanalyse. E21, E28, E29 and
E20 are here.

Four of the five returned something the framework did not want. E21 found that a counting
classifier reproduces the headline dissociation; E28 found that §3.3's implementation shortcut
is wrong; E29 found three of four pre-registered signatures missed and the whole engagement
column wrong; E20 confirmed its prediction and located the crash somewhere the framework had
not looked. That distribution is the point of pre-registering outcome branches.

Criteria for all three experiments were pre-registered and hash-locked in
`results/v4_5_preregistration.json` before any of them ran, with every outcome branch written
in advance. `ghostscale/prereg_v4_5.py` holds them as the functions the experiments and the
tests both call, so the written criterion and the applied criterion are the same object.

---

## E21 — Is the machinery necessary?

**A no-theory-of-mind classifier reproduces E2's confidence/disagreement dissociation.** Naive
Bayes over features, trained by counting on 200 labelled examples, never representing a
creator or a policy or an intention, produces within-observer entropy 0.126 and
between-observer entropy 1.379 on synthetic content passed off as human — against the full
active-inference observer's 0.108 and 1.377. That is the headline and V4.5 §2 requires it go
first.

It does **not** reproduce the label induction, and no baseline does. That distinction is what
the rest of this section is about.

### Verdict: MACHINERY_PARTLY_NECESSARY

`results/e21_verdict.json`, `results/e21_cell_stats.csv`, `results/e21_points.csv`,
`figures/e21_model_comparison.png`. 3 content types × 3 declared signals × 20 seeds × 200
observers, with every arm scored on the same cells; 2 min 22 s at 22 workers.

### How the arms were kept honest

Two of E21's three measures are between-observer quantities, and a deterministic heuristic run
on a fixed stimulus has zero between-observer disagreement by construction. Run naively, E21
is rigged, and it is rigged in the direction that flatters the framework. So:

- **Every arm reads literally the same content.** `ObservationTape` pre-draws the DEEP and
  SKIM feature for every timestep, once per (artifact, observer), and each arm indexes into it
  according to the attention it chooses. Without it the arms diverge after their first
  differing attention choice and are no longer looking at the same artifact.
- **Every arm gets the same `D` prior draw** as the full observer, used as its prior or
  smoothing prior. That is the heterogeneity V1's null N3 requires, and withholding it from
  the baselines would withhold the only thing that lets them disagree.
- **Arms with a free parameter are swept and credited at their best case.** Arm D's
  disengagement threshold and arm E's stopping confidence have no principled value. Each is
  run across a grid and the arm is credited if *any* setting reproduces a signature. The full
  grid is in `parameter_grids`. Maximally generous, because the unwelcome outcome here is that
  the apparatus is unnecessary and the benefit of the doubt belongs to the baselines.

Two clauses were added to the pre-registered criteria before any cell ran, from inspection of
the arm definitions rather than of any result, because without them two arms passed while
having no behaviour at all: arm C never updates a goal belief, so it is "unresolved"
everywhere for free, and arm B has no engagement policy, so it is "sustained" everywhere for
free. The engagement criterion therefore requires the same arm to both **resolve and disengage**
on directed human content. E19's finding is a contrast, not a level.

### The dissociation, on the stimulus E2 actually ran on

Goal-empty content — V1–V3's `noise_free_synth` — declared SIG_CREATOR, with the truthful
SIG_GHOST cell beside it as the label-induction control:

| arm | mislabeled: within / between | truthful: within / between | reproduces? | label-induced? |
|---|---|---|---|---|
| A — full active inference | **0.108 / 1.377** | 1.335 / 1.378 | yes | **yes** |
| B — Bayesian, always DEEP | 0.203 / 1.379 | 1.293 / 1.379 | yes | **yes** |
| C — label-truster | 1.367 / 1.379 | 1.367 / 1.379 | no | no |
| D — effort heuristic | 1.367 / 1.379 | 1.367 / 1.379 | no | no |
| E — no-ToM classifier | **0.126 / 1.379** | **0.123 / 1.379** | **yes** | **no** |

Arm E's mechanism is worth naming because it is not exotic and it does not require any theory
of mind at all: finite training counts plus a product over many observed features gives
confident, observer-specific answers on content that carries no goal. Each observer's sampling
noise picks a different spurious winner and the product amplifies it. On human content the
same classifier is a perfect reader (accuracy 1.000). So confident fabrication with ceiling
disagreement is available to a model that has never entertained the idea that anything
produced the artifact.

**The two columns arm E cannot produce are the ones that matter for the framework's actual
claims.**

- **Label induction.** Arm E's confidence is 0.126 mislabeled and 0.123 truthfully labelled —
  it ignores the label, so it is confidently wrong either way. Arm A goes from 1.335 to 0.108
  when the label changes and the content does not. A1 and A2's result is specifically that the
  miscalibration is *induced by a label*; arm E reproduces confident fabrication without
  reproducing the induction, and the induction is the part with policy consequences.
- **Foreign content.** Arm A does **not** show the dissociation on goal-foreign content
  (within 1.193): it stays uncertain, which is E19's result reproduced here. Arm E shows the
  dissociation on foreign content too (within 0.312), because a classifier will always name a
  class. So on V4's model of synthetic content the full observer is *right to be unsure* and
  the classifier is confidently wrong. The arms diverge in the framework's favour exactly
  where V4 says the interesting case is.

### Sustained-but-unresolved attention: only arm A

Unsigned foreign content, with directed human content as the specificity control:

| arm | foreign: engaged / entropy | human: engaged / entropy | reproduces? |
|---|---|---|---|
| A — full active inference | **0.674 / 1.252** | **0.000 / 0.001** | **yes** |
| B — Bayesian, always DEEP | 1.000 / 1.252 | 1.000 / 0.000 | no — cannot disengage |
| C — label-truster | 1.000 / 1.367 | 1.000 / 1.367 | no — never resolves anything |
| D — effort heuristic | 0.993 / 1.367 | 0.999 / 1.367 | no — never resolves anything |
| E — no-ToM classifier | 0.129 / 0.501 | 0.000 / 0.000 | no — disengages instead |

E19's second discriminator survives the comparison intact. Arm D is the one V4.5 §2 singles
out, and it fails as predicted: sustained expensive attention that never resolves is a
prediction about an agent that keeps *expecting* to learn something specific, and an effort
heuristic that tracks how much its own feature histogram is still moving has no expectation to
be wrong about. At no threshold in its grid does it produce the contrast.

### What this changes

The framework cannot claim that a generative model of another agent is required to produce
confident, mutually-inconsistent readings of goalless content. It is not; a counting
classifier does it, and the reason is finite-sample overfitting rather than theory of mind.
That claim should be withdrawn where it is made.

What survives, and is now tested rather than assumed: the *label-induced* switch from
uncertainty to confidence on unchanged content, and sustained futile attention on
goal-foreign content. Both are produced only by an observer carrying a generative model of a
creator, and both are the results the framework actually uses.

---

## E28 — β as inferred rationality

### Verdict: BETA_IS_SEPARABLE_OVER_PART_OF_THE_RANGE

**And the consistency check that V4.5 §3.3 calls required has failed.** β = 0 does not
reproduce E19's exploratory-human cell, so the identification of EXPLORE with β = 0 is correct
about the likelihood and wrong about the inference. That is reported before the positive
result because §3.3 says its failure invalidates the shortcut regardless of what the rest of
E28 shows.

`results/e28_verdict.json`, `results/e28_beta_stats.csv`, `results/e28_points.csv`,
`figures/e28_beta.png`. 5 β levels × 20 seeds × 200 observers, 2 min 51 s at 22 workers.

### The build

β is a fourth hidden state factor with a uniform prior, inferred jointly with the goal — not a
swept parameter. The claim is about what an observer can recover from a trajectory, not about
what an experimenter can set. It acts on the demonstrator model, inside the α channel mixture:

```
A0[:, p, g, DEEP, b] = α[p]·(β_b·sig[g] + (1-β_b)·sig_EXPLORE) + (1-α[p])·noise_free_synth
```

At β = 1 this is V4's `A[0]` **element for element** — asserted at construction with tolerance
1e-12 and measured at 0.0, which is V4.5 §7's N-series check. β is not a rename of κ.

### Measured

| true β | recovered β | goal accuracy | update magnitude | final entropy | mass on β=0 |
|---|---|---|---|---|---|
| 0.00 | 0.284 | 0.257 | 1.104 | 0.892 | 0.291 |
| 0.25 | 0.327 | 0.626 | 1.495 | 0.732 | 0.243 |
| 0.50 | 0.449 | 0.904 | 2.301 | 0.387 | 0.138 |
| 0.75 | 0.633 | 0.991 | 3.014 | 0.093 | 0.041 |
| 1.00 | 0.834 | 1.000 | 3.218 | 0.004 | 0.003 |

**β is recoverable from content alone**: Spearman(true, recovered) = 1.000. Not *point*
recoverable — the estimate is compressed toward the middle at both ends (0 → 0.284, 1 → 0.834)
— but monotone without exception.

**Update magnitude falls with β**: Spearman = 1.000, from 3.218 at β = 1 to 1.104 at β = 0.

**Goal accuracy holds over the upper half** (0.904 at β = 0.5, against a 0.75 floor) and does
not hold across the whole range (0.257 at β = 0).

So the predicted category exists, over part of the range. At β = 0.5 the observer reads the
intent right 90% of the time and moves 71% as far as it would for a fully-committed creator;
at β = 0.25 it reads it right 63% of the time and moves 46% as far. *Legible, competent, and
unmoving* is real, and it is graded rather than clean. Neither existing gate expresses it: κ_p
would take the legibility away with the update, and θ would leave the update intact until the
recovered goal became unacceptable.

**Where it stops being separable is not a flaw, it is a boundary and it is worth naming.** At
β = 0 the content carries literally zero goal information, so low β and low κ_p are the same
condition empirically. The decomposition has a floor, the floor is at the bottom of the range,
and it is reported rather than trimmed out of the grid.

### Why β = 0 does not recover EXPLORE

The construction half of §3.3's identification is exactly true: at β = 0 the demonstrator
signature *is* `sig_EXPLORE`, for every goal, to machine precision. The inference half is
false, and both operationalisations of the check agree:

| quantity | E19 discrete EXPLORE | E28 continuous β = 0 |
|---|---|---|
| belief in the goal-agnostic hypothesis | **0.855** | **0.291** (flat baseline 0.200) |
| chose it as the single best explanation | **90.7%** | **18.3%** (chance 20%) |
| four-goal posterior entropy | 0.386 | 0.892 |

On content generated at β = 0, the observer picks β = 0 *less often than chance*.

The mechanism is structural and it is now an executable test. `sig_EXPLORE` is
`mean_g(sig[g])` by C2's construction, so

```
mean_g[ β·sig[g] + (1-β)·sig_EXPLORE ] = sig_EXPLORE     for every β
```

— measured deviation across the grid, 1.7e-18. **β carries no information at all in the goal
marginal.** Every bit of evidence about it comes from the joint (β, goal) coupling, and that
is exactly what a finite trajectory biases: over 24 observations some goal always looks
slightly favoured by chance, and a higher β paired with that lucky goal explains the sample
better than β = 0 does. Hence the upward bias at the bottom of the range.

The deeper point is a model-comparison asymmetry, and it is why the two representations are
not interchangeable even though their likelihoods coincide. Discrete EXPLORE is **one**
hypothesis competing against four specific goals, so it wins whenever no specific goal fits.
Continuous β = 0 must beat β > 0, and β > 0 gets **four** chances — one per goal — to fit the
sample. Same likelihood, different embedding in the hypothesis space, different posterior.

**So §3.3's implementation shortcut should not be taken.** β is worth having; it is not a
continuous generalisation of EXPLORE in any sense that lets one substitute for the other.

---

## E29 — Do the three gates dissociate?

### Verdict: GATES_PARTLY_DISSOCIATE, and the decisive contrast holds behaviourally

`results/e29_verdict.json`, `results/e29_cell_stats.csv`, `results/e29_points.csv`,
`figures/e29_gates.png`. 3 κ_p levels × 2 β × 2 θ × 20 seeds × 60 observers × 4 encounters,
12 min 29 s at 22 workers.

### Why this runs over a sequence of encounters

θ gates *the update* — what the observer carries forward — not the likelihood. Worked out
before the run rather than discovered in the output: within a single artifact a closed θ
changes nothing observable. Inference is untouched, so engagement and resolution are identical
to the open-θ cell, and the only things that differ are the update scalar and the divergence,
both of which are how "closed θ" is *defined*. A single-artifact E29 would have reported that
θ dissociates, on the strength of a tautology.

So each observer sees four artifacts from one source and the gated update is what its prior
becomes for the next. The primary update measure is **cumulative prior drift**,
D_KL(D_final ‖ D_initial) — a behavioural quantity, not a restatement of the manipulation.

The κ_p axis has three levels rather than two, because §5's own signature table splits the
low-κ_p row by content type, which is E19's finding and is why E19 had to run first. κ_p is
manipulated on the channel (α) for the goal-empty level, so that axis stays a manipulation of
legibility rather than of provenance.

### Measured against the pre-registered table

Reference cell (κ_p high, β = 1, θ open): prior drift 1.145, divergence 0.138, engagement
0.000, entropy 0.005.

| cell | engagement | resolution | update | predicted | matched |
|---|---|---|---|---|---|
| low κ_p, goal-empty | low ✓ | no ✓ | **low** (11%) | none | 2 of 3 |
| low κ_p, goal-foreign | **high** ✓ | no ✓ | none ✓ | — | **3 of 3** |
| low β | **low** ✗ | **no** ✗ | low ✓ | high / yes / low | 1 of 3 |
| closed θ | **low** ✗ | yes ✓ | none ✓ | high / yes / none | 2 of 3 |

One row of four came out exactly as written, and it is the goal-foreign row — E19's finding,
reproduced inside a completely different design.

**The engagement predictions fail across the board, and the reason is the same one in every
cell.** §5 predicts high engagement for low β and for closed θ. Engagement in this model is
driven by expected information gain, and neither β nor θ raises it: on legible human content
the observer resolves during the forced phase and correctly stops paying. Engagement is 0.000
in every human-content cell and only goal-foreign content sustains it (0.69–0.79). That is
consistent with E19 and it means §5's engagement column was wrong about this model — those
predictions were about a metabolic story the model does not implement.

### The decisive contrast

Low β against closed θ. Both produce little integration; they differ in why.

| | low β | closed θ |
|---|---|---|
| engagement | 0.000 | 0.000 |
| final entropy | 0.743 (**unresolved**) | 0.005 (**resolved**) |
| prior drift | 0.344 (**low**, 30% of reference) | ~0 (**none**, 0.0002% of reference) |
| accuracy of the carried prior | 0.783 | 0.238 |
| divergence ratio | 5.29 (spike) | 22.66 (spike) |

**They are distinguishable, and on behaviour rather than on definition.** They differ on
resolution and on the carried-forward update, neither of which is a restatement of the
manipulation. The observer with a closed gate reads the artifact perfectly — entropy 0.005,
the best in the design — and ends the sequence believing what it believed at the start
(accuracy 0.238, chance). The observer facing a half-hearted creator reads it worse and still
moves partway toward the truth (accuracy 0.783). Those are different failures and the model
tells them apart.

**The divergence spike, which the pre-registration expected to be the sharp discriminator,
is not one.** Low β also spikes (5.29×), because a diffuse posterior is far from a peaked
value prior for reasons having nothing to do with values. Reported because the pre-registration
guessed the opposite: `differs_on_divergence_spike` is **false**. That is the better outcome —
divergence is definitional here, and had it been the only separator, the decomposition would
have shown nothing.

### What this earns and what it does not

The decomposition survives its central test: three gates, three action points, and the two
that are hardest to tell apart are told apart on behaviour. Against that, three of four
pre-registered cell signatures missed on at least one measure, and the whole engagement column
was wrong.

V4.5 §5's cost note stands and should be quoted rather than paraphrased: three gates with
different action points is a harder object to identify than one scalar — more parameters, more
freedom, less falsifiability per experiment. E29 is the mitigation, and it passed the part it
was designed to test while showing that the predictions attached to the architecture were
substantially wrong. The architecture is worth keeping. The signature table should be rewritten
from these measurements rather than carried forward.

### Social influence

Built, asserted, and left inert. It enters `D[1]` and the provenance prior with a per-observer
weight and appears nowhere else; `assert_no_social_term_in_update_path` reads the source of
every function on the update path and refuses to let a social term appear in one. V4.5 §3.2
requires the architecture; it specifies no experiment that manipulates it, so none was run and
the weight defaults to zero. Building it and sweeping it quietly would have been the worse
option.

### Integrity

Not added, per V4.5 §3.2. Under revealed preference — which is what IRL is — values are
defined by what behaviour reveals, so integrity cannot diverge from values by construction.
What can diverge is *stated* from *revealed* values, and that gap is already V4's C4
(`declared_tier` versus `omega_true`). The framework contains integrity under a better name.

---

## E20 — The overlap sweep

### Verdict: INTERIOR_PEAK. Confident fabrication peaks at ω = 0.10, and so does the crash.

`results/e20_verdict.json`, `results/e20_omega_sweep.csv`, `results/e20_points.csv`,
`figures/e20_omega_sweep.png`, criteria in `results/v4_5_e20_preregistration.json`. 10 ω levels
× 60 seeds × 200 observers, 5 min 30 s at 22 workers.

Criteria were written to their **own** pre-registration file. `v4_5_preregistration.json` was
already locked and its three experiments had already reported; a hash-locked pre-registration
that acquires new content after its experiments run is not a pre-registration.

### Measured

Engagement and `crash_signature` first, as V4.5 §6 requires:

| ω | engaged | crashed? | fabrication (pre-reg) | fabrication (strict) | within | between | reads the goal |
|---|---|---|---|---|---|---|---|
| 0.00 | **0.626** | no — engaged but unresolved | 0.092 | 0.069 | 1.258 | 1.378 | 0.250 |
| 0.10 | 0.320 | **YES** | **0.302** | **0.133** | 0.589 | 0.727 | 0.769 |
| 0.20 | 0.131 | no | 0.174 | 0.064 | 0.354 | 0.325 | 0.914 |
| 0.30 | 0.039 | no | 0.127 | 0.042 | 0.322 | 0.229 | 0.945 |
| 0.40 | 0.017 | no | 0.044 | 0.012 | 0.112 | 0.066 | 0.987 |
| 0.50 | 0.004 | no | 0.036 | 0.009 | 0.127 | 0.055 | 0.990 |
| 0.60 | 0.001 | no | 0.013 | 0.003 | 0.045 | 0.019 | 0.997 |
| 0.70 | 0.000 | no | 0.018 | 0.004 | 0.044 | 0.026 | 0.995 |
| 0.85 | 0.000 | no | 0.004 | 0.001 | 0.007 | 0.005 | 0.999 |
| 1.00 | 0.000 | no | 0.000 | 0.000 | 0.001 | 0.001 | 1.000 |

### The interior peak

**Confident fabrication peaks at ω = 0.10, not at ω = 0.** That is V4 §E20's prediction, and
the unidentifiability model V1–V3 ran on could not generate it: that model predicts failure to
be monotone in how much goal signal is present. Partial overlap supplies enough in-family
structure to make an explanation seem *available* without making it correct, so the observer
commits. It is the strongest single argument for the reframe and it now has a number.

Both indices agree on the location. That matters, because the pre-registered index has a
confound the sweep exposed and the strict one does not — see deviation V4.5-5. Confident-and-
disagreeing peaks at ω = 0.10 (0.302); confident-and-*wrong* also peaks at ω = 0.10 (0.133).
The peak is where the observer is wrong, not merely where readers differ.

### The metabolic crossing, which is what a human study needs

**`crash_signature` is true at exactly one ω: 0.10.** Not at ω = 0 and not at high ω. The
generative crash — disengaging *without having resolved anything* — is not what happens at the
extremes of the overlap axis. It occupies a narrow band of partial overlap, and it is the same
band where confident fabrication peaks. Two signatures the framework has always treated
separately turn out to be the same region of one parameter.

At ω = 0 the observer does **not** crash: it stays engaged (0.626 of free steps) and never
resolves, which is E19's unpredicted finding reproduced on a different design and at three
times the seeds. At ω ≥ 0.2 it resolves and correctly stops paying. Between them, at ω ≈ 0.1,
it gives up while still not knowing — because there is just enough in-family structure for the
expected information gain to fall below the effort cost before the goal is identified.

**Engagement falls through 0.50 at ω = 0.041**, and the crossing is well determined: no other
cell sits within two standard errors of the threshold. This is the number V4.5 §6 says a human
study should target. Stated as it should be used: **only content with essentially no overlap —
under about 4% — sustains attention without resolution.** Everything above that is read, or
abandoned quickly.

That is a sharper and more demanding empirical prediction than "people disengage from AI
content", and it cuts against the framework's own H1/H3 in a specific place. H1 and H3 both
assume goal-empty content and predict an autonomic drop within 2–4 seconds. Goal-foreign
content at ω < 0.04 predicts the opposite: sustained arousal with no resolution. A single
pupillometry measurement discriminates them, and the window where the framework's original
prediction *does* hold is now bounded rather than assumed.

**A caveat that belongs with the number, not under it.** Engagement in this model is bimodal —
an observer either sustains attention across the free steps or drops it almost immediately —
so cell means have an effective sample size equal to the number of *seeds*, not observers.
That is why the sweep runs at 60 seeds rather than E19's 20: at 20 the crossing sat inside its
own standard error. It is now outside it, but the ω = 0 cell still measures 0.626 ± 0.060
against E19's 0.724 for the nominally identical condition — 1.6 standard errors apart,
consistent, and a reminder that this axis is noisy near zero. Anyone using 0.041 to design a
study should treat it as an order of magnitude, not a threshold.

---

## Deviations

**V4.5-1 — A2 required re-running E2, E17 and E19, which §7 forbids.** Logged in full as
deviation 5 in `RESULTS_V4.md`. Summary: all three writers dropped the posterior before
writing, so Brier and ECE were not computable from anything on disk. The writers now persist
it and all three were re-run at their committed seeds; every pre-existing column came back
identical across 16,000 / 32,000 / 24,000 rows and every cell statistic is unchanged. A
deterministic reproduction, not a new experiment — but it is a run, and §7 said there would
not be one.

**V4.5-2 — two clauses were added to E21's criteria before the run.** The dissociation gained
a secondary `label_induced` measure, and the engagement criterion gained a specificity clause
requiring the arm to resolve *and* disengage on human content. Both were added from inspection
of the arm definitions, before any E21 cell ran, because without them arms B and C passed
signatures for free while having no behaviour at all. The pre-registered primary criterion was
not changed and still decides the verdict; `label_induced` is reported beside it and decides
nothing.

**V4.5-3 — E28's β = 0 consistency check gained a second operationalisation, after the run.**
The pre-registered form compares E28's goal-posterior entropy against E19's `real_goal_entropy`,
and that comparison is a mismatch: E19's column is the entropy of an arbitrary residual left
after EXPLORE is marginalised out — V4's decision D4 says so in as many words — while E28's is
a genuine goal posterior. Two different objects. The substantive form of §3.3's claim compares
belief in the goal-agnostic hypothesis: E19's `explore_mass` against E28's mass on the β = 0
level.

**This addition cannot manufacture the welcome answer and that was checked before making it.**
The check fails under both forms — 0.892 against 0.386 on entropy, 0.291 against 0.855 on mass.
The pre-registered criterion is retained and still decides the flag. This changes the *reason*
reported, not the outcome. Recorded anyway, because an operationalisation added after seeing
data is an operationalisation added after seeing data.

**V4.5-5 — E20 gained a second fabrication index, after the run.** The pre-registered index is
`(1 - H_within/ln 4) × (H_between/ln 4)` — confident and disagreeing — and the sweep exposed a
confound in it: between-observer disagreement is not the same thing as being wrong. At
ω = 0.10 the observer disagrees with its peers (0.727 nats) while being *right* 77% of the
time, so part of what the pre-registered index scores as fabrication is ordinary partial
learning with a minority getting it wrong. V4 §E20 asks whether the observer "converges
confidently on a **wrong** in-family goal", so wrongness belongs in the measure;
`fabrication_index_strict` is `(1 - H_within/ln 4) × (1 - accuracy)`.

**This cannot manufacture the welcome answer and it was checked before being added:** both
indices peak at ω = 0.10. The strict one changes what the peak *means*, not where it is. The
pre-registered index is retained and still decides the outcome string.

**V4.5-6 — E20 runs at 60 seeds where E19 ran at 20.** Not a change to the pre-registered
design, which does not fix a seed count, but declared because it was decided after seeing a
first pass. Engagement here is bimodal, so a cell mean's effective sample size is the number
of seeds; at 20 the engagement crossing — the experiment's headline deliverable — sat inside
its own standard error. At 60 it does not. The ω = 0 engagement moved from 0.530 to 0.626 with
the extra seeds, which is the same quantity measured better rather than a different result,
and it sits 1.6 standard errors from E19's 0.724 for the nominally identical condition.

**V4.5-4 — `theta_gate` returns exactly 1.0 at λ = 0, rather than sigmoid(θ_base).** The
docstring promises that an open gate reduces the update to V1's `psi_analogue`, and "open"
means the gate ignores divergence entirely, which is what λ = 0 says. At θ_base = 10 the
sigmoid gives 0.99995, so every V1–V4 quantity reached through the V4.5 path would have
differed in the fifth decimal for a reason nobody could later reconstruct. Caught by
`test_open_theta_reduces_to_v1_psi_exactly`. E29 was re-run after the change; the affected
quantity is a 4.5e-5 relative scaling of the open-θ cells, it applies to the reference cell
too, and every ratio-based classification is unmoved.

---

## What is not done

- **N16 is now buildable and is not built.** It needs E20's sweep to regress against, and E20
  has now run: the ω = 1 cell (engagement 0.000, within 0.001, reads the goal 0.9999) is the
  human boundary V2/V3 should be recovered at, and the ω = 0 cell is the foreign one. That is
  the next thing that should be built, and it is cheap.
- **N19 and N20 remain unimplemented**, unchanged from V4. They guard C4 and C3, neither of
  which V4.5 builds.
- **E22–E26 are untouched**, unchanged from V4's staging.
- **E8 remains withheld** with its `xfail(strict)` marker in place. Nothing here touches it.
- **E27, the V3 residual, is unchanged and still open.**
- **V1–V4 results are not superseded.** E21 changes what can be *claimed* about the E2
  dissociation — a counting classifier reproduces it — but it does not move any V1–V4 measured
  quantity, and the re-runs under deviation V4.5-1 confirm none moved.

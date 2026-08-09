# RESULTS — V5

V5 exists because a careful read-through of V1–V4.5 by the author surfaced four construct errors
and one omission. **None were found by the simulations.** They were found by someone who knows
the theory checking whether the implementation matched it.

**Status: C1–C4 built and run. E30, E31, E32, E33 and E34 have reported.** The standing
literature check (§0) has not been run and is not a spec-implementation task.

Criteria were pre-registered and hash-locked before any experiment ran, with every outcome
branch written in advance. C1 is in `results/v5_preregistration.json`; C2/C3/C4 are in
`results/v5_c234_preregistration.json`, a **separate file** for V4.5's reason — the first was
already locked and its experiments had reported, and a pre-registration that acquires new
content after its experiments run is not a pre-registration. `ghostscale/prereg_v5.py` and
`prereg_v5_c234.py` hold the criteria as the functions the experiments and the tests both call.

### The scoreboard for V5

| experiment | verdict | reading |
|---|---|---|
| **N21** | DEPTH_RECOVERED_NOT_EFFORT | 5.98× — μ is not β renamed, so E30 counts |
| **E30** | DEPTH_MOVES_NOTHING | depth is readable and inert; **C1's central claim fails** |
| **E31** | ONE_MECHANISM_OPPOSITE_SIGNS | **C2 supported**; the crash and the exploit are one mechanism |
| **E32** | TWO_DIMENSIONS | **C3 refuted, informatively**; ω and d are not one channel |
| **E33** | READS_THE_LATENT_WITHOUT_A_TRACE | **C4 half-supported**; the mark is real and unreadable |
| **E34** | not answerable in simulation | shipped as an instrument, not a result |

Two of the five corrections came back against the framework, one for it, and two split. That
distribution is the point of writing every branch in advance.

---

## E30 — μ, estimated model depth, replaces β

### Verdict: DEPTH_MOVES_NOTHING

**Depth changes what a reader recovers and does not change how much it moves them.** Update
magnitude does not scale with depth — Spearman −0.5 against a required 0.99 — while goal
accuracy holds at **1.000 at every depth**. The observer reads a deep artifact and a shallow one
equally well, correctly identifies which is which, and takes on the same amount from both.

That is the pre-registered `DEPTH_MOVES_NOTHING` branch, written before the run in these words:
*"μ is recoverable and inert, which would mean depth is not what gates uptake and C1's central
claim is wrong."* C1's motivating claim is that what gates uptake is the observer's estimate of
how complex a model the creator had. On this model, it does not.

`results/e30_verdict.json`, `results/e30_cell_stats.csv`, `results/e30_points.csv`,
`figures/e30_depth.png`. 2 attention arms × 7 cells × 20 seeds × 200 observers, 4000 observers
per cell.

### Measured — the depth grid, sustained attention

| true μ | recovered μ | goal accuracy | update (pre-reg) | update (structure) | sub-goal entropy |
|---|---|---|---|---|---|
| 1 — execution only | 1.559 | 1.000 | 3.230 | 3.313 | 1.351 |
| 2 — + sub-goal | **2.253** | 1.000 | 3.253 | **3.656** | 1.211 |
| 3 — + goal plan | 2.239 | 1.000 | 3.223 | 3.583 | 1.230 |

### What held, and it is the thing E28 could not do

**Goal accuracy is 1.000 at every depth.** E28's β collapsed to 0.257 at the bottom of its range,
because low-β content carries no goal information — so an observer that failed to read it
demonstrated nothing κ_p could not already explain, and the predicted category existed over only
part of the range. Depth has no such bottom. Shallow content is exactly as legible as deep
content, by construction and now by measurement, so `DEPTH_IS_LEGIBILITY` — the branch where μ
turns out to be doing κ_p's job under a new name — is ruled out on the evidence rather than by
assertion.

**The negative control is exact.** At δ = 0 the execution modes collapse onto `sig[g]`, so a μ = 3
artifact *is* a μ = 1 artifact and there is nothing to find. Recovered μ came back **2.000, standard
deviation 0.000, posterior entropy ln(3) to machine precision** — the observer sits precisely on
its flat prior. Recovery then rises monotonically with mode contrast (Spearman **1.000**):

| δ (how distinct the stages are) | 0.00 | 0.25 | 0.50 | 0.75 | 1.00 |
|---|---|---|---|---|---|
| recovered μ | 2.000 | 2.000 | 2.065 | 2.131 | 2.239 |

So whatever E30 reports about depth is a reading of the artifact and not an artifact of the
hypothesis space. That axis is the cleanest measurement in V5.

### What did not hold, beyond the headline

**`depth_is_recoverable` is false, and not for the reason the flag's name suggests.** Spearman on
the μ grid is 0.5 against a required 0.99. But the failure is entirely in the top two levels:
μ = 1 is separated decisively from both (1.559 against 2.253 and 2.239), while μ = 2 and μ = 3 are
tied within 0.014 — far inside their own spread of 0.32.

**The reading, and it is a limitation of the construction rather than of the theory.** The third
level's contribution is goal-diagnostic: at μ = 3 the execution modes are permuted per goal, so
the ORDER of modes tells you which goal you are looking at. But the observer already identifies
the goal perfectly at μ = 2 — accuracy is 1.000 everywhere — so the third layer supplies
information this observer does not need. **Depth saturates at two levels here because the third
one's payload is redundant with something already known.** A design where the deepest layer
carried something not otherwise available would test the top of the range; this one does not, and
the μ grid should be read as a two-level contrast that happens to have three labels.

### The secondary update measure, which changes the reason and not the outcome

`psi_analogue` is `KL(goal posterior ‖ goal prior)`, and depth is constructed so the goal is
exactly as recoverable at every depth. **The pre-registered measure therefore has no headroom
here by design** — it cannot vary with depth whatever is true of the reader. Reporting only it
would answer "does depth move the reader" with a quantity incapable of answering.

`psi_structure` is the same object one level up: how far the observer's belief about the
creator's whole intent structure moved, goal and execution mode jointly, with μ marginalised out
so the manipulation is not scored as its own effect.

| | μ = 1 | μ = 2 | μ = 3 | δ = 0 control |
|---|---|---|---|---|
| pre-registered `psi_analogue` | 3.230 | 3.253 | 3.223 | 3.236 |
| secondary `psi_structure` | 3.313 | **3.656** | 3.583 | 3.236 |

Depth moves structural uptake by about 10% and moves goal uptake by nothing. The control is
clean: at δ = 0 the structural measure sits at 3.236, below every full-contrast cell, so it is
not counting the hypothesis space.

**This does not rescue the verdict and is not offered as doing so.** It shows the same
saturation at μ = 2, its Spearman is also 0.5, and part of what it credits is the observer
knowing which mode is active at the END — transient state rather than durable uptake. It is a
second view, not a better one. `DEPTH_MOVES_NOTHING` stands.

### The attention contrast

| arm | recovered μ at μ = 1 / 2 / 3 | engagement | DEEP steps |
|---|---|---|---|
| sustained (primary) | 1.559 / 2.253 / 2.239 | forced | 24.0 |
| free to stop | 1.798 / 2.094 / 2.096 | 0.03 / 0.19 / 0.17 | 10.5 / 12.7 / 12.4 |

A reader free to stop recovers less depth, and the compression is toward the prior rather than
toward a wrong answer. Depth lives in the ORDER of execution modes, so ten steps of a
three-and-a-half-block artifact is about one and a half blocks. Two things worth noting: the free
arm's recovery is *monotone* where the sustained arm's is not, which is noise at this separation
rather than a finding; and engagement is higher on deep content (0.19 and 0.17) than on shallow
(0.03), so **the observer pays more attention to deeper work without taking more from it.** That
is consistent with E29's finding that engagement here is driven by expected information gain.

### E28 and E30, side by side

V5 decision 8: E30 is a **new design, not a re-run**, and E28 is not deleted. Under a hierarchical
creator there is no version of E28's stimulus to re-run, so the two are two constructs measured
on comparable designs.

| | E28 — β, rationality | E30 — μ, model depth |
|---|---|---|
| recoverable from content alone | yes, Spearman 1.000 | partly: shallow vs deep, not within deep |
| goal accuracy across the range | 0.257 → 1.000 | **1.000 everywhere** |
| update magnitude | **falls with β**, 3.218 → 1.104 | flat, 3.230 / 3.253 / 3.223 |
| verdict | separable over part of the range | moves nothing |

**The honest summary of C1 so far: μ is the better-specified construct and the worse-performing
one.** β's update effect was real but confounded — update fell with β partly because legibility
fell with β, and E28 said so. μ removes the confound, and with the confound removed the update
effect disappears. That is what the correction bought: a cleaner measurement of a smaller effect,
and the framework should say so rather than presenting μ as a straight improvement.

---

## N21 — does the observer recover depth, or effort under a new name?

### Verdict: DEPTH_RECOVERED_NOT_EFFORT. Dominance 5.98× against a required 3.0.

**This is the gate on E30 and it is the single most important item in V5.** C1 motivates μ with a
dissociation — fully committed but trivial is high β and low μ; an offhand sketch by a master is
low β and high μ — and assigns it to no experiment. Without a test of it, μ could have been β with
a new label, E30 would have reproduced E28's numbers, and the correction would have been cosmetic
while looking like a success.

`results/n21_verdict.json`, `results/n21_cell_stats.csv`. 4 cells × 20 seeds × 60 observers.

| | recovered μ | goal accuracy |
|---|---|---|
| β = 1.00, μ = 1 — committed, trivial | 1.552 | 1.000 |
| β = 1.00, μ = 3 — committed, deep | **2.344** | 1.000 |
| β = 0.25, μ = 1 — offhand, shallow | 1.636 | 0.744 |
| β = 0.25, μ = 3 — offhand, deep | 1.841 | 0.760 |

β's ability to manufacture depth — its effect on content that has none — is **−0.083**, against a
μ effect of **+0.499**. β does not make the observer see depth that is not there.

### N21 earned its place before E30 ran

Run first, N21 caught two defects that would each have surfaced in E30 as a small, plausible,
uninformative effect — and would have been read as evidence against C1 rather than against the
machinery.

**1. pymdp's mean-field inference could not read depth at all.** With μ in a separate factor from
the sub-goal, the μ posterior is updated from the expected LOG likelihood under the sub-goal
marginal, which by Jensen penalises any hypothesis carrying latent structure — and μ = 1 carries
none, so it wins on content that refutes it. Measured on identical feature sequences:

| true μ | exact filtering | mean-field agent |
|---|---|---|
| 1 | 1.52 | 1.04 |
| 2 | 2.20 | 1.04 |
| 3 | 2.27 | 1.06 |

The depth was in the artifact; the solver could not see it, and was *confident* — posterior
entropy 0.047. Fixed by merging μ into the sub-goal's factor, at no cost: the state space is 384
either way.

**2. V4.5's β destroys depth.** It mixes toward `sig_EXPLORE`, which is mode-independent, so a low
β washes out the hierarchy along with everything else and "offhand but deep" — the whole point of
C1's 2 × 2 — was unrepresentable. V5's β mixes toward the goal-marginal of the mode family at that
depth: the craft survives, what attenuates is what it was in service of. This **generalises**
V4.5's β rather than replacing it, and the reduction is exact — at μ = 1 the target is
`mean_g(sig[g])`, which IS `sig_EXPLORE`, asserted at 0.0.

---

## The construction, and what is asserted about it

`ghostscale/v5_model.py`. **A scalar μ was considered and rejected**, and the reason is C1's whole
content: β is a mixture weight on a flat categorical, and any scalar μ built the same way
collapses onto the same axis. Route B, composite observations over an F^k alphabet, is documented
as a fallback and was not needed — a 24-step rollout costs 166 ms in the V4.5 model and 657 ms
here, so E30 runs in minutes.

**Depth is correlational, not marginal, and this is enforced rather than hoped for.** The
execution modes average exactly to `sig[g]` and the mode chains are doubly stochastic with a
uniform stationary distribution, so a deep artifact and a shallow one have identical
time-averaged feature histograms. A reader who counts features and ignores their order cannot
tell them apart at all. Asserted at construction, measured at 5.6e-17.

Asserted at every build, each with its own failure message:

- μ = 1 reproduces V4's `A[0]` **elementwise** — measured 0.0, tolerance 1e-12
- β at μ = 1 reduces to V4.5's `sig_EXPLORE` — measured 0.0
- execution modes average exactly to `sig[g]`; chains doubly stochastic — both 0.0
- modes pairwise separated above a Jensen-Shannon floor — 0.192 against 0.10
- modes carry < 0.10 mass in the FOREIGN block — 0.0755, which is V4's own `sig_true` baseline
- three to four mode blocks fit a 24-step artifact — 3.50

**Three construction faults were caught by these assertions rather than by E30's residuals**, and
each would have been invisible in the output: a first mode family separated by 0.019 nats where
two of four modes were near-duplicates; a rebuilt family that put mode mass in the FOREIGN block,
which `assert_c1_properties` cannot see because it only inspects `sig_true`; and cyclic mode
permutations that gave one goal the identity, leaving it unable to distinguish μ = 2 from μ = 3.

### Decision 10 was revised by measurement

C2's evidence path was to be a joint prior over (provenance, μ) in one factor. That is
incompatible with the merge that fixes the solver: putting all four quantities in a single factor
needs a cardinality-1 placeholder to keep attention at factor index 2, and pymdp raises on it. So
provenance stays its own factor and the evidence path becomes a μ prior conditioned on the
observed label — unchanged in substance, arguably more inspectable, and it gives up one thing:
**the observer no longer infers provenance and depth jointly, so a reader who doubts the label
does not automatically doubt the depth estimate that label induced.** E31 manipulates this path
and must report the limitation.

---

## E31 — two gates, provenance as evidence about depth

### Verdict: ONE_MECHANISM_OPPOSITE_SIGNS. C2 is supported, and it is the only correction that was.

**What a reader takes on tracks how much thought it believes went in, regardless of which
channel put that belief there** — Spearman 0.886 against a required 0.70, across every open-θ
cell. *The pre-registration said "across all cells", and over all twelve the correlation is
−0.02, because closed-θ cells pin the update at zero by design; see deviation V5-4 before
quoting this as the pre-registered pass.* And a dishonest label inflates the depth estimate on content that has none: the exploit
gap is 0.187 against a required 0.15.

`results/e31_verdict.json`, `results/e31_cell_stats.csv`, `figures/e31_two_gates.png`.
2 content × 3 labels × 2 θ × 20 seeds × 60 observers × 4 sequential encounters.

### The headline, which is three cells

Machine content at ω = 0.10 — **the same artifact in both rows** — differing only in what the
label says:

| | recovered depth | what the reader takes on | left uncertain | invents |
|---|---|---|---|---|
| machine work, labelled honestly | 1.892 | **0.039** | 1.350 (unresolved) | 0.000 |
| machine work, passed off as human | **2.079** | **0.848** | 0.242 (resolved) | 0.025 |
| human work, labelled honestly | 2.323 | 1.148 | 0.000 | 0.000 |

That is C2's claim in one table. **The generative crash is a correctly low depth estimate** —
little reasoning behind it, so don't spend, and the reader doesn't. **The trust exploit is a
falsely high one** — the same content, one dishonest word, and the reader spends twenty-two
times as much and resolves onto an answer the content does not support. One mechanism, opposite
signs, and the update is a monotone function of the depth estimate in both directions.

### μ and θ dissociate on behaviour, not on definition

E29's decisive contrast in its corrected form. Both cells integrate almost nothing; they differ
in why:

| | low depth | closed θ |
|---|---|---|
| left uncertain | 1.350 (**never resolved**) | 0.000 (**read perfectly**) |
| what the reader takes on | 0.039 | 0.000 |
| accuracy of the belief carried forward | **0.985** | **0.238** (chance) |

The reader facing shallow content never works out what it was for and still ends up nearly
right. The reader with a closed gate reads the work perfectly — the lowest uncertainty in the
design — and finishes believing exactly what it believed at the start. Different failures, told
apart on quantities that are not restatements of the manipulation.

### The limitation, stated where a reader will meet it

The observer takes the label at face value when forming its depth prior and **cannot doubt it**,
because provenance and depth are no longer in one hidden factor — the merge that would have
allowed joint inference is incompatible with the merge that makes depth readable at all (see the
C1 section). Only content can subsequently move the estimate.

**So the exploit measured here is an upper bound.** A reader able to doubt the label would be
harder to fool. That is the right side to err on for a warning and the wrong side for a
reassurance, and it should be quoted as the former.

---

## E32 — are foreign content and an unskilled reader the same thing?

### Verdict: TWO_DIMENSIONS. They differ on all five measures, and the difference is the useful part.

C3 proposed that ω and observer inexpertise `d` are one channel: creator-side and observer-side
causes of the same gap. **They are not, and the way they fail is more informative than the
collapse would have been.**

`results/e32_verdict.json`, `results/e32_cell_stats.csv`, `figures/e32_omega_d.png`.
6 matched overlap levels × 2 arms × 30 seeds × 200 observers.

### The matching came first and is locked

"Matched effective overlap" is a free parameter, and the spec does not flag it: without fixing
the correspondence in advance one can find a `d` that matches any ω on whichever measure one is
about to report. So it is defined once, computed before any rollout, and hash-locked —
MI(features; goal) under the observer's own likelihood on the content it sees, with `d` solved
by bisection:

| ω | 0.00 | 0.05 | 0.10 | 0.20 | 0.40 | 0.70 |
|---|---|---|---|---|---|---|
| matched `d` | 0.945 | 0.935 | 0.924 | 0.898 | 0.823 | 0.612 |
| effective overlap, nats | −0.012 | 0.039 | 0.090 | 0.192 | 0.395 | 0.701 |

Achieved overlap matches target to four decimals in every cell, so the two arms really are
extracting the same amount of information by construction.

### Measured, at the widest gap

| | competent reader, foreign content | unskilled reader, human content |
|---|---|---|
| keeps looking | **0.611** | **0.001** |
| left uncertain | 1.259 | 0.663 |
| readers disagree | 1.378 | 1.310 |
| gets it right | 0.255 | 0.419 |

**The same information deficit produces opposite behaviour.** The unskilled reader disengages
almost immediately and is comparatively confident; the competent reader facing foreign content
sustains attention through most of the free phase and stays near maximum uncertainty. The
engagement gap is 0.610 out of a possible 1.0.

### The second dimension, which C3 asked to have named

**Whether the failure is detectable to the reader.** A high-`d` reader has templates aimed at
the wrong place but the content is *in support*: its hypotheses give different likelihoods, one
wins, and it wins on that reader's own sampling noise. It fails silently. Low-ω content sits
where every template reads floor, the likelihoods are near-equal, nothing wins, and the reader
keeps paying because it can still see there is something there. It fails loudly.

That was the pre-registered prediction, written before the run in place of "one variable", and
it held on every measure. **The model should carry both variables, and the second one is not
"how big is the gap" but "can the reader tell there is one".**

The zero-compute precursor from E15's and E20's committed grids pointed the same way and is
reported in the verdict as `precursor`, labelled indicative only — different feature counts,
different designs, not matched on overlap.

---

## E33 — the latent goal

### Verdict: READS_THE_LATENT_WITHOUT_A_TRACE

**A reader can know a maker better than the maker knows itself, and the margin grows the more
wrong the maker is.** But the categorical claim about generative systems does not survive: the
mark that self-blindness leaves on the work is real and **no reader in this model can detect
it.**

`results/e33_verdict.json`, `results/e33_cell_stats.csv`, `figures/e33_latent_goal.png`.
3 creator arms × 5 divergence rates × 20 seeds × 100 observers.

### The reader beats the maker's own account

| how often the maker is wrong about itself | reader recovers the real goal | maker's own account is right | margin |
|---|---|---|---|
| never | 0.887 | 1.000 | −0.113 |
| 25% | 0.889 | 0.800 | +0.089 |
| 50% | 0.895 | 0.500 | +0.395 |
| 75% | 0.883 | 0.400 | +0.483 |
| always | 0.886 | 0.000 | **+0.886** |

The reader's recovery is **flat at ~0.89 across the whole range**, and that flatness is the
result. It reads the work, weights the self-report at its stated reliability, and is unmoved by
how wrong that report is. The maker's account degrades to zero; the reader does not follow it
down. Beyond a divergence rate of about 0.1 the reader is the better source on what the maker
was doing.

### The categorical claim, and why it is only half supported

V5 decision 18 restated C4's claim in a falsifiable form: what a generative system cannot
produce is not a divergence — deception produces those too — but **a divergence that leaves a
trace in the artifact.** Three arms make that testable: self-unaware (trace), deceptive
(divergence, no trace), generative (confabulated declaration, no trace).

**The trace is real.** Measured on the artifact itself, as how far its realised structure
departs from what an unconflicted maker produces, on diverged artifacts only:

| self-unaware | deceptive | generative |
|---|---|---|
| **0.896** | 0.628 | 0.603 |

The self-unaware maker's work is marked, and the two arms that merely *say* something false are
not. That is the premise of the categorical claim, and it holds.

**The reader cannot see it.** Recovered depth on those same artifacts is 1.9709, 1.9705 and
1.9707 — a separation of 0.0004 against a required 0.10. All three arms are indistinguishable to
the observer.

**So the mark exists and is unreadable, which is a different result from there being no mark,
and the two would have been indistinguishable without the artifact-side measurement.** The
reason is the same one C3 just named: the conflict puts the work slightly outside what any
hypothesis in the reader's space predicts, and a reader with no hypothesis for "this maker was
avoiding something" has nowhere to put the evidence. **Self-blindness leaves a mark in a
vocabulary the reader does not have** — which is C4 arriving at C3's second dimension by a
completely different route, and is the most interesting thing in V5.

The generative arm behaved as designed: its confabulated declaration diverges from what drove
its output about 10–20% of the time regardless of any rate imposed on it, because it has no
self-model to be wrong *about* — it reports what its output looks like.

### What this does and does not license

It licenses: *an observer can recover a goal the creator does not represent, and beats the
creator's self-report as soon as that report is unreliable at all.*

It does not license: *a self-unaware maker's work is detectably different from a liar's.* In
this model it is different and not detectably so. **The claim that a generative system cannot
produce what a self-blind human produces survives as a fact about artifacts and fails as a fact
about readers**, and the framework should say the first without the second.

---

## E34 — where does real generative content sit on ω?

**Not a result. An instrument.** E34 cannot be run in simulation, the spec says so, and what
this ships is a prediction card mapping each band of the overlap axis to the signature a human
study would observe there — so a measurement can be *located* on the axis rather than argued
about. Zero compute; every number read from E20's committed sweep.

`results/e34_prediction_card.json`, `results/e34_prediction_card.csv`.

| band | ω | what the reader does | what a study would see |
|---|---|---|---|
| sustained and futile | 0.00–0.04 | keeps looking, never resolves | arousal held past 4 s with no resolution |
| **the crash band** | 0.04–0.15 | gives up while still not knowing, and confidently invents when it commits | arousal rises then falls sharply with no resolution event; confident but mutually inconsistent readings |
| read, then abandoned | 0.15–0.40 | works it out quickly, correctly stops paying | brief dilation, quick resolution, low disagreement |
| ordinary reading | 0.40–1.00 | reads it like human work | indistinguishable from human content |

Landmarks: engagement crosses 0.50 at ω = 0.041; fabrication peaks at ω = 0.10 (0.302); the
crash signature is true at exactly one point, ω = 0.10. **The crash and the fabrication peak are
the same band**, which the framework had always treated as two phenomena.

### Both readings are stated, because the framework cannot choose between them from inside

- **Partially foreign.** Real output sits near ω ≈ 0.10, exactly where fabrication peaks and
  attention collapses without resolving. Both headline phenomena would then describe ordinary
  encounters with generated content.
- **Nearly in-family.** Real output, trained on human data, lives almost entirely in the human
  block — high ω, where the model says content is read and correctly abandoned. That matches the
  observation that people are not in fact paralysed by AI content, and would mean the crash is a
  phenomenon of a regime generative systems do not occupy.

**A study that measures the signature decides between them.** The bands differ on engagement, on
whether resolution occurs, and on whether confident readings agree — all observable without
knowing ω.

**The caveat belongs with the card, not under it.** ω's meaning depends on V4 decision D1, which
split the feature space into disjoint human and foreign blocks because no such partition existed
at V1–V3 cardinality. Real content may have no well-defined position on this axis at all. Use
the card to classify an observed signature first, and only then ask what ω would have produced
it.

---

## Deviations

**V5-1 — N21's second clause was restated after the first version failed.** The clause as written
averaged β's simple effects across both μ levels; scored that way the null failed at 1.67 against
a required 3.0, and the criterion was restated afterwards to score β's effect on the μ = 1 row.
Nothing was pre-registered at the time — `prereg_v5.py` did not exist and was written immediately
afterwards, before N21 ran at scale — but an operationalisation changed after seeing a
measurement is an operationalisation changed after seeing a measurement.

The substance: β's effect is −0.053 at μ = 1 and +0.421 at μ = 3. β does not invent depth; it
limits how much *real* depth is recoverable, because depth is defined relative to a goal's mode
family and the plan cannot be read without partly knowing the goal. The original clause charged
that legitimate limitation as contamination. The restated clause asks what the null is for: can β
make the observer report depth that is not there.

**The threshold value was not changed** — restating which quantity it scores is the deviation;
moving the number as well would have been a second one. The original clause is retained,
computed, and reported as `dominance_ratio_original` = **1.701, which fails**. It decides nothing.

**V5-2 — E30 gained a second update measure, after the run.** `psi_analogue` has no headroom in
this design by construction, since depth is built so the goal is exactly as recoverable at every
depth. `psi_structure` measures the same thing one level up.

**This cannot manufacture the welcome answer and it was checked before being added:** the δ = 0
negative control must show no structural update either, and it does not — 3.236 there against
3.583 at full contrast. Adding the column is inert with respect to everything else: all 15
pre-existing columns came back **bit-identical** across 630 rows on a re-run, and the computation
consumes no random variates. The pre-registered measure still decides the verdict, and the
secondary agrees with it on the outcome while disagreeing on the reason.

---

**V5-3 — the C2/C3/C4 pre-registration lock fired, and it was the call sites rather than the
criteria.** E32 wrote the payload including its matching curve; E31 wrote the same criteria
without one, because E31 has no curve of its own to pass. Different payload, different hash, and
whichever experiment ran second refused to start — correctly, on its own terms. Fixed by giving
all three a single entry point that assembles the payload identically, so the hash means the
criteria have not moved rather than that the caller happened to build them the same way. No
criterion changed and no result is affected; recorded because a lock that fires is worth a line
either way.

**V5-4 — E31's primary was scored on a restriction of the pre-registered cell set, and the
restriction was never logged until an audit found it (2026-08-08).** The locked criterion reads
"Spearman(recovered mu, prior drift) **across all cells**", bar 0.70. The committed 0.886 is the
Spearman over the six open-θ cells only; over all twelve cells it is **−0.021**, because a closed
θ zeroes the update by definition and the six closed cells pin drift at ~0 regardless of recovered
depth. The restriction is openly commented in `build_verdict` and is the scientifically sensible
reading — a switch that is off cannot test tracking — but as pre-registered the criterion **fails**,
and by this repository's own rule the original decides what may be claimed: quote the tracking
result as *"on the cells where the update is enabled"*, never as the pre-registered pass. Two
riders found in the same audit: the locked `fabrication_gap` threshold (0.05) is computed and
recorded in the verdict (observed **0.025**, failing) but never enters the outcome logic, so the
committed full pass silently ignores a failed bar; and E33's `divergence_raises_recovered_depth:
true` rests on the bare sign of a Spearman across five cell means whose total range is **0.0011
μ-units** — statistically indistinguishable from zero at any defensible resampling unit — and
should not be quoted as a finding. E33's headline (the observer out-reading the maker's
self-model) does not rest on it.

**V5-5 — goals 0 and 3 share an identical sub-goal chain, contradicting the builder's own
docstring, found by audit (2026-08-08).** `build_subgoal_chains` assigns goal g the cyclic step
`(g % (n_sub − 1)) + 1`; with four goals and four modes the steps come out 1, 2, 3, 1, so goals 0
and 3 get the same cycle — a pigeonhole fact, since S modes admit only S − 1 non-identity cyclic
steps. Consequence: for that one pair, the ORDER channel contributes nothing to goal identity at
any depth. At μ = 3 the emission derangements still separate the pair, so nothing collapses; at
μ = 2 the modes are goal-generic by design and the pair is distinguished by the feature marginal
alone, exactly as at μ = 1. The docstring's "a different cyclic order per goal" is false for the
pair, and any statistic pooling goal pairs at μ ≥ 2 slightly understates the order channel. **The
committed worlds are not re-run** — re-running closed versions changes what re-running means —
and the repair is forward-only: `build_subgoal_chains_v5b` (V11) derives successor maps from
derangements disjoint from the emission permutations, asserts pairwise-distinct chains, and is
used by new work. `tests/test_v5b_chains.py` pins the original collision in place so it cannot be
"fixed" retroactively.

**C1 — μ replaces β.** μ is the better-specified construct and the worse-performing one. β's
update effect was real but confounded with legibility; μ removes the confound and the effect
goes with it. Keep μ for what it measures; do not claim it gates uptake.

**C2 — two gates, provenance as evidence. Supported, and it is the one that paid.** The crash
and the trust exploit are one mechanism with opposite signs, and the update is a monotone
function of the depth estimate whatever moved it. This is the correction to carry forward.

**C3 — ω and d are not one channel.** They differ on every measure at matched overlap. Carry
both, and name the second dimension: whether the reader can tell it is failing.

**C4 — the latent goal.** A reader beats the maker's self-account as soon as that account is
unreliable at all. The mark self-blindness leaves is real and unreadable by this observer, so
the categorical claim about generative systems holds about artifacts and fails about readers.

**C5 — the non-monotone attention gradient** is recorded in the README's limitations, stated by
the author before anyone else states it, with the observation that this repository's own
charting code had already quietly declined to use the published opacity ramp.

## What is not done

- **The literature check (V5 §0) has not been run.** It is explicitly not a spec-implementation
  task and remains outstanding. It is now the largest single thing owed: every prediction here
  was derived from theory and tested in simulation with no search for prior empirical work.
- **E8 remains withheld** with its `xfail(strict)` marker. **E27 stays open.**
- **The μ grid should be redesigned before it is used again.** μ = 2 and μ = 3 are not
  distinguishable in this construction, and a three-level axis whose top two levels are tied is
  a two-level axis with an extra label.
- **E31's exploit is an upper bound**, because the observer cannot doubt the label it
  conditioned on. A design where provenance and depth are inferred jointly would measure the
  real figure, and needs an inference engine that is not pymdp's mean-field solver.
- **E33's trace is unread, not unreadable in principle.** An observer with a hypothesis for
  "this maker was avoiding something" might detect it. Building one is the obvious next
  experiment and is not in this version.
- **A systematic or flattering latent/declared divergence was not run.** Uniform-over-others is
  the assumption-free primary and the only arm executed; a systematic variant is a second free
  parameter on the layer V5 §5 already calls the most tunable object in the spec.

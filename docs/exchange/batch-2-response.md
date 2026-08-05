# Second batch, answered

**2026-08-05.** All four tests run, plus one auxiliary the results forced, plus an audit of round
one. Verdicts, numbers, what would have falsified each, and — because it matters more than usual
here — where my own validation is imperfect.

**Reproduce:** `python runners/run_soundingline.py --only T1 T2 T3 T4 T5`. Every number below is
in `results/validation/soundingline/`. The S-2 audit is `python scripts/audit_s2_mixture.py`.

---

## Read this first: two things in round one are wrong

Both were found while building the second batch, both are in results already sent, and both
change what they were reported to mean.

### S-2's manipulation never reached the reader

S-2 varies the active terminal goal position by position and emits from it:

```python
a = Artifact(provenance=READ_TIER, goal=int(actives[t]), declared_signal=K.UNSIGNED)
deep.append(int(env.sample_feature(a, K.DEEP, art_rng)))
```

`V5Environment.sample_feature` never reads `artifact.goal` once a creator is bound. It returns
`self.creator.next_feature(rng)`, and the creator holds one fixed goal for the whole artifact.
Only `artifact.provenance` is consulted, for the alpha lookup.

The audit is decisive rather than suggestive. Switching the mixture off entirely — forcing every
position to the modal goal — leaves the feature streams **bit-identical in 100% of artifacts**,
and the maximum absolute change in `purpose_breadth` across 240 artifacts is **0.0**.

What S-2's two arms actually differ by is the artifact's true goal: the flattened arm is always
goal 0, the layered arm draws uniformly (observed 19/37/30/34). Its reported separation of
**−0.108** reproduces exactly, and matching on the goal leaves a residual of **−0.130** — larger
than the "effect", because the remainder is RNG-stream position from the extra `integers()` draw.

**S-2 is withdrawn.** Its question is re-run properly as T-2 below, against an emitter that works.
This one is my error to have inherited and not caught in round one.

### S-3's detector threshold is fitted on the test labels

```python
thr = float(np.median(np.concatenate([a, b])))
```

`a` is the concealing divergences and `b` the candid ones, at the amplification being scored. The
threshold is set from ground-truth-labelled data and re-fitted in every cell. No reader can do
that, and a threshold that moves with the data it is scored on will track a separation that is
growing for any reason at all.

Re-scored with the threshold **frozen** at the lowest amplification — what a reader who calibrated
once and then met a louder concealer would have:

| amplification | separation | accuracy, fitted | accuracy, frozen |
|---|---|---|---|
| 1.0 | 13.45 | 0.833 | 0.833 |
| 2.0 | 15.57 | 0.868 | 0.846 |
| 4.0 | 15.80 | 0.900 | 0.870 |
| 8.0 | 17.40 | 0.958 | 0.879 |

**The claim survives, at about a third of the size.** The rise is **+0.125 fitted, +0.046 frozen**,
so roughly 63% of S-3's headline was the moving threshold. "The shield gives them away" is still
true and should be quoted at +0.046.

---

## T-1 · The triangle ★

### The values vertex does not exist, and there is something in the repo that will pretend it does

`v6_model.build_values_map` looks exactly like the missing third vertex, is used live by E41, E55
and E56, and is asserted non-injective under null N26. It is `M[g % nv, g] = 1` — a deterministic
many-to-one projection of the goal. Measured rather than asserted:

| quantity | value |
|---|---|
| H(values \| goal) | **0.0000**, max over goals −0.0 |
| I(goal; values) | 0.6931 = **exactly** log 2 |
| capacity of the values vertex | 0.6931 |
| residual information values add over goal | **0.0000** |

Non-injectivity only rules out the values layer being the goal *renamed*. It is still the goal
*coarsened*. Four of the six edges through such a vertex are decided by the arity of a hard-coded
matrix, and they would have returned clean, bootstrapped, entirely artifactual numbers. `beta` is
not a fallback either: it weights how much of the emission attaches to the goal, so it is goal
legibility — a weight on the edge being measured, not a vertex.

**The triangle was run on goal – process – depth instead**, with the values vertex reported as
absent. `mu` is a genuine third latent the reader already infers. This answers chain-versus-
triangle. It does not answer the values question, and nothing here should be quoted as if it did.

### The structure: process is a source, goal is a sink

Every edge measured identically — an extra observation modality at controlled fidelity, matched
across vertices in **nats** rather than fidelity, paired artifact-by-artifact, n=200.

At 1.00 nat per step, µ=3, β=0.25 (where the goal has headroom):

| edge | gain | interval | verdict |
|---|---|---|---|
| goal → process | −0.0012 | [−0.0075, +0.0056] | **dead** |
| goal → depth | −0.0057 | [−0.0144, +0.0026] | **dead** |
| **process → goal** | **+0.1665** | [+0.1062, +0.2282] | alive |
| **process → depth** | **+0.5147** | [+0.4577, +0.5670] | alive |
| depth → goal | −0.0057 | [−0.0141, +0.0029] | **dead** |
| **depth → process** | **+0.1378** | [+0.1125, +0.1628] | alive |

The same pattern holds at µ=3/β=0.10, µ=2/β=0.25 and µ=2/β=0.10. At β=1.0 the reader gets the goal
right 100% of the time, so every edge into goal is a **ceiling, not a null**, and is reported as
one.

**Three of six edges are dead, and all three are the ones out of the goal.** The framework's chain
— intent is the key that unlocks the method — runs backwards in this model.

### E36 does not survive as an intervention

E36 is observational: it splits a rollout at the step the goal posterior settles. Reproduced here,
its gain is **+0.140 at µ=2 and +0.116 at µ=3**. The interventional edge is **−0.001**.

So E36's effect is not "supplying the goal helps the process". A settled goal posterior is a
*marker* of an artifact that was going to be readable anyway, plus the "after is simply later"
hole plate 21 already names. This is the single result here most likely to change what gets built.

### Dose-response

| edge | series (0.15 → 1.00 nats) | monotone | r |
|---|---|---|---|
| process → goal | 0.007, 0.032, 0.090, 0.158, 0.233 | yes | **0.987** |
| process → depth | 0.119, 0.344, 0.446, 0.508, 0.545 | yes | 0.941 |
| depth → process | 0.016, 0.075, 0.121, 0.130, 0.138 | yes | 0.928 |
| goal → process | 0.0005, 0.0023, 0.001, −0.0009, −0.0016 | no | −0.760 |
| goal → depth | −0.002, −0.006, −0.006, −0.006, −0.006 | no | −0.610 |

The three live edges are cleanly monotone. The dead ones are flat or drift negative.

### Bootstrapping is weak

Superadditivity — does supplying two vertices beat the sum of supplying each?

| pair → target | sum of singles | both | excess |
|---|---|---|---|
| process + depth → goal | +0.138 | +0.171 | **+0.032** |
| goal + process → depth | +0.489 | +0.499 | +0.010 |
| goal + depth → process | +0.133 | +0.127 | −0.006 |

One mildly superadditive case (~23% over the sum). The curator's "each one bootstraps the others"
gets weak support at best, and only where process is one of the two supplied.

### The asymmetry is real but most of its size is dimensionality

Goal and depth are one value per artifact; process is a new value every step. A channel at one nat
per step therefore hands the reader one nat about the goal twenty-four times over — saturating —
against twenty-four independent nats about the process. Holding the **delivered total** equal by
dropping the duty cycle:

| edge | 1 delivery | 3 deliveries | 8 deliveries | all 24 |
|---|---|---|---|---|
| goal → process | −0.0006 | +0.0001 | +0.0007 | −0.0012 |
| process → goal | **+0.0130** | +0.0522 | +0.1312 | +0.1665 |
| depth → process | +0.0463 | +0.0979 | +0.1255 | +0.1378 |
| process → depth | **+0.0894** | +0.2990 | +0.3808 | +0.5147 |

**The direction survives budget matching; most of the magnitude does not.** At one delivery each,
process → goal is +0.013 against goal → process at −0.001. That is the honest size of the
asymmetry once the process vertex is not simply allowed to be bigger. Quote the direction.

### Added in batch three: what the emission carries, before any reader sees it

Partial information decomposition of {goal, execution mode} about one emitted feature, computed
exactly from `world.subsig` with no rollouts:

| depth | total | redundant | unique GOAL | unique MODE | synergy |
|---|---|---|---|---|---|
| µ=1 | 1.4521 | 0.0000 | 1.4521 | 0.0000 | −0.0000 |
| µ=2 | 1.8783 | 0.2966 | 1.1555 | **0.0000** | 0.4262 |
| µ=3 | 1.8783 | 0.2706 | 1.1816 | **0.0000** | 0.4262 |

**Unique mode information is exactly zero at every depth.** Everything the execution mode
contributes to an emission is either redundant with the goal or readable only jointly with it,
which is a structural statement about why the edges came out the way they did. And at µ=1 it
returns zero mode information and zero synergy — null N28 recovered from the likelihood alone,
now used as a standing `identity` gate.

---

## T-2 · Automaticity and drive diversity

Run against a **new emitter**, because S-2's does not work (above). Emission is built from
`world.subsig[mu, active_goal, mode]` directly: one practised mode trajectory, the active drive
substituted per position. The curator's claim made literal.

**The mechanism half is not testable here and is not claimed.** Nothing in this model derives
drive-multiplicity from automaticity; on the mixture axis I set it. What is testable is the
reader-side half.

### Breadth rises, and the goal collapses with it

| automaticity | purpose_breadth | goal accuracy | process | true mixture breadth |
|---|---|---|---|---|
| 0.00 | 0.506 | 0.790 | −0.006 | 0.000 |
| 0.25 | 0.624 | 0.655 | −0.005 | 0.459 |
| 0.50 | 0.728 | 0.510 | −0.007 | 0.745 |
| 0.75 | 0.757 | 0.400 | −0.004 | 0.901 |
| 1.00 | 0.767 | **0.215** | −0.003 | 0.955 |

Breadth rises **+0.261** [+0.209, +0.313], monotone. But goal accuracy falls **−0.575**
[−0.645, −0.510], ending **below chance (0.25)**. Process recovery does not move, so the construct
is not confounded with shallowness — it is confounded with difficulty, which is the trap the
request named.

### The decisive control: breadth cannot tell diversity from difficulty

Hold the mixture at a point mass — one drive, no diversity at all — and lower the channel's alpha
until goal accuracy matches each mixture arm. If breadth still separates them, it reads diversity.

Excess breadth above an equally-hard single-drive artifact, differenced against the a=0 arm's own
baseline offset:

| automaticity | 0.25 | 0.50 | 0.75 | 1.00 |
|---|---|---|---|---|
| excess | **−0.012** | **−0.009** | **−0.023** | **−0.013** |

**Negative at every level.** Verdict: `BREADTH_IS_LARGELY_A_DIFFICULTY_METER`. At matched goal
accuracy, motivationally diverse work has *no more* posterior breadth than single-drive work made
equally hard to read.

*Note on n:* at n=40 this control returned the opposite verdict (excess +0.03 to +0.13). The sign
flipped at n=200. Do not quote the small-n version; I nearly did.

### Practice alone does not move breadth

The non-circular axis — `mu` at a fixed mixture. Nothing about `mu` was built to move breadth.

| mixture | µ=1 | µ=2 | µ=3 | µ3 − µ1 | excludes 0 |
|---|---|---|---|---|---|
| point mass | 0.502 | 0.492 | 0.485 | −0.017 | no |
| half | 0.641 | 0.639 | 0.642 | +0.002 | no |
| flat | 0.778 | 0.751 | 0.767 | −0.011 | no |

Flat at every level. **The curator's mechanism gets no support from the model's own practice
variable.** Artifact length also does nothing (12/24/48 steps, intervals cover zero), so the
fixed-decision-count constraint is not secretly doing the work.

---

## T-3 · Is a recovered-decision count ever well-defined?

### Two premise corrections

`delta` **ships at 1.0** — mode distinctness, "the one free knob in the depth construction", is
already spent. And the mode count cannot go below four: `goal_mode_permutations` needs one
derangement per goal, and two modes admit one, three admit two. Above six the build fails the
other way. **The admissible range is four to six**, so the entropy ceiling cannot be lowered.

### My structural prediction was wrong, and how I caught it

I predicted length would be inert, because the sub-goal is non-stationary — the chain holds a mode
for ~9.2 steps, so evidence about the *current* mode is bounded by dwell, not by artifact length.

Scored on the per-rollout **minimum** entropy, length looked like the strongest axis in the sweep
(−0.456) and beat dwell. That is an **order statistic**: a 192-step artifact gets sixteen times as
many chances to dip. Scored on the mean:

| effect | on the mean | on the minimum |
|---|---|---|
| length 192 − 12 | **−0.114** [−0.127, −0.101] | −0.456 [−0.485, −0.428] |
| dwell ∞ − 2 | **−0.206** [−0.225, −0.189] | — |

**The prediction that length is inert was too strong and is reported as wrong.** What survives is
the ranking: dwell moves the step density about twice as far as length, and both saturate.

### The answer: no, and the floor is about two modes

The clean statistic is **effective modes**, `exp(H)` — threshold-free and cardinality-free. One
would mean a decision has been recovered.

| axis | best cell | mean effective modes | steps below 0.20 of max entropy |
|---|---|---|---|
| baseline (shipped settings) | — | 3.56 of 4 | 0.000 |
| dwell | ∞ | 2.99 | 0.000 |
| length | 192 steps | 3.20 | 0.000 |
| delta | 1.0 (ceiling) | 3.57 | 0.000 |
| tier | CREATOR | 3.54 | 0.000 |
| **combined best case** | dwell ∞, n_sub 4, T=96 | **2.33** | **0.000** |

The combined best case is every axis pushed at once: a maker who never changes mode, maximum mode
distinctness, deepest maker, cleanest channel, four times the artifact length. It floors at **2.33
effective modes** as a cell mean. The single best rollout in the entire sweep averages 1.92, and
the single sharpest step anywhere reaches 1.02 — so individual moments of near-certainty do occur,
and they do not persist. Process accuracy peaks at 0.55 against chance 0.25.

**Discrete decision-counting is dead in principle, not in practice.** The posterior converges to
roughly a two-way ambiguity and never resolves further, in any regime this model can construct. As
the request put it, that is the more useful answer: instruments built on "how many decisions"
should be abandoned rather than repaired.

One warning about how S-1 asked it. A normalised-entropy threshold is unstable exactly where the
posterior sits: two effective modes reads as "below 0.5 of maximum" when the maximum is log 4 and
not when it is log 5. Two arms straddled that arithmetic and produced a 0.00 → 0.68 jump in
threshold crossings on a posterior that had barely moved. Quote effective modes, not a threshold.

---

## T-4 · Does divergence survive an uncertain reader?

Built on **S-3's own seed and affinity matrix** (leak accuracy 0.9015 here against S-3's 0.8991).

### Reader uncertainty is not the problem

Mean balanced accuracy across amplifications, at full concealment:

| degradation | level | accuracy | separation | survives |
|---|---|---|---|---|
| emission model → uniform | 0.0 | 0.885 | 15.04 | yes |
| | 0.4 | 0.884 | 5.05 | yes |
| | 0.8 | 0.772 | 0.73 | yes |
| | **0.95** | 0.546 | −0.45 | **no** |
| channels swapped, p | 0.10 | 0.887 | 14.69 | yes |
| | **0.50** | **0.891** | 11.61 | yes |
| wrong state count | 2 of 4 | 0.660 | 1.37 | yes |
| | 3 of 4 | 0.736 | 8.11 | yes |
| structured (Dirichlet) error | up to 4.0 | 0.869 | 6.85 | yes |

**Divergence is remarkably robust to the reader.** It survives an emission model mixed 80% toward
uniform, a reader that has the two channels backwards half the time, and a reader modelling half
the states that exist. It dies only at 95% flattening, which is a reader with essentially no model.

Channel swap at p=0.5 costing nothing is worth pausing on: a reader with *no idea* which channel is
which detects concealment as well as one that knows. The signal is in the disagreement, not in the
direction of the disagreement.

### Partial concealment is what kills it

S-3's concealer emits `softmax(-aff*amp)` — an exact negation, the most detectable concealment
there is, because a perfect opposite carries as much information as a perfect copy. Rotating the
display away by a controlled fraction instead (θ=1 is S-3's negation, θ=0 is candid):

| θ | amp 1 | amp 2 | amp 4 | amp 8 |
|---|---|---|---|---|
| **0.25** | 0.645 | 0.510 | 0.487 | 0.505 |
| **0.50** | 0.792 | 0.782 | 0.777 | 0.750 |
| 0.75 | 0.820 | 0.828 | 0.840 | 0.892 |
| 1.00 | 0.833 | 0.868 | 0.900 | 0.958 |

Two findings, and the second is the more consequential:

- **At quarter-concealment the signal is gone** — accuracy 0.49–0.65, separation indistinguishable
  from zero at every amplification.
- **The "louder shield gives them away" effect only exists near total concealment.** At θ=0.5 it is
  flat and slightly *decreasing* (0.792 → 0.750). The curator's counter-intuitive claim is real for
  a concealer who inverts, and does not hold for one who merely deflects.

**So: the leaked-layer programme transports.** What it does not transport to is a subtle concealer,
and that is where I would expect real text to live.

---

## T-5 · Auxiliary — is process the better detector? (No.)

Not requested. T-1 said process is a source and goal is a sink, and every instrument this project
and Sounding Line have built scores the goal. That has an obvious engineering consequence, so I
tested it: separating a hierarchical maker from foreign content (a real policy over an unmodelled
goal — the hard negative) and from pure synthetic. All features computed from the reader's own
posteriors only, no truth. Mann-Whitney AUC, bootstrap interval, label-permutation null, no
threshold fitted anywhere.

**The internal asymmetry does not transfer.** Against pure synthetic, process-side statistics win
8 of 14 contested cells with a median advantage of **+0.015** — a wash. Against the hard negative
they win 6 of 14, median **−0.002** — also a wash on average, and they lose badly wherever the goal
is hard to read: at a 1-step look and β=0.25, goal-side reaches **0.932** against process-side
**0.790**.

What *does* work, ranked by mean oriented AUC across all 16 cells:

| statistic | vs foreign | vs synthetic |
|---|---|---|
| `subgoal_step_movement_mean` | **0.880** | 0.863 |
| `subgoal_min_entropy` | 0.865 | — |
| `goal_mean_entropy` | — | **0.871** |
| `engaged_fraction` | 0.844 | — |

And in the hardest cell — one glance, low goal legibility — the winner is **`engaged_fraction` at
0.976**, in the direction *lower means a real maker*. That is E19/E20's own result doing the work:
foreign content sustains attention indefinitely because it never resolves, and a real maker lets
you stop.

**For the detection goal specifically:** the discriminating signal in this model is the *dynamics*
of the reader's belief — how much the sub-goal posterior travels step to step, and whether the
reader is allowed to disengage — not the confidence it ends on. That is a statement about which
quantity to instrument, and it is the one thing in this batch that points forward rather than
backward.

---

## Validity register

Every check run, including the ones that failed and what they caught.

| check | result |
|---|---|
| **Placebo** — a zero-nat channel must be exactly the control | **2.7e-15** across all cells and vertices, after two fixes |
| Placebo, first attempt | **FAILED.** Channel drew from the rollout's RNG; depth recovery moved 0.163 → 0.282 on nothing. Fixed with a spawned stream |
| Placebo, second attempt | **FAILED** on depth only. At c=3 the diagonal 1/3 and off-diagonal (1−1/3)/2 differ in the last bits; the near-tied depth posterior flipped on argmax. Fixed by snapping to exact uniformity |
| **Arms paired on identical content** | max paired deviation 0.0; every contrast is within-artifact |
| **Negative control: rotation** | **FAILED AS A CONTROL, informatively.** A lying process channel helped goal recovery as much as an honest one (+0.166 vs +0.167). The chains are per-goal cycles, so a constant relabelling maps a cycle onto itself — a rotation is a change of coordinates, not a lie |
| **Negative control: random** | passes. process→goal −0.241, process→depth −0.661 |
| **Negative control: swap** (another artifact's true trajectory) | passes. process→goal −0.414, process→depth −0.747 |
| Swap control, first attempt | **FAILED.** The donor reused this artifact's own depth, so the depth swap arm was bit-identical to the honest arm and returned exactly the honest gain while testing nothing |
| **Mutual-information symmetry identity** | goal→depth and depth→goal agree to **3.2e-16** in all seven cells. At one nat both vertices are fully determined, so each edge is I(goal; depth \| data), which is symmetric. A correctness check the harness has to satisfy |
| **N28** — no process to recover at µ=1 | passes. Supplying goal moves process −0.0009, interval covers zero. Supplying depth moves it +0.039 **to exactly chance** (−0.040 → −0.0002): it removes a harmful misidentification rather than creating recovery |
| **N28 from the likelihood alone** (batch three) | passes exactly. PID returns 0.0 unique-mode and 0.0 synergistic bits at µ=1, with no rollouts |
| **Merged state index order** | asserted against `mgs_index` at every (µ, g, s) |
| **Dose-response monotonicity** | live edges r = 0.93–0.99; dead edges non-monotone |
| **E36 reproduction** | +0.140 at µ=2, +0.116 at µ=3, on the unmodified rollout |
| **S-3 reproduction** | leak accuracy 0.9015 vs S-3's 0.8991, same seed and affinity matrix |
| **Order-statistic artifact** (T-3) | **CAUGHT.** Min-over-steps inflated the length effect 4× and reversed the ranking against dwell. Every verdict now reads the mean and step density |
| **Ceiling detection** | β=1.0 flagged as a goal ceiling; T-5 flags 2 of 16 cells where the question cannot be asked |
| **Threshold-free scoring** | T-3 uses effective modes; T-5 uses AUC; T-4 adds a frozen threshold |
| **Permutation null** (T-5) | every reported AUC beats it |
| **Sample-size stability** | T-2's difficulty control **flipped sign** between n=40 and n=200. Only n=200 is quoted |
| **Seed reproducibility** (found in batch three) | **FAILED.** T-3 seeded its sweeps from Python's `hash()`, which is randomised per process, so it returned 2.29 effective modes on one run and 2.05 on the next. Fixed to `crc32` and verified identical across runs |

### Where my validation is imperfect

Six places, in rough order of how much they should worry you.

1. **T-5 tests single statistics only.** No multivariate combination, no held-out split. A
   two-feature detector could beat both sides and I did not test one. The "process is not better"
   conclusion is about single features.

2. **T-1's supply channel is not the same operation as knowing.** It delivers noisy evidence the
   reader integrates through its own likelihood. A reader *told* the goal with certainty is a
   different intervention, and prior-clamping was rejected because it moves the measure's own
   reference. The dead goal edges mean "goal evidence does not help", not "goal knowledge does not
   help". I believe these coincide here; I did not prove it.

3. **The µ=depth substitution is a substitution.** Depth is a real third latent and the triangle on
   it is real. It is not the values vertex, and the curator's claim is about values. The values
   result is a proof of absence, not a measurement of the claim.

4. **T-2's mixture axis remains circular as a mechanism test.** Only the depth axis is not, and the
   depth axis is a null. A null on a non-circular axis is weaker evidence than a positive would be.

5. **T-4's degradations are all symmetric and unstructured.** A reader wrong in a *correlated* way
   — systematically confusing two states that look alike — is the realistic failure and is not
   tested. The Dirichlet arm is the closest and it is still unstructured.

6. **T-3 could not lower the entropy ceiling.** Fewer than four modes is unbuildable at four goals.
   The "never resolves below ~2.3 effective modes" result therefore comes from a world whose mode
   count could not be reduced, and a two-mode world might behave differently for reasons that are
   arithmetic rather than structural.

Two further things that are true of everything above: **there is no human data anywhere in this**,
and **the shapes are the claims — the numbers are properties of this model's dimensions.** Quote
directions.

---

## What I would build next, aimed at detection

Ordered by how much each would change the instrument, given the goal is detecting human intent
before extracting it.

1. **Build a values factor that is actually independent**, then re-run T-1 as posed. Two latents,
   goal and values, coupled but not functionally determined — so `H(values | goal) > 0` by
   construction and the six edges are six measurements. This is the largest missing piece and it is
   a day's work in the generative model, not a research programme.

2. **Instrument belief dynamics, not belief endpoints.** T-5's best single feature is how far the
   sub-goal posterior travels step to step; the best in the hardest regime is whether the reader is
   permitted to stop looking. Both are trajectory statistics. Every instrument currently in either
   project reads a final posterior. `ghostscale/methods/trajectory.py` now wraps catch22 for
   exactly this search, with a fresh-seed confirmation protocol attached.

3. **Test partial concealment on real text before building further on the leaked layer.** T-4 says
   reader uncertainty is survivable and partial concealment is not. A concealer at θ=0.25 is
   undetectable at every amplification, and real concealment will not be an exact negation.

4. **Retire decision-counting.** T-3 is a clean negative across every axis the model admits.

5. **Re-run S-2's actual question** — concentration rather than diversity — on the working emitter
   in `t2_automaticity.mixed_deep_features`, with the difficulty-matched control attached. The
   original question was never tested and remains open.

One more, which is a process point rather than a test. Both round-one errors were of the same kind:
a manipulation that never reached the thing it was supposed to manipulate, and a threshold that
knew the answer. Both would have been caught by a check that costs almost nothing — *switch the
manipulation off and confirm something changes*, and *freeze anything fitted*. Batch three turned
both into standing gates; see [docs/METHODS.md](../METHODS.md).

# Version 6 — aligning the simulation to the Intent Extraction Limit

**Written before any V6 code. Not edited afterwards. Deviations are logged in RESULTS_V6.md.**

---

## §0. What this version is for, and what makes it different from V1–V5

Every previous version asked a new question about the world. **This one asks whether the
simulation and the theory it claims to implement are the same object.**

Three passes — validation, diagnostics, repair — audited the *results*. Reading the preprint
(*Art as an Algorithmic Virus*, the Intent Extraction Limit) against the code found something none
of those passes could have found, because they all took the code's own account of itself as given:
**the formal model has terms the simulation does not implement, and in one place the two use
different mechanisms to produce the same phenomenon.**

The formal model:

```
Ψ = sigmoid( k · (ω − θ_E,C(κ)) ) · [ −ln(1−κ) ] · D_KL( Q(R|τ) ‖ P₀(R) )

θ_E,C(κ) = θ_base(E) + λ · D_KL( Q(R|τ) ‖ P_c(R) )
```

Against the shipped code:

| term | in the code? |
|---|---|
| `−ln(1−κ)` trust amplifier | yes, `metrics.trust_factor` |
| `D_KL(Q ‖ P₀)` belief movement | yes, and V6 supersedes it with a signed measure |
| `λ · D_KL(Q ‖ P_c)` value divergence | yes, the acceptance gate |
| `θ_base(E)` metabolic reserve | **NO — no depletion state exists anywhere** |
| `sigmoid(k · …)` graded gate | **NO — replaced by a binary engagement decision** |
| `κ → 1 ⟹ θ_E,C → 0` | **NO — κ and θ are independent factors that never interact** |

The third is not an omission. It is a **mechanism disagreement**. The preprint's trust exploit
works by trust *suppressing the disgust threshold* so the gate is held open while the bottom-up
signal collapses. The code's trust exploit works by the label channel *out-arguing* the content
channel, with an analytic crossover at κ = 0.538 (diagnostics D-1). Both produce the phenomenon.
They predict different things, and V6 is where that gets settled rather than papered over.

**V6 also implements six theory extensions** that came out of a walkthrough of the current-state
review, plus two corrections the author made to my own reading of the theory. Those are §4.

---

## §1. What is deliberately NOT built

Recorded first, so that the boundary is a commitment and not an outcome.

- **No creator agent.** Zahavian signalling, strategic defection, and the reputational-cost
  security argument need a maker that chooses when to lie — an agent with its own objective.
  The framework's security case is formally specified in the preprint's Appendix B and
  **remains entirely untested in simulation.** That is named here as an open hole rather than
  discovered later. It is the natural V7.
- **No recursive / fractal hierarchy.** The claim that the same extraction runs at every scale is
  real and it needs an unbounded self-similar structure. Instead V6 builds the **scaled**
  version (§4.8): the same extraction run on a sub-window of one artifact, which tests
  scale-invariance without recursion.
- **No affective / mirror channel.** A second observation modality with no distinct prediction
  yet. Revisit if §4.3 shows the reader failing in a way a body-based prior would fix.
- **No cognitive-surrender dynamics on the creator side.** Same reason as the first item.

---

## §2. Hypotheses

Stated before any run, with the branch that would count as a failure.

**H6.1 — Depletion produces carryover damage with no contamination.**
A reader whose metabolic reserve depletes on unresolved engagement will show declining
engagement *on genuine human work it has never seen*, after exposure to intent-empty content.
Predicted: engagement on a fixed human probe falls monotonically with prior exposure count.
**Fails if** the probe is flat, in which case depletion is a knob that moves only what it is
pointed at.

**H6.2 — The two trust mechanisms are distinguishable, and the coupled one predicts an exploit
under an honest label.**
Under κ→θ coupling, a reader that *correctly believes* content is machine-made still integrates
it if trust in the source is high, because the gate is held open. Under the channel-race
mechanism it cannot, because the reader's provenance belief is correct and the exploit works
through provenance belief.
Predicted: the honest-label cell shows integration under coupling and none without.
**Fails if** the two mechanisms are behaviourally identical across the whole κ range, in which
case the distinction is formal only and the code's version should be kept for being simpler.

**H6.3 — Goal recovery gates process recovery.**
The reader recovers the maker's execution chain *only* to the extent it has resolved the goal.
Predicted: process recovery conditioned on goal-correct readers substantially exceeds process
recovery conditioned on goal-incorrect readers, at matched exposure.
**Fails if** process recovery is independent of goal recovery, which would refute the ordering
claim and make process and goal two parallel readouts rather than a chain.

**H6.4 — Depth moves process uptake even though it does not move goal uptake.**
E30's null was measured on goal uptake, and depth is constructed so the goal is equally
recoverable at every level. Predicted: on process uptake, the deepest-minus-shallowest contrast
is positive and its interval excludes zero.
**Fails if** process uptake is also flat, which would say depth transmits nothing and is a
direct hit on C1.

**H6.5 — A non-invertible generator produces a distinct failure from a foreign one.**
Content on *familiar* features whose goal→feature map cannot be inverted should produce
**low uncertainty about features and no recovery of intent** — legible and empty — as against
foreign content's high uncertainty and sustained search.
Predicted: the two conditions separate on the joint (final entropy, engagement) signature.
**Fails if** they are indistinguishable, in which case the wall really is a vocabulary deficit
and the overlap axis was the right construction all along.

**H6.6 — Expertise substitutes rather than stacks.**
A reader whose hypothesis family matches the machine generator recovers machine content, **and
loses on human content by a comparable margin.**
Predicted: a crossover, not a dominance.
**Fails if** the machine-matched reader does as well on human work, in which case AI literacy is
a free upgrade and the darker reading is wrong.

**H6.7 — The tool hypothesis produces clean disengagement, distinct from both the crash and
correct reading.**
A reader holding "there is no maker here" as a first-class hypothesis should stop looking
*without* sustained futile attention and *without* fabrication.
Predicted: a third signature — resolved, disengaged, no invention.
**Fails if** it reproduces either the crash or the honest-label cell, in which case the Ghost
Scale's mechanism is redirection rather than relaxation.

**H6.8 — Aesthetic and social cues enter engagement ADDITIVELY.**
Pre-registered as additive against multiplicative. Predicted: the "ugly but endorsed" and
"beautiful but unendorsed" corners are both engaged above baseline; a multiplicative rule
predicts at least one of them collapses.
**Fails if** the multiplicative form fits better, which is a fact about the architecture and is
reported as such.

**H6.9 — RLHF decoupling: a reader over-engages when a learned depth cue is optimised directly.**
If the aesthetic cue is learned as a predictor of depth and then maximised on content with no
depth, the reader pays MORE and gets LESS.
Predicted: engagement above the honest-depth baseline with error reduction at or below zero — a
third failure mode, distinct from the crash and the exploit.
**Fails if** engagement tracks the true depth rather than the cue, which would mean the cue
never became load-bearing.

**H6.10 — Vulnerability is the gate, not the engagement decision.**
There exists a parameter region with **high engagement and a closed gate**: a reader that looks
deeply, recovers the maker accurately, and integrates nothing.
Predicted: such a cell exists and is stable.
**Fails if** engagement and integration cannot be dissociated, in which case the code's
engagement decision really is the willingness to be vulnerable and the author's objection does
not apply here.

**H6.11 — Self-report accuracy falls with depth.**
Practised structure is bundled and becomes inaccessible to the maker's own account. Predicted:
declared-goal accuracy declines with μ while latent recovery by the reader stays flat.
**Fails if** self-report is flat in depth, which would make automaticity irrelevant to
self-knowledge in this model.

**H6.12 — The extraction is scale-invariant.**
Recovery run on a sub-window of an artifact returns the same *shape* of answer as recovery on
the whole, at reduced precision.
**Fails if** sub-window recovery is qualitatively different, which would say the extraction is
tied to the artifact boundary and the fractal claim does not hold even in the small.

---

## §3. Nulls

N22 — **depletion null.** With every artifact resolvable, depletion must not accumulate:
reserve at the end of a long exposure is within tolerance of its start. *This is the null the
generational experiment never passed, and it is written first on purpose.*

N23 — **coupling-off reduction.** With the κ→θ coupling disabled, V6 must reproduce V5
elementwise on the shared quantities. Any difference is a wiring error.

N24 — **gate-gain reduction.** As k → ∞ the graded gate must reproduce the binary decision.

N25 — **no preference over provenance, still.** Every V6 addition — cue channels, tool
hypothesis, values layer — extends C with zeros. The reader may never *want* human work.

N26 — **the values map is not the goal renamed.** Two distinct goals mapping to the same values
must produce the same gate state; a values map that is a bijection on goals is rejected at
construction.

N27 — **the tool hypothesis must not absorb human work.** If NO_MAKER absorbs human artifacts it
is a uniform fallback and every result is destroyed. Same failure mode V4 decision D2 caught for
EXPLORE, and it is checked the same way.

N28 — **process recovery is not goal recovery renamed.** At μ = 1 there is no process to
recover, so process recovery must be at chance whatever the goal recovery is.

N29 — **cue channels carry no goal information.** The aesthetic and social channels must be
independent of the goal given depth, or they are a second legibility channel.

N30 — **non-invertible content stays on the human block.** Otherwise §4.3 is foreign content in
new vocabulary and H6.5 is unfalsifiable.

---

## §4. What gets built

### §4.1 Metabolic reserve (E), and the depleting reader

A scalar `E ∈ (0, 1]` carried by the reader **across encounters**. It falls when the reader
engages and fails to resolve, and recovers otherwise. It enters the gate through `θ_base(E)`:
a depleted reader has a *higher* threshold, so it disengages sooner.

**Depletion is driven by unresolved engagement, not by exposure.** A reader that looks and
succeeds pays the effort cost and is not depleted by it; a reader that looks and gets nothing
is. That is the essay's mechanism — "your brain keeps being disappointed after its search for
meaning" — and it is what makes N22 a real null rather than a formality.

No new hidden factor. E is reader-side bookkeeping that modifies the engagement threshold.

### §4.2 The κ→θ coupling, switchable

`θ_E,C(κ) = (θ_base(E) + λ · D_KL(Q ‖ P_c)) · (1 − κ)^c`, with `c = 0` recovering the current
independent behaviour exactly (N23) and `c = 1` the preprint's stated `κ → 1 ⟹ θ → 0`.

**Both are run and reported.** The point is the comparison, not the replacement.

### §4.3 The wall: a non-invertible generator

A third content family. Machine content living on the **human** feature block, produced by a real
policy over a real maker-state, whose goal→feature map is **many-to-one**: several maker states
emit the same surface distribution. The reader has full vocabulary and cannot invert.

Distinct from foreign content (disjoint support, invertible in principle) and from an unskilled
reader (mis-aimed templates on invertible content). N30 keeps it honest.

### §4.4 The machine-matched reader

A reader whose likelihood family is built from the machine generator rather than the human one.
Everything else identical. Run on both content types, against the human-matched reader — a 2×2
whose interesting cell is the machine-matched reader on human work.

### §4.5 NO_MAKER as a first-class hypothesis

One extra value on the goal factor, with a likelihood matched to intent-empty content, so a
reader can *conclude* "no maker" rather than merely failing to conclude anything. Structurally
the same move V4 made with EXPLORE, pointed at a different target, and it inherits EXPLORE's
null (N27).

### §4.6 Aesthetic and social cues

Two scalars per artifact, entering the **engagement decision** and nothing else:

- **aesthetic salience** — a surface property that (in the honest regime) predicts depth,
- **social endorsement** — an exogenous signal that density is present.

Combined additively with the model's own expected information gain, with a multiplicative form
built alongside for the pre-registered comparison (H6.8). Both channels carry zero goal
information (N29).

**The RLHF decoupling arm:** the reader learns the aesthetic→depth association on an honest
corpus, then meets content where salience is maximised and depth is zero.

### §4.7 The values layer

`values = M · goal_posterior`, with `M` a non-injective map from goals to value vectors (N26).
The acceptance gate compares recovered **values** to the reader's value prior, not the goal.

### §4.8 Scale invariance (the salvaged part of the fractal claim)

Recovery run on a sub-window of one artifact, compared with recovery on the whole. No recursion,
no new structure — a measurement over a slice.

### §4.9 Graded self-report

The maker's declared-goal accuracy falls with μ. Practised structure is bundled and becomes
inaccessible to its own author; the reader is unaffected. This corrects a claim in the working
notes that the subconscious holds *the* process goals: it holds the **practised** ones.

### §4.10 Process recovery, and Ψ as a first-class measure

- **Process recovery**: how much of the maker's true execution chain the reader's sub-goal
  posterior captured. Read off state the reader already carries in every V5 run.
- **Ψ**: the full Intent Extraction Limit with finite k, θ_base(E), λ, and the coupling — computed
  as a measure, beside the existing `psi_analogue`, which is retained and not replaced.

---

## §5. Scale and acceptance

Criteria are hash-locked before any run, in `ghostscale/prereg_v6.py`, in the same manner as V3,
V4, V4.5, V5, the validation pass, the diagnostics pass and the repair pass. Editing the locked
file after the fact makes the pass refuse to run.

Every V6 experiment runs under **exact inference by default**. V6 is the first version for which
that is true, and it is possible only because the repair pass built the exact learning path.

Every original measure is retained and reported beside its replacement, per the standing rule.

---

## §6. Pre-mortem

Written before the run. The four ways this pass could produce a worthless answer.

1. **Depletion makes everything decline and explains nothing.** Guarded by N22 and by H6.1's
   requirement that the *probe* moves, not merely the exposed condition.
2. **Too many additions at once, so nothing is attributable.** Guarded by every addition being
   independently switchable and off by default, and by the V5 reduction null (N23).
3. **The cue channels become a second legibility channel** and quietly re-derive the label
   effect. Guarded by N29.
4. **The tool hypothesis absorbs everything**, the failure V4 caught with EXPLORE. Guarded by
   N27, which is the same check.

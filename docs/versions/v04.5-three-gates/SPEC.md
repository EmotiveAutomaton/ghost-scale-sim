# Ghost Scale Simulation — V4.5 Specification

**Author of spec:** Abraham Haskins, PhD
**Target:** an autonomous coding agent, extending `ghost-scale-sim`
**Depends on:** V4 spec and `RESULTS_V4.md`. This is a **small delta**, not a new direction.

Two jobs. Finish V4 stage 2, which is the experiment that can invalidate a public claim. And
propagate a conceptual change to the trust model that the V1–V4 formalism cannot currently
express.

---

## 0. Priority order, because this is time-boxed

| # | item | compute | why now |
|---|---|---|---|
| 1 | **A1, A2** — analyses on existing CSVs | **zero** | no run needed, do them while E21 queues |
| 2 | **E21** — model comparison | small | V4 stage 2; the only unrun experiment that can invalidate a public claim |
| 3 | **E28** — β as continuous rationality | small | the conceptual change; makes the three-gate model expressible |
| 4 | **E29** — gate dissociation | medium | proves the three gates are separable, not relabelings |
| 5 | E20 — the ω sweep, with the engagement question added | medium | already specced in V4; §4 below adds a required measure |

Stop after 3 if time runs out. Items 1 and 2 are what a skeptical reader checks.

---

## 1. Zero-compute analyses

Both run on committed CSVs. No simulation.

### A1 — The mislabeling asymmetry

From `results/e2_variance.csv`, already measured, never reported:

| condition | within-observer | between-observer |
|---|---|---|
| human artifact, labeled AI | 0.092 | 0.009 |
| AI artifact, labeled human | 0.090 | 1.379 |

Same confidence, opposite outcomes. Human work falsely labeled AI is still **read accurately**;
observers agree. AI work falsely labeled human produces confident fabrication at ceiling
disagreement.

**The Ghost Scale's two failure modes are asymmetric. Mislabeling human work costs engagement.
Mislabeling synthetic work costs the model.** That is a directional design argument for erring
toward over-labeling, it is measured, and it appears nowhere.

Write to `RESULTS_V4.md` as a new section. Confirm the direction against the CSV before writing;
do not take the numbers above on trust.

### A2 — Calibration metrics

Add `brier_score` and `expected_calibration_error` to `metrics.py`. Recompute across existing
E2, E17, and E19 outputs. Regret, accuracy, and the epistemic/pragmatic decomposition already
exist; these two are the gap.

Purpose is translation, not new information. It restates the fabrication result in the vocabulary
an ML audience already uses: **the observer is not merely wrong, it is miscalibrated, and the
miscalibration is induced by a label.**

---

## 2. E21 — Model comparison (V4 stage 2, unchanged in scope)

Specified in V4 §E21. Build it as written. Restated here because it is now the highest-priority
run.

**Arms:** (A) full active-inference observer; (B) Bayesian inverse planning, no engagement
policy, always DEEP; (C) heuristic label-truster, believes the declared tier, no content
inference; (D) effort-allocation heuristic, disengages on low first-observation information, no
goal model; (E) no-ToM classifier, predicts provenance from features, never infers a goal.

**Pre-register before running.** Several baselines will reproduce *disengagement*, which is
cost-benefit arithmetic and needs no theory of mind. The signature that should require a
generative model of another agent is E2's **simultaneous** high-confidence / high-disagreement
dissociation. A heuristic can be confidently wrong; it should not produce that joint pattern.

**E19 adds a second discriminator.** Sustained expensive attention that never resolves is a
prediction about an agent that keeps *expecting* to learn. A pure effort heuristic has no
expectation to be wrong about. Arm D should therefore fail to reproduce the foreign-content
engagement result, and if it does reproduce it, that is informative about how little machinery
the phenomenon requires.

**If arm C or D reproduces either signature, the active-inference apparatus is scaffolding
rather than mechanism.** Report it in the first line of the section.

---

## 3. The three-gate decomposition

### 3.1 What is wrong with the current model

κ is a single scalar doing three jobs, and the canonical trust literature says they are separate
constructs. Mayer, Davis & Schoorman's ability / benevolence / integrity and Lee & See's
performance / process / purpose both decompose trustworthiness into a competence dimension and a
values dimension. The Ψ equation has neither cleanly:

- **Value alignment is double-counted.** It sits in θ already, as
  `θ_base(E) + λ·D_KL(Q(R|τ) ‖ P_c(R))`. If κ also carries value convergence, the same quantity
  enters twice, once multiplicatively and once through the gate. Redundant parameters are
  unidentifiable.
- **Competence is absent entirely.** MaxEnt IRL's demonstrator likelihood is properly
  `P(τ|R) ∝ exp(β·R(τ))`. Appendix A.1 writes it with **β = 1** — the creator is modeled as
  perfectly rational, always. That elision is the missing dimension.
- **Social influence has nowhere to sit**, because it is not a gate on integration. It is a shift
  on the prior, upstream of every gate.

### 3.2 The architecture

Three gates in series, each answering a different question, each acting at a different point in
the pipeline. This is E5's κ-versus-γ finding generalized: things that look like one knob act at
different positions and produce different signatures.

| gate | question | acts on | status |
|---|---|---|---|
| **κ_p** | is there anything here to read? | the likelihood | exists, but currently conflated |
| **β** | how much does this trajectory constrain R? | the demonstrator model | **missing** |
| **θ** | should I write what I found? | the update | exists |

**Social influence is not a gate.** It shifts `P_0(R)` and the prior over provenance before any
gate runs. Implement it as a prior perturbation with a per-observer weight, never as a multiplier
on integration.

**Integrity is not a values dimension.** Under revealed preference — which is what IRL is — values
are defined by what behavior reveals, so integrity cannot diverge from values by construction.
What *can* diverge is **stated** values from revealed ones, and the gap between the R inferred
from a declaration and the R inferred from behavior is a real measurable quantity. That gap is
already V4's C4 (`declared_tier` versus `omega_true`). **Integrity is declaration-behavior
calibration, and the framework already contains it under a better name.** Note this in the
write-up; do not add a separate construct.

### 3.3 β and EXPLORE are the same axis

This is the implementation shortcut and it should be exploited rather than worked around.

Under low β, the observer models the creator as not optimizing hard, so the expected output moves
away from the goal-specific signature toward the creator's policy marginal:

```
A_observer[:, p, g, DEEP] = normalize( beta · sig[g] + (1 - beta) · mean_g(sig[g]) )
```

`mean_g(sig[g])` **is `sig_EXPLORE`**. So EXPLORE is β = 0 as a discrete hypothesis, and β is its
continuous generalization. The construction is already built, already asserted against global
uniformity, and already validated by E19's positive control.

**Required consistency check:** at β = 0, E28 must reproduce E19's exploratory-human cell. If a
continuous β near zero does not recover the discrete EXPLORE result, the two are not the same
axis and the identification above is wrong.

---

## 4. E28 — β as inferred rationality

**Design.** Add β to the observer's inference as a hidden quantity rather than a fixed parameter:
the observer infers *how hard the creator was optimizing* alongside *what they were optimizing
for*. Sweep the true β used to generate human content from 0 to 1; measure the observer's
recovered β, its recovered goal, and its update magnitude.

**Measure.** Recovered β against true β; goal accuracy; `psi_analogue` (update magnitude);
engagement; final entropy.

**Predicted.** Goal accuracy stays high across the β range while update magnitude falls with β.
That is the missing aesthetic category: *legible, competent, and unmoving.* The observer reads
the intent correctly and correctly treats it as weak evidence.

**Falsification.** If update magnitude does not fall with β while accuracy holds, β is not doing
separable work and collapses into either κ_p or θ.

---

## 5. E29 — Do the three gates dissociate?

The experiment that determines whether this decomposition is real or a relabeling.

**Design.** 2 × 2 × 2 across low/high κ_p, low/high β, and low/high value divergence, on human
content held constant.

**Pre-registered signatures.** Each gate should produce a distinct behavioral pattern:

| condition | engagement | resolution | update |
|---|---|---|---|
| low κ_p (goal-empty) | low | no | none |
| low κ_p (goal-foreign) | **high** | **no** | none |
| low β | high | **yes** | **low** |
| closed θ | high | yes | **none, with divergence spike** |

The low-κ_p row splits by content type, which is E19's finding and is why that experiment had to
run first.

**The decisive contrast is low β against closed θ.** Both produce full extraction with little
integration. They differ in *why*: low β says the trajectory is weak evidence, closed θ says the
recovered goal is unacceptable. If those two cells are behaviorally indistinguishable on every
measure, the decomposition is not earning its parameters and should be reported as such.

**Cost note.** Three gates with different action points is a harder object to identify than one
scalar: more parameters, more freedom, less falsifiability per experiment. E29 is the mitigation.
If it fails, say so plainly rather than keeping the decomposition on theoretical grounds.

---

## 6. E20 addition — the metabolic question

E20 is specified in V4 and unchanged, with one required addition.

E19's unpredicted finding is that observers **do not disengage** from goal-foreign content: 0.746
engagement across free steps, no resolution, `crash_signature` false in every cell. That inverts
V1–V3's metabolic prediction, and it independently reproduces V1 E5's negative content-only
selectivity by a completely different mechanism.

**E20 must therefore report engagement and `crash_signature` as primary outcomes across the full
ω sweep, not as secondary columns.** The question is where along ω the metabolic prediction
flips: at what overlap does content stop draining attention and start holding it.

This matters beyond the model. **H1 and H3 in the preprint both assume goal-empty content**, and
both invert under goal-foreign. Goal-empty predicts an autonomic drop within 2–4 seconds;
goal-foreign predicts sustained arousal with no resolution. **A single pupillometry measurement
discriminates them**, which makes this the sharpest and cheapest empirical prediction the
framework has produced. E20's ω-crossing point is what tells a human study where to look.

---

## 7. Constraints

Everything in V1 §14, V2 §6, V3 §6, V4 §6, plus:

- **A1 and A2 add no new simulation.** If either requires a run, the analysis has been
  misunderstood.
- **E21 must be able to return "the machinery is unnecessary"**, and that outcome goes in the
  first line of its section.
- **β is not a rename of κ.** N-series addition: at β = 1 across all conditions, V4.5 must
  reproduce V4 within tolerance. If it does not, β has been wired into the wrong pipeline
  position.
- **Social influence enters the prior, never the integration multiplier.** Assert this at
  construction: no social term may appear in the update-magnitude path.
- **E8 stays withheld.** E27, the V3 residual, stays open. Neither is touched here.

# Ghost Scale Simulation — V5 Specification

**Author of spec:** Abraham Haskins, PhD
**Target:** an autonomous coding agent, extending `ghost-scale-sim`
**Depends on:** V1–V4.5 specs and all `RESULTS_*.md`.

V5 exists because a careful read-through of V1–V4.5 by the author surfaced four construct errors
and one omission. **None were found by the simulations.** They were found by someone who knows
the theory checking whether the implementation matched it. That is the correction channel this
project has not been using, and V5 is its first output.

---

## 0. Standing task, carried forward until done

**THE LITERATURE CHECK.** Every prediction in this repository was derived from theory and tested
in simulation, with no search for prior empirical work. A systematic search should now be run
against the published literature for each: the competence ceiling on intent extraction, the
label-induced confidence switch, the mislabeling asymmetry, coverage thresholds for disclosure
regimes, the poisoning/starvation dissociation, sustained attention to low-overlap content.

**This is out-of-sample in a real sense** — the predictions were fixed before anyone looked — and
it should be reported that way: *derived from theory, tested in simulation, then checked against
existing literature after the fact.* It is not new evidence and must not be presented as such.
It is cheap, it is honest, and it may find that several of these are already known.

Not to be run inside a spec-implementation session. Its own task.

---

## 1. Construct corrections

### C1 — β is replaced by μ, estimated model depth

**The error.** V4.5 introduced β as the MaxEnt rationality coefficient: *how hard was the creator
optimizing*. That is not what gates uptake.

**The correction.** What gates it is the observer's estimate of **how complex a model the creator
had** — the depth of the hierarchy of decisions behind the artifact. This is the preprint's own
decision-density construct, and it is why Duchamp's urinal works and why a fully-committed
shallow artifact does not.

The two dissociate, and the framework takes the opposite side from β:

| | rationality β | model depth μ |
|---|---|---|
| fully committed, trivial goal | high | low |
| offhand sketch by a master | low | high |

**The preprint says the second is more artful.** A rationality parameter says the opposite. β was
measuring the wrong thing.

**Implementation.** μ indexes the *hierarchical depth* of the creator's policy, not the
temperature of its action selection. A deep-model creator's artifact carries structure at
multiple scales — goal-level, sub-goal-level, execution-level — and a shallow one carries
structure only at the execution level. The observer infers μ jointly with the goal, as β was.

**Retain E28's machinery.** The inference architecture is sound; the quantity being inferred
changes. Report E28's finding as an artifact of a mis-specified construct rather than deleting
it, and re-run under μ.

### C2 — Two integration gates, not three; provenance is evidence, not a gate

**The error.** V4.5 §3.2 specified three gates: provenance (κ_p), competence (β), value alignment
(θ). Provenance does not belong in that list.

**The correction.** Provenance is **evidence about model depth.** Learning that something is
machine-generated lowers your estimate of how much reasoning produced it. That is an input to the
μ gate, not a parallel channel.

```
                provenance signal ─┐
                observer expertise ─┼─→  μ  ──→  θ  ──→  update
                surface properties ─┘
```

**Two gates, and the framework's two headline results become one mechanism with opposite signs:**

- **The generative crash is a correctly low μ.** Little reasoning behind it, so don't spend.
- **The trust exploit is a falsely high μ.** A dishonest provenance signal inflates the estimate,
  the observer over-invests, and fabricates to justify the investment.

This is tighter than three channels and it is the author's structure, not the spec's.

**Social influence** remains where V4.5 put it: upstream of both gates, shifting priors, never
appearing in the update path. The existing assertion stays.

### C3 — ω and observer inexpertise d are the same channel

**The observation.** ω measures how foreign the creator's structure is. `d` measures how poor the
reader's template is. Both produce one thing: a gap between what was generated and what can be
represented. Creator-side and observer-side causes of the same failure.

**Required test, and it is cheap.** A high-`d` observer on human content should be behaviorally
indistinguishable from a low-ω observer on foreign content, at matched effective overlap. If they
are, the model should carry one variable and say so. If they diverge, the divergence is
informative and names a second dimension.

### C4 — The latent goal: what the creator does not know about themselves

**The omission, and it is the most consequential item in V5.**

The model currently assumes the creator represents its own goal. The author's position is that a
large part of what makes something art is a goal that shapes the creator's behavior **while the
creator has no access to it**, and that an observer can sometimes extract it anyway — reading
someone better than they read themselves.

**Implementation.** Three distinct objects where the model currently has one:

- `goal_latent` — drives the creator's policy; **not represented in the creator's own model**
- `goal_declared` — what the creator believes and would report
- `goal_inferred` — what the observer recovers

The creator acts under `goal_latent`; its self-report is `goal_declared`; they can diverge. The
observer runs inference over the full space with no privileged access to either.

**Questions this makes askable, none of which the current model can pose:**

- Can the observer recover `goal_latent` when it diverges from `goal_declared`?
- Under what conditions does the observer's inference beat the creator's self-model?
- Does divergence between latent and declared *increase* extractable depth, as the theory
  implies it should?

**The prediction about generative systems, and it is much stronger than "AI content is shallow."**
A generative model has no unconscious: no goal pursued without being represented. It therefore
**cannot produce a latent/declared divergence at all**, in principle, at any level of output
quality. If that divergence is what the theory says it is, this is a categorical difference rather
than a quantitative one, and it is testable in-model.

### C5 — The Ghost Scale's attention gradient is non-monotone, and this is a real weakness

**Author's design critique, to be recorded in the paper's limitations rather than defended.**

1. **The 5% tier is close to a logical impossibility.** Prompting alone constitutes more selection
   than 5% implies. Functionally the tier is Curator with performative framing, and users should
   probably treat the two identically.
2. **The bounding box inverts the attention gradient.** The tier meant to signal "do not spend
   effort here" is rendered as the visually loudest element on the page, while Curator's reduced
   contrast makes it genuinely easy to skip. **The scale's attention gradient is therefore
   non-monotone in exactly the wrong place.**
3. **The two-tone image border has the same problem** and is visually awkward besides.

E1's finding that CURATOR costs the most DEEP looks is consistent with (2): the model has Curator
as the most expensive tier, and the visual design makes it the easiest to ignore. **The model and
the design disagree, and the design is probably right about human behavior.**

This belongs in the limitations section, stated by the author before anyone else states it.

---

## 2. Experiments

### E30 — μ replaces β (re-run of E28)

Identical design to E28, with μ (model depth) as the inferred quantity instead of β.

**Predicted:** update magnitude scales with recovered μ while goal accuracy holds across a wider
range than β achieved, because depth and legibility are less entangled than rationality and
legibility were.

**Consistency requirement:** report E28 and E30 side by side. E28 is not deleted.

### E31 — Two gates, corrected (re-run of E29)

Re-run E29's sequential-encounter design with the corrected architecture: μ and θ as the two
gates, provenance as evidence feeding μ.

**The decisive contrast is unchanged in form** — a gate that *blocks* integration versus one that
*weakens* it — but the weakening arm now varies model depth rather than rationality.

**E29's engagement predictions were wrong and are not carried forward.** V4.5 §5's signature table
is rewritten from E29's measurements, not from its predictions.

### E32 — The ω/d equivalence test

Match effective overlap two ways: foreign content read by a competent observer, and human content
read by an incompetent one. Compare engagement, resolution, confidence, disagreement, update.

**Falsification:** if the two are indistinguishable on every measure, they are one variable and
the model should be simplified. Report either outcome.

### E33 — The latent goal

Creator acts under `goal_latent`; `goal_declared` diverges with controllable probability.

**Measure:** observer's recovery rate for latent versus declared; conditions under which the
observer beats the creator's self-model; whether latent/declared divergence raises or lowers
recovered depth.

**Control arm:** a generative creator, which cannot produce the divergence by construction.
That contrast is the categorical claim in C4 and it is the reason to build this.

### E34 — Where does real generative content sit on ω?

**The question the framework now turns on**, surfaced by reconciling E19/E20 with the observation
that people are not in fact paralyzed by AI content.

E20 locates sustained futile attention below ω ≈ 0.04, and the crash plus peak fabrication
together at ω ≈ 0.10. Real generative output is trained on human data and inherits substantial
human-shaped structure, so it is plainly not at ω = 0.

**If real output sits near ω ≈ 0.10, both of the framework's headline phenomena occupy the same
band, and the band's location was predicted before the sweep ran.**

This cannot be settled in simulation. It is an empirical question about real generative systems
and it should be posed as one, in the write-up, as the thing a human study should measure.

---

## 3. What V5 does not change

- **E8 stays withheld** with its `xfail(strict)` marker.
- **E27, the V3 residual, stays open.**
- **V1–V4.5 results are not deleted or retroactively reinterpreted.** E28 and E29 stand as
  measurements of mis-specified constructs, labeled as such.
- **The null suite is unchanged and must continue to pass.**

---

## 4. README update, required before any public promotion

The landing page is at V3 state. It does not mention V4, V4.5, E19–E29, or any V5 correction.

**Required changes:**

1. **The experiments table lists e1–e6. There are more than thirty entries now.** Replace with the
   full plain-English index: what each asked, what it found, and what it *did not* find. One
   table, plain-language questions and answers, numbers inline. **Not two tables** — a technical
   table and a plain-English table drift apart within three revisions, and that is the same
   instrument-versus-claim failure this repo has caught six times.
2. **"What is being modeled" still describes synthetic content as goal-empty.** V4 replaced that
   with goal-foreign. Rewrite, and define both terms plainly, because the distinction is the
   single most misunderstood thing in the project: goal-empty is wood grain, goal-foreign is a
   page in a script you cannot read, and neither is value divergence.
3. **Add a "what died" section.** Seven hypotheses, named, including the author's own and the
   framework's own necessity claim from E21. **This is the section a stranger uses to decide
   whether to trust everything above it.**
4. **Add the E21 withdrawal explicitly.** A counting classifier reproduces the headline
   dissociation. State it plainly, high on the page, with what survives: label induction and
   sustained futile attention, neither reproduced by any baseline.
5. **Commit the figures and summary CSVs.** They are gitignored. A visitor arriving from a link
   cannot see a chart or check a number, and the README points at figure paths that do not
   resolve for them. That undercuts the only property this repository is selling.
6. **Rewrite the About field**, which is the link-preview caption everywhere the repo is pasted.
   It currently opens with three pieces of jargon.
7. **Add `LICENSE` and `CITATION.cff`.**
8. **Em-dash pass**, per the author's standing convention. Several are load-bearing and need the
   sentence rebuilt rather than the character swapped.
9. **Add RESULTS_V4.md and RESULTS_V4_5.md to the layout section.**

---

## 5. Pre-mortem

**The corrections are unfalsified.** μ, the two-gate architecture, and the latent-goal layer are
theoretically motivated and have not been tested. They are more defensible than what they replace;
that is not the same as being right. Each has a named falsification above and each must be able to
return it.

**The latent-goal layer is the most tunable object V5 introduces.** It has no external constraint,
and the divergence between latent and declared is a free parameter that could be set to produce
almost any result. Pre-register its construction before E33 runs, the way `sig_EXPLORE` was
pre-registered, and assert the construction at runtime.

**Scope, again.** V5 arrives after V4.5, which arrived after V4, which arrived after a withheld
experiment. The pattern is real. E32 is the cheapest item here and E34 requires no simulation at
all; if attention runs out, those two plus the README update are the ones that matter.

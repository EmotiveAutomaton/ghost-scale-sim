# Ghost Scale: proposed test set for the version after V21

**For the Ghost planning thread and coding agent.** Analyst proposal dated 2026-10-13. Nothing here launches work, edits V21, or changes a gear.

**Source.** The curator's 13 October walkthrough. Sounding Line tests what human text supports; these cards test whether the mechanisms work at all, in worlds where the answer is known. The claims:
- Credibility is maker inference pointed at the source.
- Uptake is gated by whether you have found the source's top goal, and whether it matches yours.
- Disgust protects by withdrawing attention.
- Preferences are set points revealed by costly departures.
- Wanting to want arises two ways: by learning what leads to what, and by association.
- Reading is constraint satisfaction over a goal hierarchy.

**Pin read.** `main` at `9db172644cc14283fb0c63a2b11a11c71acdffde`, the V21 README and three of its reports. V21 is running until 16 October; nothing here touches it.

**Construction rules carried forward** from the 10 October Ghost errata and V21:
1. Every learned comparison includes a validation-stopped direct reader.
2. The law-marginal posterior is the ceiling, and a headroom map gates admission.
3. Shift effects are reported reader by reader.
4. Every product-of-factors reader is checked for duplicated evidence.
5. Asymmetric histories are available.

V21's lessons bind too:
- A shared-maker mixture lost to the strongest matched rival by 0.42 to 1.00 nats despite headroom (`WIDE_NATIVE_SUPPORT_REPORT.md`).
- Unrelated history content did not remove the history benefit (`LEARNED_PERSISTENCE_REPORT.md`).

So no card below asks "does a maker model help" in general. Each asks a narrower mechanism question, against a rival that knows everything the mechanism knows.

**Priority:** A1, B1 and C2 first, because they answer the curator's three most specific claims. Then A2, A3, B2 and C1.

---

## A. Credibility and gated uptake

### A1. Credibility computed by maker inference (endogenous trust)

**Claim.** Trust is not a free parameter. A reader infers the source of a claim as a maker: competence (how noisy their judgments are) and goal (inform versus steer toward a fixed agenda). Uptake follows from that inference.

**Construction.**
- Each world draws artifacts with a latent property q.
- Sources are makers with competence σ_s ∈ {low, high} and goal ∈ {inform, steer}.
  - Informing sources report q plus noise σ_s.
  - Steering sources report a fixed agenda value plus noise, regardless of q.
- The reader sees a track record of k ∈ {0, 2, 8, 32} earlier source claims with outcomes, then a new claim at discrepancy D from its own reading of the artifact.
- The reader's own reading noise σ_r ∈ {novice, expert} is drawn per reader.

**Readers and rivals.**
- An exact reader that infers source type from the track record and integrates by Bayesian causal inference (the ventriloquist model of [Körding et al. 2007](https://www.cns.nyu.edu/malab/static/files/publications/2007%20Kording%20Shams.pdf)).
- A fixed-trust reader, the current Ghost mechanism, at trusts 0.2, 0.54 and 0.9.
- A label-only reader: trust set by a stated label, without the track record.
- A validation-stopped direct reader trained on (track record, claim, own reading) to predict q.
- An oracle that knows the source type.

**Measures.**
- Belief shift as a function of D, by source type and reader expertise.
- Signed error on q.
- Time to detect a steering source as k grows.

**Predictions.**
1. The exact reader reproduces the direction of [Aronson, Turner & Carlsmith 1963](https://doi.org/10.1037/h0045513). For informers inferred as competent, shift keeps rising with D over a wide range; for those inferred as weak, it peaks early. The analyst's illustrative computation of 13 October shows this shape.
2. Expert readers shift less, and stop at smaller D.
3. Ghost's 0.54 crossover appears only for short track records (small k). With long records, a steering source is detected whatever its label: the trust exploit is a short-history phenomenon.

**Refuted if:**
- the endogenous reader's shift curves match a fixed-trust reader at some single trust value across all k and source types; or
- the direct reader matches the exact reader at every k. That would be parity: the structure buys nothing.

### A2. Top-goal misidentification (the propagandist)

**Claim.** Uptake depends on the probability that you have found the source's top goal and that it matches yours. A reader that misfiles a steering artifact as informing absorbs its content. A reader whose expertise contains the steering process attends to technique and absorbs little content.

**Construction.**
- Makers with top goal inform or steer produce artifacts. Steering artifacts are built to match informing artifacts on surface features, and differ in a process signature: a selection pattern over which facts appear.
- Readers carry an expertise library either with or without the steering process. This is persuasion knowledge, after [Friestad & Wright 1994](https://ideas.repec.org/a/oup/jconrs/v21y1994i1p1-31.html).

**Measures.**
- Content uptake: the shift in the reader's world belief.
- Technique uptake: the reader's later production using the maker's operations.
- The posterior on the maker's top goal.

**Predictions.**
1. Readers without the steering process show high content uptake from steering artifacts.
2. Readers with it show low content uptake, unchanged or higher technique uptake, and a correct goal posterior.
3. The difference vanishes when steering and informing artifacts are indistinguishable in process (the control).

**Refuted if** content uptake does not differ by library when the process signature is present.

### A3. Withdrawal of attention as the gate (disgust)

**Claim.** Protection is the withdrawal of attention: precision taken off the content channel once a goal flag fires. Disgust is the trigger that draws the line.

**Construction.** Reuse the E56 and V18.2 G5 machinery. Reader policies:
- no gate;
- early guard (E56);
- procedure-only attention (V18.2 G5);
- flag-triggered withdrawal: precision on purpose and value channels drops to a floor once the inferred goal probability of a flagged class passes a threshold.

**Measures.**
- Purpose and value uptake.
- Technique uptake.
- Purpose-recognition accuracy: the cost of not looking.
- The threshold's false-alarm rate on benign makers.

**Predictions.**
1. Flag-triggered withdrawal blocks more purpose and value uptake than the early guard, at matched technique uptake.
2. It pays in lower purpose-recognition accuracy.
3. False alarms on benign makers cost technique uptake too.

**Refuted if** withdrawal blocks no more than the early guard at matched technique uptake.

---

## B. Preferences as set points

### B1. Homeostatic makers: set points from costly departures

**Claim.** A maker guards set points. Drive is distance from a set point; choices reduce drive at an effort cost. Effort spent is the strength of a preference vector. The rationale follows homeostatic reinforcement learning ([Keramati & Gutkin 2014](https://elifesciences.org/articles/04811)), where seeking reward equals staying stable.

**Construction.**
- Makers have 2 to 4 internal variables with set points h* and drive $D(h)=\big(\sum_i |h^*_i - h_i|^n\big)^{1/m}$.
- Each episode starts from a perturbed state and offers a menu of actions with known effects and effort costs.
- The maker chooses by softmax over drive reduction minus effort, at precision γ.
- Artifacts record the choices, and the effort spent when a choice departs from the maker's habit (its most frequent past choice).

**Readers and rivals.**
- An exact inverse reader that estimates set points as intervals, via ideal-point (unfolding) estimation.
- An effort-weighted reader that weights each choice by effort paid.
- An unweighted inverse reader.
- A frequency reader.
- A validation-stopped direct reader predicting held-out choices.

**Measures.**
- Set-point recovery: interval coverage and width.
- Held-out choice log loss.
- Choice consistency as a function of distance from the set point.

**Predictions.**
1. Choice consistency rises with distance from the set point. Near it, choices look random and carry little set-point information.
2. Effort-weighted evidence recovers set points with fewer episodes than unweighted evidence.
3. The direct reader matches held-out choice prediction inside the training range but fails when perturbations exceed it, where set points transfer and memorized responses do not.

**Refuted if:**
- effort weighting adds nothing at matched episodes; or
- the direct reader matches the set-point reader under out-of-range perturbations.

### B2. Wanting to want: two acquisition routes, told apart by devaluation

**Claim.** New preferences arise two ways:
- by learning that an action leads to set-point satisfaction (model-based);
- by the action co-occurring with satisfaction until it becomes rewarding in itself (associative). This is Allport's [functional autonomy of motives](https://psychclassics.yorku.ca/Allport/autonomy.htm).

**Construction.**
- Makers acquire a preference for action X during a training phase, through one route or the other.
- Then the need X used to serve is devalued: its set point is satisfied.
- This is the outcome-devaluation logic that separates goal-directed control from habit in the animal-learning literature (Dickinson; Balleine & Dickinson).

**Measures.**
- The reader's posterior on the acquisition route from pre-devaluation artifacts.
- Prediction of post-devaluation choice.

**Predictions.**
1. Model-based makers stop choosing X after devaluation, and associative makers persist.
2. Pre-devaluation artifacts carry a route signature: model-based choice of X tracks the contingency; associative choice tracks co-occurrence even when the contingency is broken.
3. A reader that models both routes predicts post-devaluation behavior better than one that assumes either route alone.

**Refuted if** the two-route reader matches the better single-route reader on post-devaluation prediction.

---

## C. Reading as inference over a goal hierarchy

### C1. Constraint satisfaction over the goal pyramid

**Claim.** Readers label as many nodes of the maker's goal hierarchy as they can and adjust all of them jointly until they cohere. The top goal is the most generally useful node. A node's strongest evidence comes from its immediate neighbors.

This has a formal version. In a tree-structured model, information from a distant node reaches a given node only through the nodes between them, so it cannot exceed what those intermediate nodes carry (the data processing inequality). Text comprehension has a classic model of this settling: Kintsch's construction-integration model.

**Construction.**
- Makers generate artifacts from goal trees of depth 4 with branching 2 to 3.
- Leaves are observable operations.
- Readers' expertise covers a random 25%, 50% or 75% of the process library.

**Readers.**
- Top-down only.
- Bottom-up only.
- Joint belief propagation over the tree.
- A validation-stopped direct reader.

**Measures.** Per-node posterior accuracy, by node depth and by expertise coverage.

**Predictions.**
1. Joint inference beats either direction alone where coverage is at least 50%.
2. Each node's accuracy depends more on its parent's and children's evidence than on the root's.
3. Top-goal accuracy degrades smoothly, not stepwise, as coverage falls. This is the curator's correction: depth is not capped; error grows.

**Refuted if** top-down-only matches joint inference at every coverage level.

### C2. Similarity helps process, records help preferences

**Claim.** This is the curator's restatement, consistent with Sounding Line L192. Similarity between reader and maker makes process inference cheap. Preferences come from the record of what the maker attended to and chose, and similarity does not supply them.

**Construction.** Cross two factors:
- the maker shares the reader's process library, or does not;
- the maker shares the reader's set points, or does not.

**Targets.**
- Process prediction: which operation or route the maker uses.
- Preference prediction: which option the maker picks when routes are fixed.

**Predictions.**
1. A shared process library improves process prediction and not preference prediction.
2. Records improve preference prediction regardless of shared process.
3. Shared set points help preference prediction only through records, not through similarity alone.

**Refuted if** a shared process library improves preference prediction by more than the practical margin with records held fixed.

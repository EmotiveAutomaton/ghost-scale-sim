# Ghost Scale Sim V16: acquired craft, self-reconstruction, and the reader's changing possibilities

**Coding-agent handoff · 9 September 2026**

**Commission:** build and run one finite, branching, theory-led simulation campaign that supplies methods, counterexamples, and transferable tasks for Sounding Line. The curator has completed the first verbal round and asked for this package. No additional theory walkthrough is required before implementation. This document specifies new work; the analyst has not implemented or launched it.

**Reviewed published sources:** Ghost Scale Sim at `82b4aa39d756904b2aecf5dd459eddd7962b3562`; Sounding Line at `6c090560e01d1ddf2386d26da047ef1a08a22697`. Both remained unchanged when checked on 9 September. The initial complete theory read carries forward from the preceding review of these same sources; relevant definitions, reader heuristics, correction history, operating contracts and current handoffs were revisited for this package.

**Resource default selected for this handoff:** a **120-hour elapsed planning ceiling, including setup and closeout**, on the currently authorized local allocation. This is a V16 design choice, not a duration newly dictated by the curator. Use CPU work as the independent core. No Gear 3, paid scientific API calls, new hardware, author contact, or human recruitment. Do not change the current gear or compete for Sounding Line's GPU. A newer coding operator is an opportunity to build better instruments; it is not evidence that a scientific reader is more capable.

## 1. The question and intended advance

**How do learned ways of acting become readable, correctable, and reusable through the things a maker produces?**

V16 should explore the larger theory through executable production, rather than inherit V15's card inventory or assume its preferred model is the next answer. The desired object is a maker with acquired procedures and limited self-knowledge, leaving an artifact that another bounded agent can interpret, attempt to reproduce, learn from, and use to predict a changed-context choice.

The campaign compares several possible explanations:

- Personal acquisition history creates reusable craft that helps explain an individual beyond generic domain knowledge.
- Generic environmental structure produces most useful skills; a personal repertoire adds little.
- Apparent personal preference is partly a consequence of what a maker knows, can perform, considers, or remembers.
- Observing one's own actions helps recover otherwise inaccessible control information, or ordinary goal-directed error monitoring does the same job more cheaply.
- Recognition attracts inquiry near a useful learning opportunity, while surprise, familiarity, competing goals, and costs produce different patterns.
- Selection, collaboration, and changes of purpose can make the final artifact systematically misrepresent an individual's production history.

The output is a **map of useful capabilities, ambiguities, and separating observations**, with examples one can inspect. A positive mean score is insufficient; an informative failure can be a major result.

### Three products that must remain separate

1. **Executable reconstruction:** a route the reader can use successfully under its own tools and constraints.
2. **Historical correspondence:** a calibrated account of what this maker actually did or knew, where the construction makes that question identifiable.
3. **Prospective maker prediction:** successful prediction of this maker's hidden continuation, repair, or response to an intervention.

Persistent motivational inference is a later, separately gated shadow branch. A useful reconstruction may be historically wrong. A correct historical account may teach an ineffective route for a differently equipped reader. Good maker identification may come from a surface signature without recovering production. Record these combinations rather than collapse them.

The ratified Sounding Line definition already concerns reconstructing how a maker transformed perceived possibilities into an artifact. V16 develops the simulator's ability to express that problem; it does not claim to introduce the definition. [Current theory](https://github.com/EmotiveAutomaton/sounding-line/blob/6c090560e01d1ddf2386d26da047ef1a08a22697/docs/theory/THE_TRIPLE_INFERENCE.md)

## 2. What the verbal round changes

The following table separates the curator's contribution from the analyst's executable interpretation. All mechanisms remain hypotheses.

| Walkthrough contribution | V16 consequence | Rival or qualification |
|---|---|---|
| A new account of a performance changes expected audience response and reveals another solution | Separate knowledge of the work's effect, an executable route, and historical production evidence | The original music prompt did not specify what extra evidence the rehearsal supplied; do not treat it as a completed discrimination |
| A creator reverse engineers subsidiary goals from unfolding behavior | Add a self-monitor with partially inaccessible control state; compare its corrective value with direct error monitoring | The benefit may be additional information or better ordinary planning, not a special self-inference mechanism |
| Old craft can serve a new purpose while subtly distorting the solution space | Learn procedures before changing the task; measure useful reuse, interference, and the cost of checking | A habit may be an efficient compression over many tasks even when a better local alternative exists |
| Fair criticism adopts the creator's actual limitations | Distinguish physical feasibility, learned competence/beliefs, and considered alternatives | Incorrect beliefs can include impossible actions; these sets are not always nested |
| Recognition creates an opportunity to refine a model | Compare inquiry under different expertise, familiarity, learning opportunities, and opportunity costs | Disinterest does not establish lower intelligence, less openness, or poor understanding |
| Returning to unfinished work involves reconstructing or replacing its goal | Add interrupted-maker experiments with separate original-goal and adopted-goal outcomes | This answers a different question from the original viewer-learning prompt; deterioration is not assumed |

Three short passages are suitable quote candidates, with the ordinary transcript provenance attached:

> You have to reverse engineer your own goals through your actions as you are going.

> You have to own the limitations of the creator to compare yourself to them.

> I might be close to learning something real.

These preserve the supplied wording; punctuation may be normalized. They do not ratify this document's models or numerical criteria. Preserve the curator's lower confidence around fatigue, quality decline, and broad personality interpretations. Do not reconstruct the unrelated, incoherent University of Minnesota sentence or guess which neuroscientist was named.

### The Murderbot observation

Treat this as a report of rapid recognition followed by directed biographical inquiry. Candidate explanations include shared experience, subject matter, learned conventions, a maker-specific production signature, and selective confirmation. Do not turn it into a diagnostic benchmark for real people's autism or a retrospective success-rate estimate.

One factual correction matters: Martha Wells says she worked in computer support, programming, and web design before writing full time. The premise that she had only been a fiction writer is incorrect. That does not establish which occupational experience caused any feature Abraham noticed. Her first-person career account is sufficient for this correction; the simulation needs no judgment about a medical diagnosis. [Wells interview](https://www.actusf.com/detail-d-un-article/interview-martha-wells-vo)

## 3. Current state and inherited debt

### Ghost

V15 ended with 150 saved verdicts: 112 discovery, 24 attack, eight transfer, and six confirmation. Two discovery instruments failed. The closure audit discloses a failed continuous-runtime contract and incomplete independent regeneration from retained unit outputs. The earlier analysis identified additional criterion and information-access defects. Preserve the original packets and their failure labels. [Closure audit](https://github.com/EmotiveAutomaton/ghost-scale-sim/blob/82b4aa39d756904b2aecf5dd459eddd7962b3562/results/v15/CLOSURE_AUDIT.json)

V16 must not be blocked on a wholesale V15 rerun. Spend at most **two setup hours** inventorying which defects affect reused components. Correct V16 dependencies before using them; add a dated V15 interpretation correction where supported. Changed V15 estimators or criteria require a separately identified scientific amendment, not a receipt rewrite. Work that cannot be completed within this allocation remains explicit debt and does not enter V16 as a validated dependency.

The table identifies the inherited defects and their V16 treatment.

| V15 location | Consequence | V16 action |
|---|---|---|
| `cards/trunk_x.py`, X12–X18 | Same-reader subtraction guarantees zero | Never reuse that contrast. Require a nonzero known-answer degradation fixture and distinct arm identities |
| `cards/trunk_p.py`, P01/P02 | Supplied true state was compared with supplied labels | Label these as privileged upper bounds; actual reader inputs must be inspected and tested |
| `cards/trunk_c.py`, C08/C14 | Coverage/onset thresholds were applied to a log-score statistic | Typed estimands with units and component receipts; invalid intended contrasts remain unmeasured |
| C04 reducer/reporting | A pooled interval accompanies a maximum; evidence-dose sorting is lexical | Each interval identifies its estimator; doses sort numerically |
| `learning_history.py` | Weak observation-count rival, privileged answer-map access, worst-item scoring | New learned-craft comparison uses matched access and genuinely prospective outcomes |
| M01 and approximate readers | Failed approximation validation | Exact inference is the first reference; approximation earns admission independently |
| B01/B02, claim/confirmation ledgers | Missing stages can be treated as success; stage joins omit actual attack mapping | Explicit dependency IDs, explicit missing state, no count-based V17 recommendation |
| Saved unit outputs and occupancy | Hash consistency does not regenerate all aggregates or establish activity | Retain raw records and actual work accounting; perform a real independent reaggregation |

Source owners are under `ghostscale/validation/soundingline/v15/`; detailed diagnosis is in the 7 September analysis, Appendix A. The current code and primary receipts take precedence if the coding agent finds a later repair. [V15 code](https://github.com/EmotiveAutomaton/ghost-scale-sim/tree/82b4aa39d756904b2aecf5dd459eddd7962b3562/ghostscale/validation/soundingline/v15)

### Sounding Line

The published September 6 operational handoff says Stage 8 and its maintenance queue drained, with zero admitted readers and zero frozen confirmations. It does not contain a running Stage 9 record; only `main` was published when checked. The separate Stage 9 handoff already supplies a broad five-day program. **The cause of the current local delay is unknown from the accessible record.** Lack of pushes does not establish lack of work, and elapsed time does not establish careful setup. [Published state](https://github.com/EmotiveAutomaton/sounding-line/blob/6c090560e01d1ddf2386d26da047ef1a08a22697/docs/STATE.md)

Do not stop, restart, or edit Sounding Line from this Ghost package. Its operator should be able to report the last completed executable test, current blocker, active owned process, remaining setup work, and next expected checkpoint from existing records. That is the diagnostic needed to explain a delay. Ghost runs independently and supplies a bounded transfer packet at closeout.

## 4. Research choices and external-code intake

### Method choices

This table states what each method actually contributes. Adaptations are new Ghost implementations unless explicitly reproduced against a published anchor.

| Method | Useful import | Scope limit and comparison |
|---|---|---|
| DreamCoder and ShapeCoder | Learning reusable production procedures; reconstructing artifacts through a learned library | A compact program need not be the maker's actual process. Full training pipelines are unnecessary for the first finite world. Compare generic libraries and independent generators |
| Stitch | Efficient abstraction discovery from an existing corpus of programs | It consumes programs, not raw images. Use for the maker's permitted training record or independently inferred reader programs, never evaluator-only true programs |
| Bounded inverse planning / SIPS | Interleaved planning and acting; goals inferred while allowing limited search | Unknown considered sets are a V16 extension. Start with exact finite inference; do not assume particles are faster |
| Laplacian option discovery | Executable skills derived from observed environmental transitions | A generic-skill rival to personal craft; do not give it the hidden complete transition graph |
| Second-order self-evaluation | Actions can inform an observer who lacks some evidence that generated them | The published model concerns decision correctness. Extension to subsidiary goals and goal memory is our hypothesis |
| Competence-progress inquiry | Allocate attempts using experienced changes in ability | Absolute change includes decline; progress can be noise and delayed gains can be missed. Compare signed progress, information gain and ordinary practice |
| MAP-Elites / adaptive environment editing | Retain distinct cases and search for nearby changes in outcome | A chosen archive is not exhaustive coverage. Compare fixed stratified sampling and validate new cases independently |
| Simulation-based calibration | Check inference against a known generative model | Ordinary marginal ranks can miss ignored evidence. Add joint/data-dependent checks and different-generator evaluation |

Primary sources, read scope, and limitations are recorded in Appendix C. The methods are challengers and components, not substitutes for the local theory.

### Smallest useful intake

**Only Stitch is an optional external runtime candidate in the initial queue.** Everything else begins as a mathematical or design reference. The core must execute using the resident Python/NumPy/SciPy stack. A native bounded motif learner is required as both fallback and rival; it must not be advertised as a complete Stitch reimplementation.

Current repository pins were checked through GitHub on 9 September. A pin is source identity, not installation verification.

| Repository | Inspected revision | Reuse decision |
|---|---|---|
| `mlb2251/stitch` | `350804b7b35807c78bd21c313785ae5152ae2985` | MIT recorded by GitHub; README/API route inspected. Optional isolated Rust CLI or separately verified binding |
| `mlb2251/stitch_bindings` | `8ba2c1c041ab384ccaa11e68226d5654474d161a` | README advertises Python ≥3.7 and wheel/source builds; Python 3.13/Windows support and code-license coverage were not validated here. Do not silently combine it with a different core revision |
| `ztangent/Plinf.jl` | `87afdb1741ba7e6765c5bae04f4b34c84a8f4608` | Julia/Gen/PDDL reference. Code license not verified; no runtime dependency |
| `mcmachado/options` | `176916c984cd1637fe5d51c8e7b677a9ce41f917` | Official option-discovery reference; current dependency/license suitability not verified. Implement the small published mathematical construction independently |
| `smfleming/Self-evaluation-paper` | `0865168239153f8dc1ff39f8e0b5f65ffc102735` | MATLAB reference; code license not verified. Use an independently implemented finite analogue, not copied source |

**Intake procedure:**

1. Use a separate reference directory and environment outside live scientific checkouts. Pin the revision before reading executable paths. Record URL, revision, license, intended component, dependencies, and input/output contract in one intake manifest.
2. A read-only clone is permitted. Do not automatically initialize submodules, Git LFS downloads, notebooks, hooks, setup scripts, or benchmark launchers. Do not deserialize executable objects as data. Inspect any installation/build entry point before invoking it.
3. For an optional runtime, verify code-license applicability, dependency resolution, target interpreter/platform, and a minimal upstream example. Record the actual executable and dependency lock without modifying a live or historical environment.
4. For Stitch, reproduce the basic README example and its declared compression-cost calculation, then independently expand the rewritten program and verify identical behavior. Representation compression is the anchor; novel maker inference is the extension. Do not use flags that disable consistency checks.
5. Cap all initial external intake at **two active coding-agent hours combined**, at most **one hour of build/debug effort for any candidate**, and at most **one GiB of fetched source/dependencies in aggregate**. These are campaign budgets, not general file-policy changes. If the environment or upstream size makes that unsuitable, record the boundary and use the native implementation.
6. No source acquisition may block the first end-to-end native case. If the runtime candidate fails, its consumers retain the native comparison or become explicitly unexecuted. Do not repeatedly chase mirrors, contact maintainers, or install a robotics/GPU stack.

No reference repository was installed or executed in this analyst review. [Stitch source](https://github.com/mlb2251/stitch/tree/350804b7b35807c78bd21c313785ae5152ae2985), [binding source](https://github.com/mlb2251/stitch_bindings/tree/8ba2c1c041ab384ccaa11e68226d5654474d161a)

## 5. The simulation substrate

Build a small shared interface, with independent mechanism implementations behind it. Do not build a general cognitive architecture before producing an executable case.

### 5.1 Two production families

**W1: finite graphic construction.** Begin with a small discrete canvas, an explicit primitive action set, finite execution budget, and terminal artifact consisting only of visible cell/shape state. Procedures can place, remove, transform, repeat, and compose legal operations. Introduce reusable subprocedures through training rather than assigning every maker a descriptive skill label. Use one tiny exhaustively enumerable size for validation and a larger bounded size for exploration. A 4×4 canvas with at most six primitive steps is an initial size to profile, not an obligation to enumerate an intractable space.

**W2: constrained assembly and revision.** A maker assembles a small part graph with dependencies, performs edits, and stops. The final artifact records the resulting parts and relations; intermediate order, rejected alternatives and controller state are private unless an access arm purchases them. Its mechanics and objective must differ structurally from W1, not merely rename its tokens. Reuse stable V15 composition concepts where helpful, but do not treat a state-averaged emission summary as a finished artifact. [Existing composition model](https://github.com/EmotiveAutomaton/ghost-scale-sim/blob/82b4aa39d756904b2aecf5dd459eddd7962b3562/ghostscale/validation/soundingline/v15/world_composition.py)

W1 is the required early spine. W2 is the planned transfer family and must earn its own gates. If it cannot be validated within its build allowance, cross-family claims are unavailable; continue eligible W1 questions and disclose the missing transfer. Do not fabricate independence by changing a renderer.

### 5.2 Makers and acquired procedures

A maker record separates:

- An initial procedural vocabulary and learning rule.
- A dated training history, with actual trials, feedback, attempts, and instruction.
- A learned transition model and procedure library, including errors and limited coverage.
- An artifact-level purpose, subsidiary control state, and current task constraints.
- Candidate generation and bounded search, including options not considered.
- Episodic memory and a separate, optionally imperfect self-monitor.
- Optional persistent tradeoffs and editorial/collaborative roles in eligible branches.

Learned procedure definitions must expand to executable primitive behavior. Training changes what the maker can execute successfully, how cheaply it can search, or what it predicts about outcomes. A renamed latent category or a constructor-written library selected by the maker label does not count as acquired craft.

**Required acquisition rivals:** primitive-only bounded planning; a small native repeated-subtree/motif learner; a pooled generic learned library; a maker-specific library. Include transition-derived options where validated. Stitch is an optional additional learner, not the definition of expertise.

For the native motif learner, enumerate repeated bounded program fragments from permitted training traces, admit only fragments with independently verified expansion equivalence, and charge library definition plus invocation cost. Freeze arity, fragment-size cap, tie-breaking and admission policy before evaluation. Keep skill testing on new tasks; compression of the training corpus is only a diagnostic.

For transition-derived options, estimate the graph from observed exploration, compute a small spectral basis, and learn tabular terminating policies. Charge both training exploration and every expanded primitive step. Report graph coverage, disconnected components, repeated eigenvalues, and option termination. Use subspace- or behavior-level comparisons where eigenvectors are non-unique; never claim that an arbitrary eigenvector sign is a personal characteristic.

### 5.3 Possibility is not one set

Maintain separate quantities for:

1. **Actual feasibility:** what the physical world, tools, and current execution constraints permit.
2. **Learned capability and belief:** which procedures the maker knows and how it predicts their consequences or success.
3. **Consideration:** which believed possibilities enter the current search.
4. **Search effort:** how much of those possibilities is evaluated before acting.

In an accurate-belief anchor, the considered legal actions can form a subset of known feasible actions. In general the maker may consider an impossible action or overlook a feasible one. **Do not enforce a universal nested-set model that eliminates false beliefs by construction.**

Use three distinct interventions: change a physical constraint; provide training or a demonstration; and draw attention to an already known option. A fourth can change search budget. A preference inference must survive comparisons with these mechanisms. The critic's better solution is scored both under the critic's own resources and after adopting the maker's actual limitations.

### 5.4 Goal, control, and self-monitoring

Keep the remembered artifact-level purpose separate from the subsidiary policy currently shaping behavior. A maker may know the former while needing to detect a habitual subprocedure that conflicts with it. Do not force every action to be consciously chosen or every automatic action to be a mistake.

Construct a finite self-monitor whose accessible evidence is explicitly different from the lower-level controller's state. It can update from private memory, visible action and artifact consequences. Compare with a direct mismatch detector that knows the current purpose and checks outcome error without estimating hidden goals.

For interruption, independently vary memory loss, new information, deliberate goal replacement, and unchanged-purpose continuation. Keep a posterior over the original goal separate from the goal adopted for the next action. Score fulfillment of the original and adopted goals separately. No universal decline in artistic quality is assumed; the simulator's quality measures are declared task objectives, not an aesthetic verdict.

### 5.5 Readers

Readers must expose their observation contract, training data, search budget, and outputs. Required competitors, enabled locally to each claim:

| Reader | Role |
|---|---|
| Surface/statistical reader | Predicts from visible artifact features; exposes shortcut success |
| Generic domain reader | Uses pooled training and no maker-specific history |
| Primitive-program reader | Searches executable routes without learned personal abstractions |
| Generic-library reader | Uses a pooled repertoire under the same evidence budget |
| Maker-conditioned reader | Infers an individual model from allowed earlier artifacts/evidence |
| Finite exact reader | Exhaustive reference under an explicitly stated finite model and access contract |
| Supplied-state reader | Privileged upper bound, clearly excluded from learned-reader promotion |

A direct predictor must be a serious rival: give it the same allowed artifact, context, and prior works, with a frozen feature or tabular representation appropriate to the finite task. Do not recreate V15's observation-count-only rival and call it a strong expertise account. Report trained capacity and computational cost separately from observation access.

### 5.6 Information boundary and likelihoods

Use physically separate serialized **reader observations** and **evaluator records**. The reader process receives only the former and a declared public world interface. Evaluator records include true programs, acquisition histories, realized candidate menus, goals, controller states and counterfactual outcomes. Prohibit generic world-object references in reader APIs.

The primary route receives a finished artifact and declared public context. Earlier artifacts, partial process traces, training records, biography-like notes, and tool demonstrations are separate access conditions with explicit acquisition costs. A language rendering may translate visible data; it must not mention hidden operation names or narrative rationales supplied by the constructor.

Artifact likelihood must marginalize over all represented histories that produce that artifact. Do not substitute the likelihood of the one generating trace. Use a scalar brute-force reference on the tiny support, then a separately checked dynamic-programming/cache implementation. Approximation is introduced only when a measured bottleneck justifies it and must report missing support. A zero-likelihood observation outside the reader's model triggers an explicit model-mismatch outcome, not silent uniform probabilities.

An executable reconstruction is returned as a program/policy and run by a separate interpreter. A fluent explanation does not count as execution. Future maker predictions are submitted before future observations or intervention results are revealed.

## 6. Instrument admission and measures

### 6.1 Admission is local to a capability

Do not use one omnibus pass/fail flag for the whole program. Record at least: generator validity, serialization/access validity, inference validity, legal construction, successful construction, prospective prediction, adaptation, and calibration. A prediction-only reader can contribute to a prediction question; it cannot support a claim that requires executable reconstruction. This does not retrospectively alter Sounding Line Stage 8's original admission rule.

Every new instrument has a hand-checkable positive, a null, a deliberate break that its gate detects, and a boundary where the intended quantity is not identifiable. Run these before unknown outcomes. In the exact tiny support, require exhaustive equality to the independently implemented scalar reference, within a frozen floating-point tolerance of 1e-10 where rational equality is unavailable. Numerical tolerances are engineering choices, not effect thresholds.

Required gate families:

| Gate | Known answer and deliberate failure |
|---|---|
| Artifact marginalization | Several traces share one artifact; probabilities must sum. The one-trace substitute must fail |
| Forbidden information | Hold reader bytes and the reader's own random stream fixed while altering private labels, history or evaluator seed metadata. Reader outputs must remain unchanged. An intentionally leaky reader must fail |
| Target realization | Interventions actually change feasibility, knowledge, consideration or purpose as specified. A changed annotation with unchanged execution must fail |
| Executable procedures | Macro expansion and primitive execution agree, including costs, errors and stopping. A swapped primitive or lost termination must fail |
| Bayesian computation | Normalization, impossible evidence, correlated latent variables and discrete ties match the exact anchor. Add a reader that ignores data and one that uses the wrong dependence |
| Estimand identity | Arm IDs, units, sign, aggregation level and interval target are checked. Same-arm subtraction cannot pass a degradation test |
| Restart/reaggregation | Interrupt a real small packet; resume without duplicate units; independently recompute every aggregate from saved rows |
| Task family independence | A renderer-only change cannot be counted as new physics or a new production mechanism |

SBC parameter ranks are supplementary. Include data-dependent quantities and observed-evidence-use tests because a data-ignoring prior sampler can pass some ordinary rank checks. Calibration under the simulator does not validate the simulator as a model of people.

### 6.2 Outcomes and proposed practical bars

Freeze the actual estimand for each card before its discovery outcomes are inspected. These bars guide promotion and resource allocation. A smaller clean effect is still reported; a large effect cannot rescue a failed instrument.

| Outcome | Definition and reporting | Default practical bar for possible promotion |
|---|---|---|
| Future prediction | Mean paired difference in log probability assigned to the hidden realized next action/outcome; nats per event | +0.02 nats over the strongest preregistered eligible rival |
| Executable reconstruction | Fraction of all submitted attempts that legally achieve the declared target; include invalid programs and timeouts | +5 percentage points successful execution; report legal-execution rate separately |
| Adaptation | Goal attainment and primitive cost after a changed constraint or purpose | +5 percentage points success, or 10% lower cost with the paired success difference established within ±1 percentage point |
| Historical narrowing | Mean log score on the actual identifiable process class; coverage reported separately; not exact-string matching | +0.02 nats on the declared historical target, with ambiguity controls intact |
| Self-correction | Correct repairs minus harmful interventions, and detection delay | +5 percentage points net correct repair at matched monitoring cost |
| Inquiry | Held-out task performance after a fixed interaction budget, plus total inquiry/training/execution costs | +5 percentage points downstream success; claim lower cost only at comparable performance |
| Sustained calibration | Coverage, Brier/log score and confidence trajectories on the declared targets | No universal numeric promotion bar; use exact calibration anchors and frozen target-specific tolerances |

For a constructive capability claim, report the unassisted legal-attempt rate and its interval; the proposed admission threshold is at least 99% observed legality on that claim's evaluation tasks. This is distinct from successful goal achievement. Always expose legality masking, search assistance, demonstrations and other support. A mask supplied by the environment is not competence learned by the reader.

Use only task-defined quality measures. Record quality under the original goal, the newly adopted goal, and any independent constraint separately. Never choose whichever definition makes an interrupted maker look better or worse after inspecting the result.

## 7. The branching study forest

There are **42 registered work cards**: 27 principal mechanism cards, three persistent-tradeoff shadow cards, eight adversarial cards, and four bridge/archive cards. The manifest must count its own rows and reject a mismatched narrative count: K=5, P=4, O=4, S=5, R=5, M=4, V=3, X=8, B=4. This is an inventory of work, not a sample size or a claim of breadth by itself.

Every scientific card is instantiated as an explicit design record: question; mechanism; strongest rival; access arms; target realization check; primary estimand and units; secondary descriptions; generator families; paired sampling unit; frozen sample rule; dependencies; relevant adversaries; repair budget; and continuation rule. A card missing one of those fields is not queue-eligible. The tables below supply the scientific content; the operator translates it into the existing manifest conventions.

### K. What acquired craft adds

Rows identify the question, its decisive comparison, and the output. All require executable world/procedure gates.

| ID | Question and comparison | Main output / dependency |
|---|---|---|
| K01 | Can reusable procedures be learned from training and used on genuinely new compositions? Compare personal motif learning, pooled learning, and primitive-only planning with equal training experience | New-task execution and cost; required core acquisition test |
| K02 | Does the personal repertoire predict this maker beyond the generic repertoire? Give readers the same prior artifacts and hide true libraries | Prospective score and executed adaptation; requires K01's valid machinery, not a positive K01 result |
| K03 | How does old expertise respond to a new purpose? Cross aligned, partially aligned and opposed goals with continued use, selective inhibition and relearning | Reuse/interference curve and checking cost; no prespecified claim that old habits are harmful |
| K04 | Can generic transition-derived options explain the apparent personal-library benefit? Compare options, motifs and primitives at matched exploration and primitive evaluations | Performance–cost frontier; optional method gate, independent of Stitch |
| K05 | Does attention allocation during acquisition predict later repertoire after instruction, feedback and exposure are separated? Record attention as actual allocated attempts/processing, independently of its eventual effect | Matched-skill transfer and errors; tests an operational acquisition hypothesis, not a definition of attention |

**Automatic continuation:** if personal libraries add nothing but generic skills help, preserve the generic-skill mechanism and route the reader comparison through it. If neither helps in a valid repeated-subproblem positive control, repair the acquisition instrument once or close its dependent branches. A flat scientific K02 result does not invalidate K01's learner.

### P. Reconstruction, historical truth and future behavior

| ID | Question and comparison | Main output / dependency |
|---|---|---|
| P01 | Can the reader produce the visible artifact without recovering the original route? Construct multiple valid histories and compare executable reconstruction with historical scoring | Joint map of execution, ambiguity and historical correspondence; core identifiability anchor |
| P02 | Which new observations separate histories that share an artifact? Compare continuation, changed tool constraints, a process fragment and another prior work | Intervention-specific equivalence classes and prospective predictions; requires P01 |
| P03 | Does an individual model accumulate useful information across works? Use evidence doses 0, 1, 2, 4 and 8 earlier works, with maker, topic, training and history groups separated | Dose curves by target and family; generic and surface rivals mandatory |
| P04 | Does reconstruction survive a different production mechanism? Train/infer with program-library makers; evaluate option-based, bounded-search and habitual makers, then reverse the direction where feasible | Misspecification, abstention and prediction; requires at least two independently validated generator mechanisms |

**Automatic continuation:** if enactment succeeds but history remains ambiguous, classify this as productive reconstruction with unresolved history. Probe only distinctions relevant to a named future task. Do not keep inventing biography until an exact historical label becomes recoverable.

### O. The maker's actual and perceived opportunities

| ID | Question and comparison | Main output / dependency |
|---|---|---|
| O01 | Can the reader distinguish inability, lack of knowledge, omission from consideration and different purpose? Create paired cases with the same initial choice but different diagnostic responses | Response prediction and calibrated cause distributions; core opportunity test |
| O02 | What does a critic's better solution establish? Compare its advantage under its own resources, the maker's actual constraints, and the maker's known repertoire | Constraint-matched improvement and which differences remain; no inference of laziness |
| O03 | Does explicit consideration uncertainty improve inference over generic bounded search? Match prior artifacts and search cost; compare fixed-menu, latent-menu and budget-only readers | Prospective gain after a reminder or a new demonstration; O01 machinery |
| O04 | What happens when believed and actual feasibility disagree? Include false affordances, ineffective tools and mistaken self-assessments alongside accurate-belief anchors | Failure prediction, model correction and overconfidence; tests the non-nested possibility model |

**Automatic continuation:** if the latent-menu reader only improves when given the true menu, record an upper bound and continue the matched bounded-search rival. If two causes remain observationally equivalent after the allowed probes, retain the equivalence rather than force a classification.

### S. Self-reconstruction and returning to unfinished work

| ID | Question and comparison | Main output / dependency |
|---|---|---|
| S01 | Can actions reveal control information absent from accessible self-knowledge? Cross intact versus partial introspection with action-conditioned versus memory-only inference | Historical/control score and the intact-information no-extra-evidence null; core self-monitor test |
| S02 | Does a self-model improve correction beyond direct goal-error monitoring? Compare both at equal observations, monitoring cost and repair opportunities | Net useful repairs, false alarms, detection delay; S01 machinery |
| S03 | Which mismatches escape notice during a purpose change? Cross obvious, subtle and initially useful old subprocedures with downstream dependency consequences | Detection and collateral effects; distinguish conspicuousness from eventual harm |
| S04 | After interruption, does an artifact help recover the original goal or merely support a new one? Cross memory quality, preserved notes, artifact access and explicit retargeting | Original-goal recovery and adopted-goal performance separately; no mandatory quality decline |
| S05 | Can a wrong self-explanation become self-reinforcing? Supply misleading but plausible memory or an artifact modified by another actor; compare inference, goal adoption and direct completion | Confidence without accuracy, correction after evidence, and useful continuation; requires S01/S04 and source reliability |

**Automatic continuation:** if self-inference gives no benefit beyond direct monitoring, keep that result. Do not add hidden privileged evidence to rescue it. If action evidence helps only after memory loss, report that conditional mechanism. A new adopted goal must never be scored as correct recovery of the old one merely because the resulting work succeeds.

### R. Recognition, learning opportunity and inquiry

| ID | Question and comparison | Main output / dependency |
|---|---|---|
| R01 | Can familiar recognition lead to useful inquiry without high surprise? Cross familiarity and learnability while matching visible cue strength | Subsequent inquiry choices and actual held-out improvement; distinguish identification from learning |
| R02 | Does an intermediate competence region explain inquiry better than novelty alone? Compare novice, improving and saturated readers on the same works | Inquiry and learning curves; no enforced inverted-U shape |
| R03 | Which inquiry policy survives different learning ecologies? Compare surprise, uncertainty/EIG, signed progress, absolute progress, uniform practice and decline/stop | Performance–cost frontier under learnable structure, irreducible noise, skill loss and delayed payoff |
| R04 | Can two equally capable readers rationally differ in interest? Change their current objective, future task distribution and opportunity cost while holding expertise fixed | Inquiry policy and task-relative utility; a counterexample to a direct disinterest–ability inference |
| R05 | Does practicing production improve later reading beyond observing equal examples? Compare enactment, observation and matched unrelated practice; test unseen makers and procedures | Prospective reading plus own execution, separately; this is an analyst-proposed branch, not an answer obtained from prompt five |

**Automatic continuation:** recognition-only success stays in the identification ledger. If an inquiry policy uses evaluator-only correctness, invalidate the comparison. If delayed-payoff tasks defeat myopic progress, pursue the preregistered finite-horizon value-of-learning comparison within the same cost budget; do not claim a general curiosity law from the surviving ecology.

### M. Maker recognition, context and multiple hands

| ID | Question and comparison | Main output / dependency |
|---|---|---|
| M01 | Does recognition track production organization or topic/style? Cross visible subject/surface cues with acquisition and control mechanisms; include misleading familiar cues | Maker identification, process prediction and correction curves separately |
| M02 | Can selective publication create an apparent individual signature? Compare all produced artifacts, randomly retained artifacts and editor-selected artifacts | Bias in inferred maker model and future prediction; retain the rejected work privately for evaluation |
| M03 | Can audience knowledge change the reader's useful reconstruction without changing the original production history? Reveal audience expectations separately from rehearsal/process evidence | Reader adaptation, historical posterior and intended effect as different outcomes |
| M04 | Whose decisions are being read in a collaboration? Cross producer, editor, selector and shared-brief effects without naming the true topology to the reader | Role-relative prediction and uncertainty; do not force a single-author label |

**Automatic continuation:** if topic or selection explains the signature, that is a substantive boundary, not a reason to erase the result. A further personal-model claim must survive the corresponding controlled intervention.

### V. Persistent tradeoffs, conditionally eligible

This is a synthetic shadow branch. It does not reopen human values or alignment claims. Its machinery may be prepared early, but its interpretation requires valid P03/O/M comparisons and actual target realization.

| ID | Question and comparison | Main output / dependency |
|---|---|---|
| V01 | Is there a persistent tradeoff signal after purpose, capability, consideration and selection are independently varied? Compare stable, changing and absent profiles | Held-out choice prediction beyond matched nuisance models; named profile dimensions are constructor-defined |
| V02 | Can inherited craft and current redirection distinguish a trajectory from stable preference? Cross training history with current purpose, constraints and costly corrections | Future choice and revision prediction; no slope inferred from two undated proxy scores |
| V03 | Which probe separates preference change from constraint or audience adaptation? Include genuinely indistinguishable public histories and known rival planners | Probe gain, residual ambiguity and transfer; public distinguishability checked before private evidence is credited |

**Automatic continuation:** if prerequisite separation fails, retain a diagnostic labeled with that failure and close the value interpretation. Do not relabel a goal, style feature or repertoire as a recovered value.

### X. Adversaries applied to the claims that need them

Each row supplies a named adversary, not a single average attack score. The manifest maps each attack to its actual consumer card IDs.

| ID | Adversary | Required observation |
|---|---|---|
| X01 | Hidden-state, seed, filename and cache leakage | Fixed reader bytes imply invariant output under private-state changes |
| X02 | Program spelling and macro-boundary shortcuts | Behavior-preserving recodings preserve relevant predictions; syntax accuracy is separate |
| X03 | Exact artifact collisions | Posterior ambiguity survives until genuinely separating evidence arrives |
| X04 | Observation and compute inequality | Advantages are shown with matched inputs and primitive/search costs, plus any unequal-access upper bounds |
| X05 | False context and unreliable memory | Confidence, prediction and adopted goals are distinct; later evidence can correct the account |
| X06 | Different generator / omitted procedure | Unmodeled worlds expose errors or calibrated uncertainty, not manufactured certainty |
| X07 | Selection, dependence and duplicate histories | Effective sampling groups and retained/rejected records prevent pseudo-replication |
| X08 | Misleading progress and false runtime success | Noise fools an intentionally broken inquiry policy; interrupted jobs and empty queues cannot generate green completion receipts |

### B. Transfer, explanatory archive and closure

| ID | Work | Acceptance product |
|---|---|---|
| B01 | Construct a read-only Sounding Line task export and a standalone reference consumer | Artifact-only and augmented-access schemas, hidden evaluator data, exact known-answer checks and provenance |
| B02 | Compare a fixed stratified case archive with bounded adaptive search for explanatory changes | Distinct validated cases per primitive-evaluation budget, evaluated on a frozen follow-up set |
| B03 | Select and freeze at most three claim packets, including a useful null/boundary if warranted | Reader, estimand, alternatives, gates, attacks, sample size, source hash and untouched lineage fixed before access |
| B04 | Execute selected confirmations and perform scientific/documentary closeout | Every outcome retained; independent reaggregation; pursuit/warrant map and next decision in plain language |

### Dependency and budget logic

Build W1 and the access/scoring spine first. K, P and O share it. S can use a small finite controller while richer acquisition is being validated. R's genuine learning claims require a learner that changes competence; a fixed outcome-distribution forager cannot silently substitute. M can proceed with valid artifact production even if personal craft adds no predictive benefit. V waits for the controls on its interpretation, not for every other scientific hypothesis to be positive.

All mandatory consumers must have a path to a validated dependency or an explicit blocked terminal state. Scientific nulls do not act as instrument failures. Missing optional dependencies do not hold unrelated CPU work. No branch is silently replaced by an easier question.

## 8. Sampling, comparison and confirmation

### 8.1 Units and coverage

The main independent unit is a **maker with an independently generated acquisition history and a paired task packet**. Several artifacts, actions or adaptations from that maker remain a cluster. Do not treat primitive actions, branches of the same execution, repeated contexts, or several renderings as independent makers. Constructor redraws and training seeds are separate levels and are recorded explicitly.

When several makers share constructor parameters, use a hierarchical analysis that resamples constructor redraws and then makers within them, or report a conditional-on-constructor claim. Power and uncertainty must reflect that shared variation. Twenty redraws are a coverage floor, not permission to treat thousands of nested makers as thousands of independent construction regimes; increase the number of redraws when between-constructor variance controls precision.

For each eligible card:

1. Use the smallest exact known-answer world first. The initial graphic anchor should be no larger than a 2×2 canvas with three primitive steps and a tightly bounded action set unless exhaustive enumeration is demonstrably cheap.
2. Run a **64-maker discovery scout per registered condition**, pairing readers/interventions on the same allowed evidence and latent maker. Include at least eight independent constructor-parameter redraws across the scout; cluster uncertainty at the level that was randomized.
3. Expand a useful discriminator to **256 makers**, with at least 20 independent redraws for any miniature promoted as a mechanism result. Expansion is for a named ambiguity, a boundary, a serious rival, or a potential capability, not merely a favorable p-value. If constructor variation is itself the target, allocate enough independent redraws to estimate it rather than inflate episodes within one redraw.
4. Permit one further discovery expansion to **1,024 makers** when the frozen contrast remains informative but imprecise. Retain the earlier scout and state why the expansion occurred. A sampled near-zero effect with poor precision is not a boundary established.
5. The exact tiny subset and the broader approximate subset retain different scope labels. Publish failed, impossible, unsupported and timed-out cases with their original denominators.

These are initial finite allocations, subject to the declared runtime budget. A smaller realized allocation gets its actual uncertainty and coverage; it does not inherit the planned strength. Timing pilots choose tractable world sizes before unknown-outcome comparisons, and every size change is recorded.

Suggested first levels are below. They are design settings, not estimates of psychological quantities. Each card uses only its named contrast; do not launch the Cartesian product of every row.

| Axis | Initial levels |
|---|---|
| Earlier artifacts | 0, 1, 2, 4, 8 |
| Acquisition experience | 4, 16, 64 training episodes, with feedback access held explicit |
| Search allowance | 8, 32, 128 expanded primitive successor evaluations |
| Relation of old and new purposes | Aligned, partially aligned, opposed, checked on an independent candidate-action reference set |
| Retained goal memory | Intact, partial, absent; misleading memory is a separate intervention |
| Consideration coverage | Low, intermediate, full among believed alternatives; record realized menus |
| Observation access | Final artifact; final artifact plus prior works; separately purchased process/context evidence |
| Source reliability | Accurate, uninformative, systematically misleading, with likelihood assumptions exposed |
| Learning ecology | Immediate progress, delayed payoff, saturation, irreversible noise, deterioration |

Begin with balanced matched contrasts. Use the boundary/archive budget to vary additional settings and interactions. Report the empirical coverage of the named axes and mechanisms, with holes; a job count is not a coverage argument.

### 8.2 Fairness of information and computation

Use paired observations, training opportunities and outcome draws where the intervention permits. Independent random streams must separate world generation, training, reader search, observations and scoring; no seed may encode an answer visible to the reader. Use a stable hash/seed construction, never Python's randomized `hash()`.

Record primitive transitions, successor evaluations, likelihood evaluations, library-building cost, training cost, memory footprint and wall/CPU time. A macro invocation expands to its real primitive cost. Learning can earn amortized savings across a declared future workload, but its initial cost remains visible.

Different inference methods need a performance–cost curve. Identical wall time alone is not a fair comparison if one method received true labels or a better observation interface. A reusable procedure may improve speed without changing the final answer; preserve that outcome.

### 8.3 Learning progress and information gain

An inquiry policy may use only feedback available to the reader. Distinguish acquiring evidence about a maker from practicing a production skill. A hidden true program, latent maker identity, or evaluator-only reconstruction score cannot become an undeclared curiosity reward.

For the simple progress rival, keep a fixed recent window per task family and compare the older and newer mean experienced competence. Compare the signed change with its absolute value. Log which feedback produced the statistic; decline can attract the absolute-change policy. This is a small adaptation of competence-progress methods, not a complete SAGG-RIAC recreation.

For the finite exact information policy, value a query by its expected reduction in uncertainty about the **declared target**, under the reader's current model and possible query outcomes. It receives no future realized answer in advance. Compare target choices: reproducing a useful artifact, predicting this maker, and recovering an identifiable history. Charge inquiry separately; show the information–cost frontier. For the finite-horizon learning policy, simulate the reader's own update on possible permitted feedback, then evaluate expected future task utility, not true future performance supplied by the evaluator.

### 8.4 Confirmation

At the confirmation freeze, select **at most three claim packets**. Eligible packets can concern a capability, a discriminating null, or a boundary. Selection is lexicographic: validity and target realization; separation of a serious rival; relevance to Sounding Line; then practical size and cost. Selection must not depend only on the largest favorable estimate.

Freeze the reader and generator code, training recipe, access contract, primary statistic, direction or equivalence margin, adversaries, sample size, and grouping structure before opening confirmation data. Use a separate never-inspected seed namespace and fresh makers/histories. Parameter regimes already chosen through exploration are discovery-selected; acknowledge that even when fresh draws are used. For a transfer claim, include a mechanism or family held out from reader fitting, not only new random seeds.

Choose the confirmation sample size once from discovery variance and the predeclared practical effect, targeting 90% power under a familywise 0.05 budget for at most three primary claims. A conservative Bonferroni planning allocation and Holm final adjustment are acceptable. For equivalence or interaction claims, power the actual estimand; do not borrow the sample size of a simple mean difference. Cap any packet at 4,096 independent maker units and the remaining compute budget. If the achievable sample is inadequate, leave it exploratory and report the detectable effect.

Do not replace a failed claim after seeing its confirmation result. No confirmation data tune a library, prior, score, or threshold. Any post-freeze defect yields a preserved failed/invalid packet and a separately labeled repair; it cannot be silently promoted using the same reserve. Register additional inferential tests in the multiplicity ledger. Descriptive curves need uncertainty but are not dozens of independent confirmations.

## 9. The explanatory archive

Start with a fixed archive of validated cases. Use a short record per case: visible artifact, permitted context, reader output, actual continuation, leading rival, intervention that does or does not separate it, and all costs. Keep evaluator truth separate from the reader export.

Use these initial categories as lenses, not a claim of an exhaustive ontology:

- Enactable and historically accurate.
- Enactable but historically ambiguous or wrong.
- Accurate historical account that the reader cannot enact under its own constraints.
- Recognizable maker without improved process prediction.
- New evidence improves prediction while leaving history ambiguous.
- Self-correction that helps, and plausible self-explanation that harms.
- Curiosity that acquires competence, and curiosity that chases noise or confirmation.
- Personal-looking patterns produced by selection or shared constraints.

Adaptive search receives the same primitive-evaluation budget as a stratified sampler. It may edit predeclared world parameters, training schedules, constraints or allowed evidence; it may not invent a new scientific target or scoring rule. A case counts as newly explanatory only after an independent interpreter/checker validates its construction and a frozen follow-up confirms the described distinction. Compare both methods on distinct valid cases found and on useful boundary coverage; archive occupancy alone is not success.

Do not choose only vivid wins. The final set should include at least one clean failure of each favored mechanism when the constructed space supplies one. If an entire category is absent, state whether it was searched, inaccessible, invalid, or simply not found.

## 10. Runtime and setup contract

### 10.1 V16-specific changes to the old execution assumptions

This handoff establishes a **finite program with a ceiling and honest early closure**. It does not inherit V15's obligation to occupy seven continuous days or its prohibition on final prose before hour 168. Preserve those historical V15 rules and its failed runtime flag; do not rewrite them to make V15 pass.

There are two immutable clocks: the campaign's initial accepted start time and each scientific packet's freeze/start time. Restarts preserve both. Time spent preparing, downloading, repairing, waiting for hardware, computing and reporting is classified separately, never omitted by restarting a clock. No claim of continuous occupancy is required; idle time remains visible.

**The early obligations are concrete:**

- By elapsed hour **4**, save one real end-to-end native case: generated history → final artifact → reader prediction/program → independent execution/scoring → retained unit record → reaggregated result. Include at least one positive and one ambiguity/null fixture. A directory scaffold or prose plan does not satisfy this checkpoint.
- By hour **12**, either the minimal W1 inference/production spine passes its gates and enters discovery, or publish a precise setup-blocked receipt. Stop platform expansion while repairing that spine. Permit one bounded four-hour repair; if it still fails, close its consumers and report an incomplete V16 setup rather than spending days preparing to prepare.
- Close optional runtime intake by hour **24**. A missing optional library selects the native rival. W2 or other mechanism construction gets a named budget and can finish later as an independent packet; it cannot delay already valid W1 work.
- Every six elapsed hours during active setup, record what executable capability was gained, the current blocker, the next concrete test and updated time estimate. During science, maintain ordinary status/heartbeat records without narrating per-rollout conclusions.

These are implementation obligations, not claims that the analyst benchmarked the build time. If the coding environment cannot meet them, the result is an exposed setup boundary and a smaller valid executed program, not an automatic extension.

### 10.2 Initial allocation

The table describes planned use of the 120-hour horizon. Work may overlap in isolated packets, and valid early closure is allowed.

| Elapsed region | Work | Budget logic |
|---|---|---|
| 0–4 h | Read current local state; dependency debt; native vertical case and tripwire gates | Maximum two hours of old-version dependency inventory; no external build on the critical path |
| 4–12 h | Minimal learned-procedure and inference spine; restart/reaggregation; first eligible scouts | Setup gate or explicit blocker at hour 12 |
| 12–36 h | Main K/P/O/S scouts; optional method reference checks; W2 and R/M construction | Independent packet freezes; optional runtime intake ends by hour 24 |
| 36–72 h | Broad eligible forest, mechanism rivals, source/selection and inquiry cases | Preserve balanced mechanism coverage before deepening one favorable card |
| 72–96 h | Boundary expansions, different-generator transfer, archive comparison, eligible value shadows | No unbounded new architecture; prepare confirmation candidates |
| 96 h | Confirmation selection and freeze | Empty selection is valid; publish eligibility reasons internally |
| 96–114 h | Frozen confirmations, or the predeclared remaining boundary ladder if selection is empty/finished | Do not use failed confirmation as a reason to select a replacement |
| 114–120 h | Finish/checkpoint bounded jobs, independent reaggregation and final write-through | No new science dispatch that cannot close within the remaining horizon |

Use only the actual available local CPU allocation, with an explicit worker cap chosen from measured memory and timing. Avoid nested BLAS/process parallelism and reserve resources the current gear requires. Do not infer a GPU allocation from this document. Profile representative packets; forecast worker-hours from real completions, not constants labeled as measurements.

At hour 120, stop scientific execution at a declared checkpoint boundary, preserve partial outputs and write the incomplete status where applicable. Size jobs/checkpoints so this does not require destroying a long valid result. Any unavoidable overrun is reported as an overrun; it does not move the original ceiling. If the finite eligible queue empties earlier and the bounded expansion ladder is exhausted, close early.

### 10.3 Predeclared expansion ladder

When work finishes faster than expected, choose only eligible, unresolved work in this order:

1. Complete missing positive/null/adversarial checks that alter interpretation of an executed question.
2. Test a serious rival on the same information and compute budget.
3. Add fresh constructor redraws or makers to a named uncertain boundary.
4. Test a new mechanism or W2 transfer already specified in this package.
5. Add a case that distinguishes executable usefulness from historical truth or future prediction.
6. Extend the fixed-versus-adaptive archive comparison.

Each expansion has a cap and an explicit expected contribution. If none applies, close. No maintenance reruns, sleeps until deadline, duplicate seed jobs, or synthetic workload are permitted as occupancy padding.

### 10.4 Process ownership and resumption

Use a new isolated V16 checkout/lineage and the existing interpreter conventions. On the user's Windows host, inspect actual runner PIDs, process creation times, ownership and fresh heartbeats before editing any imported module. Published GitHub status cannot certify local process state.

Use module-form launches, preserving the documented defense against the sibling orphan sweep. One supervisor owns status writes. Workers append or atomically submit immutable unit/packet results; they do not compete to replace the supervisor's PID. A heartbeat proves liveness, not scientific progress: record CPU work, completed units and blocked/waiting reasons separately.

The required CLI contract may be implemented in `runners/run_v16.py`; these are **commands to implement**, not existing tested commands:

```text
python -m runners.run_v16 --stage preflight
python -m runners.run_v16 --stage pilot
python -m runners.run_v16 --stage discovery
python -m runners.run_v16 --stage transfer
python -m runners.run_v16 --stage confirmation
python -m runners.run_v16 --stage close
python -m runners.run_v16 --stage resume
```

`resume` checks locks and completed unit IDs, then resumes eligible work. It must not regenerate timestamps inside lock inputs, rerun prepare/pilot, delete failures, or duplicate completed units. A scratch gate check must not write the live supervisor status. A signal handler/checkpoint path and a real interruption test are required before a long job is admitted.

Freeze the overall commission/manifest separately from each implementation packet. This permits W1 science to proceed while an independent W2 packet is built, without changing code imported by running workers. New packet code is validated and frozen before its first scientific outcome. Changed estimators or mechanisms after results are viewed create explicit amendments and fresh discovery lineages; no silent lock regeneration.

### 10.5 Failure and repair bounds

| Failure | Automatic action |
|---|---|
| Broken generator, interpreter or information boundary | Quarantine affected descendants; preserve attempts; one bounded repair, fresh packet identity and appropriate new data |
| Second substantive defect in the same instrument family | Close its dependent scientific claims for V16; retain independent alternatives |
| Valid null or effect below the practical bar | Report it; pursue one named discriminator from the ladder if useful; no tuning until it turns positive |
| Exact support too large | Reduce only the preregistered tiny validation world; retain the larger question as bounded/approximate or unexecuted, with support loss explicit |
| Approximate inference fails its exact anchor | Disable that reader; exact and independent validated readers continue |
| External build or source unavailable | Stop at the intake cap and use the native path; no general setup extension |
| Optional/new world cannot pass construction gates | Close that world, retain valid others and remove cross-world claim eligibility |
| Hardware withdrawn or shared workload interferes | Log availability, reduce workers and reforecast; keep the initial clock |
| Worker dies repeatedly | Resume with evidence of ownership; after three failures of the same root cause, quarantine the job instead of an infinite watchdog loop |
| No eligible work remains | Close with the actual scientific/operational reason |

## 11. Data, provenance and independent regeneration

Every unit must be reconstructible from an immutable scientific identity. At minimum retain: commission hash, implementation hash, dependency/environment identity, card and condition IDs, generator family/mechanism, constructor redraw, maker/history ID, seed lineage, observation hash, reader package, exact access tier, predictions before reveal, executed program, outcomes, costs, failures and timestamps.

Persist raw scientific outputs throughout the run, not only aggregates. Honor the repository's ignored per-rollout naming conventions and size limits. Large raw files remain outside ordinary Git history with a manifest, checksums, location and retention receipt; **do not call them reproducible artifacts merely because they are gitignored**. Keep compact aggregate records and a small replay bundle under the normal repo policy. Verify the actual raw archive exists, is complete and remains accessible at closeout.

Reaggregation has two distinct receipts:

1. **Full aggregate regeneration:** a clean analysis process reads retained unit outputs and independently reproduces every reported V16 aggregate, interval and denominator. It does not import the reporting reducer being checked as its answer oracle.
2. **Scientific replay:** a prespecified small, diverse subset regenerates worlds and reader outcomes from frozen code/seeds; deterministic quantities match exactly/tolerances, and platform-dependent ones follow a declared reproducibility rule. This is not a claim that every rollout was rerun.

If raw retention fails, stop affected claim promotion. If aggregate regeneration fails, correct the record and affected claims through a labeled amendment. If replay has a platform limit, report it specifically. Do not merge these three questions into one green fresh-clone flag.

## 12. The Sounding Line transfer packet

Ghost should supply tasks that expose what Sounding's reader must actually do. It should not import all of Sounding's corpora or make its own synthetic success a new admission rule for real artifacts.

B01 produces a bounded schema, a minimal standalone consumer, and paired public/private exports:

```text
public observation:
  task_id, schema_version, lineage_id, access_tier
  final_artifact, declared_context, permitted_prior_artifacts
  permitted_query_descriptions, query_costs
  target_request, reader_action_budget

private evaluation:
  task_id, true_production_record, acquisition_record
  original_goal, current_goal, controller_state
  actual_feasibility, maker_beliefs, considered_alternatives
  hidden_continuations, intervention_outcomes
  equivalence_classes_for_declared_targets, scoring_contract
```

The consumer must demonstrate that it cannot read private evaluation fields. Only the evaluator joins them by task ID after predictions are committed. Public task/lineage IDs are opaque aliases independent of condition labels, latent variables and seeds; the evaluator retains the mapping. Remove truth-revealing ordering. Schemas are a proposed interface; inspect the current Sounding Line task loader before claiming compatibility. JSONL existence alone is not a successful handoff.

Export at least one validated packet for each **eligible** capability: executable reconstruction, maker-specific continuation, changed opportunity, self-monitoring/resumption, and inquiry that changes competence. Include positive, ambiguous and misleading cases, not only success demonstrations. If a capability never passed its gates, its example is labeled diagnostic and is not an admission fixture.

Preserve the real-record branch already commissioned in Sounding Line: CoAuthor, ScholaWrite, ArgRewrite and the Stage 9 intake directions including genetic editions, creative selections, drawing records and code revisions. Ghost contributes known processes, controlled alternatives and counterfactuals; the real datasets contribute ecological constraints and independently recorded events. Match the *task and evidence interface* before transferring a conclusion. A synthetic trace field that the real corpus lacks is a boundary, not an invitation to infer and relabel it as ground truth.

The handoff identifies which validated operation each candidate real substrate could support and which field is missing. It does not amend a live Stage 9 lock, download new human data, or launch a neural-reader test from Ghost. Work that specifically concerns a language model's behavior belongs in Sounding Line's own authorized queue.

## 13. Required final products and decisions

Write one completed-study account. Begin with the question and what changed in the project model; put queue mechanics and complete numbers behind it. Produce:

1. A capability map separating execution, historical correspondence, future prediction, self-correction and inquiry-driven learning.
2. A comparison of mechanism families and their performance–cost curves, with information access visible.
3. A coverage map across evidence dose, acquisition, constraints, consideration, memory, source selection and generator families; include unsearched or blocked regions.
4. An archive of inspectable positive, negative, ambiguous and misleading cases, with separating interventions where available.
5. The Sounding Line transfer packet, validated consumer and limitations.
6. A confirmation record, including empty selection or failed claims, plus multiplicity and lineage accounting.
7. Full aggregate regeneration, bounded replay, raw retention and actual runtime receipts.
8. Updated pursuit and warrant ledgers. A next-version recommendation must cite a mechanism, capability or unresolved discriminator; a card-count rule cannot choose it.

Each headline must state whether it is a **method result**, a **constructed mechanism result**, or a **missing measurement**. Public claims remain limited to the actual finite worlds and readers. V16 supplies no human evidence of empathy, autism recognition, artistic quality, intent detection, provenance, or values.

### Decision map at closeout

| Result pattern | Next action |
|---|---|
| Personal acquired repertoire improves adaptation and individual prediction across controlled transfer | Promote the validated task/interface to Sounding's next eligible reader comparison |
| Generic skills match personal craft | Pursue generic competence as the useful substrate; keep individual reconstruction open where it adds no value |
| Enactment succeeds while history remains unresolved | Preserve constructive usefulness and conditional equivalence; do not demand a fabricated historical label |
| Self-monitoring helps only with missing internal evidence | Carry the conditional access mechanism; avoid a general claim that artists must infer all their goals |
| Direct error monitoring matches self-inference | Prefer the simpler operational mechanism for that scope, preserving the theory-level alternative |
| Recognition prompts inquiry without later learning | Record identification/selection behavior; do not promote a learning mechanism |
| Benefits disappear under a different generator or information-matched rival | Treat grammar/access dependence as the result and name the next discriminator only if one remains |
| Core construction fails within its repair budget | Report an incomplete V16 implementation and the concrete missing capability; do not produce a synthetic scientific null |

No fresh walkthrough is required for routine implementation, dependency choice or bounded failure handling. Return to the curator for a genuinely new theoretical choice only after completing all authorized independent work and presenting a concrete branch decision.

## Appendix A. Implementation ownership and precise operating changes

Use the existing repository structure. The paths below identify responsibility; avoid duplicating the handoff across several competing documents.

| Owner | Required implementation/change |
|---|---|
| A single accepted V16 specification under the existing version-document convention | Store this commission and its acceptance identity once; amendments point to it. Do not edit V15's locked specification |
| New `ghostscale/validation/soundingline/v16/` modules | Reader/evaluator contracts, the two worlds, acquisition, bounded readers, self-monitor, inquiry, measures and cards; keep the architecture small and dependencies explicit |
| New V16 runner/supervisor under `runners/` | Finite manifest dispatch, one status owner, per-packet locks, interruption/resume and final closure; reuse only inspected stable infrastructure |
| Existing gate/test conventions plus scoped V16 checks | Real known-answer, break-injection, access, estimator, replay and resume checks. A recorded gate failure must remain test-visible at publication |
| `results/v16/` and the existing validation-result convention | Manifest, capability/claim ledger, packet records, source/environment identities, costs, raw archive receipt, confirmation and closeout |
| `docs/METHODS.md` | State why each imported or independently implemented method exists, its observation requirements and its known-answer anchor |
| `FINDINGS.md` and `docs/theory/READING_INTENT.md` | Land each scientific result in both channels, updating its existing owner/afterword and carrying access limits; no ungrounded global theory upgrade |
| `docs/exchange/` | One final Sounding Line bridge and curator interpretation with the same headline wording as the final account |
| `AGENTS.md` and compatibility entry point | Add the V16-specific operating qualification below; preserve historical V13–V15 instructions and failure history |

### A1. Current-version qualification

When the coding agent accepts and begins this package, replace the current-version summary that calls V15 active with an accurate V15-closed/V16-state pointer. Add a short V16 qualification containing these exact obligations in the project's own format:

- Current execution state comes from V16 records and live owned processes, not the word “active” in prose.
- V16's accepted 120-hour planning ceiling includes setup and closeout; early closure is allowed after its finite eligible queue and expansion ladder are exhausted.
- The V15 seven-day occupancy and no-packet-before-hour-168 requirements remain historical V15 requirements and do not govern V16.
- The hour-4 executable case, hour-12 core gate/blocked receipt, bounded repair and optional-intake cap are mandatory V16 checkpoints.
- One supervisor writes live status; resume never prepares again or changes frozen inputs.
- A passed criterion, valid instrument, completed job, confirmation and finished closeout are distinct states.
- New method dependencies are optional and isolated; no live environment synchronization or unrequested gear change.

Announce the AGENTS/compatibility changes in the coding agent's handoff. Do not turn these scoped rules into undocumented changes to all historical versions or the sibling project.

### A2. Ledger schema

Represent these dimensions separately, preserving the repository's existing vocabulary where possible:

```text
execution_state: planned | running | completed | blocked | failed | checkpointed
instrument_state: untested | valid | failed | not_applicable
criterion_state: untested | held | failed | inconclusive | not_applicable
evidence_scope: fixture | discovery | robustness | transfer | confirmation
access_tier: explicit identifier from the observation contract
dependency_ids: actual IDs, including adversaries and implementation versions
pursuit_status: OPENED | PROMISING | STALLED | EXHAUSTED | PROMOTE
warrant_status: DESCRIPTIVE ONLY | ANALOGUE EVIDENCE | MECHANISM CANDIDATE |
                CONFIRMATORY SUPPORT | BOUNDARY ESTABLISHED | INSTRUMENT FAILED | VOID
```

An absent required dependency is missing, not passed. Confirmation results join through explicit claim/card IDs, not matching filenames by coincidence. A criterion's units, numerator/denominator, aggregation target and uncertainty method are mandatory. A conditional maximum and pooled mean have different estimator IDs. Compute all prose counts from the ledger and verify every row against a retained receipt.

### A3. Reading continuity and delay visibility

Preserve a read-progress record with source hashes and last fully read portions. Complete the initial required reading; after compaction, resume unread or changed material and revisit relevant owners as the current contract requires. Do not replace reading with a hash, and do not restart an already completed unchanged full-folder read merely because the session compacted.

A setup status must identify an executable milestone, not just “theory reading,” “hardening” or “nuance.” If repeated reading, dependency installation, export plumbing or environment recovery consumes the build budget, identify that category and its actual duration. This is a prospective control; this analyst has not established that any one category caused Sounding Line's present delay.

## Appendix B. Theory provenance and future integration

This appendix is a precise handoff for the next appropriate theory-maintenance boundary. **It does not instruct the Ghost operator to edit a live Sounding Line checkout.** The September 7 theory errata are still proposals in the reviewed published snapshot. Reconcile overlapping edits once; do not stack duplicate paragraphs or quotes from both packages.

### B1. Goal self-knowledge

**Owner:** Sounding Line `docs/theory/THE_TRIPLE_INFERENCE.md`, §2, the passage beginning “On who knows the goal” and its explanation of privileged episodic memory.

**Proposed prose integration:** the maker's episodic memory can provide privileged evidence about an attended purpose, but that evidence can be incomplete, forgotten or reconstructed after interruption. Subsidiary control can remain partly opaque during production. Observing actions may help infer it, while adopting a new goal for continuation is a distinct event from recovering the original goal. Preserve the original quote as a historical statement; do not silently rewrite “flawless” into a newly spoken qualification.

Attach the first September 9 quote from §2 once, with date and transcript-cleaning provenance. Keep the mechanism OPEN until measured; the Fleming–Daw connection is adjacent literature, not direct support for artistic-goal recovery.

### B2. Criticism and perceived opportunity

**Owner:** `THE_TRIPLE_INFERENCE.md`, §1's context/belief/subjective-action-set definitions, and `READER_HEURISTICS.md`, §6 “Distinguishing choice from constraint.”

**Proposed prose integration:** distinguish an unavailable action, an unknown technique, a known but unconsidered alternative, and a considered alternative rejected under the maker's purpose or costs. Incorrect beliefs can introduce imagined possibilities that are physically infeasible. Comparative criticism must specify whose tools, knowledge, constraints and search resources define the comparison. A better solution alone does not show that the creator was careless.

Attach the second September 9 quote once in the reader-side owner; use a cross-reference in the inference definitions rather than duplication. Do not add another inference vertex.

### B3. Recognition and inquiry

**Owner:** `READER_HEURISTICS.md`, §2 “Finding an entry point” and §5 “Continuation and stopping.”

**Proposed prose integration:** familiar recognition can make a promising distinction available for inquiry before any anomaly is noticed. Its pursuit value depends on expected learning, the reader's current purposes, costs, and prior knowledge. Recognition, confidence, inquiry, and acquired competence are separate outcomes. Lack of inquiry does not identify a person's intelligence or expertise.

Attach the third September 9 quote once, preserving its tentative “might.” Keep surprise-based, progress-based, familiarity, and task-value accounts as competing mechanisms. No neural sequence or diagnostic author-trait claim follows.

### B4. Habit and interrupted production

**Owner:** existing automatic-control/learning passages in `DECISION_TRACES.md` and the expertise/process distinction in `THE_TRIPLE_INFERENCE.md`.

**Proposed prose integration:** a learned routine can reduce search and support a new purpose while also causing interference. Local improvability does not establish global inefficiency. After interruption, the maker can recover an old plan, infer it from the artifact, or choose a new one. Outcomes relative to the original objective and the newly adopted objective can diverge; no general deterioration law is ratified by the walkthrough.

Use analyst prose, retaining the curator's lower confidence. The fifth response concerns a returning maker; do not record it as evidence about a viewer's learning through practice. Add no new theory file or section solely for this material. Revisit affected afterwords, preserve stable identifiers and run the existing theory-format checks at the actual edit boundary.

## Appendix C. Primary-source register and limits of this review

This is a selective methods pass for an executable commission, not a systematic survey or a reproduction of the external literature. One literature-search agent was used under Ghost's existing limit; repository inspection and package design were done by the principal analyst. Load-bearing method ideas were checked against primary sources. No external algorithm or new scientific test was run in this review.

| Source | Material inspected and role | Limitation retained |
|---|---|---|
| [DreamCoder](https://arxiv.org/pdf/2006.08381) | Prior review: wake/sleep formulation and drawing/building examples; background for acquired executable abstractions | Primitive language and training distribution shape what can be learned; no historical human-process claim |
| [ShapeCoder](https://arxiv.org/pdf/2305.05661) | Prior review: program/library discovery, inputs and limitations; artifact reconstruction comparison | Uses decomposed geometric primitives and costly search/training; not a drop-in image reader |
| [Stitch](https://arxiv.org/html/2211.16605v2) | Prior method read; current source README, API route and repository pin checked | Program compression has a different target from maker reconstruction; current binding compatibility untested |
| [Online Bayesian Goal Inference for Boundedly-Rational Planning Agents](https://arxiv.org/pdf/2006.07532) | Agent read §§3–4 and runtime appendix; principal checked planning/execution generative specification | Consideration-set inference is a new extension; runtime advantage is domain-dependent |
| [A Laplacian Framework for Option Discovery](https://arxiv.org/html/1703.00956v2) | Agent read methods/evaluation; principal checked intrinsic rewards, policies and termination | Generic state-space structure can explain skills. Full hidden graphs and emulator look-ahead are privileged access |
| [Fleming–Daw self-evaluation](https://www.princeton.edu/~ndaw/fd17.pdf) | Model families and action-informed confidence read; agent also inspected the published small MATLAB kernel | Published target is decision correctness. Artistic subsidiary-goal recovery and resumption remain V16 hypotheses |
| [SAGG-RIAC](https://www.pyoudeyer.com/RAS-SAGG-RIAC-2012.pdf) | Agent read architecture, progress and limits; principal checked §2.4.3 | Absolute progress includes deterioration; task-space choice and delayed/noisy learning matter |
| [MAP-Elites](https://arxiv.org/pdf/1504.04909) and [ACCEL](https://arxiv.org/html/2203.01302v3) | Prior review: archive/search algorithms and limitations | Explanatory diversity is our adaptation; these works do not establish scientific mechanism discovery |
| [Simulation-based calibration](https://sites.stat.columbia.edu/gelman/research/unpublished/sbc.pdf) and [richer test quantities](https://pmc.ncbi.nlm.nih.gov/articles/PMC12490788/) | Agent checked basic rank algorithm and limits; prior review checked ignored-data/joint-dependence failure cases | Internal computational consistency is not external validity; no green histogram can substitute for the information boundary |
| [Partial identifiability in reward learning](https://proceedings.mlr.press/v202/skalse23a/skalse23a.pdf) | Prior review of ambiguity and downstream-task dependence | Ambiguity may be harmless for one task and consequential for another; priors are assumptions, not observations |

Recent work is not automatically the highest-value dependency. The earlier reviewed 2026 robust-design methods remain possible challengers, but they do not resolve the immediate production, self-monitoring and information-access gaps. Large embodied environments, full language-model training stacks and unrestricted language-generated worlds are outside this finite V16 commission. A later result can justify them; their novelty alone does not.

## Acceptance checklist

- [ ] Current local state and actual resource allocation checked; historical evidence preserved.
- [ ] Manifest contains the 42 unique card IDs with explicit dependencies, units and terminal states.
- [ ] First real native case and its positive/null fixtures are saved by the early checkpoint, or the missed checkpoint is recorded.
- [ ] Core passes its independent gates before discovery; optional imports do not block it.
- [ ] Reader access, learned versus supplied state, actual versus believed possibilities, and original versus adopted goals remain separate.
- [ ] At least one serious generic/mechanism rival is run for every promoted personal-model claim.
- [ ] Raw outputs, costs, failed attempts, source locks and actual restart behavior are retained.
- [ ] Confirmation selection is frozen, independently seeded and never silently replaced.
- [ ] Final counts and claims regenerate from records; absent work is visible.
- [ ] A validated Sounding Line task interface and a textured explanatory account accompany the numbers.
- [ ] Closure states what was learned, what remains unmeasured, and why the next step follows.

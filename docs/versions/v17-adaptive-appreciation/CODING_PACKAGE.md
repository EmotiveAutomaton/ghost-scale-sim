# Ghost Scale Sim V17 — acquired craft, recipients and adaptive inference

**Coding-agent handoff, 12 September 2026.** Proposed new campaign based on repository head `5b73980a2ee8a28daf0922b26072ba4f461ff7f8`. V16 remains closed. Implement in a new V17 namespace; do not reopen its confirmations or change their meaning. This document specifies the next work and does not itself assert that it has run.

## 1. Question and intended advance

**Hypothesis:** a reader can make more useful predictions from artifacts by combining inferred reusable procedures, concrete exceptions, a model of the recipient, and selective further computation.

**Method:** construct makers whose training, actions, opportunities and recipients are recorded; expose only the declared artifact/context evidence to each reader; compare withheld behavior and recipient-effect predictions under controlled changes. Include direct prediction, memory and fixed-effort rivals. The output is a map of which methods help under which conditions, plus a small challenge usable by Sounding Line.

The scientific target is usefulness of a revisable maker model. Do not make uniqueness of historical reconstruction a universal admission rule. Score historical correspondence, executable reconstruction, prediction and recipient effects separately. A method may be useful on one and uninformative on another.

The curator favors optimistic exploration. Keep informative nulls and reversals as outcomes that redirect implementations. Neither a failed method nor an attractive synthetic positive decides the human theory.

## 2. Scope, resources and schedule

Aim for approximately **five elapsed days from beginning implementation through delivering analysis**. Aim for first useful experiments within the first day. These are soft planning targets, with no occupancy minimum, no arbitrary report embargo and no global all-families-ready gate. The current instruction supersedes inherited timing rules about starting the clock only after setup. A validity check still runs before the claim that depends on it.

Default is CPU work alongside Sounding Line's GPU campaign. Reuse current process isolation, resume and recording utilities. Agree CPU thread allocation with the actual host's available capacity in the local launch plan; do not let Ghost's parallelism starve the reader service or override current thermal policy. No cloud expense is necessary for the core.

| Approximate interval | Useful work and decision |
|---|---|
| First 0–12 hours | Reuse existing worlds, define the common task/score interface, run A's first real comparison and verify one export |
| By roughly 24 hours | A running; B/C/D each either running after its local checks or carrying one specific remaining problem. Publish a short progress note |
| Days 2–3 | Complete the broad screen; expand contrasting regions rather than every cell; E can consume completed packets immediately |
| Day 4 | Freeze up to three selected contrasts and run fresh constructor/history draws; continue useful unselected discovery work separately |
| Day 5 | Analyze completed work, ship public cases and counterexamples, state what would change the next decision |

If a family takes longer, reduce its local scope and continue other families. If a nearly complete comparison is worth a short overrun, state why and give a revised ETA. Do not silently renew the whole campaign. If the informative work finishes early, report it; filling time is not a scientific outcome.

## 3. Reuse before new infrastructure

Read current `AGENTS.md`, `docs/METHODS.md`, V16 `RESULTS.md` and `TRANSFER.md`, then the relevant existing modules. Preserve reading continuity across context compaction. The principal reuse points are:

| Existing path | Reuse / limitation |
|---|---|
| `ghostscale/validation/soundingline/v16/graphic_world.py` | Sixteen-cell board, 32 place/remove primitives, six-step default, explicit trace and primitive cost |
| `.../v16/assembly.py` | Three-part dependency-constrained assembly, rotation/removal and explicit STOP; some tasks are necessarily small |
| `.../v16/learning.py`, `craft.py` | Retain repeated-fragment learning as a serious baseline; the original four-cell world is an exact toy control, not the whole new testbed |
| `.../v16/multi_actor_world.py` | Existing production, self/other revision and selection scaffold; current style/topic readout is not an independent recipient model |
| `.../v16/mechanism_reader.py` | Exact conditional reference; its direct and latent forms can be equivalent, so label it a ceiling rather than a cheap rival |
| `.../v16/transfer_consumer.py`, `runners/package_v16_transfer.py` | JSON framing, public/private separation, source receipts and reference consumer |

Paths abbreviated with `.../v16/` share the first row's directory. The inspected [V16 code](https://github.com/EmotiveAutomaton/ghost-scale-sim/tree/5b73980a2ee8a28daf0922b26072ba4f461ff7f8/ghostscale/validation/soundingline/v16) is the starting point, not a claim that every desired faculty already exists.

New modules should live under `ghostscale/validation/soundingline/v17/`, with a thin `runners/run_v17.py` entry point and one `docs/versions/v17-adaptive-appreciation/` campaign directory. These are proposed paths. Prefer a small shared evaluator over separate runner/scorer implementations for every comparison. Do not copy the entire V16 campaign orchestration into a second independently evolving system.

## 4. Common experimental contract

Each private case records maker training, current and earlier goals, actual tool constraints, believed constraints, considered options, realized action history, contributor role and recipient state. Public evidence is an explicit projection. Absence of a variable from the reader's representation is different from uncertainty over that variable.

Use three evidence tiers:

1. **Artifact:** current finished object plus the task's declared context; no hidden trace, true goal or learned library.
2. **Artifact collection:** tier 1 plus permitted earlier finished objects. Cross-object dates appear only if actually supplied.
3. **Recorded process:** tier 2 plus selected genuine actions/feedback. This estimates the value of additional evidence; it cannot establish artifact-only inference.

A supplied true goal, complete library or true recipient model is an explicitly labeled oracle/assistance condition. The main reader must infer the relevant unknowns. The maker's legitimate access to its own training is not legitimate access for the observer.

Primary task outputs are probability distributions over finite future actions or recipient outcomes. Evaluate these with proper log loss and Brier score; retain raw probabilities. Also score terminal task success, inappropriate stopping, dependency violations and total cost where applicable. A missing output or invalid program is a separate failure, not silently a uniform forecast or a correct STOP.

Charge training/acquisition, definition storage, retrieval, proposal generation, hypothetical execution, actual execution and any selection overhead separately. Also report their sum in the deployment setting. One macro call does not make its primitive actions free. Cache construction has a cost; show cold and repeated-use costs. Equal training examples do not imply equal total computation.

## 5. Study families

### A. Reusable procedures plus concrete exceptions

**Question:** when does a compact learned procedure generalize, when does memory win, and when does their combination help?

Compare four representations on the same permitted acquisition examples: existing repeated fragments; Stitch-derived parameterized abstractions; episodic retrieval with adaptation; and abstractions plus a bounded exception store. Retain primitive-only search as a reference. Every operative method gets the same execution law and budget accounting. A pooled-library comparison is a separately labeled test of personal versus generic training distributions.

Start on the sixteen-cell graphic world. Encode its operations in a small typed expression format such as `place(cell)` and `remove(cell)` plus sequencing. Bind abstractions to observable cell/part arguments; expand them into the original primitive executor. Arity 0, 1 and 2 are the planned range. Argument binding and failed candidates consume search resources. Then test the useful comparison on assembly, where order and dependencies matter.

Cross three kinds of held-out cases: familiar combinations; new combinations of familiar operations; and an exception where a previously useful routine fails under a changed constraint. Remove exact test programs and target combinations from acquisition. Balance action/token order or permute it across paired constructors so enumeration order cannot impersonate a craft advantage.

Use budget levels 8, 32 and 128 as continuity anchors where meaningful; on the larger world, select three common budgets from a discarded pilot so all arms are not at the same floor or ceiling. Freeze that choice before the scored cohort. Report solve rate and cost together. A compression gain without task benefit is a compression result; a pooled method matching the personal one preserves generic skill value.

The hybrid store retains concrete training episodes selected by a rule frozen on development examples, not failures from the held-out test. Compare at a common memory cap and display a second curve with each method's actual storage; no uncharged full history hidden behind a compact representation.

**Reference:** [Stitch source, MIT](https://github.com/mlb2251/stitch/tree/350804b7b35807c78bd21c313785ae5152ae2985). Use its standalone program-list interface. A failed installation of this one component pauses the Stitch arm; it does not postpone the other representations. [LILO's method](https://arxiv.org/html/2310.19791v4) motivates language descriptions for the Sounding Line handoff, not compulsory retraining of its full stack.

### B. Maker, editor and recipient

**Question:** does a reader better understand a made object when it models the intended effect on a recipient as well as the maker's direct production goal?

Extend the existing multi-actor scaffold with a finite recipient whose knowledge or hypothesis distribution changes when it sees the artifact. The recipient must actually compute an outcome from observable features; do not assign success merely because a maker was told to communicate something.

Use the graphic artifact as a small instruction/signaling display and the assembly artifact as a view of a dependency-bearing object. Define recipient tasks with independently executable answers, such as identifying a target configuration or deciding a next action. These are formal communication analogues, not simulations of human emotion. Where intended uncertainty is manipulated, evaluate a declared distribution over interpretations rather than declaring low entropy universally desirable.

Compare a maker optimizing a physical result; a maker optimizing a recipient response; and a producer–editor pair with a shared brief but different subordinate goals. Manipulate the editor's knowledge of the recipient: correct, incomplete or wrong. Include selection/ratification by the original maker and preserve who proposed, executed and retained each change.

The reader predicts both the next accepted edit and the direction/distribution of the recipient effect. It also predicts outcomes under a changed recipient. Test an object whose literal cue suggests one state while a skilled communicator deliberately evokes another. This is the relevant distinction from the suitcase example: depicted effort and actual effort are different targets.

Use a finite candidate-edit set, with all methods seeing the same candidates, before adding free program search. The core contrast is an executable recipient-aware evaluator against a surface/task-only evaluator and a retrieval baseline. A supplied recipient model is a ceiling; an inferred recipient model earns the practical claim.

**Reference:** [Acting implementation, especially `gridworld/i2p.py`](https://github.com/kach/acting-as-inverse-inverse-planning/blob/5b31902bde406493b835c5b6af683d1e3c3baf62/gridworld/i2p.py). Reimplement the finite audience-posterior objective from the paper; no full animation/controller training is needed. No top-level code license was found in that checkout.

### C. Continuous precision versus selecting a strategy

**Question:** is useful adaptive inference better implemented by continuously refining one model, changing its represented variables, or choosing among different methods?

Implement three explicit rivals with a common downstream prediction objective:

- An anytime inverse reader, updated at every observation with adjustable hypothesis/sample effort.
- A reader that initially tracks cheap access/knowledge indicators and represents belief content only when needed.
- A controller selecting among cached prediction, retrieval and inverse inference.

Include a fixed-effort inverse reader, a cheap empirical predictor and a confidence-only refinement rule. Do not label an exact marginalizer “cheap direct.” Fit or tune controllers on development environments; report both the cost of learning the controller and of running it.

Manipulate evidence reliability, computation price/deadline, changed goals, and a formerly successful cue becoming misleading. Add matched cases distinguishing true belief, false belief, ignorance and accidentally correct belief. Include nonsocial predictive cues so success need not be attributed to mental-state inference. These can be implemented through the same small recipient/action task, rather than a new large world.

Record which variables were represented, which computations were performed, accuracy, calibration, switching lag and total cost. Abrupt behavior does not prove distinct mechanisms; equal performance does not prove a single algorithm. If the alternatives remain observationally indistinguishable, report that boundary and the cheapest adequate implementation.

Use expected improvement in held-out decision quality, estimated from development cases, as the operational benefit signal. Compare against entropy reduction. Never train the scheduler on the current evaluation answer or treat confidence as accuracy. [Resource-rational planning](https://cocosci.princeton.edu/papers/callawayrationaluse.pdf), [resource-limited mindreading model](https://quillienlab.github.io/Quillien%20%26%20Taylor-Davies%202026.pdf), [2026 teaching study](https://www.nature.com/articles/s41562-026-02540-2).

The inspected [AutoToM adjustment code](https://github.com/SCAI-JHU/AutoToM/blob/3f569b7ab1d0ee2702ab43ebdc724679d5fc5231/model/model_adjustment.py) offers a concrete variable-expansion pattern. Its negative-entropy utility and numerical thresholds are an explicit comparator, not our default stopping rule.

### D. Revision, dependencies and knowing when to stop

**Question:** does a useful reader model help revise an artifact without destroying an important dependency, and does outside feedback add more than another internal pass?

Build on B's recipient task and assembly's real dependencies. Compare local edit optimization; dependency-aware revision; and dependency-aware revision with a bounded recipient-feedback query. Give a matched-cost internal reconsideration arm the same computation allowance. Each method selects from the same legal opportunities; private feasible options cannot be handed to only one method.

Cross stable purpose, deliberately revised purpose, and a stale remembered purpose. Include changes that improve a local feature while harming the final intended effect. Present correct, uninformative and misleading outside feedback as separate conditions. The reader may flag uncertainty and request information; a felt-mismatch analogue is a trigger to investigate, not evidence that its diagnosis is correct.

Score final recipient/task outcome, collateral dependency damage, edit cost and stop regret. The benchmark for stopping is the expected benefit of available further operations under the declared task and cost, not whether a particular author historically stopped. Construct both appropriate-stop and inappropriate-stop cases; a stop-always policy must visibly fail.

This family tests your account of reverse-engineering one's own automatic subgoals. Include a fully informed error monitor as a strong rival: V16 did not establish a self-narrative advantage over such monitors. Goals can be revised intentionally; label goal discovery separately from accidental drift.

### E. Artifact reading and Sounding Line transfer

For completed A–D cases, ask observer readers to predict an unseen choice after a constraint/recipient change. Compare artifact-only, earlier-artifact and recorded-process evidence. This is the bridge from useful craft in a maker to useful craft inference in an observer.

Produce a small frozen pilot early, then a final reader-only bundle. The old V16 bundle remains immediately usable by Sounding Line; V17 delivery is not its prerequisite. New cases must be drawn independently of the V16 selected examples and labeled as new discovery or fresh evaluation according to actual exposure.

Select final illustrative cases by a declared rule spanning advantages, reversals and failures—not just visually persuasive successes. Provide at least one case each where a plausible history is wrong, memory catches an exception, a recipient model is wrong, extra computation is wasted, and an edit has a surprising downstream effect, **if the constructed data actually contain it**. Mark absent types as absent rather than fabricating them.

### F. Optional extension: transfer with partial sharing

If core work is progressing comfortably, add one targeted comparison of shared, separate and partially shared performer/observer parameters, with expert versus unfamiliar partner behavior. Measure observation accuracy and personal execution separately. This tests the possibility of interference without assuming it. It is lower priority than the artifact-to-future bridge and is not a condition of closing V17.

## 6. Sampling, selection and claims

Start each main family with approximately 64 independent maker/recipient histories spread across 16 constructor draws per relevant regime. Those are planning sizes, not thresholds for declaring a null. Keep the screen to the main contrasts above; do not take the Cartesian product of every method, budget, world and perturbation.

Expand informative boundaries to roughly 256 histories and at least 32 constructor draws when the actual identity space and runtime support it. Two exact copies are not two independent worlds: hash the public problem and relevant private construction, report unique counts, and cluster inference at shared constructor/history level. Repeated actions and aliases do not enlarge sample size. If a world has few distinct structures, state that limitation and use exact finite-world results where appropriate.

Reserve fresh random namespaces before looking at discovery outcomes. Freeze at most three consequential contrasts, their estimands, sample sizes and directions before fresh evaluation. Choose them by a stated rule combining practical gain, a mechanistic discriminator and transfer relevance, rather than selecting only the largest positive. Within that small family, apply Holm correction if making confirmatory claims. All other surfaces remain descriptive; confidence intervals crossing zero are not equivalence claims.

Plan precision using between-constructor variation in discovery. If the available week cannot resolve a modest effect, report the attainable interval and leave the question open. Do not extend sample size repeatedly until significance or rename an exploratory result confirmation.

## 7. Minimal validity checks, applied locally

Before a new consumer's scientific rows, run a few known cases and a deliberate failure through the literal command, score and resume path. Check only dependencies it actually uses. Required facts include: real maker traces are accepted; invalid actions are rejected; answer support varies; probabilities normalize; a true intervention changes an applicable outcome; the no-information baseline remains uninformative; and no private field enters public requests.

For new abstractions, verify expansion preserves execution and cost on the supported domain. For adaptive inference, verify actual operation counts rather than intended budgets. For recipient tasks, verify that target assignments change realized responses in both directions. For resumed runs, compare the retained rows and final aggregate with the uninterrupted small fixture.

Record failures and their repairs. A model that performs badly through a valid interface is a scientific outcome. An inaccessible file, impossible target or silently empty proposal set is an apparatus problem. The difference must not depend on whether the result is encouraging.

## 8. Outputs and acceptance

Use chunked JSONL or an existing columnar format, one compact index and selected readable examples. Do not create millions of tiny per-operation files. Retain raw predictions, inputs, cost receipts, seeds, source identities and enough replay information to reconstruct reported results.

Deliver:

1. `RESULTS.md`: hypothesis and method first; effect/cost surfaces; what changed our view; limits and next discriminating experiments.
2. One machine-readable comparison table with attempted/valid/unique-unit counts, evidence tier, method, regime, interval and claim status.
3. `TRANSFER.md` plus a reader-only bundle and separate evaluator material. Public schema must declare target kind, answer support, permitted evidence and query cost. Preserve V16 compatibility through a versioned adapter; do not alter `v16.transfer.1` silently.
4. A short expense/time account separating setup, execution, recovery and analysis, with actual CPU/GPU use distinguished from elapsed time.

V17 completes when the commissioned scope has a disposition, completed comparisons have an honest analysis, and its usable public products are delivered. It need not support the favored hypothesis. Do not change the umbrella theory on the basis of simulation alone, and do not edit old quotes as part of routine implementation.

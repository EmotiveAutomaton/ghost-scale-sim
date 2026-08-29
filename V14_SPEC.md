# V14 — The Routed Reader

**Joint reconstruction, route reliability, expertise and attention history, affect ownership, strategic communication, goal hierarchy, and epistemic foraging**

**Status:** immutable implementation and execution specification

**Prepared:** 2026-08-29

**Ghost Scale Sim snapshot reviewed:** `cc31790c8cf0c9ebfaa7af784635673a1a54fbf1`

**Sounding Line snapshot reviewed:** `8230a933fab805a4ee39c256f1e189fe46314dfe`

**Execution class:** constructed-world, CPU-first research; no human, neural, language-model, or real-artifact evidence

**Scope:** 64 mandatory cards across eight trunks plus 12 cross-cutting adversarial attacks

**Runtime:** one continuous 24-hour local execution window. Internal checkpoints are required. **No early, daily, milestone, or partial curator packets are permitted.** Emit one final packet after the window closes.

**Companion GPU program:** Sounding Line Phase 2.4 Stage 5

V1–V13 and their result records remain closed. Implement V14 only under `ghostscale/validation/soundingline/v14/`, its runners, tests, version documents, and `results/v14/`. Do not modify a historical verdict to make a repaired instrument appear to have passed originally.

---

## 0. Executive command

Build and run V14 around one question:

> Under what information conditions can a bounded reader route among action, semantic, contextual, and forensic models to jointly reconstruct a maker's process, episode goal, and persistent preference—without mistaking ease, induced affect, stale attention history, or strategic presentation for evidence?

V14 is justified because the curator's new model introduces discriminators that V13 did not test. It is not a fourth pass over common-prior labels, cost functions, or generic trust curves. The required new objects are:

1. a **joint posterior** over process, episode goal, and standing preference, compared fairly with independent and staged alternatives;
2. a **route-reliability layer** that distinguishes how easy a reconstruction feels from how well it predicts;
3. a factorial separation of **demonstrated competence** from **past attention/value history**;
4. an **affect-ownership model** separating reader response, intended audience effect, maker appraisal, world truth, communicative goal, and uptake;
5. an **inverse-inverse maker** that may select evidence to control a reader's inference;
6. a **goal-hierarchy ruler** that tests stable preference against habit, subgoal, and reward-equivalent alternatives without assuming a final intrinsic horizon;
7. an **epistemic-foraging tournament** separating novelty, complexity, prediction error, learning progress, compressibility, relevance, and information gain per cost.

The program includes the one allowed repair of V13 C03, C05, and P01 because V13 explicitly recommended those repairs. They are integrity debts, not the center of V14.

Every trunk must run independently. When an upstream empirical reader fails, a downstream method card receives an oracle posterior or deliberately degraded posterior so the branch can still test its distinct question. An oracle bypass never counts as end-to-end success.

## 1. Claim ceiling

V14 can establish:

- logical consequences and identifiability boundaries in declared constructed worlds;
- whether a joint estimator, staged estimator, or independent estimator predicts held-out constructed choices better under matched information and compute;
- whether route selection improves proper prediction or information per cost;
- whether planted competence and planted attention history have distinguishable effects;
- whether specified source states can be distinguished from artifact and process evidence;
- when goal hierarchies and stable preferences are or are not recoverable;
- which epistemic-foraging objective performs best in these worlds;
- which rulers are licensed for Sounding Line Stage 5.

V14 cannot establish:

- that humans use the implemented algorithm;
- that an action route is a mirror-neuron system, default-mode network, limbic substrate, cortical reference frame, or “shared humanity”;
- that simulated attention history is human expertise or value;
- that a simulated response is felt valence;
- that propaganda expertise protects people;
- that a final human goal or value horizon exists;
- that an artifact uniquely identifies its historical process;
- that a model, PyMDP agent, or exact estimator has empathy, trust, sincerity, fanaticism, or care.

Use `METHOD`, `CONSTRUCTED_MECHANISM`, `BOUNDARY`, `INSTRUMENT_FAILURE`, `VOID`, and `RESOURCE_BLOCKED` exactly as V13 did. Every public sentence carries one of these ceilings.

## 2. V13 record imported with corrections

| V13 finding | V14 use |
|---|---|
| A truly equal local comparator removed most general self advantage; the near-maker interaction remained, while hidden-next-goal gain was nearly zero. | Locality remains a prior proposal, not the target result. V14 asks whether joint process/goal/preference inference makes local initialization prospectively useful. |
| C03 and C05 failed controls, so the common-substrate mechanism was unread. | Run exactly one disclosed repair each. No further repair if the named gate fails again. |
| P01's matched-control correction curve failed a positive and calibration gate. | Run one disclosed repair that corrects the gate semantics and calibration instrument without redefining the target. |
| Learned precision greatly helped under the planted model but failed adaptive reallocation and double-counted duplicate evidence. | Route learning is live; safe route use is not. Duplicated and correlated evidence remain mandatory attacks. |
| Correct cost models helped; misspecified cost models reversed the result. | V14 uses cost only for route acquisition and foraging. It does not infer motivation directly from expenditure. |
| Communicative goal, reliability, content evidence, and uptake were separable in the constructed reader. | Add affect ownership and audience modeling; retain the factorization. |
| Full interaction logs revealed a production hand; partial logs and artifacts did not. | Historical and actor-level claims require access conditions. Artifact-only equifinality must produce abstention. |
| Exact active selection helped only when probes changed evidence visibility; the legacy PyMDP selector closed. | Exact EIG is the reference. Do not reopen PyMDP unless a new action model first passes exact-agreement and divergence gates. |
| Pooling readers reduced accuracy and increased overconfidence. | Diverse readers may constrain an equivalence class only through calibrated likelihood or feasible-set intersection, never naive posterior averaging. |

## 3. Constructed ontology

### 3.1 Maker state

At episode `t`, define a maker by:

\[
M_t = (K, H_{1:t}, V_t, G_t, \Pi_t, B_t, A^m_t, A^a_t, C_t, R_t, O_t).
\]

| Symbol | Constructed meaning |
|---|---|
| `K` | demonstrated domain competence: transition, tool, and observation accuracy |
| `H` | history of attended features, practiced transitions, and past reward-linked selections |
| `V` | standing preference weights that constrain choices across episodes |
| `G` | current episode goal or goal mixture |
| `Pi` | process plan or policy over action chains |
| `B` | maker belief about relevant world state |
| `A^m` | maker's own appraisal variable |
| `A^a` | audience state the maker intends to induce |
| `C` | communicative goal: inform, assist, warn, impress, recruit, conceal, or mislead |
| `R` | source reliability history, separate from current intent and content |
| `O` | opportunity set, constraints, costs, and available evidence |

An artifact is emitted from an action/process history under these variables. Several histories must be allowed to produce the same artifact. The reader never receives a variable merely because a construction name contains it.

### 3.2 Reader state and routes

A reader has its own local prior, route models, route-reliability beliefs, evidence budget, and posterior over maker states. Required routes are:

- `action`: action-chain, tool, geometry, and transition constraints;
- `semantic`: artifact meaning, claims, associations, and task structure;
- `context`: role, convention, source history, opportunity, and institution;
- `forensic`: a costly optional observation that improves material, temporal, or interaction resolution.

Operational route ease is measured separately through known inference cost, posterior entropy before target evidence, convergence steps, or a planted processing penalty. It is never called subjective fluency without a human measure.

### 3.3 Communication regimes are derived, not labels

Cross these independently:

- maker belief in a threat or benefit;
- objective content support;
- desired audience appraisal;
- desired audience action;
- evidence-selection policy;
- willingness to correct;
- private/off-audience action;
- surface intensity.

“Sincere fanatic,” “strategic propagandist,” “honest warning,” and “neutral report” name regions of this factorial only. A valid discriminator must predict a counterfactual action where those regions diverge. A template classifier does not count.

### 3.4 Goal hierarchy and non-identifiability

Generate goal graphs with episode goals, subgoals, instrumental goals, standing preferences, and optional terminal rewards. Include:

- identical local actions produced by different higher-order goals;
- potential-based reward transformations or other policy-equivalent rewards;
- stable habits that persist after current preferences change;
- stable preferences expressed through changing local goals;
- finite records with no identifiable terminal horizon.

The reader's target may be an equivalence class. Requiring one “true value” when several models are behaviorally equivalent is a scoring defect.

## 4. Shared experimental contract

### 4.1 Matched estimator tournament

For applicable J, H, and bridge cards, compare:

1. independent marginals for process, goal, and preference;
2. `goal -> process -> preference`;
3. `process -> goal -> preference`;
4. `preference -> goal -> process`;
5. recurrent joint inference;
6. oracle joint inference;
7. cheap frequency, last-choice, and surface baselines.

All non-oracle estimators receive identical observations, candidate state space, priors, and effective compute. If exact enumeration is feasible, use it as the reference. Approximate routes report divergence from exact. Free-form natural-language explanations are outside this simulation and cannot serve as an estimator.

### 4.2 Evidence trajectories

Save the full posterior after every observation. For each latent report:

- first dose at which held-out prediction improves over its baseline;
- time to criterion and evidence consumed;
- reversals after diagnostic contradiction;
- residual projection or history bias;
- confidence, calibration, and abstention;
- whether another latent supplied the information through a joint update.

### 4.3 Required gates

Every causal card carries:

- live manipulation gate;
- matched placebo gate;
- known-answer positive gate;
- surface/leakage gate;
- oracle identifiability gate;
- prospective prediction gate;
- calibration/abstention gate.

Route cards additionally require an information-divergence gate and an equal-ease/equal-accuracy control. Communication cards require surface collisions. Hierarchy cards require reward/policy-equivalence checks. Foraging cards require an unlearnable-noise control.

One repair is permitted only where a card explicitly says so. The original attempt and all failed gates remain in the record.

### 4.4 Lanes and independent units

Use disjoint `pilot`, `discovery`, `transfer`, and `confirmation` lineages. Root seeds include lane, card, world family, domain, maker, reader, and replicate. Test that ancestry does not cross lanes.

The independent unit is a generated world, maker history, source history, production team, or reader construction as declared. Repeated actions, artifacts, messages, rows, and posterior doses are not additional independent makers.

## 5. Mandatory card inventory

There are exactly 64 mandatory cards before attacks. Every card must be represented literally in the machine-readable manifest and must end in `LANDED`, `SCIENTIFIC_CLOSED`, `INSTRUMENT_FAILED`, `VOID`, or `RESOURCE_BLOCKED`.

### I — integrity, identities, and V13 repairs (8)

| Card | Required question | Primary criterion or disposition |
|---|---|---|
| I01 | Do the V13 numeric anchors used by V14 reproduce from committed inputs? | Hash and tolerance receipt; mismatch blocks inheritance only. |
| I02 | Does the manifest enumerate all 64 cards, factors, attacks, lanes, and independent-unit floors? | Recursive expected-cell validator. |
| I03 | Does the joint enumerator recover exact posteriors in tiny exhaustive worlds and remain invariant to labels/order? | Exact identity and normalization tests. |
| I04 | Are action, semantic, context, and forensic routes independently live and materially different in information? | Pairwise divergence and null-route gates. |
| I05 | Are surface collisions, equifinal histories, factor orthogonality, and lineage identities valid? | Leakage baselines at floor and collision hashes equal where required. |
| I06 | Repair V13 C03 once: within-common versus all-family prior, preserving its target and failed positive gate. | New positive calibration must be justified from construction, not lowered after data; otherwise close common-substrate card. |
| I07 | Repair V13 C05 once: self versus within-common prior, preserving reader-type interaction and convergence placebo. | Routes must converge under sufficiently diagnostic evidence or the mechanism closes. |
| I08 | Repair V13 P01 once: matched-local correction curve, preserving similarity bins, prospective endpoint, and failed calibration record. | Correct gate semantics and calibration instrument without redefining correction; no second repair. |

### J — joint partial identifiability (10)

| Card | Required question | Primary criterion or disposition |
|---|---|---|
| J01 | Is process identifiable when goal and preference are supplied? | Proper posterior and abstention on process-equivalent histories. |
| J02 | Is episode goal identifiable when process and preference are supplied? | Hidden next action/choice, not retrospective label alone. |
| J03 | Is standing preference identifiable when process and episode goal are supplied across episodes? | New-episode prediction after local goal changes. |
| J04 | Does recurrent joint inference beat independent marginals under matched evidence/compute? | Held-out log score, calibration, and ablation of cross-latent messages. |
| J05 | Which staged order is best, and is any order uniformly best across expertise/access regimes? | Predeclared order × regime interaction; no pooled “purpose first” headline. |
| J06 | At what evidence dose does each latent first improve prospective prediction? | Dose trajectories and provisional-confidence curve. |
| J07 | Does diagnostic contradiction revise all affected latents rather than only the surface label? | Revision, recovery half-life, and false-revision rate. |
| J08 | Does the joint reader abstain on exact equifinality and contract uncertainty after resolving evidence? | Risk–coverage and equivalence-class coverage. |
| J09 | Can it distinguish a changed episode goal from a changed standing preference? | Crossed intervention with held-out future choices. |
| J10 | Does the joint advantage transfer to fresh world factorization and action vocabulary? | Same frozen estimator on transfer families; domain-bound effects remain named. |

### R — route reliability, fluency, and conflict (8)

| Card | Required question | Primary criterion or disposition |
|---|---|---|
| R01 | Which route contains the most target information in each access regime? | Exact conditional information and prediction ruler. |
| R02 | Can a reader learn route reliability from feedback without receiving target labels at test? | Prospective route weighting versus fixed/equal/random. |
| R03 | Does planted ease bias route use when accuracy is equal? | Equal-accuracy/different-ease contrast. |
| R04 | Does true accuracy control route use when ease is equal? | Equal-ease/different-accuracy contrast. |
| R05 | When routes conflict, does expanding the latent set to missing goal, constraint, or strategic source improve prediction? | Gain minus search cost, with consistent-world false positives. |
| R06 | When is costly forensic access worth purchasing? | Realized information gain per cost versus exact, random, always-buy, and never-buy policies. |
| R07 | Can routes be fused without treating correlated or duplicated evidence as independent? | Accuracy and calibration under shared-cause, duplicate, and independent evidence. |
| R08 | Do learned route reliabilities transfer, or should the reader reset under a new domain? | Reset/partial-transfer/full-transfer tournament. |

### E — competence versus attention history (10)

| Card | Required question | Primary criterion or disposition |
|---|---|---|
| E01 | Are demonstrated competence and past attention/value history independently live? | Full `K × H` manipulation and orthogonality gates. |
| E02 | With competence matched, does different attention history change initial route choice or prior? | Initial bias and prospective correction. |
| E03 | With history matched, does different competence change process reconstruction? | Action/process prediction and calibration. |
| E04 | Does a learned attention bias persist after its reward/value reverses? | Decay curve and current-utility cost. |
| E05 | Can target evidence correct stale attention history without erasing genuine skill? | Bias reduction with retained process accuracy. |
| E06 | Does competence selectively improve early relevance detection? | Evidence-dose interaction; no generic intelligence interpretation. |
| E07 | Does attention history transfer more broadly or narrowly than competence across domains? | Cross-domain conditional matrix. |
| E08 | When skill was acquired through different attention paths, do equal competencies yield different maker signatures? | Same held-out skill, different residual choices; prospective only. |
| E09 | Can diverse calibrated readers shrink the compatible maker set without naive averaging? | Feasible-set/likelihood intersection versus best member and posterior pool. |
| E10 | Which object—competence, attention history, or current preference—best predicts the maker's next novel choice? | Factor intervention and hidden next-choice tournament. |

### A — affect ownership and strategic communication (10)

| Card | Required question | Primary criterion or disposition |
|---|---|---|
| A01 | Are reader response, intended audience appraisal, maker appraisal, content support, communicative goal, reliability, and uptake independently live? | Factor and schema identities; no overwritten owner. |
| A02 | When is the reader's own induced response a useful prior for intended audience effect? | Similarity/access interaction and projection cost. |
| A03 | Can intended effect be recovered while maker appraisal remains uncertain? | Partial-identifiability posterior and held-out presentation choice. |
| A04 | Can maker appraisal be recovered while intended effect differs? | Owner-swap construction and private/off-audience prediction. |
| A05 | Can derived honest warning, sincere fanatic, strategic propagandist, and neutral reporting be separated without labels? | Counterfactual evidence selection and correction behavior. |
| A06 | What minimally distinguishes sincere fanatic from strategic propagandist under matched surface and audience effect? | Source belief/private action/willingness-to-correct tournament; abstain when no discriminator is observed. |
| A07 | Does an inverse-inverse reader help only when the maker actually models the audience? | Maker mechanism × reader model interaction. |
| A08 | Does influence/source awareness improve attribution or merely suppress all uptake? | Discrimination, criterion, calibration, and selective true/false uptake. |
| A09 | Can acute response habituate while cumulative belief or policy uptake grows across exposure? | Separate fast response and slow posterior/policy trajectories. |
| A10 | Does factored trust gate uptake after reconstruction without changing content evidence or inferred goal? | Posterior-to-policy causal bridge and negative-transfer score. |

### H — hierarchy, habit, and value residue (8)

| Card | Required question | Primary criterion or disposition |
|---|---|---|
| H01 | Can repeated transition structure recover a subtask hierarchy when it exists? | Known-answer hierarchy and next-subtask prediction. |
| H02 | Do identical local actions under different higher-order goals remain correctly ambiguous? | Equivalence-class posterior; no forced actor/value attribution. |
| H03 | Can policy-equivalent reward transformations be distinguished without extra evidence? | Required non-identifiability; resolving intervention closes uncertainty. |
| H04 | Can a stable preference be distinguished from a practiced habit across changed incentives? | Counterfactual cost/opportunity and new-domain choices. |
| H05 | Can stale attention/expertise residue be distinguished from a current preference? | History reversal and hidden future choice. |
| H06 | Does multi-episode evidence support progressively higher coordinating goals without requiring a terminal horizon? | Hierarchical predictive compression and calibrated level uncertainty. |
| H07 | When full interaction records are available, can role-relative control be recovered beyond coherence and shared brief? | Hidden next intervention, crossed subordinates, and exact shared-brief rival. |
| H08 | Which inferred hierarchy level best predicts the next changed-context action? | Prospective level-selection score against flat value and last-goal baselines. |

### F — interest and epistemic foraging (8)

| Card | Required question | Primary criterion or disposition |
|---|---|---|
| F01 | Are novelty, complexity, prediction error, reducibility, learning progress, relevance, competence, and cost independently live? | Factorial identity and correlation audit. |
| F02 | Is a novel but immediately explained item still selected over a familiar unresolved structure? | Novelty versus structured residual. |
| F03 | Is a complex but compressible item preferred over a simpler unresolved one? | Complexity versus compressibility/reducibility. |
| F04 | Does random unlearnable noise lose to structured learnable error despite higher surprise? | Noise trap and realized learning. |
| F05 | Does expected learning progress outperform raw current error in a nonstationary curriculum? | Future error reduction and task sequence. |
| F06 | Does expected information gain per cost outperform novelty, surprise, and always-forensic policies? | Realized held-out gain per cost. |
| F07 | Can pursuit value stay high while warrant stays low for a hoped-for explanation? | Separate query allocation and posterior confidence under hope congruence. |
| F08 | Does an active selector transfer to new foraging ecologies and abstain when no probe is discriminative? | Transfer regret and no-action/null behavior. |

### B — Sounding Line bridge and closure (2)

| Card | Required question | Primary criterion or disposition |
|---|---|---|
| B01 | Which validated rulers license an implementation in Sounding Line Stage 5? | One row per candidate: required access, construction gate, cheap rival, endpoint, expected shape, and claim ceiling. Failed instruments cannot license a bridge. |
| B02 | Which V14 questions should be promoted, closed, or left as context after confirmation? | Final pursuit/warrant ledger, runtime audit, and recommendation; no automatic V15. |

## 6. Cross-cutting adversarial matrix

There are 12 mandatory attacks. Apply every relevant attack to every promotion candidate, plus one method positive and one valid null. The validator records applicability and forbids a silent `not applicable`.

| Attack | Threat and required test |
|---|---|
| X01 — surface | Preserve the latent and alter lengths, counts, names, labels, order, and cheap features; preserve surface and swap the latent. |
| X02 — route ease | Make the wrong route easier and the correct route harder, then reverse; accuracy and ease must not alias. |
| X03 — equifinal history | Give identical artifacts validly produced by different histories; require equivalence-class uncertainty. |
| X04 — duplicate evidence | Duplicate or paraphrase one cause; confidence must not rise as though evidence were independent. |
| X05 — wrong generative model | Change cost, competence, noise, or source-selection process while retaining familiar surfaces. |
| X06 — attention/skill swap | Hold competence fixed and swap history; hold history fixed and swap competence. |
| X07 — affect owner swap | Swap reader response, maker appraisal, intended effect, and content truth while matching intensity. |
| X08 — fanatic/propagandist collision | Match artifact and intended audience effect; vary only counterfactual belief/private/correction behavior. |
| X09 — hierarchy equivalence | Exact shared brief, reward shaping, or locally equivalent higher goal must defeat unjustified unique attribution. |
| X10 — hope and salience | Make an attractive hypothesis salient but weakly supported; make a dull hypothesis diagnostic. |
| X11 — aggregation | Verify that global means cannot hide planned sign reversals across similarity, access, or competence. |
| X12 — solver/lineage | Seed, order, solver, process count, and fresh-lane audit; exact/approximate disagreement is reported as such. |

## 7. Scores, inference, and promotion

### 7.1 Primary scores

Use log score, Brier score, calibration slope/error, risk–coverage, false-confident-attribution rate, held-out next choice/action, posterior equivalence-class coverage, information gain per cost, and regret against exact selection. Top-1 accuracy is secondary.

For hierarchies, separately score subtask boundary, level, reward-equivalence class, and next changed-context action. For communication, separately score maker belief, maker appraisal, intended audience effect, content support, communicative goal, reliability, and uptake. For foraging, score realized learning/prediction gain, not an internal variable named `interest`.

Use world- or maker-level aggregation, cluster bootstrap, or hierarchical models. Conditional effects precede any pooled number. Multiplicity is controlled within trunk during discovery; confirmation contains only a frozen primary estimand and named rivals.

### 7.2 Possible flight results

V14 has five possible constructed-world flight results:

1. **Joint reconstruction advantage:** recurrent joint inference predicts a new action better than every matched staged/independent estimator, while preserving uncertainty under equifinality.
2. **Reliable routing:** learned route reliability selects useful evidence beyond ease and survives conflict, duplicate, and domain-shift attacks.
3. **Competence/history dissociation:** competence and attention history have distinct, prospectively validated signatures and correction dynamics.
4. **Affect/source factorization:** the reader distinguishes owner, audience goal, source belief, content support, and uptake, including a valid fanatic/propagandist boundary.
5. **Learning-progress foraging:** reducible uncertainty or learning progress predicts useful evidence acquisition better than raw novelty, complexity, or surprise.

Each remains a constructed mechanism. A negative but valid equivalence boundary may be more valuable than a weak positive.

### 7.3 Promotion requirements

A flight candidate must:

1. pass every live, placebo, positive, surface, oracle, prospective, and calibration gate;
2. beat its named cheap rival with a predeclared practical margin or establish a predeclared boundary;
3. survive all relevant X attacks;
4. transfer to a fresh factorization or remain explicitly domain-bound;
5. preserve the true independent unit;
6. reproduce on untouched confirmation worlds.

At elapsed hour 20, freeze at most four confirmation candidates, no more than one per flight family unless fewer than four families qualify. Do not choose by effect size alone. Prefer a decisive boundary over an underpowered positive.

## 8. Continuous 24-hour runtime contract

### 8.1 One execution window

The runtime clock begins with the discarded pilot and ends exactly 24 elapsed hours later. Pilot, workload freeze, discovery, transfer, repairs, confirmation, and closure preparation occur inside the window. A restart preserves the original deadline. Final reporting begins only after the 24-hour execution window has closed.

Write machine-readable checkpoints continuously. Do not write or emit early, daily, milestone, preview, interim, or partial curator packets. The only curator-facing packet is generated after the execution deadline:

- `docs/versions/v14-routed-reader/RESULTS_PACKET.md`.

Generate `docs/versions/v14-routed-reader/RESULTS.md` in the same post-run closure as the stable repository summary, but do not emit or present it as a separate packet.

Internal logs and verdicts may be written through; they must not contain a premature global narrative.

### 8.2 Resource governor and coexistence

Run CPU work below normal priority. Start with the smaller of 12 workers or half the available logical cores. Record process-tree CPU, wall time, peak RSS, worker utilization, serialization, and Sounding Line GPU-lock wait/loading behavior. If Ghost causes more than a frozen 5% degradation in the Sounding Line calibration throughput or material data-loading stalls, reduce Ghost workers; never increase Sounding Line's gear to compensate.

GPU use is prohibited except for whatever the companion Sounding Line program already owns. V14's exact and legacy PyMDP paths remain CPU-only. Do not migrate the repository to the modern JAX PyMDP interface in this run.

### 8.3 Discarded pilot and workload tiers

After all card functions, attacks, and validators exist, smoke them without scientific output. Run a discarded pilot over one heavy card per substantive trunk plus the largest joint, hierarchy, and foraging rulers. Use separate pilot lineages. Measure end-to-end throughput and freeze the smallest tier forecast to occupy 20–21 hours, leaving 3–4 hours for confirmation and closure.

| Tier | Discovery worlds | Transfer worlds | Confirmation worlds | Stochastic repeats | Makers/readers per applicable world |
|---|---:|---:|---:|---:|---:|
| T0 | 32 | 16 | 32 | 2 | 32 |
| T1 | 64 | 32 | 64 | 3 | 48 |
| T2 | 128 | 64 | 96 | 3 | 64 |
| T3 | 256 | 128 | 128 | 4 | 96 |

The pilot may select a mixed tier only through a predeclared workload function that gives high-resolution hierarchy, equifinality, and rare-regime cards their declared floors. Freeze `WORKLOAD_LOCK.json`, expanded expected cells, forecast, and scientific lock before opening discovery results.

If T3 is predicted to finish early, expand useful independent work in this frozen order:

1. independent world families and makers;
2. transfer action vocabularies and competence ecologies;
3. stronger surface collisions and equifinal histories;
4. additional evidence-dose and history-decay points;
5. additional foraging ecologies and conflict magnitudes;
6. confirmation worlds allocated before discovery.

Never sleep, repeat an identical seed, clone rows from one maker, oversample an already saturated easy cell, or increase serialization/output solely to occupy time. If admissible independent work is exhausted early, write `SHORT_RUN` and the actual duration; the continuous 24-hour contract is not satisfied.

At hour 20, stop opening new exploratory branches and begin frozen confirmation and closure. At hour 24, stop launching new work and checkpoint in-flight units. Then validate the realized record and generate the final-only report. An unfinished card remains unfinished.

### 8.4 Runtime allocations

| Work family | Target window share |
|---|---:|
| integrity and V13 repairs | 12% |
| joint inference | 20% |
| route reliability | 14% |
| competence versus history | 16% |
| affect and communication | 16% |
| hierarchy and value residue | 10% |
| foraging | 7% |
| confirmation, validation, closure preparation | 5% |

Shares govern scheduling breadth; they are not evidence quotas. No positive result may consume another trunk's minimum floor before breadth completes.

## 9. Implementation layout

Preferred paths:

- `V14_SPEC.md` — this immutable handoff;
- `docs/versions/v14-routed-reader/SPEC.md` — implementation translation;
- `docs/versions/v14-routed-reader/RESULTS.md` — final read-first report;
- `docs/versions/v14-routed-reader/RESULTS_PACKET.md` — full appendix;
- `ghostscale/prereg_v14.py` — criteria, gates, repairs, promotions, and locks;
- `ghostscale/validation/soundingline/v14/` — all world, estimator, and card modules;
- `runners/run_v14.py` — resumable 24-hour scheduler;
- `runners/run_v14_confirmation.py` — frozen confirmation only;
- `runners/report_v14.py` — final-only reporting;
- `runners/validate_v14_program.py` — read-only structure/result/runtime validation;
- `tests/test_v14_gates.py`, `tests/test_v14_metamorphic.py`, `tests/test_v14_fresh_clone.py`;
- `results/v14/` and `results/validation/soundingline/v14/`.

Suggested shared modules:

- `world.py` — maker state, histories, opportunities, and surface rendering;
- `joint.py` — exact joint, staged, independent, and oracle estimators;
- `routes.py` — route observations, reliability, ease, costs, and conflict;
- `history_skill.py` — competence and attention-history generators;
- `communication.py` — appraisal owners, audience model, source behavior, and uptake;
- `hierarchy.py` — subgoals, reward equivalence, habits, and role graphs;
- `foraging.py` — novelty, error, reducibility, learning progress, and EIG;
- `schemas.py` — card, result, runtime, lineage, completion, bridge, and claim schemas;
- `cards/trunk_*.py` — one callable per literal card ID.

## 10. Machine-readable record and validation

Before scientific execution, create:

- `results/v14/QUEUE_MANIFEST.json`;
- `results/v14/EXPECTED_CELLS_TEMPLATE.json`;
- `results/v14/SOURCE_LINEAGES.json`;
- `results/v14/CONSTRUCTION_IDENTITIES.json`;
- `results/v14/prereg_v14_structural_lock.json`.

After the pilot, create and lock:

- `WORKLOAD_LOCK.json`;
- `EXPECTED_CELLS.json`;
- `ROUTE_INFORMATION.json`;
- `ATTACK_MATRIX.json`;
- final scientific lock.

During and after execution, maintain:

- `RUNTIME.json` with parent and child CPU;
- `COMPLETION.json` with verdict and source hashes;
- `COVERAGE.json`;
- `CONFIRMATION_REGISTRY.json`;
- `CLAIM_LEDGER.json`;
- per-card verdicts and compact aggregate tables.

Required tests include:

1. exact joint posterior identities in exhaustive tiny worlds;
2. same-evidence and effective-compute equality across estimator routes;
3. factor orthogonality and independent manipulation;
4. matched-surface and exact-equifinality fixtures;
5. route-information divergence and no-information closures;
6. ease/accuracy crossing identities;
7. competence/history swap tests;
8. affect-owner and source-variable non-aliasing;
9. reward-shaping/policy-equivalence non-identifiability;
10. unlearnable-noise foraging trap;
11. duplicate/correlated evidence calibration;
12. row-order invariant aggregation at the correct unit;
13. disjoint pilot/discovery/transfer/confirmation ancestry;
14. process-tree runtime accounting and immutable deadline across restart;
15. report guard preventing every curator packet before closure;
16. recursive manifest/expected-cell/attack/card count validation;
17. clean-clone regeneration of hashes, aggregates, and final scientific fields.

Use stable explicit seeds; never Python's salted `hash()`. Keep validators read-only unless an output path is explicitly supplied. Smoke every card, attack, resume path, and report suppression on a scratch root before starting the pilot. Preserve all prior failed verdicts beside repairs.

## 11. Research anchors and design consequence

These are design anchors, not evidence for V14's outcomes.

### Structured social inference

- [Baker, Saxe, and Tenenbaum: action understanding as inverse planning](https://pubmed.ncbi.nlm.nih.gov/19729154/)
- [CLIPS: cooperative language-guided inverse plan search](https://arxiv.org/abs/2402.17930)
- [Grounding Language about Belief](https://arxiv.org/abs/2402.10416)
- [LaBToM](https://aclanthology.org/2025.tacl-1.30/)
- [Storytelling as Inverse Inverse Planning](https://pubmed.ncbi.nlm.nih.gov/37962526/)
- [AutoToM: uncertainty-guided automated agent-model expansion](https://arxiv.org/abs/2502.15676)
- [LLM-augmented inverse planning](https://arxiv.org/abs/2507.03682)

The import is structured latent inference, sequential updating, multimodal likelihoods, uncertainty-triggered expansion of the latent set, language-model hypothesis proposal, and an audience-modeling maker. Proposal and warrant remain separate, and the cooperative-planner assumption is crossed rather than adopted.

### Attention history, expertise, and fluency

- [Learned value magnifies attentional capture](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0027926)
- [Persistence of value-driven attention](https://andersonlab.sites.tamu.edu/wp-content/uploads/sites/36/2016/09/JEPHPP_2013.pdf)
- [Expert–novice relevance selection in chess](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2014.00941/full)
- [Self-generated cognitive fluency](https://pages.ucsd.edu/~pwinkielman/VonHecker-Hanel-Jin-Winkielman_Fluency-mental-models_CE-2023.pdf)

The import is a factorial separation. Reward-linked attention can persist, skill can change early relevance selection, and self-generated ease can alter evaluation; none proves that expertise is merely attention history.

### Affect source, persuasion, and uptake

- [Influence awareness and affect misattribution](https://link.springer.com/article/10.3758/s13428-022-01879-4)
- [Affect labeling](https://pubmed.ncbi.nlm.nih.gov/17576282/)
- [Cognitive reappraisal](https://pubmed.ncbi.nlm.nih.gov/12495527/)
- [Persuasion, influence, and value](https://pmc.ncbi.nlm.nih.gov/articles/PMC12175252/)

The import is to separate awareness, attribution, appraisal, belief, and uptake. Recognition of persuasive intent is not defined as successful resistance.

### Hierarchy and exploration

- [HIRL: hierarchical inverse reinforcement learning](https://arxiv.org/abs/1604.06508)
- [Multiple suboptimal experts and reward ambiguity](https://proceedings.neurips.cc/paper_files/paper/2024/file/9bcd1fa0c05e5f25ba7a1261f1852e82-Paper-Conference.pdf)
- [A typology of computational intrinsic motivation](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/neuro.12.006.2007/full)
- [Learning progress as intrinsic motivation](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/neuro.01.1.1.017.2007/full)

The import is to compare reward-equivalent hierarchies and to distinguish raw unpredictability from reducible error and learning progress.

## 12. Required final curator packet

After the full window, organize the final packet in two passes.

### Pass A — read first

1. Where the constructed project world moved, in plain language.
2. The strongest three to five results or boundaries.
3. Which competing explanations gained or lost probability.
4. What, if anything, Sounding Line Stage 5 is licensed to use.
5. What remains unmeasured, especially every human and neural gap.
6. Questions for the curator only where an answer changes the next branch.
7. Recommendation: continue Ghost, retain selected rulers, or pause it.

Then print exactly:

> **STOP READING HERE**

### Pass B — analyst appendix

Include one plain-language paragraph per card before metrics; full conditional matrices and intervals; posterior trajectories; all gates; all X attacks; V13 repair provenance; equivalence classes; exact/approximate disagreement; discovery/transfer/confirmation separation; effective independent sample; multiplicity; runtime forecast versus actual; process-tree CPU; Sounding Line coexistence; checkpoints; file paths; commit and hashes; dirty state; clean-clone receipt; and B01/B02 bridge ledgers.

There is one final packet and no early packet.

## 13. Pre-mortem

V14 is persuasive but worthless if:

1. joint inference wins because it receives more evidence or compute;
2. a goal label leaks the process or preference;
3. a retrospective fit is called standing-value recovery without a new choice;
4. ease is measured once and relabeled reliability;
5. attention history is defined to equal competence;
6. stale habit is defined to equal current value;
7. reader-induced appraisal becomes the maker's appraisal;
8. communicative goal becomes source reliability;
9. fanatic and propagandist have telltale templates;
10. understanding becomes blanket distrust in the uptake score;
11. identical artifacts receive confident unique histories;
12. a hierarchy is unique only because alternative rewards were omitted;
13. random noise wins a curiosity ruler because it remains surprising;
14. multiple readers are naively averaged;
15. active inference returns without a live action set;
16. V13 failed cards are silently overwritten by repairs;
17. a pooled mean hides a conditional sign reversal;
18. one maker emits thousands of nominal independent rows;
19. duration is manufactured with duplicate work or sleep;
20. an internal checkpoint becomes an early curator packet;
21. a constructed mechanism is reported as human, limbic, motor, neural, or moral evidence.

## 14. Definition of done

V14 is complete only when:

- all 64 mandatory cards have valid dispositions;
- all 12 attacks have explicit applicability and outcomes;
- the three V13 repair debts preserve their original failed records and use at most one repair;
- joint, staged, and independent estimators are information- and compute-matched;
- process, goal, preference, competence, attention history, appraisal owners, communicative goal, reliability, and uptake are separately live;
- every promoted result constrains a hidden future action or establishes a declared boundary;
- exact equifinality and reward equivalence cause uncertainty;
- route selection passes divergence and ease/accuracy controls;
- unlearnable noise cannot win the foraging tournament by surprise alone;
- discovery, transfer, and confirmation lineages are disjoint;
- the run occupies one continuous 24-hour window or is honestly marked `SHORT_RUN`;
- no early curator packet exists;
- the final-only report, result hashes, expected cells, completion ledger, runtime, and clean-clone receipt agree.

A fully valid negative program can be complete. A dramatic positive without prospective prediction, the strongest rival, relevant attacks, and untouched confirmation is incomplete.

## 15. Handoff sentence

> Build all 64 cards and 12 attacks before scientific execution; preserve and repair the three named V13 instruments once; compare joint, staged, and independent reconstruction with the same information and compute; make route ease compete with route reliability; factor competence from attention history; separate every owner and stage of affective communication; require fanatic and propagandist hypotheses to predict a divergent counterfactual; treat goal hierarchies as equivalence classes where warranted; make learnable uncertainty compete with novelty and noise; run useful CPU work continuously for 24 hours beside the locked Sounding Line GPU program; write internal checkpoints but no early packets; confirm only frozen effects; validate from a clean clone; and export only bounded constructed-world rulers.

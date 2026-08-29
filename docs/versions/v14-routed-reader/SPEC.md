# V14 — The Routed Reader. Implementation translation and build plan

**Written 2026-08-29, before any code, as the viability assessment and plan the curator asked
for.** The scientific program is [`V14_SPEC.md`](../../../V14_SPEC.md) at the repository root
(the curator's immutable handoff) and is not narrowed here. This page records whether it can be
built as written, what V13 already supplies, what is new, the judgment calls the spec leaves
open, the build order, and the pre-mortem. It becomes the implementation translation as the
build proceeds; deviations after data go to `RESULTS.md`.

## Viability

**Fully viable as written; no card requires a construction the machine cannot run.** Roughly
half of the program's machinery already exists in V13 and copies under `v14/` unchanged in kind
(V1–V13 stay closed; nothing under `v13/` is touched). The other half is new science: a joint
hypothesis space, a route layer, a history-versus-competence generator, affect owners with an
audience-modelling maker, reward-equivalent hierarchies, and a foraging tournament. Sixty-four
cards and twelve attacks against V13's 152 and twenty; the per-card workload at the spec's tiers
is five to seven times lighter than V13's expanded T2, so the binding constraint is the opposite
of V13's — filling a 24-hour window with admissible independent work rather than fitting into
one. Exact enumeration is feasible for every estimator the spec names: the joint space is
process plans × episode goals × preference profiles, a few hundred states, and staged and
independent estimators are exact computations over the same grid.

Five judgment calls are taken here rather than silently; the first is the one that decides
whether the joint trunk measures anything, and it is the only one worth the curator's answer
before the build:

1. **What a "staged" estimator is.** With exact inference over one grid, conditioning order does
   not matter — Bayes is order-invariant — so `goal → process → preference` computed exactly is
   identical to recurrent joint inference, and J04/J05 would be identities (V12's T-trunk lesson).
   The staged estimators are therefore defined as **plug-in pipelines**: infer the first latent
   from the evidence, commit its posterior mode (or a hard sample under the same budget), condition
   the next stage on that commitment, and so on; the independent estimator is the product of
   marginals each computed with the other latents integrated under the prior; the recurrent joint
   estimator is the exact joint posterior; the oracle receives the other two latents' truths. All
   receive the same observations, grid, priors and enumeration budget (compute is matched by
   counting grid evaluations, recorded per estimator). Under this definition the joint advantage
   is the cost of premature commitment plus cross-latent coupling — a measurable quantity, not a
   tautology — and it can be zero, which is a result.
2. **Route ease.** Ease is planted and operational: a per-route processing penalty in budget
   units and the number of evidence doses a route needs to reach a fixed posterior entropy, both
   measured before target evidence. It is never called fluency. X02 crosses ease against accuracy
   by construction (the wrong route cheap and sharp, the right route dear and slow, then
   reversed), which is the equal-ease/equal-accuracy control the route cards require.
3. **The coexistence signal.** The spec caps Ghost at a 5% degradation of Sounding Line's
   calibration throughput. Sounding Line publishes a scheduler status with GPU-lock hours and
   per-stage estimated and realized minutes, and no calibration-throughput number. The governor
   therefore reads the companion's per-stage realized-over-estimated wall ratio and its GPU-lock
   wait from that file, records both in `RUNTIME.json`, and reduces Ghost workers when a stage
   running beside Ghost exceeds its own estimate by more than 5% relative to stages that ran with
   Ghost idle. If no Sounding Line stage runs during the window (none is running today and no
   Stage 5 directory exists yet), the record says so and the rule is vacuously met. Ghost never
   touches the GPU (accelerators hidden at worker start, as in V13).
4. **"Reader response."** In a constructed world the reader's own induced response is a planted
   reader-side appraisal function applied to the artifact's surface — a variable the reader owns,
   distinct by construction from the maker's appraisal and the intended audience effect, and
   crossable against both (X07). It is not a felt valence and is never scored as one.
5. **Equivalence-class scoring.** Where the spec declares a target an equivalence class (J01,
   J08, H02, H03, X03, X09), the score is posterior mass on the class and the calibration of the
   reader's uncertainty across it; a card that scores "the one true value" where several models
   are behaviourally equivalent is an instrument defect and its gate says so.

## What V13 supplies (copied under `v14/`, adapted; V13 untouched)

| V13 object | V14 use |
|---|---|
| `world.py` families, groups, makers, artifact channels, anomaly and regime machinery | the rendering layer; extended with the maker-state tuple (competence K, attention history H, preference V, episode goal G, plan Π, belief B, appraisals A^m/A^a, communicative goal C, reliability R, opportunity O). The poe-family sentinel semantics are kept and every card is smoked against a poe world (V13's crash class) |
| `exact.py` channel-factorized likelihoods, tempering, exact EIG | the likelihood engine under `joint.py`'s new grid; `Model.eig` is the foraging reference |
| `priors.py` matched routes (entropy + distance) | the local prior proposal (spec §2 row 1) and the equal-local comparator for I06–I08 |
| `attention.py` selection/precision policies, no-information nulls | route acquisition budget and the identity-at-neutral gate |
| `costs.py` opportunity records, cost vectors, actors | route acquisition and foraging costs only (spec §2 row 5); no motive inference from expenditure |
| `goals_trust.py` assertion/evidence/truth, reliability, uptake | the base of `communication.py`; the factorization is retained and extended |
| `hierarchy.py` role graphs, central/shared-brief twin, rewrites | H07's exact shared-brief rival and the interaction reader |
| `projection.py` correction rulers, robust readers, ensembles | I08's correction curve; E09's calibrated-likelihood intersection (never posterior averaging) |
| `cards/__init__.py` unit/reduce contract, seven-gate battery, cells, receipt, finish | unchanged in kind; the receipt keeps V13's "every cell in every unit" rule and is exercised at smoke this time |
| `manifest.py`, `schemas.py`, `common.py`, `atomicio.py`, `runtime.py`, `determinism.py` | queue manifest, tiers, expected cells, ledgers, lane seeds (`zlib.crc32`, never `hash()`), atomic working state, process-tree CPU |
| `runners/run_v13.py`, `watchdog_v13.py` | the resumable scheduler and death-resume, plus the 24-hour contract below |
| `runners/run_v13_confirmation.py` (binding freeze, recorded amendment, `--supersede`) | the confirmation runner, plus the hour-20 freeze and the ≤4 / one-per-family cap |
| `validate_v13_program.py`, `fresh_clone_v13.py`, `report_v13.py` | the read-only validator (lane-aware from the start), the clean-clone receipt, the final-only report with a deadline guard |
| `prereg_v13.py` three locks | `prereg_v14.py`: structural → workload → scientific, with `SOURCE_LINEAGES`, `CONSTRUCTION_IDENTITIES`, `ROUTE_INFORMATION`, `ATTACK_MATRIX` added to the hashed set |
| `tests/test_v13_*` | the seventeen required tests, seven of which exist in kind |

## What is new

| module | must contain |
|---|---|
| `joint.py` | the hypothesis grid over (plan Π, episode goal G, preference V) with process-equivalent plans declared; exact joint, three staged plug-in pipelines, independent marginals, oracle, and the cheap baselines; evaluation counting for compute matching; posterior trajectories after every observation (§4.2) |
| `routes.py` | four route observations rendered from the same artifact (action: transition and tool constraints; semantic: claims and task structure; context: role, convention, source history, opportunity; forensic: a costly optional observation with material/temporal/interaction resolution); planted ease; learned reliability from feedback without test labels; conflict detection; latent-set expansion (R05) with a search cost; fusion that models shared causes and duplicates (R07); reliability transfer/reset (R08) |
| `history_skill.py` | competence K (transition, tool, observation accuracy) and attention history H (attended features, practiced transitions, reward-linked selections) as independent generators with orthogonality gates; decay after reward reversal; cross-domain transfer matrices; acquisition-path signatures at equal skill (E08) |
| `communication.py` | the eight-way factorial of §3.3 (belief, support, desired appraisal, desired action, evidence-selection policy, willingness to correct, private action, intensity); appraisal owners (reader response, maker appraisal, intended audience effect); the audience-modelling maker (inverse-inverse) as an evidence-selection policy; habituation versus cumulative uptake (A09); the uptake gate (A10). Regime names label regions only; every discriminator predicts a counterfactual action |
| `hierarchy.py` (v14) | goal graphs with subgoals, instrumental goals, standing preferences and optional terminal rewards; potential-based and other policy-equivalent reward transformations; habits persisting after preference change; finite records with no terminal horizon; V13's role graph for H07 |
| `foraging.py` | items with planted novelty, complexity, compressibility, learnable error, learning progress, relevance and cost; the unlearnable-noise trap; policies (novelty, surprise, learning progress, EIG per cost, always-forensic, random, never); realized held-out gain, not an internal "interest" variable |
| `cards/trunk_{i,j,r,e,a,h,f,b,x}.py` | one callable per literal card id; 64 + 12; each causal card carries the seven gates; route cards add divergence and equal-ease/equal-accuracy; communication cards add surface collisions; hierarchy cards add reward-equivalence checks; foraging cards add the noise control |
| `runners/run_v14.py` | the 24-hour contract: deadline written at pilot start and immutable across restarts; hour-20 freeze; hour-24 stop-and-checkpoint; `SHORT_RUN` marker; the coexistence governor; no packet before the deadline |
| `runners/report_v14.py` | refuses to run before the deadline file says the window closed (test 15); reads every lane |

## Build order

Each phase ends at a boundary a fresh session could resume from; every phase commits.

1. **Skeleton and record (½ day).** `prereg_v14.py` with the 64 cards, 12 attacks, factors,
   lanes, independent-unit floors, criteria and repairs; `QUEUE_MANIFEST.json`,
   `EXPECTED_CELLS_TEMPLATE.json`, `SOURCE_LINEAGES.json`, `CONSTRUCTION_IDENTITIES.json`, the
   structural lock; the recursive validator (test 16) green on the empty program.
2. **Substrate (1 day).** `world.py`, `joint.py`, `routes.py`, `history_skill.py` with their
   identity tests first (tests 1–7): exact joint identities in exhaustive tiny worlds, compute
   equality across estimators, factor orthogonality, matched-surface and equifinality fixtures,
   route divergence and no-information closures, ease/accuracy crossings, competence/history
   swaps. Nothing scientific runs until these pass.
3. **Communication, hierarchy, foraging (1 day).** `communication.py`, `hierarchy.py`,
   `foraging.py` with tests 8–11 (owner non-aliasing, reward-equivalence non-identifiability,
   the noise trap, duplicate-evidence calibration).
4. **Cards (1½ days).** Trunk by trunk in the spec's order I, J, R, E, A, H, F, B, then X. Each
   trunk is smoked at six worlds on a scratch root **including one poe-family and one
   equifinal-history world**, with the receipt enforced at smoke (V13's two full-scale
   surprises). Every gate must be able to fail: no gate encodes an expected result.
5. **Scheduler, confirmation, reporting (½ day).** `run_v14.py` with the 24-hour contract and
   governor; `run_v14_confirmation.py` with the hour-20 rule and cap; `report_v14.py` with the
   deadline guard; `validate_v14_program.py`; `fresh_clone_v14.py`; the watchdog with deadline
   preservation; tests 12–17.
6. **Pilot and locks (2 hours).** Full scratch smoke of every card, attack, resume path and
   report suppression; the discarded pilot over one heavy card per trunk plus the largest joint,
   hierarchy and foraging rulers on the pilot lineage, measuring **end-to-end card wall time**
   (V13's per-unit power law over-forecast by 1.7×); the workload function selects the tier and
   expansions that forecast 20–21 hours; `WORKLOAD_LOCK.json`, `EXPECTED_CELLS.json`,
   `ROUTE_INFORMATION.json`, `ATTACK_MATRIX.json`, the scientific lock.
7. **The window (24 hours).** Discovery, transfer, the three V13 repairs, confirmation from hour
   20, closure at hour 24; internal checkpoints only.
8. **Closure (half a day, after the deadline).** Validator, clean-clone receipt, the final-only
   packet, `RESULTS.md`, the landing in `FINDINGS.md` and `READING_INTENT.md` §13 in one pass,
   B01/B02 ledgers.

Agent effort: about four working days of build before the pilot, then the window, then a day of
closure. The build is larger per card than V13's because every card carries a tournament and a
trajectory record, and smaller in count.

## The three V13 repairs (I06–I08)

Each runs once, preserving the original failed verdict beside the repair and the original
target. C03: the positive floor is re-derived from the construction's information (the expected
gain of the true within-common prior over the all-family prior at one artifact, computed
exactly), not lowered; if the repaired gate fails, the common-substrate card closes. C05: the
convergence placebo is made sufficient by construction (evidence dose at which both routes reach
the same posterior within tolerance, verified on the planted world); if routes do not converge
under sufficient evidence the mechanism closes. P01: the "near needs no more correction than far"
gate is restated as the spec intends (correction *rate*, not residual, per similarity bin) and the
calibration instrument scores the prospective endpoint; the target and bins are unchanged.

## Pre-mortem, mapped to the spec's list

| spec risk | design that prevents it |
|---|---|
| joint wins by more evidence or compute (1) | evaluation counting per estimator, equality asserted by test 2; identical observation lists by construction |
| a goal label leaks process or preference (2) | labels are never inputs; I05 leakage baselines at floor |
| retrospective fit called value recovery (3) | every J/E/H criterion scores a hidden future choice |
| ease relabelled reliability (4) | ease planted and crossed against accuracy (X02); reliability learned from feedback only |
| history defined as competence (5, 6) | independent generators with orthogonality gates; swap tests (X06) |
| reader appraisal becomes maker appraisal (7, 8) | owner variables crossed (X07); communicative goal and reliability are separate generators |
| fanatic/propagandist templates (9) | matched artifact and audience effect (X08); discriminators must predict a counterfactual action or abstain |
| distrust as understanding (10) | A08 scores selective true/false uptake, not suppression |
| confident unique histories (11, 12) | equivalence-class scoring; reward-equivalent alternatives generated, never omitted |
| noise wins curiosity (13) | the unlearnable-noise trap is a required control on every foraging card |
| naive averaging (14) | E09 admits only calibrated-likelihood or feasible-set intersection |
| PyMDP without a live action set (15) | PyMDP is not reopened; exact EIG is the reference |
| repairs overwrite failures (16) | V13 verdicts are read-only; repairs write under `v14/` with provenance |
| pooled means hide reversals (17) | conditional cells precede pooling; X11 checks every planned reversal |
| nominal rows (18) | the independent unit is the world/maker/reader as declared; row-order-invariant aggregation (test 12) |
| manufactured duration (19) | expansions only from the frozen list; `SHORT_RUN` otherwise |
| early packet (20) | the report refuses before the deadline; test 15 |
| human claims (21) | every sentence carries a ceiling; the validator greps the packet for the forbidden vocabulary |

V13's own lessons, added: smoke includes the rare families and enforces receipts; the pilot
measures end-to-end card time; the runner survives silent process death with the deadline
preserved; the validator and the report read every lane from the first commit; `git log` is
checked for another session's commits before editing runners.

## Questions before the build

1. **Judgment call 1 (staged = plug-in commitment).** This is the definition under which J04 and
   J05 measure something. If the curator intends a different staged estimator — for instance a
   soft handoff of the full stage posterior, which under exact inference collapses into the joint
   — say so now, because it changes the J trunk's criteria before the lock.
2. **Judgment call 3 (coexistence signal).** The per-stage realized-over-estimated ratio from
   Sounding Line's scheduler status is the only throughput signal it publishes. If Stage 5 will
   publish a calibration-throughput figure, name the file and the rule reads it instead.

Everything else proceeds under the defaults above.

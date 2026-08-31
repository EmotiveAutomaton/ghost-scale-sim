# V15 — The Boundary Map. Implementation translation

*This page is a build document, not a results page. Spec §9.1 forbids any result prose before the
168-hour deadline, so nothing here reports an outcome: it records what was built, what was decided
while building it, and where each decision lives in the code. The single curator packet is
[RESULTS_PACKET.md](RESULTS_PACKET.md) and it does not exist until the window closes.*

The handoff specification is [V15_SPEC.md](../../../V15_SPEC.md) at the repository root (filed
beside this page when the version closes).

---

## 1. What the program is

112 mandatory cards in twelve trunks, 24 cross-cutting attacks, and a locked balanced coverage
stream, run in one continuous 168-hour window. The question is a boundary, not a verdict:

> Under which combinations of latent coupling, evidence overlap, scarcity, history, context and
> model misspecification does a reader need a coupled and expandable maker model to predict what
> happens next — and when is a cheaper factorized reader equally good?

V14 answered a neighbouring question by comparing one coupled world with one factorized world and
finding a +0.011-nat advantage. That prices one easy-to-factor construction. V15 makes coupling,
overlap, dose, dependence structure, missingness, temperature, equifinality and maker–reader
similarity into continuous or crossed axes, and the estimand is a conditional surface.

## 2. Layout

| path | what |
|---|---|
| `ghostscale/validation/soundingline/v15/ontology.py` | the shared vocabulary and the knobs. No generative code. |
| `.../world_chain.py`, `world_composition.py`, `world_communication.py` | three independently written generator families |
| `.../exact.py`, `particles.py`, `expansion.py`, `architectures.py` | the eleven reader architectures and the budget accounting |
| `.../learning_history.py`, `foreground.py`, `persistent.py`, `strategic_source.py`, `routes.py`, `foraging.py`, `hierarchy.py` | the trunk subsystems |
| `.../causal_distance.py` | the audit that separates a readout, a planted signature and inference through behaviour |
| `.../coverage.py` | the locked balanced coverage stream |
| `.../runtime_contract.py` | the 168-hour window, the opening guard and the occupancy receipt |
| `.../cards/trunk_*.py` | one module per trunk, one `unit_<ID>` and one `reduce_<ID>` per card |
| `ghostscale/prereg_v15.py` | the three locks, the effect-size table and the instrument choices |
| `runners/run_v15.py` | the resumable, work-conserving scheduler |
| `runners/run_v15_confirmation.py` | the frozen-lineage confirmation |
| `runners/report_v15.py` | deadline-guarded, final-only reporting |
| `runners/validate_v15_program.py` | read-only validator |
| `tests/test_v15_{gates,metamorphic,runtime,fresh_clone}.py` | the suites |

## 3. The three things V15 had to remove from V14

Spec §2 names three V14 positives that were construction identities, and each is removed here in a
way that can be checked rather than asserted.

**Competence and history were supplied channels.** In V15 competence is not a parameter: it is the
measured accuracy of a learner trained by a randomized curriculum, and the reader never receives a
history feature. Card E01 is labelled `CONSTRUCTION_IDENTITY` and does nothing but establish that
four training mixtures can be brought to the same final skill, which is the precondition for the
rest of the trunk.

**Acquisition paths carried fixed planted signatures.** Curricula are randomized per maker — item
order, item frequency and which items are blocked all vary — so there is no planted class to
recover. The history estimator uses permutation-invariant behaviour features, because item identity
is not comparable across makers.

**The off-audience source action was generated from the hidden belief and read with the matching
likelihood.** In V15 the private action comes out of a planner with a cost, and motives are grouped
into *surface profiles*: sincere and strategic share one, mixed and contrarian the other. The
artifact therefore recovers the collision class and can do no better than within-pair chance on the
motive; only a purchased counterfactual probe separates inside a class. The belief prior is a
property of the profile, so the private channel cannot leak the motive either.

## 4. The rule that shapes every card

**A gate bar is never a criterion bar.** A gate asks whether the apparatus works — is the
manipulation live, is the placebo inert, does a fixture whose answer is fixed by construction
return that answer. A criterion carries the pre-registered magnitude. V14 conflated them on 26 gate
lines and recorded three small *real* effects as `INSTRUMENT_FAILED` at scale, because a gate
demanding the criterion's magnitude fails exactly when the finding is a modest true positive.

In V15 the separation is structural rather than conventional: `cards.battery` has no parameter
through which a magnitude could reach a gate, and `tests/test_v15_gates.py` fails the suite if any
committed gate carries a nonzero bar.

The same error recurred here in a second form and was caught by the smoke pass: three gates in
trunk M asserted the card's own *direction* — that the approximations are close to exact, that
relevant history helps more, that the direct predictor beats surface. A card discovering a negative
answer would have been filed as a broken instrument. Each is now a known-answer check.

## 5. Where the effect sizes came from

Spec §8.2 forbids reusing V14's 0.02-nat bar and asks for a fraction of a live positive control on
the same score. Before anything was registered, the construction's own spans were measured: the
distance from the cheapest reader to the state oracle is **0.30 nats** at the atlas's reference
settings and **0.54 nats** under scarcity. The architecture bar is **0.015** — five per cent of the
smaller span. Accuracy bars are fractions above each card's own chance floor. Every bar and its
basis is in `prereg_v15.SESOI` and repeated in each card's verdict.

## 6. Decisions made while building, and why

Each of these was settled by validating an instrument against a known answer, never against an
outcome. They are recorded in `prereg_v15.INSTRUMENT_CHOICES` and repeated here because they are
choices.

| decision | why |
|---|---|
| coupling is a **marginal-preserving mixture** with the weight solved to a target mutual information | a log-linear tilt is not monotone in mutual information: past a point the prior collapses toward a point mass and coupling falls back to zero, so a bisection walks the wrong way and returns degenerate worlds |
| the phase axis is **realized** coupling, not the nominal knob | the three families reach different ceilings (chain 1.04 nats, composition 0.63, communication 0.59) by different constructions |
| overlap is measured against a **uniform reference prior** | measured under the world's own prior it reads 0.89 at zero overlap as soon as coupling is on, which makes the two axes of the surface impossible to separate |
| the independent rival is the **marginals of the factorized-prior posterior** | assigning home routes per component while leaving the shared action channel in each one triples the policy evidence and puts the exact posterior *behind* an approximation, which is impossible for a correctly specified Bayes posterior and was the tell |
| PID is **exact Williams–Beer I_min, two sources**; three components use **exact Shapley over subsets** | I_min has no agreed unique extension past two sources; Shapley is unique, additive and computable over 2³ subsets |
| self-directed practice carries a **0.40 outcome visibility** | at zero, practice cannot improve past its own first guesses and cannot be skill-matched to an instructed history, which is E01's precondition |
| foraging selection is proportional to **value ^ 2.5** | proportional to raw value, every controller collapses onto the random floor on a nine-item ecology; greedy makes the reported avoidance an artifact of the tie-break |
| a changepoint-aware controller **discounts its stale counts** on detection | a detector that fires and keeps averaging over pre-change observations cannot re-engage: ten new observations sit underneath ten stale ones |
| G01's surface collision is built by **rejection** | switching's action marginal is a mixture of softmaxes and simultaneous control's is a softmax of a blend — different families, so no parameter search closes the gap in an arbitrary world |
| the coverage stream is **not materialized** | at the size the opening guard requires it is hundreds of megabytes; the locked definition plus a hash chain over block digests regenerates and verifies any block without storing it |

## 7. The runtime contract

One immutable UTC deadline at `start + 168h`, inherited by restarts. Hours 0–150 discovery,
transfer, attacks and the coverage stream; hour 150 freeze; 150–166 confirmation on untouched
lineages; 166–168 clean clone, aggregate regeneration and reconciliation; the single packet after
168.

The opening guard (`runtime_contract.opening_guard`) refuses to start unless the core's
conservative upper forecast fits under 150 hours, the core plus locked coverage survives a machine
three times faster than the pilot, confirmation and integrity have their own reserved worker-hours,
the cells and ordering are hashed, and the recovery and early-report guards pass. Card I08 and
attack X24 both feed it deliberately bad fixtures.

**Why the queue cannot empty.** V14's scientific queue emptied after 6.8 wall hours and the runner
then waited fourteen hours for a freeze. Here, when the card queue is exhausted the scheduler
dispatches the balanced coverage stream, which is sized so a three-times-faster machine still could
not finish it. If it empties anyway, or workers wait for the deadline, `RUNTIME_FAILED` is set and
there is deliberately no softened form: spec §9.4 permits the results to be reported after that and
does not permit the seven-day contract to be claimed.

**Launch is always module form.** The sibling project's orphan sweeper kills any python whose
command line matches `runners[\/]run_`. It killed V14's runner seven times, cost that program its
first window, and retro-explains four of V13's "unexplained" silent deaths. `python -m
runners.run_v15` does not match, and attack X24 reads the launcher on disk to check the fix has not
rotted.

## 8. What the validator refuses

`runners/validate_v15_program.py` is read-only and safe to run against a live program. It refuses:
a manifest that does not enumerate all 112 cards and 24 attacks in the spec's trunk counts; a
causal card with no declared hidden event; a `LANDED` verdict with no criterion evaluated; a gate
carrying a magnitude; a `SIMULATOR_DISCOVERY` claim whose causal-distance audit caps it lower;
any of the forbidden vocabulary in `validate_v15_program.FORBIDDEN_VOCABULARY` anywhere in the
record — the words that would make a constructed-world result read as evidence about people,
medicine, nervous systems or history; and **an export that names a card as unrun while the committed record
has it resolved**, which is the V14 stale-bridge failure made mechanical.

(The list is deliberately literal and deliberately excludes `diagnostic`, which is house
vocabulary: V14's validator matched the stem and flagged its own legitimate usage. This page is
scanned by that check like everything else, which is why it names the constant rather than
spelling the entries out.)

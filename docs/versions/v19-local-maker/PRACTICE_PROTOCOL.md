# E1: persistent production learning and exact replay

Question: does choosing training actions improve later production beyond the
same observed transitions and identical persistent updates, and how does a
restricted starting-state distribution change the result? Exact replay is an
identity control. Any active-versus-exact-replay difference is an instrument
failure. Active-versus-demonstrator differences may reflect coverage and policy.

This is an intervention on the admitted three-unit local world. The learner
chooses one of its three existing local goals before the native goal-conditioned
operation executes. The execution actor is fixed at meaning-oriented, high skill,
correct belief and no presentation routine. The learner is explicitly assigned
functional success: final claim on with matching evidence. It is not inferring
the actor's purpose or learning the original goal-selection law. Native edit,
repair, tool, presentation, inspect and undo semantics remain unchanged. Conditional
operation probabilities use each admitted world's unchanged action rate. No new
goal ontology or claim of learned human expertise is introduced.

The learner retains transition counts over step (3), initial context (2), current
artifact (8), previous artifact/undo buffer (8), selected goal (3), and next
artifact (8): 9,216 floating-point counts. Every count starts at one half. After
each observed transition, increment exactly one count. There is no optimizer,
latent teacher, hidden-maker feature, or learned sequence model. Within an episode,
the undo buffer and current step are observed. Domain action names and the assigned
success function are public; the world action rate is evaluator-only.

Plan from the empirical conditional transition frequencies by exact backward
induction over the remaining three-step horizon. Ties choose the first goal in
the fixed meaning/dependency/presentation order. Active training uses epsilon
greedy selection with epsilon 0.20 and uniform exploration. Evaluation is greedy.
Recompute planning values after every observed transition. This is a finite
model-based learner; planning computation and the entire count table are charged.

Compare active acquisition, offline replay of the active learner's exact ordered
transition log, and off-policy observation of the native actor's goal choices.
All have the same initialization, count update, capacity, feedback count and
checkpoint schedule. Replay updates immediately in the original order and must
match after every transition, not just the last checkpoint. Demonstrations use
native local-goal weights and conditional operations, with the same exogenous
operation coins and initial contexts as the active arm. Different selected goals
can therefore produce different paths despite paired randomness.

Freeze training budgets at 32, 128 and 512 episodes, each containing exactly three
feedback transitions. In the matched-start condition, alternate the two admitted
initial artifacts. In the restricted-start condition, use only the first. Test
both initial artifacts with equal weights, requested functional purpose fixed.
Report each initial context separately as well as the equally weighted average.
The restricted-start condition changes support, not the number of feedback items;
do not assume a monotonic mismatch effect. Because initial context is a feature,
the unvisited context in the restricted condition retains its prior table.

Use the first eight development coefficient lineages, both declared training draws,
and both initial feature-seed identifiers as policy random seeds. These seeds
control acquisition, not neural initialization. No test or confirmation lineage is
used. Exogenous operation and policy coins have separate fixed namespaces. Pair
arms, conditions and budgets within lineage/draw/seed. Intervals resample lineages,
with training-draw and policy-seed variation reported separately.

At every checkpoint retain all count tables, greedy policies, exact production
success under the native stochastic transitions, the exact-law optimal reference,
an untrained-table baseline, and a uniform-goal baseline. Measure both total
visited state-goal count and coverage weighted by the exact optimal test-policy
occupancy. Save full observed training transitions and exact-state evaluation
records. The reader role contains observed states, chosen goals and after-states
only, with anonymous content-derived stream names. Coefficient identities,
action rates, oracle policies, probabilities, scores and compatibility metadata
stay in the scientific/evaluator role.

Before outcome inspection require: native transition equivalence for every
artifact, undo buffer, step and goal; known controllable success; an uninformative
action placebo; exact replay equality after every update; no-update identity;
and exact transition mass. Original plus adjacent and extracted-source replays
share E1's four-CPU-hour card. This tabular task consumes zero shared tiny-model
settings. It does not supply a frozen neural latent state for the primary contrast
or establish local-process reconstruction.

## Independent prepared alternatives

A2 remains the fixed-size original/random/diversity/training-residual portfolio
comparison in [the prior protocol](BOUNDED_RESET_PROTOCOL.md), with ridge grid
0.001/0.01/0.1 and a five-hour cap; it is not yet implemented or admitted.

R3 remains the [retained whole-stream purchase calibration design](../v18-selective-acquisition/exploratory-loop/NEXT_DESIGN_2026-09-20.md):
calibrate raw surprise and surprise minus predictive entropy on 3,072 supplied-law
streams, then freeze thresholds before 1,920 separate evaluation streams. Both
sensors drive the same paid expansion, with unchanged references, copied-source
identity and scalar threshold/forecast controls. Compile its disjoint input
inventory and controls before dispatch, retaining its three-hour V19 cap. It is
independent of production learning and tiny reader capability; the current
registry is not an implemented handler. Earlier elapsed-window limits remain
historical and are not reset or reused.

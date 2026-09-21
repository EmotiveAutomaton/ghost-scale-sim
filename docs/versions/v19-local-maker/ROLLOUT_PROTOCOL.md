# Learned forward transitions and extra computation

Does composing learned transitions predict a native endpoint better than a direct
table trained on the same paths, and does increasing rollout count merely reduce
sampling error? This F1 opening comparison concerns forward prediction conditional
on a proposed operation sequence. It is not historical process reconstruction.
All arms receive the initial artifact, the three proposed legal operations, and
the maker's skill and belief bits as declared mechanics metadata. Supplying these
bits deliberately isolates transition learning from hidden-maker inference. No
goal, purpose, routine, coefficient law, final artifact or intermediate state is
an evaluation input. Query sequences come from native paths, so their distribution
is policy-selected; this is not a randomized intervention on operation selection.

Freeze original training lineages 193000–193031 and development lineages
190000–190007. Two training draws, 190301 and 190302, each select 64 complete
paths per training lineage under native probability weights. Interleave the
lineages before taking nested totals 32/128/512/2048. Retain every selected path,
its three transition labels and endpoint. Both estimators receive exactly these
paths. No new sequence-model setting, test or confirmation lineage is consumed.

The forward model is a categorical count table indexed by skill, belief, current
artifact, undo buffer and operation. Its eight possible next artifacts have unit
pseudocounts. Undo-buffer evolution is supplied public state bookkeeping: the
next undo buffer is the current artifact. This mechanism is shared with the
known-law reference. The direct model is a count table indexed by skill, belief,
initial artifact and the entire operation sequence, with the same eight-output
unit pseudocounts. It can inspect the same transition labels but its frozen
endpoint estimator uses the endpoint sufficient statistics. Capacities and
factorizations differ; no capacity-matched superiority claim is available.
Unvisited rows stay uniform. There is no optimizer, hyperparameter search or
development selection.

Eight fixed arms: direct probability table; 16 sampled direct endpoints; 1, 4,
and 16 learned forward paths; exact finite propagation of the learned table;
16 forward paths under a wrong-law control; and exact known-mechanics execution.
All Monte Carlo forecasts use the same total-one uniform endpoint pseudocount,
(counts + 1/8)/(sample count + 1). The 1/4/16 forward arms share prefixes of
16 deterministic random paths. Direct sampling uses the same final-step uniforms;
wrong-law sampling uses the same full uniforms but toggles presentation in each
sampled transition. This is a specified disturbance, not an exhaustive law search.
Exact learned propagation separates estimator error from Monte Carlo error and
its count-dependent smoothing. The direct sampling arm separates additional
sampling work from forward-model structure. Sample draws are conditional
computational randomness, not independent training initializations.

Enumerate the full native development population, group identical query inputs,
and verify a unique endpoint per query under the supplied mechanics. Score
endpoint logarithmic loss in nats and squared probability error, with natural
trajectory weights within lineage, then equal lineage weights. Preserve all
forecasts, true endpoints, per-query mass, state visitation, transition/direct
counts, two training draws, and component times. Compare paired differences and
10,000 coefficient-lineage bootstrap resamples with seed 190501; report draw
ranges separately. This finite-architecture development comparison is exploratory,
not confirmation. The query labels and all native paths are evaluator-only;
training paths have an explicit auxiliary-supervision role.

Require controls before fitting: deterministic inspect/undo/repair/tool execution,
skill restriction, correctly evolved undo state, uniform unseen table, normalized
tables and exact propagation, known deterministic-model endpoint, 1/4/16 shared
prefixes, direct Monte Carlo accounting, wrong-law disturbance, proper scores,
and a complete isolated fixture with strict reader fields. Freeze source, design,
environment and schedules before outcomes. Use the existing serial worker and
both full replays. Initial F1 cap is five CPU hours, within the ten-hour F-family
allocation and the unchanged shared exploration pool. Failures and verification
count. This is a transition-learning pilot, not completion of broader F transfer.

Two alternatives remain concretely prepared, requiring their own admission.
The G-P1 joint-process readout uses frozen raw/latent/bank features, equal complete
goal-operation labels, a training-only categorical alphabet, explicit unseen-label
mass, all four nested budgets, five feature seeds and two draws. The D-family
composition alternative crosses the two already implemented tool semantics with
an independently specified evidence-repair semantics, tests held-out combinations
against original/singly expanded/jointly expanded supplied families, and preserves
the difference between empty support and compatible omission. Freeze that rule
and its finite population before execution. Neither depends on F1 or C1 winning,
and neither is yet an implemented handler. No automatic F2 promotion follows.

# Persistent learning helps production; active acquisition has no demonstrated advantage

Active practice showed no advantage over observing demonstrations. After 512 episodes with both starting contexts represented, production success was 86.15% for active learning, 87.20% for demonstrations, and 49.19% without learning. Exact replay matched active learning throughout. This is a constructed-method result, not evidence about intent.

The question is whether choosing training actions improves later production beyond
the same observations and updates, and whether starting-state coverage changes that
comparison. A persistent learner chooses one of the existing three local goals,
then observes a native operation's consequence. The executor has high skill and a
correct belief; its local-goal choice is intervened on, not inferred. Functional
success means a final claim-on artifact with matching evidence. This is a production
learning task with assigned success, not local-process reconstruction.

Each arm retains the same 9,216-count transition table with one-half pseudocounts
and the same finite-horizon planner. Active acquisition uses fixed 20% uniform
exploration; demonstrations use the native actor's policy. Replay sees the active
arm's exact ordered transitions and makes identical updates. Every episode gives
three next-artifact feedback items. Eight development coefficient lineages, two
training draws and two acquisition seeds are paired; none is a confirmation lineage.

The table reports mean exact production success as percentages, evaluated equally
over the two initial contexts. Rows give training episodes. Columns distinguish
training on alternating contexts from training only on the first context. Active
and replay results are identical in every cell; higher success is better.

| Episodes | Both contexts: active/replay | Both contexts: demonstration | First context only: active/replay | First context only: demonstration |
|---|---:|---:|---:|---:|
| 32 | 73.97 | 74.64 | 60.59 | 58.81 |
| 128 | 80.75 | 81.53 | 62.07 | 62.57 |
| 512 | 86.15 | 87.20 | 67.25 | 68.18 |

The untrained table scores 49.19%, the uniform-goal baseline 42.87%, and the
known-law optimal planner 98.37%. At 512 episodes, active-minus-demonstration
success is -1.06 percentage points, with a 95% paired lineage interval
[-2.61, 0.59] when both contexts are represented. Restricting starts gives -0.92
points [-2.19, 0.21]. These are inconclusive contrasts, not proof of equivalence.
The four matched-context draw/seed contrasts range from -4.18 to +3.32 points;
three favor demonstrations and one favors active acquisition. Restricted-context
contrasts all favor demonstrations, ranging from -1.33 to -0.42 points.

Against the untrained table, active learning gains 36.96 points [35.57, 38.18]
with both contexts and 18.07 [17.08, 19.12] with restricted starts. Matching initial
support improves average success by 18.89 points [16.86, 20.93] for active learning
and 19.02 [16.80, 21.23] for demonstrations. Intervals resample the eight paired
lineages conditional on these draws and acquisition seeds; repeated episodes are
not independent worlds. No binary success practical margin was predeclared.

The restriction is substantial: context is an explicit table key. The unvisited
context keeps all pseudocounts and its original policy, scoring 40.50% in every
restricted arm at 512 episodes. In the visited context, active and demonstration
success are 94.01% and 95.85%; with matched starts their per-context scores are
90.43%/81.86% and 88.69%/85.71%. Thus restriction concentrates experience on one
context and prevents transfer to the other by construction. It does not isolate a
general embodiment or covariate-shift mechanism.

Coverage weighted by the known optimal policy's evaluation occupancy reaches
87.56% for matched active acquisition and 99.29% for matched demonstrations;
restricted values are 48.36% and 50.00%. This records which useful state-goal
combinations were visited; it is not an intervention proving coverage mediates the
small active/demonstration difference. The retained counts and full logs make a
support diagnostic possible without more sampling. Both methods still trail the
known-law optimum. No monotonic mismatch law is established from two conditions.

**Validation:** independent native conditional transition construction, backward
induction and forward occupancy reproduce all 576 saved tables/policies, values,
coverage statistics and score rows. All 192 observed logs contain 294,912 transition
rows including replay, or 196,608 newly acquired rows. Active and replay ordered
logs are identical; identical initialization and one-count updates establish state
identity at every prefix. All paths have native support and restricted-context
tables retain their prior. Independent score discrepancy is at most 3.4e-16.
Known controllable-success and action-inert fixtures pass. Both complete replays
match 1,355 deterministic files, with timing bound separately.

An initial review assertion requiring identical policies across different numerical
reduction orders failed and is retained. Across 576 tables, 3,271 state choices
differ at near ties; their maximum learned action-value gap is 1.2e-16. The final
review evaluates the original saved policy unchanged, independently verifies its
values and does not replace it with a newly selected policy. This leaves possible
sensitivity of acquisition to floating ties as a limitation. It does not change
the exact replay identity within the frozen arithmetic. The review reconstructs
logged updates, not an independent rerun of every acquisition random coin.

**Warrant:** exploratory production-learning advantage against an untrained table,
inconclusive active-versus-demonstration contrast, and exact matched-update identity.
Miniature — architecture untested; no neural capability, inverse intent, historical
process or human-expertise conclusion. **Pursuit:** retain support mismatch as a
specific explanatory rival. The tiny-reader capability comparison remains core;
fixed-size portfolio and independent purchase-calibration designs remain prepared.

[Scientific evidence and separate reader export](../../../results/v19/G19-E1-practice-1/EXPORT_MANIFEST.json).

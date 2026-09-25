# Finite context precision allocation: frozen implementation contract

Implements the prepared context-precision protocol without changing its
population or budgets. The capacity is exactly 81 assignments of half, single
or double precision to four contexts. The lexical tie order is float16,
float32, float64 within context order 0,1,2,3. All minimizing assignments are
retained, and the first is identified only for deterministic display.

The teacher is the supplied law, used only in evaluator/scientific records.
There are no fitted labels, training schedules or new observations. Eight
retained development laws use the same sixteen states and eight endpoints.
Each stored context uses 256, 512 or 1,024 bytes. The three law budgets are
1,280, 1,536 and 2,048 bytes; posterior state and query schedules remain
separate costs. Independent actual-array byte counts reject undercharging.

Original and stored law rows are separately normalized once in double precision.
Per-context ranges use the maximum supported endpoint log-likelihood range.
For each assignment, sum those four ranges and multiply by observations per
context (2, 8 or 32). The posterior total-variation bound is tanh of one quarter
of that sum. Minimize the unrounded log-range sum, avoiding false ties from
saturation of the bound. Equality of computed sums defines numerical ties;
this is not interval-arithmetic certification. Retain every assignment, its
exact byte count, all three sums/bounds, feasible uniform-precision baselines,
selected ties and dominance by cost/range. A dominated assignment can remain a
minimizing tie if cost differs but its bound does not.

Positive reference support lost in storage yields an infinite range and a
vacuous bound of one, not a fabricated floor. If a rounded observation has
zero probability under every state, its posterior is undefined. No attainment,
reachable-error or unrestricted optimal-compression claim is admitted.

Required controls include independent exhaustive scalar enumeration, equal
contexts and exact cast, infeasible budgets, a probability-per-byte greedy
counterexample, context permutation, support loss and corrupted byte charges.
Source-bound full-support synthetic timing must cover all assignments, raw
arrays, serialization and original plus two complete executions. Independent
reconstruction and all original-row regroup are reserved within the separate
1,800 CPU-second cap. Prior jobs and snapshots remain unchanged.

# Precision regret: frozen implementation contract

Implements the prepared minimax log-range regret alternative. Retain the same
eight development laws, all 81 four-context assignments of half, single and
double precision, byte budgets 1,280/1,536/2,048, and four permutations of query
counts (13,1,1,1) scaled to lengths 16/64/128. No new fit or observation is used.

For each assignment and query-count vertex subtract the best feasible log-range
at that vertex. Select every feasible assignment minimizing its largest vertex
regret. Use exact computed-float equality and lexical half/single/double ordering;
record independent scalar tie sensitivity. Infinity minus infinity is undefined,
and assignments with any undefined regret are ineligible. An all-undefined
comparison has no winner. Actual retained-law arrays must be finite to report.

Retain all absolute robust, balanced and vertex optima, all minimax-regret ties,
every vertex regret, absolute bounds and byte cost. Report the best and worst
values over competing tied sets, with uniform feasible baselines. Teacher access
is the supplied evaluator law; architecture is exhaustive finite enumeration,
capacity is 648 assignments across eight laws with four vertices and three
lengths. No learned architecture, training schedule, test lineage or tiny setting.

The criterion is log-range regret against a query-informed feasible allocation.
Its maximum over the specified convex hull occurs at a vertex because it is
a maximum of linear contrasts. This statement does not extend to arbitrary
nonlinear posterior-loss regret, realized inference loss or unseen query sets.

Controls precede retained outcomes: independent exhaustive scalar reconstruction,
all denominator-four convex mixtures, context permutations, zero regret,
infeasibility, undefined infinities, and a known disagreement between absolute
minimax and minimax regret. Complete-support synthetic timing, source freeze,
original and two full replays, independent numerical review and full write-through
must fit the separate inclusive 1,800 CPU-second cap within family/global limits.

Robust source-prior retention and finite allocation stability margins are two
separate prepared alternatives. No earlier cap is enlarged or failed run erased.

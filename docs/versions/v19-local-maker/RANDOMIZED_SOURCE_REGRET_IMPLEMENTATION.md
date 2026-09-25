# Randomized source-prior regret: frozen implementation contract

Compare maximum minimum expected source coverage with minimum maximum expected
opportunity loss over the same finite library. Use the existing 28,672 source
rosters, 28 distinct byte-budget problems, three supplied priors and 58 masks.
No new observations, fits, protected lineages or tiny settings are consumed.

Architecture and capacity: enumerate exact rational basic lotteries with support
sizes one through three. Solve normalization and active prior-regret equalities
by fraction Gauss-Jordan elimination, reject negative weights and violated
constraints, deduplicate full weight vectors, retain every optimal basic lottery,
and select the lexicographically largest full weight vector in ascending mask
order. The prior-specific optimum is the retained unrestricted feasible optimum;
the lottery itself ranges only over the frozen mask library. Optimal basic
solutions do not enumerate the continuum of all optimal mixtures.

Teacher inputs are evaluator-only source priors, structural byte costs and the
verified prior-specific optimum/library. No reader input or training schedule.
Run the complete paired population once, followed by adjacent and extracted-source
full replays. Emit 57,344 rows with all rational decisions retained separately.
Record expected regret for each prior, its maximum, coverage-lottery regret,
deterministic minimax regret, minimum expected-mass cost, worst realized regret,
expected byte cost and maximum realized byte cost. Every positive-weight mask
must separately satisfy the budget. The prior is fixed before the draw.

Controls precede outcomes: strict randomization advantage, coverage/regret
disagreement, identical-prior and empty/full identities, degenerate ties,
prior/item permutations, invalid parent optima/regrets/costs, and independent
full-constraint rational enumeration. Complete-support synthetic timings must
include all rosters, libraries, rows, serialization and hashes. The separate
inclusive cap is 1,800 CPU seconds including failures, controls, three executions,
independent reconstruction, original-row regroup and documentary work.

Independent review reconstructs source rosters, costs, prior-specific optima,
all basic lotteries using a separate solver, every weight and regret, realized
bytes, tie choices, every original row, paired/equal-law summaries and both
complete replays. This is a finite-library decision comparison; forecast accuracy,
learned access, historical process correspondence and human intent are untested.

Precision-choice stability and realized-risk-constrained storage remain prepared
independent alternatives. Their admission is separate.

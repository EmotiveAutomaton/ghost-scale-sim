# Realized-risk-constrained channel regret: executable contract

Does limiting the coverage loss in every realized outcome change the expected
regret benefit of contingent storage lotteries? Freeze 28 retained libraries,
15 correctly supplied mixtures, cue reliability interval [1/3,1], original
structural byte budgets and the complete population. No new fits or observations.

Each deterministic three-label policy receives an exact expected coverage line.
Its realized coverage loss is the largest difference from the best unrestricted
library mask for a positive-prior latent source, over labels possible anywhere
in the interval. This support includes all three labels because 1/3 is included;
zero-weight latent sources do not contribute. The coverage-optimal fixed mask
defines the first loss threshold. The other thresholds are halfway from that
threshold to one, and one. A fixed-mask tie selects the largest mask identifier.

Exclude policies violating each threshold before optimizing. Keep the original
unrestricted informed reference at both endpoints and at two-thirds reliability.
Enumerate every singleton and strictly interior two-policy lottery equalizing
the two endpoint regrets. These contain an optimum of the finite minimax problem;
the continuum of equivalent lotteries is not enumerated. Worst expected regret
over the interval occurs at an endpoint because the informed reference is a
maximum of affine functions and a lottery's score is affine. A selected lottery
must satisfy the realized-loss threshold and byte budget in every supported draw.

Use exact common-denominator integer arithmetic for score and regret comparisons.
Store all policy masks, rational scores, endpoint regrets, realized losses and
charges. Store every candidate as two policy indices, first-weight numerator and
denominator, lower and upper regret numerators, and their shared denominator.
Retain all exact ties; select the largest lexicographic (first policy, second
policy, first weight) among them. Reuse a stored candidate table only where two
thresholds have identical surviving policy lists. Retain excluded policies and
explicit level-to-table references; this changes storage, not the comparison.
Report expectation and realized loss separately. No positive benefit is presumed.

All 57,344 roster/budget rows join to 15 mixtures and three thresholds, giving
2,580,480 joined evaluations over 1,260 distinct library/mixture/threshold problems.
Retain paired and equal-law populations for independent review. Reader inputs:
none. Numerical acceptance requires independent ancestor reconstruction, every
policy and candidate, exact tie and threshold, expected and realized costs, all
joins and regrouping, source/input/environment bindings and both full replays.

Controls cover known risk tradeoff, singleton and identical-prior nulls, ignored
zero-weight sources, full candidate equality with a separate Fraction solver,
an independent convex hull, nested feasible sets, unaltered informed reference,
prior/policy permutations and corrupted charges. Complete-support synthetic timing
precedes scientific admission. The prospective inclusive cap is 1,800 CPU seconds,
including setup, failed attempts, controls, original, two replays and review.
This finite supplied-law method establishes neither forecast accuracy nor process
correspondence. The previously blocked randomized-channel attempt stays retained.

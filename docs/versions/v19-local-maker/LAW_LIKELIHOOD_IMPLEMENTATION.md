# Supplied-law likelihood envelopes: frozen implementation contract

Use all eight retained development observation laws, sixteen maker states, four
query contexts and eight endpoints. Cast original bytes to float64, float32 and
float16, then normalize each cast row once in float64. Normalize the original
reference once by the same convention. Retain both normalization changes.

For each supported context/endpoint, retain all positive-reference-state ratios
of rounded to reference likelihood, logarithms, extrema and their range. Original
zeros remain excluded support. A cast zero on positive reference support yields
an explicitly unbounded envelope, posterior total variation at most one. Empty
rows fail. No floor, fitted value or observed score selects an operation.

If a positive likelihood ratio is between a and b, its normalized tilt changes
a distribution by at most (sqrt(b)-sqrt(a))/(sqrt(b)+sqrt(a)). To see this, let
z be the mean ratio. Convexity of absolute deviation bounds half its normalized
mean absolute deviation by (b-z)(z-a)/(z(b-a)). Maximizing over z gives z=sqrt(ab)
and the stated bound. A two-point distribution on the extrema attains it, with
reference mass sqrt(a)/(sqrt(a)+sqrt(b)) on b. Constant ratios give zero.

For trajectories with the same prior and transition law, the likelihood ratio
is a product of observation ratios. Its log range is no larger than the sum of
per-observation ranges. Marginalizing trajectories cannot increase total
variation. The frozen conservative envelope uses lengths 8, 32 and 128 times
the largest context/endpoint range, and applies tanh(total range/4). This is a
uniform upper bound, with no claim of reachable or attained worst-case error.

Architecture is finite arithmetic on supplied laws; capacity is 24 complete law
variants, 72 original summary rows and every statewise ratio. There is no fit,
outcome teacher or training schedule. Original and two complete replays, scalar
ratio reconstruction and equal-law regroup are mandatory before acceptance.
Controls include rational prior grids, the analytical attaining prior, exact
likelihood products, exact-cast and constant-ratio nulls, support loss, impossible
observations, invalid rows, permutations and an undersized-bound failure.
The inclusive cap is 1,800 CPU seconds with complete-support synthetic timing.
Protected lineages, shared training settings and earlier blocked caps are unchanged.

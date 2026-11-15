# Bounded posterior and hidden-ledger repairs

P's four frozen convergence levels all failed after the compiler repair.
The largest split R-hat was 4.35 at the last level; minimum effective sample
size was 4.25. None of those posterior estimates is accepted. The owner
explicitly approved one further numerical repair within the original allocation.

The repaired conditional update integrates out each page mixture analytically,
using the Dirichlet–multinomial likelihood of latent label counts. The primary
density uses the integer rising-factorial identity; the independent density
uses grouped log-gamma ratios. The prior, labels, observation model and all
accuracy gates remain unchanged. Ten joint Metropolis moves update the maker
mix per latent-allocation sweep; page mixtures are regenerated afterward,
the required partially collapsed ordering. Adaptation still ends before
retained draws. Rerun the same finite validation ladder under a new capsule,
at most 1,800 CPU seconds. No further automatic P repair follows from failure.

V's first hidden-reader compilation failed because integer subtraction promoted
a shutdown array to a wider integer type. Its first packet-authorized repair
uses a consistent 64-bit shutdown array. Model, seeds, observations, filter and
resolution gates remain unchanged. The failed capsule is retained and charged.

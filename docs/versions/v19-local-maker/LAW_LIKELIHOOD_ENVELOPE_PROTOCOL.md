# Rounding and accumulated likelihood: prepared independent alternative

Can a small supplied-law forecast error still permit appreciable posterior
distortion after repeated observations? The one-step absolute probability bound
does not answer this relative-likelihood question. Use the same eight retained
development laws and float64/float32/float16 storage; no new fit or history.

Normalize the original and each cast positive row once in float64, retaining
both original bytes and normalization changes. For each context/endpoint, retain
every positive-reference state's likelihood ratio and the range of its logarithm.
Reference zeros are support exclusions; cast zeros on positive reference support
make the bound unbounded and must remain explicit. No probability floor.

For lengths 8/32/128, multiply the largest per-observation log-ratio range by
length. The maximum posterior total-variation bound is tanh(range/4), capped by
one through that formula. This compares inference using the same prior and same
transition law with exact versus rounded observation likelihoods. It is a uniform
bound over possible observations and latent trajectories, not a claim that its
worst case is reachable or that actual errors attain it. Total variation is half
the sum of absolute differences between normalized state probabilities.

Before admission, prove the two-point extremal ratio formula, check it against
independent rational-prior enumeration and explicit likelihood products, retain
constant-ratio and exact-cast zero controls, support-loss and empty-row controls,
state/context/endpoint permutations and an undersized-bound failure. Retain all
statewise ratios and extrema, original and two complete replays, independent
scalar reconstruction and equal-law regroup. No learned-access, historical
process correspondence or human-intent inference follows.

Prepared, unimplemented and unadmitted. The separate inclusive ceiling is 1,800
CPU seconds within the existing family and global ceilings, including controls,
failures, all executions, reconstruction and documentary work. Full-support timing
and source/plan freeze are required. No blocked card's cap is extended.

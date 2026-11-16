# S.b owner-approved numerical repair R2

The owner approved one additional bounded log-calculation repair on November 16,
2026, after the first repair failed because this Windows NumPy build gives
`longdouble` the same exponent range as ordinary floats. Both previous failed
excluded preparations remain retained. No S.b scientific draw preceded this repair.

The independent posterior update now adds scalar log likelihoods to retained log
weights, subtracts their maximum, and normalizes using `math.fsum` over the
rescaled exponentials. It returns log weights directly, preserving finite support
even where exponentiating an individual normalized weight would underflow.
Production continues to use its existing SciPy log-sum calculation. Initial
priors, the Bayesian model, grid, seeds, matched parameters, policy, observations,
scientific thresholds and comparison tolerances are unchanged from
[S_LEARNING.md](S_LEARNING.md). R1's source is retained separately.

Freeze the source and this protocol before testing. Require the original known
answers plus finite-support, normalization, independent-update and additive-log
shift checks at log weights spanning 2,000 nats. Reconstruct each distinct retained
failed fixture and compare with its original measures at the original tolerances.
Then run the unchanged eight-world, 400-trial largest-cell profile, its complete
seed replay, and independent reconstruction. All gates must pass. Validation
uses at most 600 CPU-seconds and 1,800 wall-seconds, charged to healing; a failure
is retained and does not authorize another attempt.

Only after passing, admit the original 64 S.b settings under S_LEARNING_R2:
four odds, two costs, two temperatures and four makers. Each retains forty
excluded pilot worlds, 200 scientific worlds in eight blocks, full seed replay
and independent reconstruction. Use the measured profile to reserve cell cost
inside S's unchanged ceiling. The already-declared COST-learning follow-on
remains conditional on its frozen practical-vanishing trigger and separate
validation. This repair admits no other descendant or additional repair.

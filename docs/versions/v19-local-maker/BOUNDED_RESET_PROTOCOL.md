# B2 bounded reset and horizon successor

Question: after one roll-in pass destabilized free-running prediction, can a fixed
history-based reset limit that error, and does either frozen updater remain useful
at a longer horizon? This is the named bounded-reset continuation of B2. It does
not tune or refit the failed head, and it does not assume that a reset will help.

Retain B1/B2's eight development lineages, two training draws, two feature seeds,
eight streams per lineage, original 32-observation prefixes, frozen one-step and
rolled-in heads, initial mean bank, direct-history head and probability repair.
Append 96 reset episodes for each same stationary maker from its retained endpoint
law. Tail randomness is keyed by lineage, draw and stream under the fixed
`v19-B2-horizon-tail` namespace. Independent and copied conditions share that tail;
copied observations repeat each adjacent even-position source. This changes the
horizon and reset intervention, not the number of independent worlds. Test and
confirmation lineages stay untouched. Every original 8/32 forecast must reproduce.

Compare free one-step and rolled-in updating, each with and without a reset, the
frozen cached-history predictor, exact filtering, and the two privileged
teacher-forced diagnostics. Before processing each new source after eight more
unique sources, replace the reset arm's bank with the frozen direct-history head
evaluated on the preceding public evidence cache. Never use an exact bank for a
reset. Copies do not update the bank or trigger resets. There is one reset interval,
eight unique sources, with no outcome-based interval selection.

Retain forecasts at all 128 positions and score paired 8/32/128 prefixes. At these
positions the scored forecast precedes the next reset: it is not the cache output
immediately substituted by a reset. The count feature retains its original
normalization by log(33), so values beyond 32 unique observations extrapolate the
training range. Report that numerical extrapolation separately from drift.
Teacher outputs on copied positions remain exact pass-throughs, not learned scores.

Save all extended streams, exact reference targets, reset flags, pre-reset cache
forecasts, method forecasts and invalid raw-probability flags. Charge a maintained
32-bin cache and extra direct-head evaluations to resetting. Report sunk fitting
cost by reference; this packet performs zero fits. Do not infer an end-to-end
speedup from algorithmic counts or aggregate diagnostic timing.

Admission requires known-answer update/copy/reset controls, retained-prefix and
retained-forecast identities, source/input/environment hashes, and full adjacent
and extracted-source replay. This packet and both replays share B2's original
seven-CPU-hour cap, including the already consumed roll-in work. No family or
global allowance is enlarged. A negative result ends this single reset strategy;
no interval search follows without a new identified scientific cause.

## Independent prepared alternatives

**A2 portfolio:** use the old informative world class and retain the equal-class
negative controls. Compare original, fixed random, diversity and training-only
targeted banks at identical entry count and observable label budget. The targeted
rule ranks entries using only training residual covariance; the law-aware optimum
is a separately labelled upper reference. Freeze one ridge grid (0.001, 0.01, 0.1),
paired development selection and complete held-out law protocol before fitting.
This addresses error direction and query relevance, independent of recurrent
capability. The five-CPU-hour opening A2 cap applies. It is a prepared design,
not an implemented or admitted handler.

**E1 persistent practice:** retain one fixed-capacity learner, initialization,
optimizer, example order, feedback and reset rules across a learner's active
trajectory and its exact offline replay. Identity of weights and predictions is
the first positive control. Compare that pair to a separately declared off-policy
demonstrator at equal feedback; vary two frozen support-mismatch levels and report
coverage explicitly. A difference between exact replay and the active trajectory
under identical updates is an instrument failure. This tests distribution coverage
and persistent learning, independent of predictive-bank success. Freeze the policy,
training schedule and new-source ownership before fitting; initial cap four CPU
hours. It is a prepared design, not an implemented or admitted handler. It must
not silently consume a shared tiny-sequence-model setting.

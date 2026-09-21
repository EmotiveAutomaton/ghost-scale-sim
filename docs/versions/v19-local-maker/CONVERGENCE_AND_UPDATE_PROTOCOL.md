# Bounded convergence comparison and recursive-state screen

These two independent successors are frozen before their outcomes. The first
addresses the nonzero final gradients in the probability-head diagnostic. The
second opens B1's recursive-state comparison on repeated episodes of the existing
local world. Neither consumes a tiny sequence-model setting, selects test outcomes,
changes the local-goal ontology, or replaces an earlier result.

## Same-objective convergence diagnostic

Reuse every input and representation of G19-A-probability-head-1. Keep zero
initialization, categorical cross-entropy averaged over training histories, L2
penalty 0.01 on non-intercept coefficients, all labels, budgets and populations.
Replace only the fixed gradient schedule with SciPy L-BFGS-B: at most 1,000
iterations, 2,000 function evaluations, gradient tolerance 1e-7, relative function
tolerance 1e-12 and at most 30 line-search steps. Save returned solver status,
gradient norm and objective traces; a successful library status does not establish
that the gradient tolerance was reached. No best-checkpoint selection, penalty
grid, feature refit or training-data change. Preserve the predecessor's unfloored
softmax scoring for this direct optimizer contrast and keep its floor diagnostic.

Report changes in training objective, final gradients, every target/evidence/class
curve and paired lineage areas. Persistent poor held-out performance at a small
gradient narrows the problem to this penalized objective/features, not an abstract
information limit. Improved training without held-out gains separates optimization
from generalization. Cap: four CPU hours including controls and both full replays.

## B1 repeated-episode predictive updating

The scientific question is whether a learned bank can update from one new
observation without accumulating more prediction error than a same-label direct
or recurrent reader. A stream consists of 32 independently reset three-operation
episodes by the same persistent maker in the existing local world. This extends
the number of observed episodes, not the episode horizon or its local goals.
Contexts are drawn uniformly; only each context and final artifact are observed.
An additional copy condition repeats every second episode's public evidence and
source identity. All arms know source identity and skip repeats; copies are not
treated as independent evidence. Evaluate paired prefixes of length 8 and 32.

Use the first 32 frozen training coefficient lineages, the first eight development
lineages, two training/sampling draws and two feature seeds. Generate eight streams
per lineage and draw in each condition. Training and development lineages are
disjoint; test and confirmation lineages remain reserved. Save the complete
conditional endpoint law for each maker/context and all streams. The exact filter
starts uniform over sixteen makers, multiplies the likelihood for each genuinely
new observation, and produces four eight-category fresh-episode forecasts.
Full transient enumeration produces this law; no old 24-state algebra is used.

This is an explicitly auxiliary, oracle-teacher comparison: every learned arm
gets the same training-prefix exact observable forecast distributions. The bank
updater additionally receives the preceding exact predictive bank during one-step
training. At evaluation it starts at the mean training prior bank and feeds back
its own output; a separately labelled teacher-forced evaluation receives the exact
previous bank as a privileged error-accumulation diagnostic. These are distinct
input conditions; a recursive advantage cannot be attributed solely to architecture.
No maker labels, local goals, law coefficients or evaluation identities are fitted.

Compare full-history recomputed counts, exactly matching cached counts, a learned
bank updater, a fixed recurrent reservoir with a learned forecast head, a constant
training-label mean and the privileged exact filter. The direct feature vector
contains 32 context/endpoint frequencies and log(1+unique observations)/log(33).
The updater uses the previous 32 bank entries, current 32-way observation and the
same count scalar. Both use fixed 128-coordinate tanh maps padded to 512 inputs.
The reservoir has 128 coordinates, fixed Gaussian input weights and recurrent
weights scaled to spectral norm 0.5; it changes state only on a new source.
Every fitted output head has 129 coefficients per one of 32 categories, using
ridge 0.01 with unpenalized intercept and the same fixed probability repair.
Fit once on all new-source prefixes; no target or checkpoint selection.

Report endpoint forecast logarithmic loss, squared probability error, invalid raw
forecast rates, paired prefix/condition/lineage differences and seed/draw variation.
Count retained state and full-history versus cached work separately; component
CPU times are measurements within native accounting, not extra charges. A matched
cached/full-history identity is a control, not evidence for recursive superiority.
This does not test joint reconstruction of the preceding local process. Skill and
purpose changes and rolled-in fitting are prepared B2 discriminators after the
stationary/copy comparison, not implicitly executed in B1. Cap: five CPU hours
including admission and both full replays. Known-law repeated likelihood, duplicate
identity, informative/no-information observations and reader-field exclusions must
pass before dispatch.

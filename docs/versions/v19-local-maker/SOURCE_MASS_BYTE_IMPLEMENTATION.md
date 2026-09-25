# Source mass per byte: implementation choices before outcomes

The existing protocol is unchanged. Use exact rational source weights, represented
by a shared integer denominator. Sparse dynamic programming stores the best
integer mass and recent-time bit vector for each attainable integer byte count.
The largest vector resolves equal-mass ties exactly. Greedy policies skip items
that do not fit; zero-byte positive-mass items remain selectable.

Retain every original fixed-byte source roster and its law/evidence/draw identity.
The input projection contains source times and structural costs only: endpoint
values, posterior probabilities and measured prediction errors cannot select a
policy. Reconstruct structural costs from the frozen complete schedules and
preserve the parent bindings. Identical deterministic allocation problems may be
cached, but every original row remains in the output and equal-law summaries.

Translate both original budgets into the weighted remainder schema: the usable
source capacity stays identical, while all policies pay 56 additional fixed bytes.
Thirteen floating values (eight copy masses, four remainder masses and one source
normalizer) replace twelve integer counts. Group masses, thirty-two shared-prior
floats, per-source metadata and joint-table entries remain counted consistently.
The common half/quarter capacities are the smaller of the original recent and
midpoint-roster byte totals. They are not fixed source counts.

Architecture and capacity are finite exact integer allocation; there is no fit,
teacher outcome or training schedule. The admitted population must keep eight
laws, both evidence conditions, both source draws and all seven available schedule
checkpoints. Independent review must rebuild costs from structural sets, use a
separate exact subset method for every distinct allocation, check every selected
time and byte identity, and regroup all original rows before numerical acceptance.
Native and portable handlers are implemented; admission awaits controls, complete
synthetic-support timing and immutable source/plan freeze. The separate 1,800 CPU
second inclusive cap and existing family/global ceilings are unchanged.

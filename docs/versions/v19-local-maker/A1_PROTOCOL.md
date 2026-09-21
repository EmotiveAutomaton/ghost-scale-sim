# A1: state error before learned-law fitting

Use retained coefficient indices 0 through 7, all sixteen old cells and both
state-support halves in each of the two pinned training arms. Use retained fit
seeds 701 and 809 for question-only, direct, flat and split readers. This mechanical
reuse substitutes existing fit identifiers for the new-fit seed namespace in the
initial registry. No fit, history or coefficient is chosen by its measured effect.
The first two retained fits were chosen before this leaf's new outcomes.

For each history compare the exact posterior bank, each retained learned bank,
and zero-mean Gaussian bank error scaled to the learned bank's squared error.
Noise is centered separately within each probability vector. Scaling uses the
evaluator posterior: this is a supplied-law diagnostic, never a deployable reader.
The same noise direction is paired across the two training arms. Probability
repair follows the existing fixed clip/nonnegative normalization rule; record
error before and after it. Do not claim matching after nonlinear repair.

Both readouts consume the same bank: tighter-tolerance (1e-14) supplied linear
readout with fixed probability repair, and supplied-law convex-hull projection.
Retain invalid raw linear probabilities and hull optimality gaps. Score the same
two farther questions and actual retained state, with loss and Brier score separate.
Report all four world classes and the equal-class population. The exact reference
continues to use the stipulated uniform 24-state prior, not the neural allocation's
optimal prior. This is a diagnostic rather than G-P1 or a new training replication.

The ceiling is three CPU hours inclusive of this leaf's work. Classwise error
sensitivity directs A2 portfolio/regularization work; parity directs B1. Wide
uncertainty does not choose a winner. Preserve two fitting seeds within lineage;
neither seeds nor repeated questions count as independent worlds.

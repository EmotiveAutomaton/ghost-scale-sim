# Asymmetric source cues: frozen implementation contract

Does uncertain directional error change which cue-contingent storage policy
protects against missed source probability? This implements the prepared
asymmetric-channel protocol. Retain the same 28 storage libraries, 15 supplied
source mixtures, feasible masks and byte budgets. No new observations or fits.

For source j, label j has probability r, label j+1 modulo three has probability
(1-r)b, and label j+2 has probability (1-r)(1-b). Reliability r is unknown in
[1/3,1]; directional bias b is unknown in [0,1]. Enumerate every deterministic
three-label policy over the feasible library. A policy chooses one feasible
mask for each observed label, with no coupling across labels.

Minimize maximum expected coverage regret relative to the channel-informed
optimum over that same unrestricted policy library. Each pairwise policy-score
difference is multiaffine in r and b. Its maximum on the rectangle is at a
vertex; maximizing also over comparator policies commutes with this finite
vertex maximum. Preserve all four vertex scores, including duplicated perfect
channels, every policy, byte charge, optimum tie and greatest lexicographic
representative. The vertex proof applies to this supplied channel family.

Compare the selected robust policy with the accepted symmetric-channel robust
choice, the nominal symmetric two-thirds choice, and optimal fixed storage.
Retain each arm's worst regret and signed differences. Diagnostics evaluate all
nine combinations of r=1/3,2/3,1 and b=0,1/2,1: expected mass, expected regret,
gain over fixed storage, expected bytes, maximum bytes on positive-probability
events and maximum realized coverage loss relative to the known-source optimum.
Expected regret and realized coverage loss remain distinct quantities.

Preserve the 57,344 roster/budget rows joining to all 15 mixtures, 860,160
evaluations and 420 distinct decision problems. Preserve source pairing, both
draws and equal weight for all eight coefficient lineages. Known supplied
priors, mixture, channel family and structural costs are evaluator information;
there are no reader inputs, learned labels or protected-lineage use.

Controls cover directional reversal, exact rectangle optimization with interior
checks, the symmetric restriction reproducing its reference, point-channel
identity, identical priors, zero-prior support, relabeling, byte corruption and
duplicate perfect vertices. Require full-support synthetic timing, isolated
controls, immutable source/input/plan/environment, independent reconstruction
and both complete replays. Inclusive prospective cap: 1,800 CPU seconds.
The existing single-worker queue owns execution. This is a constructed-method
decision comparison; no forecast, process-correspondence or human-intent claim.

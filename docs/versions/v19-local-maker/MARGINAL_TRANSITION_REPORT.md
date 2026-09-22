# Current maker state and future changes

We tested whether knowing the maker's current state is enough for later prediction. Equal current states can yield different future endpoint forecasts under all eight supplied laws. Independent enumeration of 113,376,896 hypothesis-pair/time instances and 19,968 forecast summaries, plus both complete replays, verifies this constructed-method counterexample. The witnesses are not established reachable posteriors; historical process correspondence and human intent remain unestablished.

The fixed roster contains sixteen stationary hypotheses and all purpose/skill
changes after steps 8 through seven steps before the horizon: 560 hypotheses
at 32 observations and 3,632 at 128. At checkpoints 8/16/17/20/32 we partition
every hypothesis by its current and future maker, for every remaining step.
Each cross-product block retains every unordered pair sharing its current maker
but differing later. The 29,472 blocks represent 318,624 pair/time instances
at horizon 32 and 113,058,272 at 128. Reusing laws, contexts and hypotheses does
not create independent evidence samples. No new observations or fits occur.

The table reports each horizon/checkpoint, its number of pair/time comparisons
multiplied by eight development laws and four contexts, and the weighted mean
total variation: half the sum of absolute differences between endpoint
probabilities, from zero for identical predictions to one for disjoint ones.
This is a finite descriptive average over all retained pairs and times, with
their multiplicities; it is not a sampling-confidence estimate.

| Horizon, observations | Starting checkpoint | Pair/time instances across laws and contexts | Mean forecast distance |
|---|---:|---:|---:|
| 32 | 8 | 3,899,392 | 0.258275 |
| 32 | 16 | 2,562,048 | 0.232627 |
| 32 | 17 | 2,295,808 | 0.230694 |
| 32 | 20 | 1,438,720 | 0.225750 |
| 32 | 32 | 0 | not defined (no pairs) |
| 128 | 8 | 795,172,864 | 0.248269 |
| 128 | 16 | 744,683,520 | 0.244443 |
| 128 | 17 | 737,830,912 | 0.244002 |
| 128 | 20 | 716,624,896 | 0.242722 |
| 128 | 32 | 623,552,512 | 0.238187 |


Every eligible comparison exceeds the fixed 1e-12 maximum-coordinate display
tolerance; none is an exact binary64 forecast alias. Same-time and stationary
controls produce zero different-future pairs. The largest observed total
variation is 0.417566, retained descriptively rather than selected as a primary
witness. Uniform-law fixtures preserve forecast aliases despite distinct states.

Independent review reconstructs the entire roster and every membership block
with direct tuple transitions. A separate complement-of-within-group pair count
and explicit small-roster expansion check completeness and absence of duplicate
unordered pairs. Scalar endpoint arithmetic rebuilds all 3,840 law/state-pair/
context scores, all 19,968 time summaries, and the ten rows above. The maximum
discrepancy is 1.11e-16. Six isolated producer/reviewer controls pass, including
omission, membership and score corruption. Both full replays match all 23
deterministic outputs. Plans, 495-source archive bindings, environment, parent
law inputs and output hashes verify. The parent law's own numerical validation
is inherited, not a fresh audit of its likelihood construction.

**Warrant:** a constructed-method counterexample to a current-state-only
guarantee on the full simplex of complete hypotheses. Point-mass witnesses
need not be reachable under the admitted prior and observation histories.
Numerical differences are not symbolic-law certificates. This does not show
that every useful reader needs the whole history, or that learned banks fail.
Miniature — architecture untested. No historical-path or human-intent claim.

**Pursuit:** the next frozen comparison groups saved posterior weights by their
complete remaining state schedule and measures forecast preservation and cost.
Nine isolated controls pass; original and both full replays are source-admitted
with dispatch pending at this publication snapshot. Bounded changed-primitive
feedback and retrospective-evidence updating remain independent prepared
alternatives. The existing resource ceilings and untouched confirmation reserve
remain fixed; this report does not close the week.

Reader inputs are unchanged. SCIENTIFIC.zip contains the complete factored pair
enumeration, laws and summaries; VERIFICATION.zip and VERIFICATION_SOURCE.zip
contain independent reconstruction. These are evaluator/scientific reproduction
archives, never reader evidence. [Primary evidence](../../../results/v19/G19-B-marginal-transition-1/README.md),
[protocol](MARGINAL_TRANSITION_PROTOCOL.md), [next comparison](FUTURE_SCHEDULE_QUOTIENT_PROTOCOL.md).

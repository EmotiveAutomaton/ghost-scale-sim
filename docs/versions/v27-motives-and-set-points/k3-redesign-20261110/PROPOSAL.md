# K3 redesign proposal: inference before production

Prepared 10 November 2026. **Proposal only; implementation and execution await
owner approval.** V27 remains closed. This is a proposed separate allocation,
K3-RD1, and does not admit a fourth V27 healing or any of its 34 held cards.

## Decision requested

Approve a **48-CPU-hour validation project** to build a reader that conditions
its hidden-state sampling on observed choices, and test whether it is accurate
and affordable enough for the original dynamic question. Keep Gear 2's maximum
of four single-thread, below-normal workers and its host-load reduction to two.
No GPU or environment synchronization. Production would require a second,
costed approval after validation; this proposal commissions no scientific draws.

The question stays: **can restoring actions distinguish a moving set point from
dulling sensitivity or increasing gain, and do external probes help?** Changing
the numerical reader must preserve the world, observation channel, rival models
and scientific criteria. Success here would validate a constructed method, not
establish a human mechanism or validate an engine.

## Why another particle-count increase is inadequate

The [retained H3 result](../../../../results/v27/K3_H3_RESULT_2026-11-09.json)
failed the unchanged 0.25-nat numerical comparison bar. A nat is a unit of
logarithmic evidence. At 524,288 particles, the two runs differed by 0.374715
nats in the dulling-versus-alternatives comparison at record 509. Their largest
type-posterior difference was only 0.008027, and their crossing classifications
agreed. This is numerical instability, not evidence for or against the theory.

The failed fixture was a sharpening world with probes every sixteen records
on average. Its worst discrepancy concerned a low-probability dulling rival.
That explains how a small probability difference can coexist with a failed
log-evidence check. It does not justify dropping this negative control or
loosening its tolerance after inspecting the outcome.

The last fixture consumed 4,739 CPU-seconds, predominantly for two new
524,288-particle likelihood passes; lower resolutions were retained. A rough
planning calculation for 15 dynamic settings, 16 whole-world blocks, four makers
per block, and original/replay/independent passes is:

`4,739 / 2 × 3 × 15 × 16 × 4 / 3,600 = 1,895.6 CPU-hours`.

This extrapolates one fixture, assumes similar pass costs and excludes extra
reader work; it is **not a measured production forecast or a lower bound**.
It nevertheless shows why the existing approach needs a cost gate. On those
assumptions, meeting the former 50-hour K3 studies ceiling would require about
38-fold improvement. More workers reduce elapsed time, not CPU expenditure.
The calculations and source bindings are in [EVIDENCE.json](EVIDENCE.json).

## Recommended reader

Use a fully adapted auxiliary particle filter tailored to the finite consumption
law. Each need has three possible consumption events. For a two-need state,
there are only nine joint next-loss branches; three needs have twenty-seven.
Enumerate those branches, weight them using the current observed action and
restoration size, and sample from that conditional distribution. Enumerate the
initial stock distribution exactly: 441 possibilities for two needs and 9,261
for three. Retain exact static likelihoods wherever the existing recursion is
tractable.

This is a change in how the same likelihood is estimated. The implementation
must preserve the original timing: previous restoration and consumption with
clipping, current observed shock, current dynamic update, then current choice
and restoration size. No future observation or realized hidden consumption may
guide the current step. Merge identical branch destinations by summing their
probabilities; never merge merely nearby floating-point states.

The methodological basis is observation-informed importance sampling and
auxiliary filtering, described by [Doucet, Godsill and Andrieu (2000)](https://www.cs.ubc.ca/~arnaud/doucet_godsill_andrieu_sequentialmontecarloforbayesfiltering.pdf)
and [Pitt and Shephard (1999)](https://shephard.scholars.harvard.edu/sites/g/files/omnuum7741/files/filterSim.pdf).
These papers motivate the algorithm, not the theory. Full adaptation still
approximates the history distribution with finitely many particles; it does
not guarantee that K3's long-history errors disappear. The proposed advantage
for this particular simulator is an inference from its nine-branch transition,
to be measured rather than assumed.

Record per-hypothesis prefix likelihoods, effective sample sizes, ancestry
diversity, support loss and CPU cost. Keep separate deterministic numerical
seeds for hypotheses, replicas and population sizes. Retain the current common
random-number baseline for comparison; demonstrate that any coupling preserves
each candidate's marginal law. Neither observation noise nor likelihood floors
may be introduced to make the primary easier.

## Design corrections required before admission

1. **No-probe aliasing is a prediction to test.** The packet's pre-check at old
   deficits does not prove equal distributions over complete action histories.
   Report separation without probes as a finding that challenges the prediction,
   not as an invalid instrument. Retain the exact static restore-only
   band/lower-edge identity in its stated scope; do not generalize it to every
   changing-precision law.
2. **The clock is evaluated retrospectively.** The existing `sustained` statistic
   finds the earliest prefix after which the margin stays above one nat through
   the observed horizon. Preserve that statistic and its censoring. Do not call
   it an online stopping time. Derive the first twenty-record deficit from
   evaluator-only state for all five laws; the current P0/P+ implementation
   discards its working streak. Check that S/P-/G event times remain identical.
3. **Implement the declared noisy-stock channel explicitly.** `stock_noise`
   appears in the registry but is unused by the K3 generator and reader. A later
   noisy-visible-stock card needs actual observed stock samples and their
   likelihood, with a noise-zero control and a demonstrated effect at noise one.
   It cannot be counted as built because the flag exists. This is outside the
   primary and must not leak exact stocks into its hidden reader.
4. **Validate each expanded observation law.** Overshoot creates exact size
   constraints; three needs change branch count and hypothesis count; noisy
   stocks change the observation likelihood. A validation pass on two needs,
   zero overshoot and noiseless actions does not admit all textures.

These are proposed successor requirements. Completed V27 science, claim IDs,
failed repairs, original protocols and source capsules remain byte-for-byte
preserved. No old scientific conclusion is reversed by this proposal.

## Validation and bounded work

The table allocates the proposed new CPU budget. Every phase includes its
failed attempts, native child costs and relevant replay work.

| Phase | CPU-hour cap | Required output and stop condition |
|---|---:|---|
| A: implementation and exact controls | 12 | Independent scalar reference, causal observation interface, exact initialization and branch normalization; stop on unresolved correctness or resource failure |
| B: excluded numerical validation | 24 | All fifteen original primary configurations plus one fresh excluded world per configuration; stop at the first failed frozen validation gate or cap |
| C: cost and coverage assessment | 8 | Full original/replay/independent cost probes and explicit dispositions for all 34 held cards; no primary or texture production |
| D: verification and report | 4 protected | Reproducible validation package, costs, limitations, public-safe report and a production recommendation |
| **Total** | **48** | **Validation only; no automatic extension** |

The 48 hours would be a separately approved allowance. Unspent hours in closed
V27 are not automatically reopened, and its preparation overrun is not erased
or moved. Keep the original outer dates as limits for this proposal: no new
jobs after 15 November at 8 p.m. Pacific, drain by 10 p.m. A late approval may
leave only a partial validation package. Four workers give a twelve-hour ideal
compute floor for the full allocation; dependency and implementation time make
one to two elapsed days a planning estimate, not a completion promise.

Exact controls must exercise **dynamic** updates, not just the static reader:
enumerated two-need prefixes within a declared state cap, long one-need reset
fixtures with analytically determined histories, clipping, the twentieth deficit
record, precision floors/caps, moving-edge updates and zero-support cases.
Noiseless visible-stock identities remain separate ceiling controls. Independent
code must reconstruct transition weights and normalization without invoking the
production kernel. A cap-exceeded exact reference is unavailable, not passed.

Use old excluded fixtures 0, 1, 5, 8, 11 and 14 for bounded calibration of the
candidate. Proposed calibration populations are 1,024, 4,096 and 16,384 with
four independent replicas. Before computing validation outcomes, freeze source,
the selected two adjacent resolutions, the complete fixture roster and seed
namespaces. Old fixtures are regressions, not untouched validation. The fifteen
fresh excluded fixtures are generated only after that freeze. None enters
scientific sample counts.

At each of the selected resolutions, use eight independent replicas. Preserve
the original scalar tolerances and require every pair of deployed-estimator
runs across these two levels to meet them:

- Maximum dynamic-type posterior difference: 0.05 across all recorded prefixes.
- Maximum S/P-/G comparison-margin difference: 0.25 nat across all prefixes.
- Sustained crossing times: within sixteen records, with identical sides of the
  128-record practical bar and compatible censoring.
- Retain static absolute log-likelihood checks at 0.1 nat, independent step
  checks at 1e-12 and original/reference path agreement at 1e-9 where applicable.

All replicas are retained. No failed replica may be omitted or averaged away.
Exact controls also compare absolute likelihoods so that agreement between
similarly biased approximations is not the only test. Report uncertainty in
numerical inference separately from variation between simulated worlds. A
validation failure produces a diagnosis and stop; larger populations or another
algorithm are not automatically authorized by this allocation.

Phase C opens only after the relevant primary validation passes. Freeze cost
probes for no probes, both positive probe rates, and the largest intended
settings. Count all hypotheses and all reader passes. Keep each native job at
most twelve projected CPU-hours and eighteen wall-hours, with measured CPU
enforcement and recoverable records. A row for every held card must say
validated, unvalidated, unsupported, resource-deferred or outside this scope.
This prevents a primary-only pass from being presented as a fully built queue.

## What would run after a later production approval

First preserve the full two-need dynamic primary: five generating laws, all
three probe rates, sixteen independent whole-world blocks and four makers per
block. Keep the hidden reader, all five rival laws, the same set-point prior,
the 512-record horizon and the original scientific bars. P0 and P+ remain
negative controls. Scientific seeds are fresh and disjoint from calibration and
validation. Retain complete replay and independent reconstruction.

Admit texture, precision, follow-ups or rereads only through individually
validated, measured cards. Retain the packet's shed order if a later production
allocation cannot cover everything: three needs first, then the highest probe
rate. Any such omission must be visible in the approved production manifest;
do not quietly narrow the primary. Existing completed static comparisons are
referenced, not repeated for occupancy.

## Alternatives and failure interpretation

An exact compressed-state solver remains a reference/research alternative, but
the observed state explosion makes it an unpriced full-scale replacement.
Rounded states, shortened histories or a coarser dynamic world would change the
problem and require another explicitly named study. Increasing the old particle
count alone has an unfavorable measured cost. Relaxing the failed margin gate
after seeing its result is not recommended.

The main risk is that hidden-history impoverishment, rather than one-step
proposal quality, dominates. In that case the proposed filter may still fail
or exceed its budget. A failed validation would establish a computational
limitation, not lack of distinguishability in the world. A successful validation
would still not establish the theory: the subsequent primary must run and may
find that probes are unnecessary or that none of the readers reaches the bar.

## Handoff and reporting

Approval would create a new immutable manifest and separate controller state;
the old V27 closeout and stopped monitor remain preserved. While the new work is
active, keep fixed six-hour reviews plus earlier failure, depletion and
actionable phase-boundary wakes, with one review owner. Use the repaired exact
event handoff and record each review's report. Local CLI execution is not proof
of a message delivered into this chat; user-visible delivery must be stated and
verified separately. Retire the new finite monitor when validation ends.

The approval deliverable is this proposal. The next deliverable, if authorized,
is a validation-and-cost report with a concrete production decision, not an
automatically running replacement campaign.

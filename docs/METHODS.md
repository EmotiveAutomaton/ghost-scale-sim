# The methodology layer

**What `ghostscale/methods/` is for, why each piece exists, and what it costs to run.**

Every module in here was added because something specific went wrong. None of it computes a
finding; all of it computes a reason to believe or disbelieve one. If a piece of this stops
earning its place, delete it; scaffolding that survives on ceremony is worse than no scaffolding,
because it makes a project look checked.

---

## The rule that makes it safe to add all this

> **Nothing in `ghostscale/methods/` may be required to reproduce a published number.**

`ghostscale/v1` through `v10` are closed: pre-specified, run, reported, left alone. What makes
them reproducible is a deliberately small core dependency set, now fixed by the committed
`uv.lock`; long-term reproducibility still depends on Python and platform compatibility and on
upstream artifacts remaining available. So every third-party dependency here is an **optional extra**, and every
module that leans on one degrades to a recorded skip rather than an exception:

```bash
pip install -e .                 # reproduces every published result. Unchanged.
pip install -e ".[methods]"      # + PID, sensitivity analysis, catch22, FDR, equivalence
pip install -e ".[explore]"      # + tsfresh, arviz, dabest
pip install -e ".[dev]"          # + hypothesis, for the metamorphic tests
```

A skipped gate counts as passing and is **reported separately**, so a skip can never be mistaken
for a check that succeeded.

---

## Gates: `methods/gates.py`

Five kinds of standing control, recorded in verdict JSONs under `"gates"`. Three verdicts carry
none: S-1, S-45 and S-6, under `tests/test_gates.py`'s named exemption for modules that audit
somebody else's statistic rather than running a manipulation of their own. The exemption is
arguable for S-1 and S-6 (both construct planted data, which is a manipulable arm), and tightening
it would mean retrofitting gates and re-running both; recorded here rather than hidden.

| kind | the relation | the real defect it is aimed at |
|---|---|---|
| `placebo` | a manipulation at **zero** strength reproduces the control **exactly** | a side channel drawing from the rollout's RNG, which moved depth recovery 0.163 → 0.282 on nothing; and a `1/3` that is not uniform in floating point, which flipped an argmax on a near-tied posterior |
| `live` | a manipulation at **full** strength **changes** the output | S-2, whose goal mixture was drawn and discarded; feature streams were bit-identical with it off |
| `positive` | a task with a **known answer** returns it, through the whole stack | the gap this repository had. N28 says the instrument does not fire on nothing; only a positive control says it fires on something |
| `identity` | a quantity with a provable symmetry or bound satisfies it | nothing that broke, which is why it is worth keeping. T-1's two goal↔depth edges agree to 3e-16 because both reduce to a symmetric conditional mutual information |
| `no_oracle` | a statistic does not move when a label it should not see is permuted | S-3's threshold, fitted on labelled test data; E45's efficiency result, where the reader held the world's own emission map |

### A gate records; a test fails

This is the whole enforcement design and it is deliberate.

- A gate that **raises mid-run** kills a 300-second sweep over a control that may have been
  expected to fail. That trains people to switch gates off.
- A gate that only **prints** gets ignored. That is how S-2 shipped.

So gates write into the verdict and the run continues. `tests/test_gates.py` then walks every
committed verdict and fails the suite on any broken control, putting the hard stop at the moment
a result would become public, which is where one was needed and missing.

### Three categories, and they are not the same thing

```
failed_names         a control that broke. Stops a release.
documented_failures  a gate marked expected_to_fail: a KNOWN defect, recorded so the
                     evidence travels with the result instead of living in a commit message.
unexpected_passes    a documented defect that has started passing. Somebody fixed it and did
                     not update the gate, the one nobody would otherwise notice.
```

Two gates are currently `expected_to_fail`, and both are load-bearing documentation:

- **S-2 `mixture_reaches_the_reader`**: observed **0.0**. Forcing the mixture off changes nothing,
  because `V5Environment.sample_feature` ignores `artifact.goal` once a creator is bound. This is
  why the module is withdrawn, and the number is now attached to the verdict permanently.
- **S-3 `threshold_is_fitted_on_test_labels`**, the detector threshold is the median of the pooled
  *labelled* divergences, re-fitted per cell. T-4 re-scores it frozen: the headline rise falls from
  +0.125 to +0.046.

A module with a documented failure must also carry `withdrawn`, `WITHDRAWN` or `QUALIFIED` at the
top of its verdict, and `test_documented_failures_are_explained` enforces that, a known defect has
to be legible to someone reading the first screen, not only to someone who scrolls to the gates.

---

## Provenance: `methods/provenance.py`

Every verdict now carries which module wrote it, at what content:

```json
"produced_by": {
  "module": "ghostscale/validation/soundingline/t1_triangle.py",
  "sha256": "3f2a…",
  "git_commit": "b4f5612",
  "git_dirty": true
}
```

`git_dirty` is the field that earns its place: a result from a dirty tree cannot be reconstructed
from the commit alone, and knowing that at read time is the difference between "I can check this"
and "I think I remember".

---

## Separation, not `excludes_zero`: `methods/overlap.py`

`excludes_zero` answers *is the interval on one side of zero*, which is a question about
**precision**. At n = 200 paired rollouts almost anything separates from zero.

The motivating case is in this repository. T-1's `goal→process` edge at µ3/β1.0 returns **+0.0017**
with `excludes_zero: true`, and it is 0.3% of the `process→depth` edge on the same scale, it flips
sign one cell over, and the budget-matched version flickers across duty cycles. A reader has to be
told all of that in prose to know the flag means nothing.

The overlap coefficient is the shared area of the effect and null densities, in [0, 1], unitless;
so a result in nats and a result in AUC are directly comparable. Scored against the strongest null
available, and **the null is named in the output** so a strong claim can never be quoted off a weak
one:

```
placebo  → the manipulation at zero strength. Controls the harness.
permuted → random values through a likelihood that still claims fidelity. Controls the content.
swapped  → another artifact's true values. Strongest: everything but correspondence held fixed.
```

**Two calibration mistakes were made building this, and both are worth knowing about.**

First, the overlap was computed on per-unit values. On those, T-1's *strongest* edge
(`process→depth`, +0.5147) scored 0.335 and read as "not separated", which is not a fact about
the edge, it is a statement that individual artifacts vary by more than the mean effect, which is
true of nearly every result here. The overlap is now taken on the **bootstrap distributions of the
mean**, which is the sanity-checks paper's actual framing and the scale τ = 0.2 was written for.
The per-unit number is still reported as `per_unit_overlap` because it is a real effect-size
measure, just not a verdict.

Second, `separated` was the wrong name. The scrambled nulls do not sit at zero: a channel that lies
confidently *hurts*, so the swap null for `process→goal` sits at −0.41. Separating from it means
*this effect needs the channel to correspond to the truth*, not *this effect is nonzero*. T-1's
`goal→process` edge at **−0.0012** separates cleanly from a swap null at −0.0514 while being a
negative effect. So there are now three fields:

```
separated_from_null       distinguishable from the scrambled channel
effect_exceeds_null       on the helping side of it
supports_a_positive_edge  both, AND above zero, the only one that licenses "this edge is alive"
```

**And a third thing, which is the one that actually mattered.** None of those fields is a
magnitude, and neither is `standardised_effect` in the way you would hope: at β = 1.0 the reader is
saturated, the null's spread collapses, and the +0.0017 edge scores a standardised effect of
**1.15, larger than the +0.5147 edge's 0.81**. Every confidence-flavoured statistic here agrees
that the tiny edge is real, and they are all correct. Confidence was never the problem.

So `relative_magnitude()` reports each effect as a fraction of the largest one measured **the same
way in the same family**, and that is what finally says it: the +0.0017 edge is 0.3% of the largest
edge into the same vertex. `excludes_zero` was misleading because it answered a confidence question
that nobody was really asking, and the fix was not a better confidence statistic.

---

## Partial information decomposition: `methods/pid.py`

T-1's superadditivity test is a hand-rolled synergy measure. PID is the principled version, and it
runs on `world.subsig` **exactly, with no rollouts**, the emission likelihood is the joint
distribution.

| depth | total | redundant | unique GOAL | unique MODE | synergy |
|---|---|---|---|---|---|
| µ=1 | 1.4521 | 0.0000 | 1.4521 | 0.0000 | −0.0000 |
| µ=2 | 1.8783 | 0.2966 | 1.1555 | **0.0000** | 0.4262 |
| µ=3 | 1.8783 | 0.2706 | 1.1816 | **0.0000** | 0.4262 |

Two things worth having. **Unique mode information is exactly zero at every depth**; everything
the execution mode contributes is redundant with the goal or readable only jointly with it, which
no previous measure here could state. And **µ=1 returns zero mode information and zero synergy**,
recovering null N28 from the likelihood alone; it is wired in as an `identity` gate on T-1 and as a
positive control in `tests/test_metamorphic.py`, alongside XOR (which must read as one bit of pure
synergy).

---

## Sensitivity: `methods/sensitivity.py`

Plate 5 randomises everything the theory specifies and counts survivals: one scalar, and its honest
reading is "83% of randomly parameterised models of this shape do it too". A survival count cannot
say **which** parameters matter, so an architectural finding and a one-setting finding look alike.

Sobol indices split the variance instead. `S1` is what a parameter explains alone; `ST` includes
every interaction it takes part in. **ST near zero is proof a parameter is not carrying the
result.** A large `ST − S1` means it only matters in combination, the same distinction PID draws
between unique and synergistic, reached from the other direction.

Cost: `N*(2D+2)` evaluations. At ~3 ms a rollout and a dozen parameters, `N = 256` is about a
minute. `morris()` is the cheap ordinal screen for cutting a long parameter list first.

---

## Trajectory features: `methods/trajectory.py`

T-5's best detection signal was not a posterior but how far one **travels**: `subgoal_step_movement`
and whether the reader is allowed to disengage. Both are trajectory statistics; every instrument in
this project and in Sounding Line reads an endpoint.

That result came from about a dozen features written down by hand. `catch22` is a canonical set
distilled from ~7000 candidates by removing redundancy, and a per-step posterior entropy series is
exactly its input. This is the cheapest available route to an exploratory measure nobody proposed.

**With the honesty protocol attached.** Extracting 22 features and reporting the best is a
multiple-comparisons problem wearing a lab coat. In a simulator the fix is free and stronger than a
reusable holdout: `confirm_on_fresh_seeds` re-runs the winner on seeds that did not exist when it
was chosen, and requires the sign to hold **and** half the magnitude to survive. T-2's difficulty
control flipped sign between n=40 and n=200; a feature picked out of 22 is at least that fragile.

---

## Multiplicity and equivalence: `methods/inference.py`

**FDR.** Batch two reports several hundred bootstrap intervals and corrects none of them.
Benjamini–Hochberg is right here and Bonferroni is not: BH controls the expected *proportion* of
false claims among those made, which is the correct target when deliberately scouring a space.
`control_fdr` reads the `{difference, interval}` shape the verdicts already use. Its p-values are
inverted from intervals under a normal approximation, adequate for **ranking**, and flagged in the
output as not quotable in their own right.

**Equivalence.** Half of batch two's findings are nulls. "The interval covers zero" is the absence
of a claim; "the effect is bounded below 0.02 with 95% confidence" is one. `equivalence()` demands
a `bound_source` string and records it verbatim, because an equivalence bound pulled out of the air
is worse than no test, it converts an arbitrary choice into an authoritative-looking verdict.
`smallest_effect_of_interest` builds one from a fraction of a live effect measured on the same axis
in the same run, which is the only construction here that imports no outside convention.

---

## Metamorphic tests: `tests/test_metamorphic.py`

Relations between **two** executions, with Hypothesis generating the inputs. Both batch-one defects
were invisible to every existing test and neither was a statistics problem; what finds them is
"change this and the output must change", "permute that and it must not".

Currently asserted: a zero-nat channel is bit-exactly uniform at every cardinality; channel
information is bounded and monotone; `fidelity_for_nats` inverts `channel_nats`; **a placebo
channel changes nothing end-to-end through the full stack**; **a full-fidelity channel changes
something**; overlap is symmetric, bounded and translation-invariant; `auc(a,b) + auc(b,a) == 1`
(the relation that makes `auc_oriented` well defined); the paired bootstrap never moves its point
estimate; PID atoms sum to the joint information; XOR reads as one bit of pure synergy; and N28
holds in the likelihood.

---

## Running it

```bash
make gates      # the gate + metamorphic suite only, ~40 s
make test       # everything
python runners/run_soundingline.py --only T1 T2 T3 T4 T5
```

## If you add a module here

1. Per-rollout data goes to `*_points.csv` (gitignored); the aggregate goes to `*_summary.csv`.
   **The naming is the policy**, and `test_no_oversized_committed_csv` enforces it.
2. Call `methods.provenance.stamp(verdict, __file__, gate_report)` before writing.
3. Give it at least one `live` gate and one `placebo` or `positive` gate. If you cannot think of a
   known answer your module should return, that is worth an hour before writing any more of it.
4. Add it to `GATED` in `tests/test_gates.py`.

## V18: allocation crossed with paid checking

The [V18 commission](versions/v18-selective-acquisition/CODING_PACKAGE.md) reuses the
unchanged V16 repeated-fragment learner, four-cell executor, exact local inhibition
and cost-bounded search through a thin serialized adapter. Offered cues, processed
records, later own targets and evaluator truth have separate interfaces. Shared
offers retain identical realized actions. Failed feedback cannot contribute an
eligible repetition, but failed observed endpoints remain in the population under
the [pre-run clarification](versions/v18-selective-acquisition/PRERUN_CLARIFICATION.md).

Checking consumes the primary search envelope; a separately labeled extra-cost
diagnostic restores full search. Final execution, acquisition, information queries,
learning and storage retain separate costs. Primitive search is the paired removal
diagnostic. Descriptive contrasts average targets then histories within constructor;
2,000-draw paired constructor bootstraps do not imply confirmation or equivalence.
The budget extension reuses acquisitions and is admitted by runtime alone.

Immutable eight-constructor blocks bind the source, seed list and stored traces.
Targeted known answers cover no-information access, real allocation, positive physics,
failed/empty learning, ties, paid checks, native ownership, corruption and literal
resume. An independent verifier reimplements execution and exhaustive search before
reaggregating every cell and contrast. The whole small study, including acquisition,
is replayed and retained in an executable archive. These are computation/provenance
checks within one architecture; the result remains **miniature — architecture untested**.

## V17: local admission and chunked comparisons

V17's [first implementation](versions/v17-adaptive-appreciation/IMPLEMENTATION_PLAN.md)
uses strict finite forecasts with proper log loss and Brier score, explicit evidence
projections, typed executable procedures and separately counted acquisition/storage/
online costs. A zero probability on the true event records infinite log loss; a
missing or malformed output is not replaced with a uniform forecast or STOP.
The initial construction slice scores terminal task success separately from those
future prediction consumers. Thirty-one targeted checks include the literal
command's resume and corruption paths. Small immutable JSONL chunks carry raw
inputs, outputs and replay identities; source/design locks reject changed resumes.
Constructor-cluster bootstrap intervals are descriptive. Repeated histories and
structurally equivalent constructor permutations do not enlarge independent support.

The [later full setup](versions/v17-adaptive-appreciation/SETUP_CHECKPOINT_2026-09-12.md)
adds actual typed Stitch learning, a finite Bayesian recipient, development-fitted
computation selection, dependency-aware revision with paid feedback, and unseen
native-maker choices under three evidence tiers. Cold acquisition and declared
operation/storage costs remain separate from measured process CPU and elapsed
time. The one-worker queue provides chunk resume and bounded owner-aware recovery.

A separate verifier reimplements proper-score arithmetic, physical execution and
revision regret, and independently reaggregates means and denominators. Forecast
ties use declared support order, preserved across sorted JSON serialization.
Bounded replay uses the packet's frozen source and compares full deterministic
cases and rows. Neither check independently validates the generative assumptions
or confirms a descriptive contrast. Creation-trace repairs retain old/new source
and data, reuse the same seeds, and do not enlarge sample size. Reader exports use
an explicit source whitelist, opaque aliases, separate evaluator truth, and an
actual extracted-consumer boundary probe. Fresh confirmation requires its own
frozen selection and admitted consumer; the screen engine refuses that claim.

## V16: acquired craft, bounded confirmation and retained reconstruction

V16 asks whether acquired procedures, earlier works and paid inquiry improve
construction or inference about a constructed maker. Its implementation lives in
`validation/soundingline/v16/`; the optional `methods/` layer is not required to
reproduce its numbers. The [accepted coding package](versions/v16-acquired-craft/CODING_PACKAGE.md)
and frozen packet records specify the native worlds, permitted reader evidence,
paired alternatives, practical bars and adversaries. Evaluator histories remain
separate from public observations and predictions committed before revelation.
These tests concern constructed mechanisms and their measurement. They contain
no human data and establish no real-text or human-behaviour result.

Discovery conditions retain separate denominators. Multiple histories sharing a
constructor are nested observations; multiple artifacts or reader alternatives
do not create additional independent makers. The finite expansion ladder uses
fresh namespaces while retaining the original conditions, estimands and practical
bars. Each completed source receives its own controls, including guarded access,
recorded prediction comparisons, cost accounting, constructor grouping and a cold
reader process. Syntax checks and the specifically declared physical recodings
have different scopes. Failed instruments and interrupted attempts remain in the
record. An execution receipt alone does not establish scientific validity.

Confirmation freezes at most three card packets after the final discovery
dispositions. Selection orders validity and target realization, serious rivals,
relevance, then practical size and cost. The supported capability primary is one
paired fraction difference in one discovery-selected condition, with one fresh
constructor per independent packet. Its range is externally bounded by [-1, 1].
The one-sided lower bound is the paired mean minus
`sqrt(2 * sample_variance * log(2/alpha) / n) + 14 * log(2/alpha) / (3 * (n-1))`,
clipped below at -1. The variance uses n-1; zero observed variance retains the
second penalty. This is the empirical Bernstein bound of
[Maurer and Pontil, Theorem 4](https://www.cs.mcgill.ca/~colt2009/papers/012.pdf),
applied to the negative paired difference. The primary tests the original
practical bar, and its p-value inverts this bound.

Capability sample planning uses an explicit moment-matched law on [-1, 0, 1],
with discovery variance and a mean one practical increment beyond the null bar.
Very small variances are increased only as needed to make that planning law
possible; both the original variance and the increase are retained. Twenty
thousand simulations at each candidate allocation must give a 99% lower bound
on power of at least 90%. This is power under that declared law, not a uniform
guarantee across bounded distributions or power at the null boundary. Native
targets without a justified external range remain exploratory under this
confirmation instrument.

The optional S02 null concerns all four paired repair differences across two
memory conditions and two complete error-monitoring rivals. Each constructor
contributes two distinct histories and one joint event: whether any component
exceeds the declared equality tolerance. An exact binomial upper bound on this
event rate bounds every absolute mean difference by the tolerance plus twice
that rate. Its sample size powers this joint event at a stated interior
equivalence alternative. It is not borrowed from a simple mean test. Every
packet is capped at 4,096 primary maker histories, with auxiliary comparator
training retained and costed separately. Bonferroni planning allocates 0.05/3;
one Holm ledger covers exactly the frozen primaries. No failed confirmation is
replaced, and descriptive native contrasts add no confirmed claims.

Closeout uses separate processes to reconstruct native physics and recalculate
all reported contrasts from retained unit records. Confirmation arithmetic has
an additional independent implementation: an analytic inversion of the mean
bound, SciPy binomial/beta calculations and a separate Holm calculation. Bounded
whole-unit replays regenerate worlds, acquisition, predictions, continuations,
program execution and scores. Opaque transport aliases, event timestamps and
observed OS resource samples have explicit comparison exceptions; scientific
seed identities and deterministic physical costs must match. ZIP archives carry
member manifests and receive full member-byte verification after creation.
Complete archival coverage, independent calculations, bounded replay coverage
and documentary consistency are separately required before campaign closure.
The portable replay inventory includes every file required by its frozen packet
locks, including admission tests and provenance records. ZIP membership alone is
insufficient: the extracted source-lock check must also succeed.

Independent calculation can proceed card by card after each native completion.
Cached calculations bind the original summary, raw manifest, completion receipt
and analysis source; the phase finishes only after every frozen card and
denominator is checked. This permits overlap with later discovery without using
unfinished outcomes or creating new observations. Known-answer tests exercise
the actual calculation command as well as missing-source, changed-source and
incomplete-packet refusals. The final archive compares current raw bytes with
the independent analysis input inventory before accepting complete coverage.

The completed-study tables project the original condition-specific estimates,
uncertainty, access contracts and costs without calculating replacement effects.
Confirmed claims join by exact claim and card identities, with their original
regime and multiplicity decision. A failed positive criterion does not establish
equivalence. Documentary verification binds the reviewed prose and revised
theory owners to these tables and the three scientific retention/reproduction
proofs; its interpretation remains an explicit agent review, not an automated
test of scientific truth.

## V17 continuation: bounded confirmation and durable scientific progression

The continuation retains raw constructor units as compressed JSON in SQLite WAL
transactions, with immutable source/plan/controller identities and public/private
construction hashes. Resume skips committed units and preserves failed attempts.
Final verification independently checks raw probability scores, counted-cost
identities and native physics before a completion receipt.

At most three contrasts are frozen before reserved fresh namespaces. Their sample
sizes use discovery between-constructor variation and a conservative simultaneous
Hoeffding precision bound. Paired histories are averaged within constructor draws;
Holm correction applies to the same family, including incomplete primaries as
probability one. Descriptive branch allocation never extends a primary on its
observed significance. Counts distinguish draws, unique constructions and
structural support.

The study tree promotes contrasting regions from a completed expansion cohort,
then retains separate robustness namespaces. Final illustrative selection uses
the lowest case hash within observed advantages, reversals and failures, marking
absent types explicitly. An extracted reader consumer must reproduce its reference
forecasts and refuse private-file access. This checks a trusted consumer boundary,
not a hostile-code sandbox. See the V17 continuation plan for the full contract.

## V17 final reconstruction, archival proof and replay

The final independent verifier, `runners/verify_v17_closeout.py`, imports no scientific
scorer. It reconstructs raw unit statistics, denominators, unique-construction counts,
means, between-constructor variances, descriptive intervals and costs. It separately
recomputes constructor-paired Hoeffding bounds and Holm decisions. Four known-answer
checks cover missing-output denominators, corruption rejection, pairing/direction,
fixed bounds and family correction. All 1,110,041 units and 735 method/target surfaces
passed. This uses stored row scores, independently checked by the original full-row
proper-score/physics verifier; these are distinct proofs.

The verifier checks admitted source bytes, metadata, checkpoint/event digests and
embedded dependency payloads. It creates a complete local scientific ZIP, rereads
every member and compares the original source bytes again. Its first attempt omitted
the source archive's embedded manifest from the allowed inventory and refused it;
the corrected inventory validates that manifest and all admitted members. Both attempts
remain on record. An archive is local retention, not public raw-data availability or
an off-device backup.

`runners/replay_v17_continuation.py` uses extracted frozen source and the admitted
native learner to regenerate the first and last committed unit of all 51 phase/branch
surfaces. All 102 units / 792 cases / 11,480 rows matched exactly, with new learner
caches and repeated controller training. It is bounded deterministic replay, not
independent reimplementation or full regeneration. The public bounded replay archive
contains evaluator truth and must not be supplied as blind reader evidence.

Final outcome-selected reader requests are distinct from the early pilot selection.
The original exporter reused its pilot wording; `runners/finalize_v17_transfer.py`
creates a documentation-only successor archive, validates member hashes and actually
executes its extracted consumer. All eighteen forecasts match exactly and its private
read probe is denied. Original archives and receipts remain unchanged. This guard
protects a fixed trusted consumer, not arbitrary hostile code. The completed
[study and proof inventory](../results/v17/closeout-1/README.md) records all limits.


## V18.1 physical verification and support accounting

The new continuation adapters keep known/supplied physical models separate from
evaluator truth and charge checking, search, ordering, retrieval and final execution
within the main online envelope. Original V18 enumeration retains its separately
named primitive-budget convention. Training/learning, storage and worker/child CPU
are separate quantities. Native3 legal-state/action enumeration validates the new
assembly adapter; independent dictionary/set physics checks larger programs.

Full raw reconstruction verifies reported outcomes and matched evidence. Bounded,
outcome-independent case selections are then replayed from fresh extracted source
to check reader decisions and work counts. These proof scopes are distinct. Every
proof implementation is retained by hash. Law/context identity comes from physical
fields, not constructor or draw IDs: the first G4 export's 18 draw clusters were found
to contain 16 physical contexts, corrected without changing its 72 blind cases or scores.
No duplicate draw is promoted into independent structural evidence. See the
[interim record](versions/v18-selective-acquisition/continuation/RESULTS.md).

An independent breadth-first checker verifies native target reachability and
whether a route without temporary worsening exists. Larger targets are checked
against privately retained constructive paths or legal query-menu demonstrations.
These target checks are separate from replaying whichever program a reader chose.
The direct structural compiler makes another model assumption testable: current
assembly labels expose topological order. A charged reset-and-rebuild plan can
use that public order without identifying the precise dependency graph. Its paired
comparison is admitted before any final representation contrast is selected.

The attachment-mask rival reuses completed G2 observations and probes, binding each
new row to its original case, raw block and forecast row. Its cold table keys retain
attachment and stopping state but discard orientation, using the declared action
semantics without the evaluator graph. Independent reconstruction uses set-based
keys, recalculates scores and counted work, and verifies exact pairing. A one-block
cache saves file decoding only; no scientific table fitting is amortized or free.

The opaque-label intervention deterministically permutes retained five- and seven-part
G2 worlds, states, actions, observations and queries. It preserves the physical DAG,
evidence and public candidate menu while removing the numeric topological-order cue.
The equally informed direct compiler derives a safe order from the union of those
public candidates, charges the checks and selections, and refuses a cyclic union.
Relabelings of retained structures are context interventions, not new architectures
or untouched confirmation support.

The cyclic-union discovery replaces the shared safe public order with twelve individually
acyclic candidate laws whose public union contains a reciprocal dependency. Neutral evidence
retains the full menu; one fixed parent observation isolates the true law. Explicit dependency,
conditioned-direct and primitive candidate-set action methods receive the same observations.
The primitive rival's admission gate honestly exhausted the inherited 32,768-operation envelope
on a seven-part positive control, so that failure was retained and a declared 65,536-operation
boundary was added before science. Complete reconstruction independently checks candidate cycles,
program physics, forecasts, evidence identity and work. The extracted-source replay selects one
case per size/topology stratum without outcomes.

The first final packet fixes 64 previously unused seven-part contexts in each of 18
topology, presentation and donor strata, with four histories nested in each context.
Its primary is the within-context success-fraction difference between two fixed support
observations and none. Direction, a five-point practical margin, error probability
0.025 and the one-sided empirical-Bernstein formula were frozen before generation.
Complete reconstruction precedes scoring; a separate fresh-source replay checks one
outcome-independently selected case per stratum. Forecast error and direct compilation
are secondary boundaries and cannot replace the primary.

The second and last final packet fixes 64 untouched contexts in each of six
five/seven-part topology strata, with four histories nested in each context. Physical
signatures include the true world, public candidate laws, initial state and target;
all 96 discovery signatures are excluded. Its primary is the within-context success-
fraction difference between one fixed parent observation and one decision-selected
observation for the dependency learner at 32,768 operations. Direction, a 20-point
practical margin, error probability 0.025, the one-sided empirical-Bernstein formula
and a no-extension rule were frozen before generation. The first-block verifier's
omission of a stored action trace is retained as a verifier-only failure; the repaired
checker changed no science. Complete reconstruction and extracted-source replay pass
before the frozen analysis is applied. Secondary action methods and cost comparisons
cannot replace the primary, and no third final contrast is permitted.

The post-primary cyclic cost diagnostic reuses all 384 exposed F2 contexts and adds no
independent support. It executes the unchanged decision selector under a separately
recorded 131,072-operation preparation envelope, then gives the selected observation
to each unchanged action method under the original 32,768-operation envelope. The
observation's execution/checking cost remains online; only query-selection computation
is separated. A failed 65,536 selector calibration, which left seven contexts without a
query, is retained. Selector, action and combined receipts are all recorded. This is a
descriptive decomposition of the existing result, not equal-total-work evidence, a new
primary or a result-dependent extension of F2.

The fresh cyclic target screen fixes two independent reciprocal target-relevant cycles
in each public four-law candidate family before outcomes. Its low-cost selector may use
only the changed public target parts and candidate parent relations: it greedily chooses
a parent query that maximally partitions the remaining public candidates, with stable
public tie-breaking. One query must leave two laws and two must isolate one. The historical
late-label fixed pair is retained as a designed placebo that cannot separate any of the
four laws. Dependency learning, an equally informed conditioned-direct compiler and
primitive candidate-set search receive the same observations and separate acquisition
from action work. The packet samples 192 fresh seven-part contexts across fork, chain
and grouped topologies, nests four histories within each, reconstructs all rows from the
frozen source and performs outcome-independent extracted-source replay. This is a
descriptive screen, not a third final contrast; it does not test action/routine evidence.

The charged physical query-preparation diagnostic reuses the complete-menu packet
byte-for-byte after its outcomes were exposed. For each action query, a public-only
selector searches for a setup path that reaches the same query state under every
candidate law still compatible with prior evidence. Candidate simulation, robust
setup search, the actual setup actions, the observed action and downstream task
execution share the 32,768-operation envelope. The verifier independently executes
every setup under every prior-compatible law and checks that the stored query state,
legality, trace and candidate filtering agree. Each query receives a freshly
provisioned task-initial object; provisioning/reset is explicitly outside the count.
The diagnostic adds no support or primary and cannot repair the supplied-family limit.

The candidate-family misspecification screen samples fresh seven-part contexts and
removes evaluator truth from each supplied four-law family before freezing the public
three-law contract. The same outcome-independent two-action sequence and evidence are
given to explicit dependency, conditioned-direct and primitive candidate-set methods;
each must treat an empty compatible set as detected misspecification and abstain. A
labeled forced-candidate control measures unsafe collapse, the stored-episode reader
tests whether missing output alone supplies detection, and a known-law arm is only a
ceiling. Independent reconstruction checks truth exclusion, every candidate filter,
every observation outcome and every submitted program. Query-state preparation remains
apparatus-provided, so empty-support detection here is not open-world anomaly detection.

The physical candidate-family misspecification diagnostic reuses that truth-excluded
packet byte-for-byte after its outcomes were exposed. For every attempted action query,
the selector searches only the surviving supplied laws for a setup path with a shared
terminal state. It then executes that path under evaluator truth and records the actual
setup outcome as public evidence; the intended action query executes only if its state
was reached. Candidate filtering, setup search, candidate simulation, physical outcomes,
the intended query and downstream action share one 32,768-operation envelope. The
verifier separately executes every retained setup under every then-compatible supplied
law and under truth. No actual setup failed in this packet; the measured boundary is
topology-dependent evidence and work exhaustion. The diagnostic adds no structural
support or primary, and fresh-object provisioning/reset remains outside the count.

Campaign unit accounting retains the previously filed conservative identifier count
when a later refresh would deduplicate identifiers shared by reused branch plans. The
final-primary filing caught and corrected such a decrease before commit. Scientific
outputs and CPU sums were unchanged; support claims continue to distinguish reused
identifiers, deterministic relabelings and untouched structural contexts.

## V18.2 finite maker-state validation

The new sibling namespace reuses V16 execution and acquired fragments. A separate
set-based executor and direct probability enumeration check the policy and posterior.
Reader packets reject private fields; future truth changes cannot alter features.
Learned split/flat predictors pass a known-answer training control and numerical
gradient check. Held-out artifacts score discovery; they do not select epochs or axes.
See the [design](versions/v18-selective-acquisition/maker-state/README.md).

The final packet preserves source-bound original summaries and exposes corrected
`CURRENT_SUMMARY.json` records. Nonlinear geometry bases are fitted after the
nonlinear transform; matched-probe comparisons use identical probe sets. Paid
observations are charged to every receiving arm, and uptake budgets reserve final
execution as well as acquisition and search. Superseded packets remain visible.
A separate assembly executor checks recorded legal actions, and public-law direct
compilers test whether construction needs the acquired library at all. All
intervals remain descriptive and grouped by maker/world lineage; repeated fits,
repairs and selected examples do not add independent observations. See the
[final scope and limits](versions/v18-selective-acquisition/maker-state/FINAL_REPORT.md).

## V18.3 purpose-sensitive inquiry and source-bound validity

The finite native board crosses decision rule, acquisition interference, considered opportunities and repeated sources. Three reader purposes share potential query outcomes; stopped selectors and exactly-three-attempt arms separate query choice from count. Nonresponse is an outcome, and every attempted query remains charged. Selector prices refer to declared likelihood-table entries rather than hardware instructions; observation, practice, planning, execution and whole-run CPU are separate costs. The robust stopping rule charges all of its likelihood evaluations. A known-answer cost crossover detects the original undercharge. Independent execution and probability references, strict public-packet parsing, corruption/resume tests, every retained mean, and fixed evenly spaced extracted-source replay establish scoped validity. Inference intervals average paired architecture cells and repeated probes within coefficient draw.

## V18.3 finite compression and transfer diagnostics

Exhaustive enumeration of all deterministic partitions of eight observation histories certifies the best old-question code only within that finite class. Maximum storage, code entropy and new-question loss are separate measures. A generator-informed new-question decoder measures retained information, without claiming learned transfer. Independently calculated code loss and fixed source-extracted replays validate the computation. [Protocol](versions/v18-selective-acquisition/research-extension/PROTOCOLS.md).

The final numerical audit qualifies this certificate to tolerance 1e-10 and reports
the full range of future losses among numerical old-task optima. It preserves the
original choice rather than selecting with future outcomes. Accuracy and calibration
use explicit numerical-tie rules; native actions use common seeded tie coins.
Scoring repairs retain originals, unchanged probabilities and proper losses, and
independent scalar reconstruction. [Review](versions/v18-selective-acquisition/research-extension/FINAL_REPORT.md).

## V18.3 learned memory, interventions and complete cost accounting

Ghost exchanges strict serialized public histories and test queries with an isolated
CPU PyTorch process using an existing optional environment. Evaluator truth stays
outside that input boundary. Development-only capacity and checkpoint selection,
optimizer/RNG resume, immutable input/weight identities, known learning controls,
and independently reconstructed means precede bounded extracted-source forecast
replay. Replay is not full retraining. Fit seeds, role swaps and repeated probes
are averaged within world lineage, not counted as independent evidence.

Frozen historical-role probes receive equal additional supervision but retain a
restricted decoder; failed decoding does not prove information destruction.
Intervention-trained readers face equally informed flat and direct rivals,
behavior-only and shuffled-target controls, wrong mappings, incompatible partitions
and compatible coordinate permutations. These diagnose useful alignment to a
supplied intervention law without identifying a unique ontology.

Cost records separate fit search, new-history encoding, cached query decoding,
single-observation update, model parameters, per-maker state, public world data,
native practice and action, serialization, failed work and verification. All readers
may cache. Fit-only amortization projections are explicitly conditional on the
measured workload and do not equate accuracy. Process CPU and wall time are distinct;
conservative uncertainty charges remain visible. A sampled peak-memory observation
is not a complete campaign peak. Source-aware revision deduplicates explicit roots;
the separate provenance-to-practice policy keeps the stronger public-law checker.

## V18.4 prospective exploration and replay

Adaptive discovery proceeds through agent-reviewed source-bound packets. Completed
batches trigger analysis and new designs; they do not create automatic scientific
confirmation. Compression selectors receive only declared training questions;
held-out questions cannot resolve old-objective ties. Cardinality, code entropy,
old-purpose loss and future-purpose loss stay separate. Complete mean reconstruction
and fixed source-extracted whole-unit or weight-forecast replay precede publication.

Neural coverage checks count question exposure within every latent combination,
not merely over the full dataset. A pre-execution repair removed a query/state-cycle
alias and restored the longest test histories. Long fits keep two resumable
optimizer/RNG checkpoints plus development-selected weights; an interrupted flat
fit reproduced uninterrupted weights exactly. Verification occupies the same
single-worker queue and shares the inherited cumulative CPU accounting.

The adaptive-state packet averages static and role-specific transition models
under common public evidence, scoring before each new observation. Copied source
IDs contribute no repeated likelihood while elapsed time still advances state.
An enumerated two-state filter validates the update. Native artifact matching
executes the chosen program, but it is not a claim about acquiring a procedure;
proper prediction, matching and abstention costs are reported separately.

The V18.4 native handoff check found the original Windows priority call returned
failure while the worker remained at normal priority. The Win64 pseudo-handle
needed an explicit pointer type. The repair checks the API return and actual
priority, adds explicit low-priority child creation, and passed a real venv-child
inheritance fixture. The already running worker and supervisor were lowered after
native ownership checks; immutable scientific snapshots were preserved. Earlier
work in this opening wave did not have verified low priority. Its recorded CPU
and numerical results remain actual observations; scheduler-priority claims are
qualified rather than silently retrofitted.

L1 reports actual combination/question exposure separately in each training regime;
legacy test labels do not establish novelty after the support swap. The first main
packet retains nine development-selected weight sets, all raw forecasts, and
independent replay of 64 fixed rows per fit/test file. Repeated queries and fit
seeds are averaged within coefficient-draw lineage. Question-transfer failure is
not interpreted as proof that a hidden state lost the relevant information.

U2 makes the observation-horizon comparison within one retained 96-step trajectory.
Its seed excludes horizon, condition, switch time and copy span; all steps consume
the same evidence/target draw positions. Exact pre-change and nested-prefix controls
detect future-condition leakage. Absolute switches and equal post-change windows
separate evidence age from changing pre/post-change proportions; they do not turn
sequential prefix differences into a randomized causal estimate. Independent
arithmetic fixtures check window boundaries, and every full trace remains retained.

L1 support swaps reuse the test coefficient draws and histories. Report the
contrast as paired, translate raw condition names into actual exposure, and do
not count the second support regime as another independent test-world sample.

L3 freezes both verified complementary-support encoder sets and fits separate
linear and fixed-random-feature nonlinear behavioral readouts. Fresh identical
label access, a raw-history rival, a question-only rival, context-stratified
target permutation and two nested balanced label budgets distinguish decoder
access from an undeclared increase in supervision. All capacities and controls
remain reportable outcomes; no preferred ranking is a validity gate. Training
alone sets standardization and random features; development alone selects ridge
penalty and temperature. The farther question family remains absent from both.
Public exports retain copied encoders and fitted readouts, and independent replay
reconstructs predictions from them. This bounded decoder ladder cannot establish
information destruction or that the original predictor used recovered features.

The L1 full-support comparison holds total history and label counts fixed, so
examples per individual maker combination fall relative to half-support training.
Report both test halves as familiar combinations and retain that density caveat.

L2a reuses completed full-support weights and test cases for a paired supplied-law
decoder intervention. It converts old-question prediction banks into new answers
with fixed pseudoinverse cutoffs, reports finite-law span and conditioning, and
retains every invalid raw output before declared floor-and-normalize scoring.
This adds generator knowledge without new fitting labels. Whole-file replay
preserves batch arithmetic before potentially unstable inversion; scalar controls,
observational-alias fixtures and immutable input checks validate the computation.
It does not establish passive identification, recursive closure, independent
replication or success of a new bank-training objective.

The diverse L1 rotation matches total labels while reducing exposure per original
question. Added compositions must be reported as covered; farther queries alone
remain question-family holdouts. Use shared-history paired contrasts and retain
the supervision-density and optimization caveats.

P2 uses uniform case allocation to match the marginal and conditional target
entropy of each supplemental purpose under whole-vector history rotations.
The passive task remains fixed, so between-task compatibility and relation to
future purposes can change. Seven cyclic shifts balance every nonself pairing
without enumerating all permutations. A marginal-target control deliberately
removes history information. Exact supplied-law decoding and every retained
shift are reported; extra target content is distinct from its useful alignment.

The complementary diversity comparison is a paired support swap on shared worlds.
Its repeated direction is not an independent-world replication; all forecasts and
fit seeds remain grouped by lineage in the verified means.

R1 separates state updates, finite-family selection, Bayesian averaging and paid
access to a predeclared alternative law. Identical unique records are reordered
under a stationary maker; marked copies affect neither updates nor score counts.
A fixed trigger reads only earlier observed-program surprises. Adding a law
refits retained history and charges that work; expanded-from-start rivals pay
the same access fee. Sequential and final expected losses, native artifact-match
probabilities and two stipulated cost sensitivities remain distinct outcomes.
Every unit's final posterior and evidence are independently reconstructed by a
batch likelihood product. Final static-model evidence must be order invariant;
purchase timing and prequential forecasts need not be. This is finite catalog
access, not law invention, human revision or a learned uptake mechanism.

The full-support diversity comparison completes the six-regime paired support/query
slate. Both test halves contain trained combinations; the legacy novelty labels
are not used to claim held-out maker support. Fixed total labels leave per-question
exposure as an unresolved explanation, alongside decoder and optimization effects.

S1 enumerates binary truth, four copy graphs, every six-report string and every
terminal path of at most two paid inquiries. The reader's posterior determines
actions; separate true joint masses evaluate them. One-step and two-step decision
value are compared with restricted purchases, a graph-information heuristic and
fixed common traces. Actual and assumed audit reliability are separate factors.
Independent latent-root sums and path likelihood products validate inference;
brute-force contingent-policy enumeration checks the bounded decision recursion.
Purchase prices are stipulated utility costs. Proper truth loss, binary decisions,
graph probability error and net utility remain distinct outcomes. Initial shared
source error does not imply shared errors in the subsequently purchased channels.

U2 reports nested cumulative prefixes beside equal-length windows since the switch.
A cumulative sign reversal can include changing pre/post-change weights even when
both methods improve locally. Common random draws remove separate-horizon seed
confounding; they do not remove these different estimands or create independent
replications from windows, copy spans or alternative switch times.

The frozen-state ladder recovers aligned signal, but the nonlinear question/world-only rival is stronger on taught compositions. Decoder success alone therefore cannot establish useful history; farther transfer and decoder limitations remain distinct. The next learning intervention needs an equally informed no-history rival.

The frozen old-question bank contains useful target information under a supplied law: truncated inversion greatly improves farther scores. Full mathematical span coexists with ill conditioning, invalid raw probabilities and clipping; the next comparison must isolate feasible bank decoding and prior-only supplied-law performance before attributing the improvement to learned sufficient state.

L4 separates equal original-question exposure from equal total labels. Its 480/768
history counts balance every original-question/state/history-length combination.
The larger original-menu control matches the diverse arm's histories and updates;
the smaller control matches original-question examples. Common diverse development
removes unequal model-selection access, while exposing compositions in every arm.
Only farther questions remain absent from training and selection. The no-history
neural rival keeps the direct architecture's nominal parameters but zeros every
history feature and ignores length; query identity learning and history-change
invariance validate that access contract. The changed sample and fitting budgets
are explicit costs, not claimed controlled-away nuisances.

P2 reports each content-preserving cyclic shift as well as its mean and range.
A near tie is retained as a descriptive result; a nonzero shift does not certify
that useful partition geometry disappeared. Uniform history allocation supports
the information invariants, while changes in cross-task compatibility remain.

R1 result review establishes that coupling alternatives with uniform skill priors
are predictively equivalent under a fixed skill permutation. Retain the executed
comparison as catalog-access evidence; do not infer a selection/averaging result
from aliased candidates. Equal-label priors reweight predictive equivalence classes.
The post-result alias control checks repertoire and full likelihood identities;
every stored fixed/selection/mixture forecast was also compared.

S1 publication retains all graph-conditional scores beside equal-graph averages.
An exact net-utility benefit can be tiny and disappear under miscalibration; it
is not promoted to an audit-success gate. Graph squared-probability error sums
four components and is distinct from truth logarithmic loss and binary utility.

R2 replaces aliased initial candidates with distinct decision laws and crosses
equal predictive-class priors with a doubled supplied-class mass. Selection is
over classes; the doubled mass is an explicit prior change, not equivalent to
selecting individual duplicate labels. Scalar state/class products reconstruct
every prefix forecast and final answer independently, including the prior. A
split-mass duplication control checks invariance for Bayesian averaging. Costs
and fixed past-only purchase thresholds are retained; no method ranking gates
validity. Copy repetition stays a control rather than redundant science cells.

The first L4 result reports its question/world-only baseline before interpreting
any paired exposure intervention. Original questions, selection-exposed compositions
and untouched farther forms are separate estimands. A failure to beat the no-history
rival on new forms is not a failure of the known-answer instrument.

L2c projects retained eighty-coordinate old-answer banks onto the convex hull of
twenty-four supplied-law state banks. Analytic constrained optimization is checked
by independently reconstructed simplex duality gaps and scalar mixtures. The same
public-world prior and history posterior separate law access from history use;
their uniform prior is stipulated, not inferred from deterministic sample indices.
Projection validity, latent identifiability and future logarithmic loss remain
separate claims. Fit seeds average within reused history before reporting means.

L4 restored-exposure reporting pairs the same test lineages across training arms.
Equal original-question counts exclude dilution as a sole explanation in that
contrast, while changed history count and optimizer updates remain explicit.
The equally trained no-history rival tests whether deterioration requires memory.

# Exact maker-bank sufficiency — 21 September 2026

Can a bank of future predictions preserve the maker's preceding operation order? In the exact three-unit world, two reachable witnessed histories have identical persistent-maker posteriors but different operation orders. Every exact fresh-episode bank that depends only on that posterior therefore loses this distinction. The frozen example holds in all 32 lineage/context pairs. This is a conditional analytic method counterexample, miniature — architecture untested; it does not establish impossibility for learned banks or a result about human intent.

The distinction is available in the supplied evidence. One history replaces the
presentation, inspects, then inspects again; the other inspects, replaces the
presentation, then inspects again. Both end at the same artifact. A reader retaining
the witnessed history can identify which operation came first. Compressing that
history to predictions about new, reset episodes can discard this known order.
This is information loss under a particular representation, not ambiguity in the
complete witnessed input and not a failure to infer an unobserved operation.

## Independent source-derived argument

Fix a coefficient lineage, context and persistent maker. Write `a` for the action
rate and `g_M`, `g_D`, `g_P` for the probabilities of meaning, dependency and
presentation goals. These are strictly positive and sum to one. The executor's
goal weights depend on claim/evidence, maker traits and the request; neither the
display bit nor the undo buffer enters those weights.

Replacing presentation changes only the display bit. Inspection changes no bit.
Thus both paths preserve claim/evidence at every step and use the same goal
probabilities. At either of the first two steps, inspection is available under all
three goals, giving total probability `1-a`. At the third step, dependency's
alternative is undo, so inspection has probability `(1-a)(g_M+g_P)`. Replacing
presentation has probability `a*g_P` at either early step. After summing over the
unobserved goals, both full witnessed paths have the same probability for that maker:

```
(1/64) * (a*g_P) * (1-a) * ((1-a)*(g_M+g_P))
```

The factor `1/64` is the common uniform maker/context prior. Every factor is
strictly positive under the declared laws, establishing reachability for every
maker and context, not merely for an arbitrary simplex posterior. The order of
the early factors changes but their product does not. Summing and normalizing
these equal maker likelihoods yields identical posterior vectors.

The fresh-episode endpoint bank is a fixed maker-to-answer matrix applied to that
posterior. Equal posteriors therefore imply equal banks, regardless of the number
or conditioning of those posterior-only questions. Meanwhile, the first witnessed
operation is known and different. No decoder receiving only this bank and the
same query can return both distinct correct first-operation answers. Giving the
decoder the original history, an order flag, or another history-dependent input
changes the representation under examination.

The full transient executor retains different intermediate undo buffers. That
does not invalidate this witness: neither chosen path performs undo, and goal
weights do not depend on the buffer. A law that conditions choices on display,
undo memory or early-step index would need a fresh proof. This is not a theorem
about all possible local-maker worlds.

## Executed evidence and controls

The frozen handler reuses the eight admitted local lineages, four contexts,
16 persistent makers and all 110,592 retained trajectories. It groups all visible
packets at four evidence levels. Per lineage there are eight artifact-only groups,
32 with added context, 168 with a first-operation witness, and 704 with complete
witnesses: 7,296 groups altogether. These are enumerated groups, not independent
sampled worlds. There are no new fits or confirmation lineages.

All 32 prescribed path pairs have positive recorded evidence mass. The smallest
is 0.00162587 under the joint maker/context prior. The largest recorded absolute
posterior-coordinate discrepancy from the common expression is 4.17e-17; the
recorded bank discrepancy is zero. The first-operation target differs by one in
probability. These floating-point observations check the execution; the exact
claim rests on the factorization above, not the numerical tolerance.

The synthetic exact-alias, normalization and near-singular-but-invertible controls
pass. The pre-dispatch fixture separately exercised both complete transient paths
under every maker/context on a fixture law. Rounded posterior screens at eight,
ten and twelve decimals remain numerical candidates only. No exhaustive alias
frequency, all-tier nonidentification or unique local-goal reconstruction is claimed.

Complete adjacent and freshly extracted-source replays reproduce all 43 bound
files, including the eight raw input archives. Review verified source, plan,
environment receipt, input identities, all output hashes and all 7,296 group
structures. It independently checked the analytic factorization and reachability
from the executor. It did not independently resummate the numerical groups: those
numbers are supported by the frozen execution and complete replay. This limitation
does not turn rounded equality into the proof.

## Scope, exports and continuation

**Warrant:** exact conditional method counterexample for posterior-only banks and
these two reachable path shapes. It identifies a structural rival to the learned
readout result; it does not identify the cause of that result's measured loss.
Approximation errors in a learned bank can retain additional history, so its
failure does not follow from this theorem. No neural capability or human intention
claim follows. General architecture severity remains untested.

**Pursuit:** retain transient process information alongside persistent-maker
predictions and measure which queries require each. The already admitted
same-objective optimizer comparison and independent recursive updater remain
useful. Their results cannot be inferred from this counterexample. A later
bank-plus-history comparison should match labels and capacity and separately
score operation order and future prediction before any fit.

The scientific/evaluator archive preserves raw trajectories, all grouped targets,
the original paired packet and analytic witnesses. The separate reader archive
contains eight distinct complete-witness inputs, indexed only by visible-content
hash. Repeated lineage instances and evaluator pair IDs are excluded. This is an
outcome-independent, deliberately selected proof fixture, not a random evaluation
sample. Reader content contains no hidden maker, local goal, posterior or proof.

[Scientific evidence and role manifest](../../../results/v19/G19-A-process-sufficiency-1/EXPORT_MANIFEST.json)
and [frozen protocol](SUCCESSOR_PROTOCOL_2026-09-21.md).

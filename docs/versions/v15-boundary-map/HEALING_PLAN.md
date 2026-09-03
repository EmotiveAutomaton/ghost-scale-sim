# V15 — healing plan

*Instrument repairs and pending curator decisions during the 168-hour window. This page carries
no result prose (spec §9.1): it records what was repaired in the apparatus, what was not, and
why. Card states below are record-completion states, never held criteria.*

## Repaired and re-run (2026-09-02, hour ~43)

Re-run through `runners/run_v15_amendment.py` on the original lineage, tier and seeds; originals
preserved under `amended/` beside each lane's verdicts; swap recorded in
`results/v15/AMENDMENTS.json` with both hashes.

| card | what was wrong with the instrument | repair | state before → after |
|---|---|---|---|
| X23, X24 | every row hardcoded repeat `0`, so the expected-cell receipt counted 80 units where 240 had run; both criteria were already HELD | rows carry the real repeat index | RESOURCE_BLOCKED → LANDED |
| H03 | the placebo gate observed the card's own criterion statistic with a tolerance of twice the criterion bar — the V14 gate/criterion conflation in a third form (a magnitude reaching a placebo gate through `tol`); the suite's two structural checks look only at live/no-oracle `expected` fields and the battery signature, so it passed them | placebo is now the reader's named topologies scored against shuffled truth labels, averaged over eight seeded permutations, tolerance three binomial SDs with a 0.05 floor | INSTRUMENT_FAILED → LANDED (criterion evaluated separately) |

## Not repaired, deliberately

| card | why it stays as recorded | what a repair would be |
|---|---|---|
| M01 | the 240-particle reader collapses on T3-sized worlds (3.9% unique particles, non-normalized posterior). That is the anchor doing its job: the approximate reader fails the exact anchor at scale. Raising the particle count is an estimator change, not a bug fix, and M02, M06 and M12 have already run with the 240-particle reader | a lock amendment on the particle count (`particles.DEFAULT_N` and the `n_particles` architecture default), then re-running M01, M02, M06 and M12 on an amendment lineage. Curator decision |
| C11 | the true preference vector is itself feasible under the card's own predicate only about half the time, even for a 0.99-competence actor (300-world pilot, 2026-09-02). The feasibility predicate disagrees with the choice generator, so the positive gate is right to fail; a gate edit would hide a real defect in `persistent.feasible_reward_set` | reconcile the predicate with `choose` in the locked `persistent.py` (lock amendment), then re-run C11. Curator decision |

## Runtime repairs (2026-08-31, hour 0.26–3.95)

Recorded in `CLAUDE.md`: the relaunch loop (44 refuse-and-exit relaunches), the resume path,
the heartbeat, the lock restore from HEAD. About 3.7 window-hours carried no science; the
per-stint occupancy receipt does not show that gap, and whether the seven-day contract can be
claimed at closure is a curator decision.

## Standing constraints learned this window

- `V15_SPEC.md` stays at the repository root until closure: the validator's vocabulary scan
  covers this directory and the spec quotes the forbidden words as prohibitions.
- Never launch any `run_v15` stage beside the live runner; it rewrites `RUNNER_STATUS.json` and
  the watchdog will start a second runner. Scratch smokes go through `run_card` from a file.
- `tests/test_v15_gates.py` cannot see a magnitude that reaches a placebo or positive gate through
  `tol`. A third check is worth adding after closure, not during the window.

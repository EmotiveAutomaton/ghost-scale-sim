# Bounded validity review, 14 September 2026

REVIEW.json records a deterministic check of retained V17 data. The first and
last saved units of every expansion and confirmation branch and the latest saved
unit of every robustness branch were checked, including the curtailed branch.
All 78 sampled units, 600 cases and 8,680 measurement rows passed the archived
verifier's projection, physical execution, probability, proper-score, cost and
budget rules. Stored unit summaries and sampled paired confirmation values also
matched. The 37 admitted source members and frozen snapshot payloads matched.

audit_retained.py extracts the verified source archive into a separate directory
and uses a read-only database connection with indexed, bounded sample queries.
It does not start a scientific stage or optional learner. This is a spot check,
not full reaggregation or a random estimate of the error rate. Full independent
verification and final scientific interpretation remain due at closeout.

The pooled-assembly failure is unchanged. Its last retained valid unit passed;
the failed constructor supplies no result and remains in the failure ledger.
The original worker continues other branches until the immutable sampling end,
17 September at 18:09:19 Pacific. No new scientific setup is needed for that queue.

TEST_REPAIR.json records a separate operational repair. CI at 2621d7f failed the
existing V15 C11/M01 gate check and three V16 interruption fixtures. Those fixtures
copied the real V16 acceptance/deadline, which expired on 14 September. They now
use the existing separately identified fixture-clock helper. The real scientific
campaign record, dispatch deadline checks and saved historical results are unchanged.
Nineteen targeted tests passed, including actual process interruption/resume and
cross-checkout ownership/clock-immutability checks. Windows restart-test children
also use consoleless launches.

The earlier completed phase records were pushed and verified at 2621d7f. This
review does not turn a retained failure or an unfinished campaign into a pass.

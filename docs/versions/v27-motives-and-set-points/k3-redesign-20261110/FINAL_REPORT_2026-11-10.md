# K3-RD1 final validation report

Can an observation-conditioned hidden-state reader make the dynamic K3 set-point
comparison numerically stable enough to justify a production study? **No.** Both
approved adjacent population pairs failed the frozen calibration gate on the
first fully completed fixture. K3-RD1 is closed, and no K3 production is
recommended or authorized.

This is a constructed-method failure. It shows that this numerical reader did
not meet its predeclared stability requirements; it is not evidence for or
against moving set points, human motives, or any real-text mechanism.

## What completed

All 37 known-answer controls and 13 targeted tests passed. The bounded review
verified the frozen 1,072-file source manifest and the plan, result, raw-fixture
and per-hypothesis bindings for 29 completed jobs. The complete fixture-5 roster
contains four retained replicas at each of 1,024, 4,096 and 16,384 particles.
No scientific draw occurred.

The table reports numerical disagreement across all 28 retained run pairs for
each adjacent population pair. Posterior error is the largest absolute change
in the five-law posterior; margin error is the largest absolute change in the
three declared log-evidence comparisons.

| Population pair | Failed / checked pairs | Largest posterior error | Frozen posterior limit | Largest margin error | Frozen margin limit | Crossing failures |
|---|---:|---:|---:|---:|---:|---:|
| 1,024 / 4,096 | 28 / 28 | 0.461729 | 0.05 | 5.154218 nats | 0.25 nat | 16 |
| 4,096 / 16,384 | 28 / 28 | 0.312091 | 0.05 | 3.452349 nats | 0.25 nat | 7 |

An independent reaggregation and the frozen assessment function produced the
same maxima and failure counts. Because neither allowable adjacent pair can now
pass every calibration fixture, continuing the remaining calibration paths
could not change the admission decision.

## Retained failures and cost

The controller stopped four already-started fixture-1 replicas when the review
confirmed the gate failure. Their raw fixtures and 28, 26, 26 and 19 completed
hypothesis checkpoints remain local; none was restarted or omitted. Forty-two
planned jobs were never started. The generic native queue receipt names the last
stopped replica, but the scientific stop reason is the earlier completed
fixture-5 gate failure.

Phase A charged 9,421.859375 CPU-seconds (2.617183 CPU-hours), including controls,
completed paths, the four stopped replicas and owned services. Phases B and C
never opened. Phase D used 43.8125 measured native CPU-seconds through report
generation; its remaining work was verification and administrative closeout.

## Scope left unvalidated

The 15 old plus 15 fresh validation fixtures were not generated as a validation
roster or run. The conditional full-reader cost probes were not run, so this
allocation establishes no production cost. Texture-specific observation laws
remain unvalidated, and the noisy-stock likelihood remains unsupported. The
explicit disposition of each of K3 and K3-T001 through K3-T033 is recorded in
`FINAL_DISPOSITIONS.json`; none is production-ready.

Sounding Line receives no real-text evidence from this failure. ToMpathy receives
no dynamic set-point identification result and no engine admission. A different
algorithm, a larger population ladder, a changed gate, or a new healing attempt
would require a new owner decision and allocation; none is recommended by this
closeout.

Primary aggregate record: `results/k3-redesign-20261110/FINAL_VALIDITY.json`.

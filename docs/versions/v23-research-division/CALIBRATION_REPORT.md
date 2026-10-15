# V23 stopping calibration and implementation checkpoint

Can pooled adaptive stopping avoid discarding useful families without nominating too many nulls? On the declared held-out simulation grid, the selected policy stops 725 of 13,181 strong families (5.50%; 95% upper bound 5.90%) and nominates 18 of 6,247 null families (0.288%; upper bound 0.455%). Both pooled targets pass. Thirty-three individual settings exceed at least one point-rate target, so this is a qualified method result, not a universal guarantee. Fresh-process full replay and independent count/interval reconstruction pass.

The frozen policy uses a kill probability of 0.05 and prior scales 0.08 for the
family mean and 0.05 for its spread. The useful margin remains 0.02; nomination
still requires probability at least 0.90. Two complete burn-in batches, the
16/16/48/144/144/144/144 world ladder, factor-three survival and protected late
variants remain. Integration refines from 121 through at most 1,921 nodes until
every variant's tail probability and mean agree within 0.002.

Nine permitted policies were tuned on 180 settings with 16 repetitions each.
The selected policy was frozen before 128 new repetitions per setting: 23,040
held-out simulated families. Strong families have best true gain above 0.03;
null families have best true gain at most 0.02. Intermediate families do not
enter those two denominators. The targets compare pooled 95% Wilson upper bounds
against 10% false stopping and 20% false nomination. Every cell remains in the
[validation record](../../../results/v23/calibration/VALIDATION.json).

The table compares false stopping of strong families. Counting policies see all
variants at all looks; the pooled policy halves its active set. These are not
equal-resource comparisons and do not establish that pooled stopping is always
better than counting.

| Stopping policy | Wrongly stopped strong families | Rate |
|---|---:|---:|
| Pooled calibrated rule | 725 / 13,181 | 5.50% |
| Two consecutive nonpositive batches | 3,913 / 13,181 | 29.69% |
| Five consecutive nonpositive batches | 211 / 13,181 | 1.60% |
| Seven consecutive nonpositive batches | 57 / 13,181 | 0.432% |

Here a counting-rule batch is positive only when at least one variant's
normal lower bound exceeds 0.02. The anticipated blanket advantage over counting
is not established: the five- and seven-batch rules stop fewer strong families
on this grid while consuming their different declared allocation.

The first attempt stopped on insufficient quadrature resolution. Its one
numerical repair retained the targets and seeds. The second attempt stopped on
a Windows status-file sharing error; the third changed only atomic I/O/retries,
with every numerical source byte unchanged. All attempts, costs and source
capsules remain retained. The successful original and replay together charged
935.391 CPU seconds. Two failed attempts charged 114.922 seconds. Setup and
services are accounted separately under the unchanged 90-hour ceiling.

This calibrates Ghost's declared Gaussian/slow-starter approximation conditional
on fixed fits. It does not calibrate Sounding Line's different decision rule,
non-Gaussian errors, arbitrary variant counts, correlated effects, or any human
dataset. Full replay regenerates all simulations; retained calibration products
are cell counts and policy summaries, rather than a separate per-trial trace.

## Execution readiness

The final scoped engineering suite passed all 66 tests, with no skips or errors.
This includes native process accounting, independent posterior controls, ledger
conformance, raw-score reconstruction, finite monitoring and admission rejection.
It is not a full production validation of every family or a historical repository
suite. [Engineering receipt](../../../results/v23/ENGINEERING_VALIDITY.json).

V21 is closed and published. V23 opened at 2026-10-15 05:04:15 UTC and ends no
later than 2026-10-22 05:04:15 UTC. No family discovery, confirmation or largest-
cell production pilot has been admitted. The other thread's canonical schema
exists provisionally but is not yet committed; no dirty schema is a campaign pin.

R1-R4 have complete retained-data pipelines, known-answer controls, independent
posterior reconstructions, pilot entry points and a small engineering replay.
New mechanism modules and adapters exist, with independent numerical checkers
for N2/N3/N6/N8. Their complete production pipelines and the remaining checkers
are unfinished. All four twins remain to be built. See
[implementation status](IMPLEMENTATION_STATUS.md) and the machine work order.
This checkpoint is not a claim that all commissioned studies are runnable.

The durable monitor is disarmed during the operating turn. It collects failure,
completion and committed-schema events; routine checks remain fixed at twelve
hours. A new actual review delivery is not yet observed. No V21/V22 recurring
wakes remain. The finite V23 program retires its monitor at verified closure.

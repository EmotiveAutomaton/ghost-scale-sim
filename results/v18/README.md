# V18 results and reproducible record

Can focused acquisition preserve useful craft, and when does checking before use
protect construction from interference? The [completed study](../../docs/versions/v18-selective-acquisition/RESULTS.md)
answers within a four-cell, capacity-one miniature. The core and permitted budget
extension are complete; all findings are descriptive constructed mechanism results,
**miniature — architecture untested**.

Each row below identifies an artifact and what it establishes.

| Artifact | Scope |
|---|---|
| [PLAN.json](PLAN.json), [ADMISSION.json](ADMISSION.json) | Frozen cases, source hashes, deadlines and passing targeted admission. The original specification and pre-run clarification are among the source bindings. |
| [EXTENSION.json](EXTENSION.json), [RUN_COMPLETE.json](RUN_COMPLETE.json), [blocks/](blocks/) | Runtime-only extension decision, complete finite block inventory, immutable byte bindings and block costs. Execution completion alone does not assert instrument validity. |
| [COMPARISONS.json](COMPARISONS.json), [cells_summary.csv](cells_summary.csv) | All 28 cells, 52 paired contrasts, constructor means/intervals, acquisition counts and exposure diagnostics. |
| [VERIFICATION.json](VERIFICATION.json) | Independent reconstruction of physics, search, selection, learning and costs; reaggregation; complete deterministic replay of 21,504 transfer executions. |
| [EXAMPLES.md](EXAMPLES.md), [ILLUSTRATIONS.json](ILLUSTRATIONS.json) | Six observed outcome-selected examples; raw acquisition, checked steps, proposals and executed endpoints. Illustrations are not new evaluation data. |
| [replay.zip](replay.zip), [ARCHIVE.json](ARCHIVE.json) | Complete small-study traces and frozen source, with every archived member reread. Contains evaluator truth, not a blind reader handoff. |
| [PORTABILITY.json](PORTABILITY.json) | Actual execution after extracting the archive, including all-row replay and comparison equality. |
| [VALIDATION.json](VALIDATION.json), [TIME_ACCOUNT.json](TIME_ACCOUNT.json) | Targeted and wider regression outcomes, retained setup failures, measured scientific CPU/wall time and their limits. |
| [SCOPE.json](SCOPE.json), [CLOSEOUT.json](CLOSEOUT.json) | All planned blocks and deferred scope accounted for; documentary bindings and finished disposition. |

## Table units and denominators

Success, primitive-relative success and paired gains use fractions; multiply by
100 for percentages or percentage points. Target averaging precedes history and
constructor averaging. Descriptive 95% intervals use 2,000 constructor bootstrap
draws, with no corrected-significance or equivalence claim. Constructor vectors
remain available in each contrast. The zero interaction and observed ceilings do
not establish equivalence or generality.

Checking, search and submission fields count primitive executions. Their sum is
repeated online work. Final submission is outside the checking/search envelope.
Successful-only costs average only successful rows, with their count retained;
different methods can condition on different successes. Acquisition costs are
reported separately per history, not repeatedly charged for every test target.
Queries, record scans and stored actions are separate units, not hardware time.
Extra-cost checked arms retain a full search budget in addition to their checks.

The sample is 64 constructor configurations and 256 histories at both budgets,
with 22 observed cell-label/order arrangements and one architectural family.
There are 10,752 target executions per budget. Each core policy condition averages
two compatible or four changed targets. Primitive and extra-cost rows are diagnostics.
The extension is paired resource sensitivity, not independent replication.

## Replay

Extract `replay.zip` to a fresh directory and run from its root:

```powershell
# Use the project's explicit scientific interpreter path on the local workstation.
python -B -m runners.report_v18 --root run --output reproduced --replay
```

The replay requires only the Python standard library; no installation or old campaign
stage is needed. The archive includes all 16 raw compressed blocks, frozen source,
the source and case manifest, extension decision and reference report. Full scientific
replay means these finite recorded executions, not an independently designed world.
Public archive availability is distinct from an off-device archival guarantee.

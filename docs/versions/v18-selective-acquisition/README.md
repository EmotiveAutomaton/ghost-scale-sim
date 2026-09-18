# V18: selective acquisition and unwanted carryover

Original V18 is closed. Its separately commissioned [V18.1 continuation](continuation/README.md)
has its own unchanged specification, clock, [interim results](continuation/RESULTS.md)
and [reader bridge](continuation/BRIDGE.md). The original design below remains historical.

Accepted bounded commission: [CODING_PACKAGE.md](CODING_PACKAGE.md), copied without
alteration from the 18 September handoff. V17 remains closed.

The [pre-run endpoint clarification](PRERUN_CLARIFICATION.md) preserves errors and
counts accidental exposure. Transfer goals are untrained; endpoints need not be
unseen. The original specification is retained unchanged.

## DESIGN CHECK

Question: can allocating acquisition toward a focal topic preserve useful craft,
and when does paid local checking protect later construction from interference?
This is a descriptive constructed mechanism study, not belief or value learning.

Reuse unchanged V16 `prepare`, `perform`, `learn_processed`, `inhibition`,
`construct` and execution helpers through a new serialized V18 adapter. Offered
cues, admitted acquisition records, own transfer goals and evaluator truth have
separate schemas. Selections precede outcome exposure. A common offer produces
identical realized actions across arms. All failed trials and empty libraries stay.

The capacity-one learner, fixed lexicographic tie break, primitive enumeration
order and exact local goal check are assumptions. Constructors permute motifs over
four fixed cell labels; every paired arm uses the same 0..7 primitive order.
Reliability variation and these permutations do not supply new architectures.

Core: 64 configurations x 4 histories, two compatible targets and four changed
targets per history, seven methods including primitives and two extra-cost checks.
This is 10,752 target executions and 2,048 primary policy-condition evaluations.
Target averaging precedes history averaging; descriptive paired intervals resample
the 64 constructor means. No significance threshold or confirmatory claim.
The only extension reuses the identical acquisitions at budget 128, admitted solely
from measured runtime and remaining resource/deadline allowance.

Primary checking plus search shares budget 32 (128 in the extension). Final
execution, acquisition, queries, learning and storage remain separate costs.
Extra-cost checks retain the full search budget. Safe timeout submits an empty
program; this is construction failure, not an enacted belief or value change.

Independent verification reexecutes stored programs and the search enumeration,
checks selection, learning, targets and costs, and derives the complete cell and
paired-contrast tables from verified traces. Literal command tests cover resume,
source changes, corruption and ownership. No old campaign stage is invoked.

## Ownership and limits

- `v18/study.py`: evidence interfaces, paired acquisition and transfer adapter.
- `v18/runtime.py`, `runners/run_v18.py`: one worker, immutable manifest and blocks,
  pinned source, CPU/time ceilings, native owner lock and resume.
- `v18/verify.py`, `runners/report_v18.py`: independent execution and reaggregation;
  separate source-to-report checks, inspectable examples and portable traces.
- This directory: scientific design, outcome and scope; `results/v18`: compact
  manifests, admission, tables, proof and replay products.
- Codex in the commissioning conversation owns final analysis and documentary
  write-through to FINDINGS, the existing theory owners and exchange record.

One single-threaded CPU worker; no GPU, cloud or environment change. The local
acceptance record fixes the first observed work timestamp and Sunday deadline.
Maximum 18 worker CPU hours, admissions end by Sunday 13:00 UTC, delivery by
Sunday 15:00 UTC (or the earlier start-relative ceilings). Close early when done.
Healthy external computation needs no periodic model check. Existing consoleless
launcher and native ownership are reused. Process output is file-backed.

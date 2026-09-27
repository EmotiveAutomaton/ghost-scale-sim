# Documentation map

Start with the status of the work, then follow its evidence. This index describes
where documents belong; it does not replace scientific verdicts or live native
state. Updated 27 September 2026.

## Current and proposed work

| Status | Entry point | Use |
|---|---|---|
| Active V19, closeout still required | [V19 index](versions/v19-local-maker/README.md) | Accepted contract, interim synthesis, detailed catalog and primary records |
| Proposed V20, awaiting implementation approval | [V20 index](versions/v20-contribution-reconstruction/README.md) | Original September 27 specification and implementation plan; no runtime started |
| Closed V1–V18 allocations | [Version index](versions/README.md) | Specifications, reports and retained failures; historical instructions do not reopen them |

## Reading order for an operating agent

1. [Repository instructions](../AGENTS.md), then the current version's README.
2. Its accepted specification and primary status/coverage records; inspect the
   separate local handoff and actual native owners before changes.
3. The relevant [hypotheses](theory/READING_INTENT.md), their
   [format/source rules](theory/README.md), [methods](METHODS.md) and named result.
4. The [exchange](exchange/README.md) for the question and response.
5. A proposed version's plan only when working on that approved scope. A filed
   specification is not a launched or accepted campaign.

## Document responsibilities

| Location | What belongs here | What establishes authority |
|---|---|---|
| `versions/<version>/` | Original specifications, implementation plans, protocols, version reports and indexes | Accepted contract, dated decisions and linked primary evidence |
| `../results/<version>/` | Verdicts, raw/export manifests, accounting and verification receipts | Frozen identities and explicit execution/review/acceptance states |
| [theory/](theory/README.md) | Stable hypotheses, curator source material and current claim status | Evidence rows and their updated interpretation; not a new file for each experiment |
| [METHODS.md](METHODS.md) | Instrument definitions, validation logic and methodological limits | The implementation and its independent controls |
| [FINDINGS.md](../FINDINGS.md) | What was run and what was found | Named result and version report |
| [exchange/](exchange/README.md) | Requests, responses and cross-project interpretation | Authorship and source role; a response file is not proof of delivery |
| [research/](research/) and [EVIDENCE.md](../EVIDENCE.md) | Research notes and literature context | Provenance and limitations; these do not silently amend the theory |
| [archive/](archive/README.md) | Superseded prose and earlier navigation | Exact retained bytes or explicit provenance, with a replacement pointer |

The outer workspace has a separate source-document index for original intake,
private notes and machine-local material. Those files are not public scientific
evidence merely because they are nearby. Runtime sources, logs, environments and
locks stay at their existing operating paths.

## Audit passes

| Pass | Purpose | Report |
|---|---|---|
| A1 validation | Check the recorded answers | [Results](audits/a1-validation/RESULTS.md) |
| A2 diagnostics | Check whether the instruments can answer | [Results](audits/a2-diagnostics/RESULTS.md) |
| A3 repair | Record what was fixed and what changed | [Results](audits/a3-repair/RESULTS.md) |

Historical orientation: [HISTORY.md](HISTORY.md) covers the early program;
[WALKTHROUGH.md](../WALKTHROUGH.md) retains the illustrated narrative. Neither
is the live queue. [The prior documentation map](archive/documentation-2026-09-27/README.md)
is archived rather than silently discarded.

## Maintenance

Keep entry pages short: replace their current summary instead of prepending every
batch. Put complete findings in the version report, FINDINGS, theory table and
summary, and exchange as required. Preserve stable scientific paths; organize
large campaigns with a catalog rather than breaking source-bound references.
File new intake with its version immediately, preserve its bytes and provenance,
and label it proposed until acceptance. Never turn a dated snapshot into live
state by copying its counts without its date.

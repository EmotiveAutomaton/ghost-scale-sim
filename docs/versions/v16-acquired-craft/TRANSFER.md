# V16 transfer interface

B01 exports recorded synthetic tasks and runs a standalone reference consumer.
It is a task-interface handoff, not evidence about a human or language-model reader.
The finite selection is seed index zero in every condition of K01, P01, P03, O01,
O04, S02, S04, S05, R01, R03 and R05, retaining every reader arm. Selection does
not depend on success. The case catalog distinguishes success, failure, ambiguity
and misleading evidence after predictions are committed.

## Files and execution

The current transfer packet lives under `results/v16/transfer-fixture-2/`. Its public
manifest names only public observations, opaque case and lineage aliases, declared
source hashes and dependencies. Raw exports are retained under the ordinary ignored
public/private/predictions directories, with a tracked checksum manifest. The complete retained archive has its own
[coverage proof](../../../results/v16/closeout/raw-archive-2/RECEIPT.json).

The original `transfer-fixture-1` is preserved with its provenance. Its inquiry
examples came from the first instrument, subsequently invalidated by recoding
and tied-action controls. The second packet uses the separately corrected inquiry
source and consumer; it is the handoff to use. Re-exporting the other recorded
cases does not add independent scientific observations.

The versioned [reader bundle](../../../results/v16/transfer-handoff-1/reader.zip)
contains exactly the reviewed public whitelist plus a checksum inventory.
[Its instructions and receipt](../../../results/v16/transfer-handoff-1/README.md)
describe extraction and the standalone command. Evaluator mappings and committed
predictions remain outside this reader-facing ZIP.

The reader-facing distribution is the explicit whitelist in
`PUBLIC_DISTRIBUTION.json`. `PUBLIC_REPORT.json` is its safe aggregate receipt.
The original `COMPLETION.json` includes the evaluator's source-to-alias mapping;
it remains local/private and is excluded from public commits and reader delivery.

- `public/consumer/`: standalone `consumer.py` and nine byte-identical reference
  modules under `v16_reference/`. Python and NumPy are required; the export records
  the tested versions. Neither Ghost Scale nor Sounding Line must be installed.
- `public/observations/`: the bounded V16 public schema. The declared context
  contains the reference operation, its public input and reader strategy.
- `public/stage9/`: the current Stage 9 evidence shape, exactly a text prefix and
  a mapping of opaque option identifiers to full continuation strings. Every
  finite prediction option is present. Executable/open tasks have empty options.
- `private/`: complete original evaluator truth, source unit and sampling
  identities, expected reference predictions, and scoring contracts.
- `predictions/`: actual standalone outputs. An immutable commitment receipt
  precedes the evaluator's task-ID join.

From the exported consumer directory, run `python -s -B -u -m consumer`. It receives
one public JSON observation per standard-input line and returns one JSON response.
The export driver performs the same operation in a hidden owned child process.
The consumer verifies its actual copied source hashes and rejects imports from
Ghost Scale. Trusted imports finish before the CPython audit guard closes file,
process and network operations. A real private-file read is attempted and denied.
This is an access check for fixed trusted Python implementations, not a general
sandbox for hostile native extensions.

Independent cases are shuffled. Random 128-bit task and lineage aliases contain
no condition or seed encoding. Observations within an interaction retain their
recorded temporal order. Inquiry and self-resumption exports replay the recorded
public branch; they do not grant a new policy credit for an unexecuted alternative.
Returned feedback may update competence, while the evaluator's hidden command
mapping and future execution scores stay private.

## Current Sounding Line interface

Read-only inspection on 2026-09-10 used Sounding Line revision
`6c090560e01d1ddf2386d26da047ef1a08a22697`. The inspected working-file hashes are:

| Inspected file | SHA-256 of the inspected bytes |
|---|---|
| `runners/stage9/reader.py` | `c3e12059fadeb6b56b140be0c0fe35d72a2e64b9d5be50132ff426438cbbc434` |
| `runners/stage9/record_runtime.py` | `0d1d1d0d273468bc1b6cac7b1ceb8e2ee3688a4a16adfdb1abd62067787a6e6a` |
| `runners/readout_repair.py` | `60d25e5d08cffa773a96829781ea7b3396c77faecc44fba41126c0962e780034` |
| `runners/stage7/reader/contracts.py` | `bfd6b6637f82d02fc27060492727ea03f765337a7e6e7377ab949727dcd336da` |

Stage 9 requires an explicit task, model/scorer identity and evidence hash in
addition to this evidence shape. B01 supplies no invented identity or runnable
scientific lock. Its exact synthetic predictive probabilities also differ from
Stage 9's sequence-likelihood distribution conditional on offered text options.
The legacy VisibleEvidenceV1 allowlist is not the current ordinary Stage 9 loader.
Its zero-sum normalization fallback is not used here: impossible synthetic evidence
remains an explicit model mismatch.

No sibling source, data, environment, queue or lock is modified. No human data is
downloaded or copied, and no neural-reader operation is launched.

## Operations and real-record limits

This table maps candidate substrates to a possible operation, the evidence needed,
and the boundary that remains. It does not report a result or new admission.

| Candidate real substrate | Operation its recorded evidence could support | Required distinction or missing field |
|---|---|---|
| CoAuthor | Predict later handling of an offered suggestion from the current document and permitted earlier handling | Released event replay can establish insertion/handling; it does not supply a writer's acquired routines or counterfactual response. Mixed human/model agency remains explicit. |
| ScholaWrite | Predict a validated next released edit from current text and a permitted earlier edit | Continuity, time order and project-scoped author identity are required. Annotated revision labels are not the writer's reported goal; released edits are not raw keystrokes. |
| ArgRewrite | Compare artifact-only and actual before/after views; predict a genuine later whole-draft revision where it exists | Preserve essay/version groups and multiple-purpose exclusions. An after-text label cannot be moved into earlier evidence or treated as a hidden production program. |
| Genetic editions | Test useful local reconstruction against documented alternatives and partial order | Local apparatus supplies selected correspondence, not complete production chronology or all considered alternatives. |
| Creative selections | Separate produced work from released work when both and the selection record are available | Without rejected candidates and an independently recorded selector, publication bias cannot be identified from final artifacts alone. |
| Drawing records | Compare reproduction with recorded stroke/edit correspondence | A finished drawing has no observed stroke chronology. Actual recorded strokes still do not reveal all unchosen routines or considered alternatives. |
| Code revisions | Execute a reconstruction and compare it with recorded revisions or future edits | Tests can establish behavior within their scope. Commit order does not establish internal acquisition, complete editing history or the programmer's original goal. |

The first three mappings follow the inspected Stage 9 projection and source-boundary
code. The remaining mappings are proposed evidence requirements from the V16
commission. No uninspected corpus is asserted to contain those fields.

## Validation and limits

The admission tests run the exported child process, independently enumerate a
known finite prediction, execute learned commands on unseen compositions, retain
the irreducible-noise boundary, reject extra evaluator fields, deny private reads,
require prediction commitment, and detect modified predictions. Actual exported
cases must then reproduce their frozen reader outputs exactly.

The corrected packet contains 44 cases and 1,574 reader tasks; all matched the
original frozen outputs exactly. X01 exercised this actual standalone consumer
across changed aliases, interleaved requests and a fresh process. X02 checked all
1,574 public tasks under the applicable representation/physical scope. The
consumer also completed five collision calibrations, four context calibrations
and two misspecification calibrations. These are the declared fixed-consumer
controls, with their original positive/null witnesses and limits.

The 2.83 MB public ZIP was fully reread, extracted, and used to start the actual
standalone consumer. Its loaded source hashes and a mechanically selected public
task prediction matched the retained originals; an actual private-file read was
denied. This packaging check supplements the complete original task comparison.

Instrument admission, successful transport and consumer-specific
X01/X02/X03/X05/X06 coverage remain distinct from scientific expansion and confirmation. Raw aggregate
reproduction, bounded whole-unit replay, and final archive accessibility are also
separate receipts. The B01 replay does not stand in for any of them.

## Final V16 handoff

The [completed study](RESULTS.md) binds the final numerical and validation record.
The public reader ZIP above is the blind B01 task interface. The separate
[scientific replay ZIP](../../../results/v16/replay-bundle-1/replay.zip) includes
selected evaluator truth, whole-case plans and all frozen source dependencies for
reproduction. It must not be supplied as blind reader evidence. Its 196 executed
checks cover 168 distinct cases; actual extraction verified all 3,046 dependencies
and 226 source files required by 38 packets. It is bounded replay, not a rerun of
every observation in the complete archive.

The next proposed interface test would compare bounded readers under matched
information and actual computational cost after a change in production family.
That recommendation grants no new scientific admission. No real-text or neural
reader ran as part of V16, and no synthetic profile is a label for human values.

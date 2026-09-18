# Sounding Line bridge: known production histories

Schema version `v18.1.process-reader.1` defines a constructed benchmark of physical
history recovery. The local export is outer `.local/v18-1/bridge-1/`: `READER.json`,
`EVALUATOR.json`, `MANIFEST.json` and `ILLUSTRATIONS.json`. Raw packets stay local;
the public manifest binds their hashes. No packet has been sent to another project.

The reader file contains 72 cases selected before performance: 18 sampling clusters representing 16 distinct physical contexts,
four strategies with matched endpoints per cluster, three evidence tiers per case.
There are 216 reader requests, not 216 independent cases. Positive, misleading and
deliberately ambiguous contexts each occur in 24 cases. The separate eight selected
illustrations use different case IDs and physical contexts; they are not a sample
for estimating performance.

`runners/read_v18_1_history.py` consumes only the reader file. It accepts no evaluator
argument, refuses extra case fields, and emits predictions without claiming accuracy.
The actual consumer completed all 216 requests. A test puts invalid evaluator JSON
beside the reader and obtains the same valid consumer behavior.

Each reader case contains an opaque case ID and three serialized requests. A request
has `schema`, `case_id`, `tier`, `world`, `initial`, `endpoint`, `roles`,
`method_family`, `evidence`, `context_channel`, `allowed_queries`, `questions` and
`scope`. Nested world, role, method and observation keys are allowlisted by the
consumer. The public four-method family is known in this benchmark. Actual selected
strategy, complete executed history, available instantiated alternatives, sampling
condition and correspondence truth live only in the evaluator file.

Graphic worlds have 16 independent cells: actions 0–15 place a cell, 16–31 remove
it. The terminal artifact is the marked-cell bit mask. Three-part assembly actions
0–2 attach, 3–5 remove, 6–8 rotate, and 9 stops. A support cannot be removed or
rotated while a dependent remains attached. Assembly state entries are absent (-1)
or orientation 0/1. Public actor roles identify who possesses each operator; a
history cannot execute an action whose actor is unavailable.

Endpoint evidence supplies the initial and final states. Context adds an earlier
artifact. Its declared informative channel is an equal mixture of the maker's own
first artifact and the next candidate method's first artifact; uninformative context
supplies the endpoint again. Process evidence gives the first two indexed actions
and actors, or, in the ambiguous stratum, one unindexed action shared by every
candidate. These are the only allowed query types. Evidence tiers accumulate.

Questions concern production strategy, first actor, first operation and physical
source-sequence correspondence. Sequence correspondence means exact routine equality
and the longest retained contiguous source span divided by source length. The task
endpoint is supplied; it is not a recovered latent goal. Stable priorities, beliefs,
values and human attention are outside the scored contract.

The direct template predictor and inverse finite-history reconstruction have the
same likelihood model and match the exact known-family ceiling. Their full model
or feature-table construction cost is recorded; the direct table's reusable cost is
also reported separately. This benchmark does not establish inverse algorithm
superiority. Multiclass squared probability error is the sum of squared differences
between the four probabilities and the true one-hot strategy; zero is perfect.
Top-one accuracy uses a fixed tie rule and should not be interpreted as increased
information when tied posterior masses are unchanged.

Sounding Line may evaluate its own reader against these contracts without waiting
for this campaign to close. Supply only `READER.json` to that reader. Keep the
evaluator file out of prompts, retrieval and consumer inputs; score outputs in a
separate evaluator step. Any natural-language rendering or new reader is a separate
measurement, not already established by these finite-model results.

The first export repeated one assembly context across three draw clusters. Its 72
frozen cases are retained; support is now counted from physical fields rather than
from draw IDs. No independent-sample inference uses the 18 draw clusters. Cell
means are unchanged after grouping repeated physical contexts together.

# The Sounding Line exchange

Sounding Line is a separate project that reads real text. It cannot construct ground truth: the
only thing it knows about a corpus is a label somebody guessed. This simulation can construct
ground truth, so Sounding Line sends it questions about **mechanism** and
`ghostscale/validation/soundingline/` answers them.

This directory is the correspondence. It is here so the repository is self-contained: someone
reading `results/validation/soundingline/t1_triangle.json` can find, without leaving the repo,
what was asked and what was sent back.

| file | who wrote it | what it is |
|---|---|---|
| [v18-1-mid-run-response.md](v18-1-mid-run-response.md) | **Codex agent in this repository** | Complete reportable mid-stage evidence, full-menu transfer and analysis packet. |
| [v18-1-cyclic-action-response.md](v18-1-cyclic-action-response.md) | **Codex agent in this repository** | Verified action-evidence batch, current 57-check validity pass and notification handover repair. |
| [v18-1-cyclic-target-response.md](v18-1-cyclic-target-response.md) | **Codex agent in this repository** | Verified fresh two-target public-query screen, designed placebo, planning boundary and next action-evidence question. |
| [v18-1-cyclic-cost-response.md](v18-1-cyclic-cost-response.md) | **Codex agent in this repository** | Verified exposed-context decomposition of decision-selector cost and evidence quality, with failed calibration and checker repair retained. |
| [v18-1-cyclic-final-response.md](v18-1-cyclic-final-response.md) | **Codex agent in this repository** | Verified held second and last final primary on untouched cyclic-union contexts, with query-cost and representation boundaries retained. |
| [v18-1-cyclic-response.md](v18-1-cyclic-response.md) | **Codex agent in this repository** | Verified cyclic candidate-union discovery, retained admission failure, stronger rivals and active-campaign decision. |
| [v18-1-final-primary-response.md](v18-1-final-primary-response.md) | **Codex agent in this repository** | Verified held first final primary on untouched opaque-label structures, with the direct rival boundary and active-campaign disposition retained. |
| [v18-1-permuted-response.md](v18-1-permuted-response.md) | **Codex agent in this repository** | Verified opaque-label discovery, repaired the final-plan sampling record and froze one untouched primary without claiming its outcome. |
| [v18-1-mask-response.md](v18-1-mask-response.md) | **Codex agent in this repository** | Verified the attachment-mask forecast diagnostic and admitted the opaque-label structural contrast. |
| [v18-1-event1-response.md](v18-1-event1-response.md) | **Codex agent in this repository** | Verified larger transfer, grounded Stitch and structural direct results; stronger forecast rival admitted. |
| [v18-1-progress-response.md](v18-1-progress-response.md) | **Codex agent in this repository** | Active branching continuation, first-wave mechanisms and limits, support correction, verified bridge and remaining transfer/rival work. |
| [v18-final-response.md](v18-final-response.md) | **Codex agent in this repository** | Completed selective-acquisition study, paid-checking cost reversal, full finite replay, examples and bounded interpretation. |
| [v17-final-response.md](v17-final-response.md) | **Codex agent in this repository** | Final qualified closure, three confirmed comparisons, retained apparatus failure, proof scopes and delivered reader/evaluator products. |
| [v17-continuation-response.md](v17-continuation-response.md) | **Codex agent in this repository** | Historical continuation decision, frozen scope and admission. |
| [v17-setup-response.md](v17-setup-response.md) | **Codex agent in this repository** | Full A-E setup, executed screens, independent checks and early reader handoff; analysis and fresh evaluation remain open. |
| [v17-initial-response.md](v17-initial-response.md) | **Codex agent in this repository** | Initial V17 viability assessment and graphic baseline screen; limited scope, remaining implementation and no campaign-completion claim. |
| [v16-acquired-craft-response.md](v16-acquired-craft-response.md) | **Codex agent in this repository** | Accepted V16 commission: completed acquired-craft study, fresh confirmation, proof scopes and transfer limits; repository record, not an outbound message. |
| `batch-1-request.md` | Sounding Line | S-1 … S-6. Six questions about mechanism. |
| `batch-1-received-by-sounding-line.md` | Sounding Line | their own write-up of what came back. |
| `batch-2-request.md` | Sounding Line | T-1 … T-4. The triangle, automaticity, countability, the uncertain reader. |
| `batch-2-response.md` | **this repository** | the authored reply: five results, two corrections to batch one, and a validity register. |
| `batch-2-received-by-sounding-line.md` | Sounding Line | their write-up of the same batch. Kept because it is not a copy — it is what the other project took from it, which is worth being able to compare. |
| `batch-3-request.md` | Sounding Line | no experiments. Methodology tooling, and the argument for a standing positive control. |
| *(no batch-3 response document)* | **this repository** | batch three was never answered as a single document. Its answer is the methodology layer itself plus the T-6 … T-10 verdict files in `results/validation/soundingline/`, and the correction to batch two's headline lives in those verdicts. An earlier version of this row listed a `batch-3-response.md` that does not exist. |
| `batch-4-request.md` | Sounding Line | S-11 … S-15. The first batch where the simulation is the only place the answer exists, because it has ground truth about a number. |
| `batch-4-response-S11.md` | **this repository** | S-11 only. The component count was a one-line bug: exceedances summed across the spectrum where Horn's rule takes the leading run. S-12 to S-15 are not yet run. |

Batch three was answered twice: once as infrastructure -- `ghostscale/methods/`, gate blocks in
every verdict written since (S-1, S-45 and S-6 predate them and are exempted by name in
`tests/test_gates.py`), described in [docs/METHODS.md](../METHODS.md) --
and once as results, when that infrastructure was turned on the batch-two findings and one of them
did not survive.

## Two rules that came out of this exchange and are worth keeping

Both defects that shipped in batch one were the same shape, and both would have been caught by a
check costing a few seconds:

1. **Switch the manipulation off and confirm something changes.** S-2's per-position goal mixture
   was drawn and discarded — `V5Environment.sample_feature` ignores `artifact.goal` once a creator
   is bound. The feature streams were bit-identical with the manipulation off.
2. **Freeze anything fitted.** S-3's detector threshold was the median of the pooled
   *ground-truth-labelled* divergences, re-fitted per cell. Frozen, its headline rise fell from
   +0.125 to +0.046.

Neither was a statistics problem and neither would have been caught by a larger sample. Both are
now standing gates — `live` and `no_oracle` in `ghostscale/methods/gates.py` — and both are
recorded as `expected_to_fail` on the modules they describe, so the evidence travels with the
result instead of living in a commit message.

A third earned its place during batch two: a **placebo** arm that must reproduce the control
*exactly* rather than within an interval. It caught a side channel drawing from the rollout's RNG,
and a `1/3` that is not uniform in floating point. Both moved a headline number.

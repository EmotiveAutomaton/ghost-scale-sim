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
| `batch-1-request.md` | Sounding Line | S-1 … S-6. Six questions about mechanism. |
| `batch-1-received-by-sounding-line.md` | Sounding Line | their own write-up of what came back. |
| `batch-2-request.md` | Sounding Line | T-1 … T-4. The triangle, automaticity, countability, the uncertain reader. |
| `batch-2-response.md` | **this repository** | the authored reply: five results, two corrections to batch one, and a validity register. |
| `batch-2-received-by-sounding-line.md` | Sounding Line | their write-up of the same batch. Kept because it is not a copy — it is what the other project took from it, which is worth being able to compare. |
| `batch-3-request.md` | Sounding Line | no experiments. Methodology tooling, and the argument for a standing positive control. |
| `batch-3-response.md` | **this repository** | what the tooling found when pointed at the existing results: T-6 to T-10, and a correction to batch two's headline. |
| `batch-4-request.md` | Sounding Line | S-11 … S-15. The first batch where the simulation is the only place the answer exists, because it has ground truth about a number. |
| `batch-4-response-S11.md` | **this repository** | S-11 only. The component count was a one-line bug: exceedances summed across the spectrum where Horn's rule takes the leading run. S-12 to S-15 are not yet run. |

Batch three was answered twice: once as infrastructure -- `ghostscale/methods/`, the gate blocks
now in every verdict, and `tests/test_gates.py`, described in [docs/METHODS.md](../METHODS.md) --
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

# Recursive predictive-bank updates — independently verified

Can a predictive bank update from new observations without losing useful forecasts? At 32 independent episodes, the learned updater lowers logarithmic loss by 0.01369 nats versus cached history and 0.06668 versus a fixed recurrent state, but remains 0.02396 nats worse than the privileged exact-previous-bank diagnostic. The history comparison reverses at eight episodes. This is an exploratory constructed-method result, miniature — architecture untested; teacher privilege and initialization remain rivals to accumulated update error, and no historical-process or human-intent conclusion follows.

Each episode resets the three-unit artifact while retaining the same maker. Only
context and endpoint are observed. Streams have 32 episodes, with paired prefixes
of 8/32 observations. The copy condition repeats every second observation and
source identity; all readers skip it. It therefore contains 4/16 unique observations
at these scored prefixes. It does not test unknown source dependence.

Thirty-two training and eight development coefficient lineages, two training draws,
two feature seeds and eight streams per lineage are retained. Every head learns
the same exact observable endpoint distributions. The updater additionally receives
the exact preceding bank during training; the teacher-forced evaluation receives
it at evaluation too. The free-running bank starts at the mean training prior,
whereas the teacher has the correct world-specific preceding bank. No maker labels
or evaluation truths enter ordinary fitted inputs. The recurrent transition is a
fixed 128-coordinate reservoir; only its forecast head is learned. This is not a
trained recurrent transition or tiny causal sequence model.

Logarithmic loss measures forecast error in nats; lower is better. The table gives
bank-minus-rival differences with 95% paired lineage intervals. Rows specify source
condition and prefix length; columns identify the rival. Both feature seeds and
training draws remain inside each of the eight resampled lineages.

| Source condition and length | Bank minus cached history | Bank minus fixed recurrent state | Bank minus exact-previous-bank diagnostic |
|---|---:|---:|---:|
| Independent, 8 observations | 0.00569 [0.00259, 0.00895] | -0.01899 [-0.02232, -0.01596] | 0.02285 [0.01741, 0.02879] |
| Independent, 32 observations | -0.01369 [-0.01951, -0.00739] | -0.06668 [-0.08038, -0.05299] | 0.02396 [0.01879, 0.02946] |
| Copied, 8 observations | -0.02082 [-0.05250, 0.00395] | -0.00245 [-0.01274, 0.00475] | 0.01801 [0.01393, 0.02336] |
| Copied, 32 observations | -0.00284 [-0.00477, -0.00013] | -0.04140 [-0.04824, -0.03456] | 0.02946 [0.02083, 0.03817] |

At 32 independent episodes, bank loss is 1.91879, cached history 1.93248,
fixed recurrent state 1.98548, no-history mean 2.00209 and privileged exact
filter 1.89320. The bank/direct conditional interval lies within ±0.02 nats,
so this is a small ranking advantage within the declared practical margin.
At eight independent episodes, cached history has the lower loss. The bank has
useful history relative to the mean, but no universal recursive superiority follows.

The independent-source free/teacher gaps are 0.02285 at eight and 0.02396 nats at
32. This does not establish growth of drift with length. It identifies a useful
training-input and initialization diagnostic for the single rolled-in successor.
In copied streams both scored positions are duplicates. The teacher-forced arm
then returns the exact preceding bank directly: its score is an oracle pass-through,
not a learned one-step update. Its zero squared error and tiny difference from the
floored exact reference follow from that implementation. The record is preserved,
and this condition is excluded from claims about learned teacher-forced competence.

Raw forecast validity is separate from repaired scores. At 32 independent episodes,
the free bank requires repair for 1.76% of query distributions and the teacher-forced
updater for 9.77%; the latter can still have lower repaired loss. Full-history and
cached-count inputs and forecasts are identical by construction and verification.
Counting work favors caching: 528 prefix-observation visits versus 32 sequential
visits for the full 32-step trajectory. These are algorithmic counts, not an
end-to-end measured speedup or fit-cost break-even. The bank stores 32 predictions
and a count, the cache 32 counts and a count, and the reservoir 128 state entries
and a count, excluding model parameters and source-deduplication storage.
Component timings and all four fit records remain in the scientific evidence.

**Validation:** all 896 score rows reconstruct within 9e-16; all 56 saved forecast
files reconstruct exactly from saved parameters. Exact filtering over the retained
endpoint laws reproduces 40,960 training/development prefixes. All full-history/
cache, copy-skip and role allowlist checks pass. Adjacent and freshly extracted-
source complete replays match every deterministic file. This independent check
does not re-enumerate every transient path or refit the heads: those are covered
by reproducibility and source inspection, not independent scientific derivation.

**Warrant:** exploratory method advantage/reversal, conditioned on the stipulated
stationary worlds and privileged training teacher. **Pursuit:** implement the
previously prepared single roll-in pass, with a two-pass exact-teacher control,
and the independent task/skill projection diagnostic. Neither needs a tiny model
or a bank win. A future longer horizon requires its own source-bound manifest.
No preceding joint-process reconstruction, human intent or practice benefit is tested.

[Scientific evidence and separate reader export](../../../results/v19/G19-B1-update-1/EXPORT_MANIFEST.json).

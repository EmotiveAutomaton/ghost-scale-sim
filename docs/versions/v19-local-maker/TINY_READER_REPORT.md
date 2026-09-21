# Small sequence readers pass the predictive capability pilot

Can small sequence readers retain useful predictive information from observed local-world episodes? Both readers pass all declared capability checks. At 32 independent observations, forecast loss falls by 0.08342 nats for the transformer and 0.09496 for the recurrent reader relative to the fixed training mean. This establishes a constructed predictive instrument, not local-process correspondence, selective causal access or human intent.

The task predicts a fresh three-unit artifact in four public contexts from earlier
visible endpoint tokens and supplied copy flags. Training uses exact conditional
observable-distribution teachers on 32 training coefficient lineages. Development
uses eight separate lineages, two training draws and two initialization seeds per
architecture. No maker labels, previous exact bank, law matrix, lineage identity
or development target enters the child learner. The setting is still a simulator
with distribution teachers; it is not learning solely from sampled outcomes.

Each draw provides 512 streams across independent and copied conditions, with
12,288 unique prefixes and four target context distributions per prefix. Both
models receive the same data. A one-block, four-head transformer has 10,656
parameters; a one-layer recurrent reader has 10,512. Both train for 32 epochs,
512 optimizer updates, using the frozen final checkpoint. The two architectures
consume the combined two-setting allowance; the eight paired fits are instances
of these settings, not eight searched architectures. No further setting is admitted.

The table reports logarithmic forecast loss, in nats, averaged over four questions,
eight development lineages and both training draws; learned arms also average two
seeds. Lower is better. Rows give evidence condition and observed positions; copied
streams contain half as many unique observations. The exact-law column is a
privileged conditional reference, and the mean column ignores the input history.

| Evidence | Positions | Fixed mean | Transformer | Recurrent | Exact law |
|---|---:|---:|---:|---:|---:|
| Independent | 8 | 2.00145 | 1.94612 | 1.94035 | 1.93136 |
| Independent | 32 | 2.00209 | 1.91867 | 1.90713 | 1.89320 |
| Copied | 8 | 1.99664 | 1.96062 | 1.95751 | 1.95062 |
| Copied | 32 | 2.00218 | 1.92956 | 1.91751 | 1.90543 |

All 32 fit/condition/length checks pass the predeclared capability rule: at least
0.02 nats better than the mean, at least half its gap to the exact reference
recovered, and nonconstant predictions. This is an engineering admission, not a
confirmatory significance threshold. The lowest individual improvement is 0.03493
nats. Every individual recurrent-versus-transformer contrast favors the recurrent
reader; its advantage is smaller than the 0.02-nat practical margin.

At 32 independent observations, paired 95% coefficient-lineage intervals for
mean-relative improvements are [0.06216, 0.10314] for the transformer and
[0.07115, 0.11714] for the recurrent reader. Recurrent-minus-transformer loss is
-0.01154 nats [-0.01530, -0.00806]. With copied observations the corresponding
improvements are 0.07262 [0.05631, 0.08772] and 0.08467 [0.06473, 0.10306]; the
architecture difference is -0.01206 [-0.01591, -0.00813]. These intervals average
within paired lineages and condition on two seeds and two draws; repeated queries
are not independent worlds. Every fit's gain, headroom and variation is retained
separately. At 32 independent observations, transformer gains range 0.07623–0.09076
and recurrent gains 0.08653–0.10344; neither interval captures a universal training
population. No untouched test or confirmation lineage has been consumed.

Independent NumPy equations reconstruct GRU updates, causal attention, feedforward
layers, normalization and output probabilities from the saved parameters without
Torch or producer imports. All development states and forecasts agree within
9.3e-7 and 1.8e-7 respectively; all 384 proper-score strata agree within 1e-12.
Copied positions preserve identical hidden states and probabilities. Both full
training replays match all 54 deterministic files, including every parameter array,
forecast and learning curve. Execution-specific optimizer/RNG checkpoint files
are separately hash-bound, not required to have identical serialization bytes.
Resident and isolated Torch admission controls, including known-answer learning,
causal-prefix invariance and checkpoint continuation, precede scientific dispatch.

The first review failed an incorrect hard-coded training-count assertion (half the
actual input count). The corrected verifier derives counts and optimizer steps
from frozen input shapes. That failed review and its cost are retained; no fitted
artifact, criterion or frozen worker source changed. Independent verification does
not reconstruct every optimizer step; the two complete source replays check that
trajectory. No intervention alignment, change/stay counterfactual fixture, local
goal readout or joint historical process evaluation has been performed here.

**Warrant:** exploratory constructed-method capability advantage against a fixed
mean, conditional on the declared distribution teachers and source flags.
Miniature — architecture untested. The models are now candidates for a separate
causal-access admission; prediction alone cannot establish selective mediation.
**Pursuit:** freeze donor/recipient factor fixtures before any interchange fit.
Independent whole-stream purchase calibration and fixed-size portfolio selection
remain useful alternatives, so the week is not complete after this pilot.

[Separate reader, training-teacher and scientific/evaluator evidence](../../../results/v19/G19-06-tiny-reader-1/EXPORT_MANIFEST.json).

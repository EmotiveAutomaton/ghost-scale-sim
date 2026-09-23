# Frozen goal-confidence calibration: execution contract

This implements the prepared fixed-bin protocol, using accepted goal-decision
forecasts and native targets. All sixteen development laws, two training draws,
five fit seeds, four budgets, four readouts and two forecast variants are retained.
Only complete witnessed histories are admitted. No fit, sampling, new teacher,
protected lineage or tiny setting is used.

At each of three positions, select the smallest-code fine goal attaining the
largest saved marginal probability. Independently check these marginals against
the saved full fine-goal distribution before use. Save the selected goal, raw
confidence and native expected correctness for every frame and lineage.

Use bins [0,.1), [.1,.2), ..., [.9,1], represented by the fixed binary64 edges
computed from integer tenths. Exact interior edges enter the higher bin; only the
last bin includes one. Values within 4e-12 above one retain their saved value and
enter the final bin. Only proper binary losses use a separate probability clipped
to [0,1] within that normalization tolerance; record the clipped mass/count.
Negative values and larger excess are instrument errors. Native correctness has
the same tolerance and explicit proper-score clipping. No tolerance changes modes.

For each lineage and step, preserve native frame weights and equal-frame weights
as separate populations. Save all ten bin weight, confidence and correctness sums,
including empty bins. Empty bin means are undefined. Signed calibration error is
weighted confidence minus native correctness. Absolute bin discrepancy sums the
absolute confidence-minus-correctness mass within each bin. Squared confidence
error averages the squared difference between confidence and native correctness.
Expected binary Brier loss is c(1-p)^2+(1-c)p^2; binary logarithmic loss is
-c log(p)-(1-c) log(1-p). These proper losses target whether the selected goal is
correct, not which goal is true. Impossible positive-weight success/failure terms
give infinite loss, encoded as a null finite value plus explicit infinite flag
and incompatible mass, never silently truncated. Zero-weight terms contribute zero.

Fine-goal logarithmic loss and native modal accuracy must reproduce every parent
cell. Native self-forecast reports have zero calibration discrepancy; their proper
loss remains native uncertainty. Uniform, known overconfidence, reversed confidence,
empty bins, exact/adjacent edges, normalization drift, impossible events, zero
weights, roster corruption and reader/evaluator split require isolated controls.

Compare the bank with raw history, frozen latent state and matched frequencies at
each budget, forecast, weighting and step. Fixed seed 191014 gives 10,000 paired
lineage resamples, averaging draws/seeds inside each law. Preserve fit variation
separately. Bootstrap only finite metrics; any nonfinite pair is explicitly
undefined, with its count. Summarize bin sums across the roster before dividing;
pooled bin discrepancy differs from the mean of lineage-wise discrepancies and
both remain labeled. Log-budget areas retain the fixed normalized trapezoid rule.

Source/input freeze, isolated controls and conservative timing for original,
adjacent replay, extracted-source replay, independent reconstruction and operating
work must fit the inclusive 1,800 CPU-second D card. Execution uses the existing
single-worker queue. Reader evidence contains no confidence, target or score.
This is exploratory constructed-method calibration, not human intent or historical
goal correspondence. Retrospective evidence updating and complete class-wise goal
reliability remain independent prepared alternatives.

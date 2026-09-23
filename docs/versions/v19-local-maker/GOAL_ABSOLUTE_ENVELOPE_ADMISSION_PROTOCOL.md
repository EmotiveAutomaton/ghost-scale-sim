# Absolute residual envelope: executable admission

How much frame-level absolute probability error is hidden by averaging within
fixed probability bins? Implement the prepared envelope protocol on unchanged
goal-decision forecasts. Use the accepted five-, ten- and twenty-bin partitions,
integer-ratio binary64 boundaries, and probability one in the last bin.

For every law, population, position and goal, compute the weighted sum of the
absolute forecast-minus-native residual before bin averaging. Subtract each
accepted binned absolute discrepancy. The triangle inequality makes these gaps
nonnegative, and nested refinement makes them nonincreasing, up to -1e-12
roundoff tolerance. Preserve negative roundoff rather than clipping it.
Retain per-frame signed residuals and weighted absolute contributions in separate
evaluator arrays; unchanged reader packets contain none of these fields.

Use all sixteen development laws, two training draws, five fit seeds, four label
budgets, four readouts, two forecasts, three positions and three goals. Native
and equal-frame populations remain separate. Expected outputs: 92,160 rows,
320 group bundles, 320 forecast bundles, 320 raw residual bundles, 225,280 frame
forecasts, 704 reader packets and 5,120 parent loss/accuracy identities. Reproduce
all accepted five-, ten- and twenty-bin discrepancies before interpreting gaps.

Freeze bootstrap seed 191021 and 10,000 paired sixteen-law resamples, keeping
both draws and five seeds inside each law. Report 576 means, 1,728 gap estimates,
3,024 predictive-bank versus baseline budget contrasts and 756 normalized
log-budget areas. Preserve draw and seed variation separately. No practical
threshold, adaptive partition, fit, new sample or protected lineage is admitted.

Controls cover constant signed error, opposing errors hidden in all partitions,
unequal weights, zero weights, native self, frame/class permutations, every bin
boundary, probability one, independent scalar absolute sums and scalar signed
bin sums. Include deliberate parent-score corruption, full synthetic execution,
input and roster failures, native runtime and portable dispatch checks.

Require both complete replays, independent scalar reconstruction of all raw
arrays and scores, parent identities and original/reconstructed regrouping,
followed by a third direct-index paired regroup before numerical acceptance.
The inclusive 1,800 CPU-second D card includes failures, tests, benchmark,
execution, replays, review and export. A full-size synthetic benchmark including
raw-array compression must establish tractability before source admission.

This is an exploratory constructed-method diagnostic, miniature — architecture
untested beyond the declared roster. It bounds the effect of aggregation without
improving a forecast, identifying historical goals or establishing human intent.
Retrospective updating and exact forecast-collision uncertainty remain independent
prepared alternatives.

# Squared residual decomposition: executable admission

Does a small average error inside a probability bin hide variation among its
frames? Implement the prepared squared-error protocol on the accepted frozen
goal-decision forecasts. Use ten integer-tenth bins with probability one in the
last bin. Preserve raw binary64 probabilities without clipping or recalibration.

For each law, population, position and goal, retain bin mass, weighted first and
second residual sums, conditional residual means, second moments and variances.
Empty bins have zero mass/contribution and undefined conditional moments. The
weighted squared mean residual and the weighted within-bin variance sum to the
unchanged unbinned mean squared error. Preserve small negative roundoff instead
of clipping it; values below -1e-12 fail the numerical guard. This is a diagnostic
against evaluator probabilities, never teacher information given to a reader.

Use all sixteen development laws, two training draws, five fit seeds, four budgets,
four readouts, two forecasts, three positions and three local goals. Native and
equal-frame populations remain separate. Expected outputs: 92,160 metric rows,
320 moment bundles, 320 forecast bundles, 225,280 frame forecasts, 704 reader
packets and 5,120 parent score identities. Retain native arrays separately.

Freeze bootstrap seed 191020 and 10,000 paired sixteen-law resamples, keeping both
draws and five seeds inside each law. Report 576 means, 1,152 within-readout
component estimates, 1,296 predictive-bank versus baseline budget contrasts and
324 normalized log-budget areas. Preserve draw and seed variation separately.
No practical threshold, effect-selected bins, new samples or fits are admitted.
Untouched confirmation and test lineages remain reserved.

Validate constant signed error, opposing error with equal and unequal weights,
native self, empty bins, boundaries, probability one, permutations and independent
pairwise variance. Test input/roster corruption, parent identity, existing runtime
and complete portable replay dispatch. Before numerical acceptance require both
complete replays, independent scalar moment and centered-variance reconstruction,
all output bindings, parent score reconstruction and a third direct-index regroup.
Original and rebuilt outputs, source, plan, environment, tests, failures and timing
remain distinct retained evidence. Reader packets exclude all evaluator truth.

The inclusive 1,800 CPU-second D card covers tests, timing, failures, execution,
replays, independent review and export. Measured synthetic timing must fit it
before admission. Use the existing serial queue, one numerical thread, CPU only
and below-normal priority. This is an exploratory constructed-method diagnostic,
miniature — architecture untested beyond the roster. It cannot establish
historical goals, process correspondence or human intent.

Retrospective evidence updating and the unbinned absolute-error envelope remain
independent prepared alternatives; neither is implemented by this handler.

# V19 update review — 21 September 2026

Can a predictive bank update from new observations without losing useful forecasts? At 32 independent episodes, the learned updater lowers logarithmic loss by 0.01369 nats versus cached history and 0.06668 versus a fixed recurrent state, but remains 0.02396 nats worse than the privileged exact-previous-bank diagnostic. The history comparison reverses at eight episodes. This is an exploratory constructed-method result, miniature — architecture untested; teacher privilege and initialization remain rivals to accumulated update error, and no historical-process or human-intent conclusion follows.

[Complete comparison, controls and limits](../versions/v19-local-maker/UPDATE_REPORT.md). Both complete replays and independent numerical reconstruction verify; reserved lineages remain untouched.

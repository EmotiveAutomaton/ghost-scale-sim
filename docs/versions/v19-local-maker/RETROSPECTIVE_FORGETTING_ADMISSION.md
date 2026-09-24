# Bounded retrospective forgetting: prospective admission contract

Does a recent-source window retain the state needed to interpret a later report
about any earlier observation? The comparison is fixed before native outcomes.
Use all 28,672 retained posteriors, 505,856 sources, eight development laws, both
evidence conditions, two draws and seven available checkpoint/horizon cells.
Preserve the 48 unavailable checkpoint strata. No new fit or observation.

The four arms retain all sources, sources after half the original checkpoint
time, sources after three quarters of that time, or none. Strict integer time
cutoffs are floor(checkpoint/2) and floor(3*checkpoint/4); selection uses original
times, including gaps where duplicates were excluded. Each original source keeps
probability 1 divided by the original source count. A report either copies its
original endpoint or is generated independently, at the five existing copying
probabilities 0, 0.25, 0.5, 0.75 and 1. All eight endpoint reports are evaluated.

Recent sources retain their joint past-state/future-group distribution. Forgotten
sources retain their total mass, original endpoint histogram and context counts.
Their independent likelihood uses the original uniform sixteen-maker prior,
without history or future-group conditioning. Copy counts from all sources are
retained. The remainder therefore multiplies the existing future-group weights
by a constant report factor; it is never deleted or renormalized away. This
explicit approximation, including its supplied-law privilege, is the estimand.

The reference executor evaluates these compact statistics through an equivalent
full-hypothesis calculation. Independent fixture tests materialize the joint tables.
Float64 storage counts include future-group weights and mixed joint cells for
each retained source; deterministic cells reuse group weights. Int32 storage
counts include twelve histogram counts, three indices per retained source and
two indices per structural joint cell. Counts are conservative without cross-time
deduplication. Shared prior probabilities (32 float64 values), law tables, schedule
structure and evaluator source bindings remain separate. Raw evidence retains
all source/window identities, remainder masses, histograms, report probabilities,
support masks, regret, total-variation bounds and storage counts. Immutable inputs
reconstruct every forecast. Reference timing is not optimized compact latency.

Evaluate expected multiclass squared-loss regret under the correct report law,
mean report-weighted largest future-probability error, and updated future-group
total variation. Squared errors sum over eight endpoints and average every future
time and four contexts. Exact unsupported report mass is separate from supported
regret. Undefined raw values remain NaN. Each of 128 Cartesian identities gets
equal weight, two draws stay inside a law, and eight laws get equal weight.
Evidence, checkpoint, copying probability and window stay separate. No practical
margin or confirmatory interval is specified: outcomes are descriptive estimates.

Admission requires scalar Bayes and explicit one-hot proper losses, direct compact
joint-table reconstruction, dense/sparse/constant/disjoint laws, source permutation
with preserved times, full-retention and certain-copy identities, an informative
older-source positive, omitted-remainder corruption, full native fixture integration
and portable dispatch. Full-support synthetic timing covers all seven structural
cells and both source densities with three complete 32-row archived batches each.
The inclusive card cap is 3,600 CPU seconds for all preparation, attempts, original,
two complete replays, independent reconstruction and final review. Source, input,
architecture, zero fitted capacity, teacher access and schedule freeze together.

The coding package permits an informative B-family continuation up to 24 CPU
hours after timing. This card releases that existing provision only after its
complete-support timing passes. The 80 exploratory and 96 total CPU-hour limits
and protected 16 hours remain unchanged. No confirmation lineage is consumed.

Two-report dependence and context-balanced retention are separate prepared
alternatives. A failure of this comparison does not close those questions.
The numerical result requires independent reconstruction and original-row regroup
after both complete replays. This is a supplied-law constructed-method experiment,
miniature — architecture untested beyond its roster, with no learned provenance,
historical process correspondence or human-intent claim.

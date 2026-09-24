# Exact retrospective provenance intervals: implementation contract

Does bounded uncertainty about copying leave useful bounds on future claims?
The accepted source-mixture factors fix the population: all eight laws, both
posterior source conditions and observation draws, seven available checkpoints,
28,672 posterior rows and all 505,856 distinct-source queries. No missing cell
is filled. Use the previously prepared intervals [0,1], [0.25,0.75], [0,0.5]
and [0.5,1], all eight report endpoints and every remaining future coordinate.

At the old endpoint, the posterior equals the old posterior plus an independent
update difference times (1-alpha)*p/(alpha+(1-alpha)*p), where p is the independent
probability of that endpoint. This coefficient decreases with alpha. Its endpoint
values give the exact posterior segment, every coordinate envelope and the worst
discrepancy of the interval midpoint or either endpoint. At other endpoints the
posterior is the independent update whenever supported. A copying probability of
one excludes those reports; it is not silently assigned an independent posterior.

Retain complete segment endpoints, midpoint coefficient, report probability bounds,
all support masks and old-endpoint widths and worst midpoint differences. Parent
source identities, laws, posterior arrays and signed-update reconstruction remain
bound. The complete future-coordinate maximum and mean total variation scale
exactly along this one-dimensional segment; no future coordinate is sampled or
replaced by a total-variation bound. Individual coordinate extrema need not be
simultaneously attainable as an arbitrary rectangular combination.

Each source retains old-endpoint width and an equal-supported-endpoint mean,
plus the fraction of otherwise possible endpoints excluded by each fixed endpoint
decision. Equal endpoint weighting is descriptive counting, not a report likelihood
under an unknown copying probability. Average sources within each posterior, then
128 rows per draw, keep two draws paired within laws, and weight eight laws equally.
No fitted architecture, added sample, random seed or protected lineage is used.

Before dispatch require direct hypothesis Bayes calculations over interior alpha
values, all future coordinates and endpoints on synthetic fixtures; constant,
sparse and disjoint laws; degenerate and invalid intervals; impossible reports;
endpoint inclusion and correct exclusion of impossible alpha/report pairs; complete
producer integration and changed-input/missing-factor rejection. Freeze source,
input hashes, capacity and this schedule after isolated controls and complete
synthetic timing. The inclusive cap is 3,600 CPU seconds, including failures,
verification, both complete replays and independent reconstruction. The current
B-family and global ceilings and protected reserve remain controlling.

The completion reviewer must independently reconstruct scalar interval bounds,
coordinate extrema and fixed-point worst-case discrepancies, verify both complete
replays, and separately regroup original rows before numerical acceptance. This
is a supplied-law constructed-method certificate; no learned provenance probability,
historical process correspondence, minimal state or human intent is established.

Two-report dependence remains prepared. Unknown source identity is a second
independent alternative in [its protocol](RETROSPECTIVE_SOURCE_IDENTITY_PROTOCOL.md).

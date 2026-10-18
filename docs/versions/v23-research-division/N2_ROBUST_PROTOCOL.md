# D10: descriptive robust reanalysis of N2

Declared October 18 under item D10 of the October 17 packet. No new discovery
cases, no reopened stopping decision, no confirmation. Reuse both complete
N2 burn-ins: 124 settings, 32 whole worlds and 64 cases per world. Four structural
settings stay visible and are excluded from the 120-setting descriptive count.

Choose a generic fixed likelihood exponent separately for each setting from
V19's declared roster 1/.75/.5, minimizing log loss on that setting's rows in
the existing independent 4,096-label pre-discovery validation bank. This gives
33 or 34 labels per setting, an explicit tuning limitation. Preserve its hash,
row assignments, losses and selected exponent before reading test predictions.
The exponent applies uniformly to all records; it does not inspect response
times, latent mark types, or test labels. It is not the original comparator's
diagnosticity-matched tempering.

Reconstruct the naive and timing-weighted log odds from public choices and the
already retained timing weights. Verify predictions and stable original log
losses against saved evidence before computing the new comparison. Use stable
softplus arithmetic and an independent scalar calculation, without score floors
or clipping. Retain every new case loss and its source identity. Require all
253,952 cases and both original raw gain series to reproduce.

For each setting pool the 32 whole-world means equally. Estimate variance of
its mean from those worlds. In the across-setting inverse-variance summary,
replace each variance by the maximum of itself and the median across the 120
nonstructural settings. Normalize weights by the median before division to
avoid overflow; reject nonfinite inputs or a zero/nonfinite median. Report the
floored descriptive mean for the original and generic-tempering comparisons.
This is a descriptive weighted summary, not the original hierarchical posterior.

Report the share of the 120 settings whose point gain over generic tempering
exceeds .02 nats, with all per-setting gains and whole-world bootstrap intervals.
This is a descriptive above-margin count, not a formal partial-conjunction test
or multiplicity-adjusted claim. Retain original fixed dispositions unchanged.
Full fresh-process replay, independent case-score reconstruction, identity and
coverage checks, and known-answer floor/overflow controls precede acceptance.
Cap both passes and checks within 1.5 CPU-hours, charged to n2 if available,
otherwise conditional as authorized. Constructed-method reanalysis only.

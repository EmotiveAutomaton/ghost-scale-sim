# M: first numerical healing, structural zero variance

The first retained M failure is a false finite per-maker short-gap correlation.
In study world 150 of M-CORE-0004, maker 72 has three pairs sharing the same
first observed value. Its first-value variance is mathematically zero. The
original sum-of-squares route reports unavailable; the independent route's
roundoff gives a tiny positive denominator and includes the maker in an average.
This changes the average and its interval enough to fail the unchanged gate.

Use a centered streaming covariance for the per-maker short-gap correlation and
an independent two-pass centered calculation. Both explicitly detect identical
values before division. No tolerance clips small but nonzero variance. Pooled
pair measures, Kalman readers, data, model, seeds and all validation gates remain
unchanged. Retain both calculations and all failed or stopped cells.

Before admission, test the identical-anchor unavailable case, a triangle with
known correlation one half, and independent centered arithmetic, alongside the
original controls. Replay every retained M original with a completed summary
and raw manifest, reconstruct it independently, and require unchanged raw bytes.
Only the diagnosed per-maker mean may change. Repeat the largest eight-world
full replay/reconstruction cost profile. The excluded validation uses one Gear 2
slot and at most 1,800 CPU-seconds from healing.

Only a passed, source-bound validation admits the original 552 settings as M_R1.
The two previously complete settings are affected by the instrument hole and
receive corrected successors; they are not new independent replications. All
other settings retain their original planned seeds. Keep the 15-hour M ceiling,
forty excluded pilots, 200 scientific worlds and eight blocks per cell. This is
M's single automatic repair; a failure stops further automatic M repair.

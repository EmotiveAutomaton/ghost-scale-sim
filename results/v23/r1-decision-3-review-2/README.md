# R1 discovery review

The [family report](../../../docs/versions/v23-research-division/R1_REPORT.md)
records a discovery nomination, not confirmation. All 36 settings and 101,376
cases are covered; three original/replay pairs have distinct process owners.

- `REVIEW.json`: numerical acceptance, packet/replay/source hashes and fit metadata.
- `CURVE.json`, `SPECIFICATION_CURVE.svg`, `HEATMAP.svg`: every setting, raw and
  shrunk estimates, descriptive intervals and design axes.
- `TRAJECTORY.json`: the fixed-policy decisions after the three accepted batches.
- `DIAGNOSTICS.json`, `READER_STRATUM_SCORES.csv`: independently reconstructed
  proper scores, source-goal strata, primary, skepticism and matched direct gaps.
- `SECONDARY_DIAGNOSTICS.json`, `SECONDARY_SCORES.csv`: per-batch content, purpose
  and technique diagnostics bound by the accepted full replay.
- `RAW_HASH_INDEX.json`: every scientific original/replay file hash; only CPU
  timing is excluded. Raw evaluator data and immutable source archives are retained
  separately, with truth inaccessible to readers.
- `ARTIFACT_INDEX.json`: hashes of the initial review/visualization products.
- `ENGINEERING_VALIDITY.json`: sixteen scoped controls for the later confirmation
  implementation and its unchanged repair pipeline; no production confirmation.

Reader log loss uses natural logarithms; lower is better. Half-Brier is half the
sum of squared probability errors; accuracy is the correct modal forecast rate.
Source goal zero means informing, one means steering. Positive primary gain is
masked-reader loss minus visible-reader loss. Other contrast orientations are
stated in the report. Intervals are descriptive whole-world summaries after
adaptive discovery and do not claim simultaneous coverage or a human effect.

Review attempt 1 stopped on a scoring-key mismatch before writing a result. The
corrected read-only review used the same saved science and produced these files;
the failed helper's charge is retained separately. No accepted worlds were rerun.

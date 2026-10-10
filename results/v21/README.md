# V21 public evidence

[Current status](CURRENT_STATUS.json) is a dated snapshot. Native process ownership,
heartbeats, immutable plans, raw forecasts and private operations remain in the
separate local campaign directory. No status count substitutes for a live check.

- [Opening duplicate-evidence audit](opening-duplicate-audit/ACCEPTANCE.json):
  786,432 verified forecasts on reused V20 worlds, with complete frozen replay.
- [Admission controls](ADMISSION_VALIDITY.json) and
  [verifier I/O repair](VERIFIER_REPAIR.json).
- [Frozen source archives](SOURCE_BINDINGS.json): opening and core source hashes.
- [Program and accepted protocol](../../docs/versions/v21-supporting-mechanisms/README.md).

The opening casebook has separate `reader/` and `evaluator/` directories. Only
`reader/` is a blind input. Forecasts, targets, truth, exact posteriors and summary
outcomes are evaluator material even when the data are synthetic. Raw numerical
arrays stay local under completion hashes; public files contain aggregates,
provenance, validity and bounded readable cases.

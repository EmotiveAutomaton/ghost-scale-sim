# V22 B1: Costly departures and set-point recovery

Do costly departures identify preferences more efficiently? Effort weighting narrows the constructed set-point intervals but lowers coverage relative to exact unit weighting. At 32 native records, coverage remains 98.26% and future-choice log loss improves by only 0.0090 nats. The direct reader already trails inside its training range and does not worsen after larger perturbations, so the proposed inside-parity/outside-failure account is not established. These are finite constructed-mechanism results.

[Report](../../../docs/versions/v22-mechanism-tests/B1_REPORT.md).

- `PROVENANCE.json`: source/plan/environment and exact independent/full-replay receipt bindings.
- `ANALYSIS.json`: paired-world mechanism estimates, separate targets and readers.
- `SUPPLEMENT.json`: paired high-label metric differences and additional controlled contrasts.
- `STRATA_summary.csv`: all original target/control strata, each reader and both proper scores; rows are aggregates, not cases. Empty numerical fields indicate nonfinite or missing-world estimates, not zeros.
- `HEADROOM.json`: original immutable matched gates for both label budgets.
- `FITS.json`: validation-stop metadata and weight hashes.

Log loss uses natural logarithms; lower is better. Full multiclass Brier is the sum of squared probability errors; lower is better. Accuracy is the modal correct fraction; higher is better. Confident error is an incorrect modal answer with probability at least 0.9; lower is better. Differences name their direction. Bounds are descriptive whole-world 95% percentile intervals, never case-level standard errors. C1 targets are marginal nodes, not a joint score. The compact public view omits per-world arrays; exact retained aggregate hashes are in `SUMMARY.json`.

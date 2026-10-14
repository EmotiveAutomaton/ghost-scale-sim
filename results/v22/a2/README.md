# V22 A2: Steering signatures and selective uptake

Does recognizing steering reduce content uptake while preserving technique? With a correctly mapped signature, the constructed full-library reader changes its content belief by 0.0167 on average versus 0.1606 for the missing-process reader, while technique-production probability is 89.94% versus 89.46%. But the content gap persists without the signature, so the proposed vanishing-gap control fails. This supports a qualified library-and-signature mechanism, not a clean signature-only explanation.

[Report](../../../docs/versions/v22-mechanism-tests/A2_REPORT.md).

- `PROVENANCE.json`: source/plan/environment and exact independent/full-replay receipt bindings.
- `ANALYSIS.json`: paired-world mechanism estimates, separate targets and readers.
- `SUPPLEMENT.json`: paired high-label metric differences and additional controlled contrasts.
- `STRATA_summary.csv`: all original target/control strata, each reader and both proper scores; rows are aggregates, not cases. Empty numerical fields indicate nonfinite or missing-world estimates, not zeros.
- `HEADROOM.json`: original immutable matched gates for both label budgets.
- `FITS.json`: validation-stop metadata and weight hashes.

Log loss uses natural logarithms; lower is better. Full multiclass Brier is the sum of squared probability errors; lower is better. Accuracy is the modal correct fraction; higher is better. Confident error is an incorrect modal answer with probability at least 0.9; lower is better. Differences name their direction. Bounds are descriptive whole-world 95% percentile intervals, never case-level standard errors. C1 targets are marginal nodes, not a joint score. The compact public view omits per-world arrays; exact retained aggregate hashes are in `SUMMARY.json`.

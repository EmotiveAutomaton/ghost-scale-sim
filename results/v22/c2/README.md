# V22 C2: Similarity, process evidence and preference records

Do shared processes and records help different inference targets? In the independent constructed world, eight preference records improve loss by 0.423 nats without a shared library and 0.460 with one. A reliable shared-library benefit itself is not established when similarity is hidden. Disclosing the sharing relation identifies the binary preference without records, and a coupled preference witness also helps. The unconditional claim that similarity cannot inform preferences fails; the result depends on what is disclosed and coupled.

[Report](../../../docs/versions/v22-mechanism-tests/C2_REPORT.md).

- `PROVENANCE.json`: source/plan/environment and exact independent/full-replay receipt bindings.
- `ANALYSIS.json`: paired-world mechanism estimates, separate targets and readers.
- `SUPPLEMENT.json`: paired high-label metric differences and additional controlled contrasts.
- `STRATA_summary.csv`: all original target/control strata, each reader and both proper scores; rows are aggregates, not cases. Empty numerical fields indicate nonfinite or missing-world estimates, not zeros.
- `HEADROOM.json`: original immutable matched gates for both label budgets.
- `FITS.json`: validation-stop metadata and weight hashes.

Log loss uses natural logarithms; lower is better. Full multiclass Brier is the sum of squared probability errors; lower is better. Accuracy is the modal correct fraction; higher is better. Confident error is an incorrect modal answer with probability at least 0.9; lower is better. Differences name their direction. Bounds are descriptive whole-world 95% percentile intervals, never case-level standard errors. C1 targets are marginal nodes, not a joint score. The compact public view omits per-world arrays; exact retained aggregate hashes are in `SUMMARY.json`.

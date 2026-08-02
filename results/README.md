# What is in here

Committed summary CSVs and JSON verdict files — everything a number in the README or a chart in
`figures/` depends on. Raw per-reader CSVs are **not** committed, because `e4_raw.csv` alone is
16 MB. Regenerate them with `python run_all.py`.

## Two layouts, and the reason is historical rather than principled

**Versions 1 to 5 wrote flat into this directory.** `e1_summary.csv`, `e2_cell_stats.csv`,
`e17_verdict.json`, and so on — one file per experiment per artefact type, all at the top level.

**Version 6 onward writes one subdirectory per version.** `v6/`, `v7/`, `v8/`, `v9/`, `v10/`, plus
`validation/`, `diagnostics/` and `repair/` for the audit passes.

The flat files have deliberately **not** been reorganised. Roughly forty experiment modules and
every chart script resolve these paths, the layout is baked into committed verdict files that record
where they were written, and the whole point of a sealed record is that re-running it reproduces
what is here. Tidying a data directory is not worth the chance of silently breaking that. The
inconsistency is visible, it is explained here, and it costs nothing but appearances.

## What lives where

| path | what it holds |
|---|---|
| `*.csv`, `*.json` (top level) | versions 1 through 5: summaries, per-cell statistics, verdicts, and the hash-locked pre-registration cards |
| `v6/` … `v10/` | one directory per version, each with `summary.json`, per-experiment verdicts, and the pre-registration card for that version |
| `validation/` | one verdict per check, plus the side-by-side tables |
| `diagnostics/` | one verdict per instrument check, plus the recovery sweeps |
| `repair/` | one verdict per repair item, plus the matched-pair sweep in which every reachable experiment was run under both code paths |
| `figures/` | intermediate data some charts are built from |

## How to read a verdict file

Each records the criterion **as it was locked before the run**, the measured value, and the pass or
fail. Where a criterion was restated after seeing a measurement, **both** are present — the original
is retained, still computed, and reported as failing if that is what it does. The restatement never
replaces it.

That convention is the reason these files are committed at all. A results document can be rewritten;
a verdict file with a content hash cannot be, quietly.

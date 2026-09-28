# V20 results

V20 is running under the fixed Friday 2 October, 05:00 Pacific finish. Execution,
verification, accepted discovery and confirmation remain distinct states.

- [Retained-history learning curves](../../docs/versions/v20-contribution-reconstruction/HISTORY_CURVE_REPORT.md), [validity](history-wave-1/VALIDITY.json), [forecast](history-wave-1/FORECAST.json).
- [Learning budgets and score repair](../../docs/versions/v20-contribution-reconstruction/SCORE_BOUNDARY_REPORT.md), [validity](score-boundary-1/VALIDITY.json), [continuation forecast](score-boundary-1/FORECAST.json).
- [Opening scientific report](../../docs/versions/v20-contribution-reconstruction/OPENING_REPORT.md), [validity](opening-wave-1/VALIDITY.json), [measured continuation forecast](opening-wave-1/FORECAST.json).
- [Setup and launch report](SETUP_REPORT.md), [validity](SETUP_VALIDITY.json).
- [Current dated status](CURRENT_STATUS.json), [opening forecast](OPENING_FORECAST.json).
- [Acceptance](ACCEPTANCE.json), [finite study forest](FOREST.json).
- [Claim ledger](CLAIM_LEDGER.json), [pursuit ledger](PURSUIT_LEDGER.json).
- [Protocol](../../docs/versions/v20-contribution-reconstruction/PROTOCOL.md).
- [Exact replay source](SOURCE.zip), [source bindings](SOURCE_MANIFEST.json).

`packets/` holds stable card/result identities. Raw observations and forecasts are
retained locally under immutable completion hashes; their absence from Git is not
an assertion that they were re-created. Only reader/ is a blind input; training,
evaluator truth, summaries and casebooks have separate roles. Private machine
state and human source passages stay outside this repository.

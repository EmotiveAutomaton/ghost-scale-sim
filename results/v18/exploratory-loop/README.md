# V18.4 portable evidence

This cutoff-closed adaptive campaign has separate admitted plans, retained superseded
unexecuted plans, raw scientific bundles and completed verification receipts.
`PLAN_LEDGER.json` records admission/supersession, not scientific completion.
`sources/<sha256>.zip` supplies the source archive named by each plan.

Each published packet contains its plan, summary, extracted-source replay proof,
executed replay driver, scientific manifest and ordinary ZIP split into chunks
under 40 MB. Concatenate the parts in `BUNDLE.json` order, then verify the archive
SHA256 before extracting. `python -m runners.export_v18_4 unpack --root PACKET
--output NEW_DIRECTORY` performs those checks. Extract its `SOURCE.zip` into a
separate source directory and place the supplied replay driver at
`runners/replay_v18_4.py` there; invoke that module with the packet data root and
a new proof output path. Neural forecast replay requires the recorded optional
CPU PyTorch environment. No complete retraining is implied by forecast replay.

Raw cases, evaluator truth, forecasts, fit curves and selected weights are public.
Machine-local process logs, ownership and intermediate optimizer checkpoints
remain in private operating storage. No current-world human evidence is present.
See the [study](../../../docs/versions/v18-selective-acquisition/exploratory-loop/RESULTS.md)
for claims, limits and next research decisions.

Final [coverage](COVERAGE.json), [validity](FINAL_VALIDITY.json),
[accounting](RESOURCE_ACCOUNTING.json) and [closeout](CLOSEOUT.json) distinguish
28 science packets (one timing pilot), 28 adjacent replays and useful unimplemented
designs. Fourteen finite packets contain 38,480 assigned evaluations; fourteen
neural/access packets are separate. These are not independent-sample counts.
All 12,585 scientific files and archive hashes passed the documentary audit.
No scientific work ran after cutoff.

# Inspecting and replaying V18.3

Every packet has its own source and environment identity. The working checkout can
contain later reporting or repair code; it is not a substitute for that packet's
source archive. The campaign uses planted simulation data, with evaluator truth
retained separately from reader inputs.

For finite A/B/C/D/F/H packets, extract `RAW.zip` beside the packet's `PLAN.json`
to restore `raw/` and `blocks/`. `SOURCE.zip` supplies the scientific source.
`REPLAY_DRIVER.py` is the separately hashed audit entry point. Its receipt records
complete retained-unit and aggregate-mean checks and eight fixed whole-unit source
replays. This is a bounded replay, not a claim that every unit was regenerated.

For neural packets, concatenate the binary parts in `BUNDLE.json` order to obtain
one ordinary ZIP archive. Each part and the assembled archive have SHA-256 hashes.
`python -m runners.export_v18_3_neural unpack --root PACKET --output NEW_DIRECTORY`
performs safe assembly, extraction and checks every scientific artifact against
`SCIENTIFIC_MANIFEST.json`. It refuses an existing destination or altered bytes.
The bundle includes reader inputs, evaluator truth, retained forecasts, selected
weights, all fit curves and the original source. Operating telemetry and
intermediate optimizer snapshots stay local; the incomplete original E attempt
has a separate scientific retention archive.

Extract the packet source into a separate directory and run from there with the
recorded Python/NumPy environment. Neural forecast replay additionally requires
the recorded CPU PyTorch environment. `REPLAY_DRIVER.py` supplies the audit runner
whose hash is in `INDEPENDENT_REPLAY.json`; when it was added after the scientific
freeze, copy it into that extracted source's `runners/` under the corresponding
replay module name. No source file listed in the plan may change.

The replay entry points require `--root`, `--output` and `--campaign`. The last is
an audit operating directory with a local `ACCEPTANCE.json`, ownership locks and
CPU-attempt records. A new external reproduction should declare its own audit
allowance and deadline there, rather than reuse or reset this campaign's historical
clock. The expected acceptance fields are `prior_cpu_seconds`,
`cumulative_cpu_ceiling_seconds` and an ISO-8601 UTC `report_start`. No local paths,
credentials or operating receipts are required from the original workstation.

Neural replay reconstructs every retained lineage mean and checks 128 fixed,
evenly spaced forecasts in each retained prediction file using selected weights.
It does not retrain those weights. Separate positive-learning and interruption/
optimizer-resume controls were executed. Calibration, factorial contrasts, native
source uptake, costs and worked examples have retained source hashes and raw
analysis points. Their scope and any unavailable comparison remain explicit.

The base test environment can collect the optional PyTorch tests as explicit
skips. Those skips are not passes: the campaign separately executes them with the
pinned shared CPU interpreter. Closed historical failures remain in the repository
and must not be erased to make the overall test suite appear green.

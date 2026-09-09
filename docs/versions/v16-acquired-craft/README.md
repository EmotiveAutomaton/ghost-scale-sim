# V16 — acquired craft

[The accepted coding package](CODING_PACKAGE.md) is the authoritative commission,
filed byte-for-byte from the repository-root handoff on 2026-09-09. Its SHA-256 is
75a50d5ae239a2a24b3ca5f33e4be9ad5962992d954c1fa7759ace9e1985986c.

Implementation and runtime state are separate from acceptance. Consult
[CAMPAIGN.json](../../../results/v16/CAMPAIGN.json), packet locks and
RUNNER_STATUS.json; a planned card is not an executed experiment.

Work uses an isolated checkout and the existing scientific interpreter without
synchronizing dependencies. The native finite implementation is independent of
V15 scientific code. V15 remains closed with its failures and retained evidence.
No V16 scientific finding was admitted by acceptance. Current operational
coverage is recorded in [PROGRESS.json](../../../results/v16/PROGRESS.json);
the [implementation status](IMPLEMENTATION_STATUS.md) distinguishes scouts,
instrument qualifications, and unfinished commission work.

Raw unit records are immutable and retained locally with checksummed manifests.
A curated full-replay bundle and verified archive handoff remain closeout requirements.
V16 extends the per-rollout naming convention to *_points.json files because
nested paired records cannot be represented faithfully as flat CSV rows.

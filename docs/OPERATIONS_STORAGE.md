# Scientific evidence storage

The working repository contains source, aggregate results, public proof bindings
and reports. Machine-local campaign directories retain small queue records,
source capsules, accounting and launch state. Large persistent outputs belong on
the owner-designated external evidence volume, configured outside Git.

The owner also commissioned same-repository git-annex integration. Keep readable
results in ordinary Git; `raw/data/` holds annex pointers and `raw/catalog/` their
provenance. The standard external directory remote uses hard links to completed,
verified raw payloads and preserves the original paths. Never annex private
operating material or substitute a raw pointer for a reportable aggregate.
Read [the results and raw-data guide](RESULTS_AND_RAW_DATA.md) and coverage index
before claiming that an archive is registered or published. No second physical
copy is created by the object-store links.

Current V31 continuation uses external job directories through directory links
at the original local paths; per-job records accompany the data, while central
queue control and accounting remain local. The frozen worker creates its own
output subdirectory. Scientific plans, seeds, source capsules and validity
gates are unchanged. Admission checks free space on the actual output volume and
protects a separate local metadata floor. An unavailable output volume holds new
starts. Network performance remains subject to the original job wall limits.

Completed output is transferred by `runners/archive_storage.py`, using an explicit
reviewed source plan. It inventories regular files without following nested links,
copies bytes, verifies destination SHA-256 hashes, and preserves the old path via
a directory link. Only then does it reclaim the manifest-covered local files.
Transaction manifests remain on both volumes. A partial transfer is resumable;
unverified source data is retained. A relocated directory is not a new scientific
replication or an extra independent backup.

An empty output directory from a failed or excluded job stays at its original
path. The transfer records `EMPTY_RETAINED` on both volumes, without creating a
payload directory or discarding the failed attempt. Resume checks that it is
still empty. This disposition permits later bundles to proceed; it is not a
scientific success or a claim that raw evidence existed there.

Historical parent folders may mix raw data with source snapshots, service state
and protected caches. Narrow those plans to independently preflighted output
subdirectories and retain the other content locally. An inaccessible cache is
not authority to alter permissions or skip evidence during a parent cutover.
Standalone database/archive files need a separate file-aware transfer, including
database sidecars where present; record them as retained local scope meanwhile.

Keep active outputs out of historical transfer plans. Do not perform raw-data
audits during a folder's cutover: wait for its completed transaction, or pause
the transfer service and verify it has stopped. Public aggregate files are
independent of these raw-data moves. Service status, copied bytes, verified bytes
and reclaimed bytes are separate facts; a prepared migration is not completion.

Before admitting a future campaign, estimate retained bytes for originals,
replays, failed attempts and reconstruction products. Record a campaign storage
allowance and route large output to the external volume before scientific starts.
Keep local source caches and metadata bounded. Do not silently delete failures,
drop replay evidence, lower validity thresholds or fill the working disk to keep
workers occupied. Consult the private storage policy and latest campaign handoff
for the current destination, service ownership and in-flight transfers.

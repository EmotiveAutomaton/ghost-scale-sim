# Scientific evidence storage

The working repository contains source, aggregate results, public proof bindings
and reports. Machine-local campaign directories retain small queue records,
source capsules, accounting and launch state. Large persistent outputs belong on
the owner-designated external evidence volume, configured outside Git.

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

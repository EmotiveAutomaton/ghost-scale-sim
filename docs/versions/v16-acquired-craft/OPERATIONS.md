# V16 operational boundaries

A supervisor acquires both the local result-directory lock and ownership keyed
by the accepted campaign identity. Windows uses a global named mutex so two
checkouts cannot own the same campaign; POSIX uses a user-scoped temporary OS
file lock. A process-local registry also rejects recursive ownership from another
root in the same process. Lock release follows actual process termination;
resumption preserves the scientific packet and accepted clocks.

The Windows namespace choice follows Microsoft's
[kernel object namespace contract](https://learn.microsoft.com/en-us/windows/win32/termserv/kernel-object-namespaces).
The executed process tests establish competing-root rejection and recovery on
this host. They are separate from the scientific reader boundary.

The acceptance lock records hashes of the unchanged campaign and commission
manifest. Migration checks existing packet clocks before writing that lock.
Every subsequent entry rejects changed acceptance or manifest bytes.

A single supervisor serializes status writes from work reports and a 15-second
liveness timer. The timer updates parent CPU and elapsed time; it does not change
the completed-unit count. Full raw/reader resource records remain the evidence for
scientific costs. SIGTERM requests a checkpoint; abrupt OS termination is also
covered by a real interruption/resume test.

The general CLI now follows the frozen finite dependency plan in
`results/v16/operations/QUEUE_PLAN.json`. Each invocation dispatches one admitted,
eligible job, preserves existing scientific packets on resume, and keeps missing
dependencies explicit. Three worker failures with the same recorded cause
quarantine that job. Instrument repair limits are accounted separately.

The final source-control and confirmation handlers have their own source-bound
admissions and actual interruption/resume checks. Their scientific executions
still depend on the preceding frozen jobs. A successful known fixture does not
mean its future scientific allocation has executed.

Independent calculations may start on an immutable card after its valid native
completion, while later cards continue. They do not read an unfinished card,
generate new independent observations, or own campaign status. Their completed
phase receipt requires the entire original packet, unchanged source identities,
and every registered card. Global aggregate coverage, bounded scientific replay,
complete raw archival coverage and documentary write-through remain separate
proofs required by the final completion guard.

The final archive excludes the live job pointer because the ordinary supervisor
advances it when closing. The frozen queue plan, recorded failures and scientific
outputs remain required archive members. Pre-close and final administrative
snapshots preserve the transition separately. Runtime recovery establishes its
tested preservation and ownership properties; it does not claim uninterrupted
occupancy or erase earlier failures.

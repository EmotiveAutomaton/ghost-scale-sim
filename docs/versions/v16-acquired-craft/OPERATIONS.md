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

General finite-queue dispatch, repeated-root-cause quarantine, bounded repair,
transfer, confirmation and closeout remain implementation obligations. These
ownership tests do not certify those unfinished mechanisms.

# Afternoon storage review: November 18

The archive is running again after a retained access-denied failure. The prior service stopped at a protected test-cache inside an old source snapshot, before moving that historical parent. Completed archives and scientific results were preserved.

The original phase completed 8,712 bulk bundles and three pilot bundles, reclaiming 614.50 GiB. Their transaction records are terminal. This check also independently rehashed all files in three bounded completed bundles, checked their mirrored receipts and verified the preserved original paths. It was not a new full rehash of every historical archive.

The successor preflight cleared 6,815 raw-output directories (240.81 GiB; 1,728,106 files). Source snapshots, operating records and protected caches stay local. Eight explicitly held items total about 73.97 GiB: standalone historical database/archive files and five mixed output trees requiring a narrower selection. No permissions were changed and no retained evidence was discarded.

The successor runs below normal priority with the existing tested copy, full destination-hash verification, original-path preservation and local-reclamation checks. It carries the prior infrastructure CPU charges forward. At this snapshot it has completed 17 bundles. The working volume has 643.95 GiB free; the external volume has 25.67 TiB free.

The annex selector now recognizes completed successor receipts and rejects unfinished transfers; seven bounded selection checks pass. The annex service remains stopped and new entries are not yet published. Readable scientific results remain ordinary Git files.

V31 science remains at its eligible endpoint. This is storage infrastructure only. Full migration is unfinished, and the retained items are not represented as externally available. The many small files make a finish-time estimate uncertain; the next recommended manual progress/resource check is 14:30 Pacific. Automatic agent wake delivery remains unavailable.

[Machine receipt](../../../results/v31/storage-check-afternoon-20261118/STORAGE_CHECK.json), [scientific queue review](QUEUE_REPORT_20261118.md), [storage contract](../../OPERATIONS_STORAGE.md).

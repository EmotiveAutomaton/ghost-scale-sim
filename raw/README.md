# Raw scientific evidence

This directory is part of the original Ghost Scale Git repository. Files under
`data/` are git-annex pointers identified by SHA-256 and byte size. Their large
contents live in the external `ghost-archive` store. They do not enter ordinary
Git history. [Per-bundle catalogs](catalog/) bind each file to its completed,
verified archive transaction or completed external job. [Coverage](INDEX.json) distinguishes registered
content from the migration still in progress.

The readable results remain in [`results/`](../results/), with reports in
[`docs/versions/`](../docs/versions/README.md). Reading the findings, estimates,
uncertainty intervals, controls and reported verification does not require the
external volume. Raw evidence supports deeper reconstruction and robustness work.

With git-annex installed and the archive mounted, enable the existing remote:

```text
git annex enableremote ghost-archive directory="<archive-root>/annex-objects"
git config remote.ghost-archive.annex-readonly true
git annex verifyremote ghost-archive
git annex whereis raw/data/<file>
git annex get --from=ghost-archive -- raw/data/<file>
git annex fsck -- raw/data/<file>
```

For remote verification, confirm that the displayed configuration is the expected
`type=directory encryption=none` and that the local directory is your archive.
Newer git-annex versions require this verification after enabling a restored remote.

Retrieve only the files needed for an analysis. `git annex drop -- raw/data/<file>`
can release a retrieved local cache after annex checks its external availability.
It is not an instruction to remove the external archive. A normal GitHub clone
contains pointers, catalogs and readable results; raw retrieval needs archive
access. The archive's machine-specific mount location is configured locally.

Completed raw payloads enter this tree only after their archive copy is verified.
Active output, incomplete transfers and private operating files are excluded.
Annex admission establishes storage integrity, not a scientific pass: failed
experiments retain their original disposition in the version's scientific record.
Newer completion records may still await their separate aggregate audit/report.

The external object store uses hard links to verified archived files. Those names
share the same underlying bytes and count as one physical copy. This avoids a
second full archive allocation but does not create an independent backup.

[Results and storage guide](../docs/RESULTS_AND_RAW_DATA.md).
[Storage checks and retrieval evidence](VALIDATION.json).

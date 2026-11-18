# Results and raw evidence in one repository

Reports, aggregate results, uncertainty intervals, controls, claim records and
public verification receipts remain ordinary Git files in this repository. The
storage migration selects the separate local runtime tree; it does not relocate
tracked `results/` or `docs/` files. At annex setup, all 388,532 tracked result
files and all tracked documentation retained their existing index identities.

Start with the [current theory return](versions/v31-four-settings/THEORY_RETURN.md),
the [version index](versions/README.md), [findings](../FINDINGS.md) or the
[claim ledger](theory/CLAIMS.json). Each reported scientific batch links its
compact numerical evidence and verification. No external disk is needed to read
those records on GitHub.

The owner additionally commissioned git-annex integration for raw archives.
[`raw/`](../raw/README.md) now provides a versioned directory of content-addressed
raw files in the same repository. Git stores small pointers and provenance
catalogs; the configured external store supplies the actual content on request.
Git-annex explicitly supports this combination of ordinary files and externally
stored large content. [Git-annex documentation](https://git-annex.branchable.com/git-annex/).

This distinction matters: making a local directory link does not publish its
target's bytes, and ordinary GitHub Git rejects individual files over 100 MiB.
Keeping raw content external also avoids growing Git history by hundreds of
gigabytes. [GitHub large-file limits](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github).

The standard directory special remote is named `ghost-archive`. Its actual mount
location remains local configuration. The verified filesystem supports hard
links, so annex object names can reference the existing external raw bytes
without a second large allocation. Original scientific paths remain intact.
Retrieval verifies content hashes. The remote is configured read-only for normal
annex operations; registration creates new links but never replaces or deletes
an existing archive file. [Directory remote](https://git-annex.branchable.com/special_remotes/directory/).

`runners/annex_storage.py` admits completed migration receipts or fully hashed
external-job snapshots bound to immutable completion, worker and plan records. It binds their
destination, rejects unsafe paths and changed files, and checks that every staged
raw entry is the expected small pointer. It selects numerical arrays, tabular
payloads, compressed data archives and explicitly named raw JSON records;
operating plans, source capsules, logs and notification state stay out. The
catalog preserves source-bundle identity and each file's SHA-256 and byte size.
Registration and publication are separate from copying, scientific validation
and a later review of the findings.

Recent tests retain full originals, replays and independent reconstruction
products. Their useful compact results must be landed in `results/` after the
existing scientific audit, even when raw files are already available through
annex. No result may live solely on the external volume after being reported as
published. Every new campaign estimates both local metadata and external raw
storage before admission.

Current annex coverage and remaining migration scope are in
[`raw/INDEX.json`](../raw/INDEX.json). Raw registration follows completed archive
transactions; an unfinished transfer is not represented as available content.
The archive and its hard-link object store are one physical copy, not a backup.
The repository stays on `main`; `git-annex` is the auxiliary metadata branch,
not a separate research branch.

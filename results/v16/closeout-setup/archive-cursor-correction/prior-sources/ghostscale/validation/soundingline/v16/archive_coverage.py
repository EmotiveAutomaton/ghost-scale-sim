"""Complete current-file coverage by verified raw ZIP members, with preserved versions."""
from pathlib import Path
from .records import read, digest
from .raw_archive import identity, verify


def chunk_inventory(archive_root, receipt_paths):
    root = archive_root.resolve()
    chunks = {}
    for receipt_path in receipt_paths:
        receipt_path = receipt_path.resolve()
        if not receipt_path.is_relative_to(root):
            raise ValueError("archive batch receipt escapes the archive root")
        receipt = read(receipt_path)
        if receipt.get("execution_state") != "completed" or receipt.get("instrument_state") != "valid":
            raise ValueError("archive batch has no valid completed receipt")
        for row in receipt["chunks"]:
            name = row["relative_archive"]
            path = (root/name).resolve()
            if not path.is_relative_to(root):
                raise ValueError("archive chunk escapes its retained root")
            if name in chunks and chunks[name] != row:
                raise ValueError("archive batch receipts disagree on a retained chunk")
            chunks[name] = row
    if not chunks:
        raise ValueError("empty archive inventory cannot establish completeness")
    return chunks


def coverage(archive_root, receipt_paths, required, *, verify_bytes=True, report=None):
    if not required:
        raise ValueError("an empty required raw inventory cannot establish completeness")
    chunks = chunk_inventory(archive_root, receipt_paths)
    found = {}
    checked = []
    for name, row in sorted(chunks.items()):
        path = archive_root/name
        plan = read(path.parent/"member_manifest_points.json")
        if digest(plan) != row["plan_sha256"]:
            raise ValueError("archive chunk member inventory changed")
        if verify_bytes:
            proof = verify(path, plan)
            if proof["archive_identity"] != row["archive_identity"]:
                raise ValueError("archive chunk bytes differ from its completed receipt")
        for member, saved in plan["files"].items():
            if required.get(member) == saved:
                found.setdefault(member, []).append(name)
        checked.append({"relative_archive": name, "members": len(plan["files"]),
            "archive_identity": row["archive_identity"], "plan_sha256": row["plan_sha256"]})
        if report:
            report(name, len(found), len(required))
    missing = sorted(set(required)-set(found))
    return {"instrument_state": "valid" if not missing and verify_bytes else "incomplete coverage" if missing else "inventory only",
        "required_files": len(required), "covered_files": len(found), "missing_or_different_files": missing,
        "required_bytes": sum(value["bytes"] for value in required.values()),
        "matched_chunks_by_file": found, "chunks": checked,
        "all_archive_member_bytes_reread": verify_bytes,
        "verified_complete_accessible": not missing and verify_bytes,
        "scope": "Every required current path must match a retained member's SHA-256 and byte length; archived older versions are retained without substituting for current evidence"}


def snapshot(repo, results_root):
    """Raw evidence and implementation, excluding final administrative self-reference."""
    repo, results_root = repo.resolve(), results_root.resolve()
    if not results_root.is_relative_to(repo):
        raise ValueError("scientific result root escapes its repository")
    paths = set()
    for path in results_root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(results_root)
        if relative.parts[0] == "closeout" or path.name in {"RUNNER_STATUS.json", "OWNER.lock", "CLOSEOUT.json"} or path.suffix in {".tmp", ".pyc"} or "__pycache__" in relative.parts:
            continue
        paths.add(path)
    paths.update((repo/"ghostscale/validation/soundingline/v16").glob("*.py"))
    paths.update((repo/"runners").glob("*v16*.py"))
    paths.update((repo/"tests").glob("*v16*.py"))
    mandatory = {repo/name for name in ["pyproject.toml", "uv.lock", "docs/METHODS.md", "docs/versions/v16-acquired-craft/CODING_PACKAGE.md"]}
    if any(not path.is_file() for path in mandatory):
        raise ValueError("required commission, method or environment record is missing")
    paths.update(mandatory)
    result = {}
    for path in sorted(path for path in paths if path.is_file()):
        if path.is_symlink() or not path.resolve().is_relative_to(repo):
            raise ValueError("required raw evidence contains an escaping link")
        result[path.relative_to(repo).as_posix()] = identity(path)
    return result

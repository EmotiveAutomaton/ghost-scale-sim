"""Archive explicit retained setup/failed sections without promoting their validity."""
import argparse
from pathlib import Path
from ghostscale.validation.soundingline.v16.runtime import REPO
from ghostscale.validation.soundingline.v16.records import read, write, now
from ghostscale.validation.soundingline.v16.raw_archive import manifest, create
from ghostscale.validation.soundingline.v16.record_integrity import source_locks


def run(root, output, sections, batch):
    if not batch or any(part in batch for part in ["/", "\\", ".."]):
        raise ValueError("supplement requires an explicit local batch name")
    checks = source_locks(root, REPO)
    common = {root/"CAMPAIGN.json", REPO/"pyproject.toml", REPO/"uv.lock",
        REPO/"docs/versions/v16-acquired-craft/CODING_PACKAGE.md", Path(__file__).resolve(),
        REPO/"ghostscale/validation/soundingline/v16/raw_archive.py"}
    for path in (root/"packets").glob("*.json"):
        common.add(path)
        common.update(REPO/name for name in read(path)["identity"]["files"])
    reports = []
    for section in sections:
        if "/" in section or "\\" in section or section in {".", "..", "operations", "packets", "boundary-expansion-1"}:
            raise ValueError("supplement requires a named completed setup or retained failed section")
        source = root/section
        if not source.is_dir() or not source.resolve().is_relative_to(root.resolve()):
            raise ValueError("supplement section missing or outside the scientific result tree")
        paths = sorted(common | {path for path in source.rglob("*") if path.is_file()})
        plan = manifest(REPO, paths, scope={"section": section, "batch": batch,
            "evidence_state": "Snapshot of retained setup or failure evidence; original invalid/failed labels remain unchanged",
            "scientific_sample": False, "complete_campaign_archive": False})
        directory = output/"supplements"/batch/section/"attempt-1"
        receipt = create(REPO, directory, plan)
        report = {"section": section, "relative_archive": directory.relative_to(output).as_posix()+"/raw.zip",
            "member_count": receipt["verified_members"], "uncompressed_bytes": receipt["uncompressed_bytes"],
            "archive_identity": receipt["archive_identity"], "plan_sha256": receipt["plan_sha256"]}
        reports.append(report)
        print(report, flush=True)
    result = {"execution_state": "completed", "instrument_state": "valid", "recorded_at": now(),
        "source_checks": checks, "chunks": reports, "complete_campaign_archive": False,
        "scope": "Only explicitly named retained sections; later writes require a separately retained supplement"}
    write(output/"RECEIPTS"/(batch+".json"), result)
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, default=REPO/"results/v16")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--section", action="append", required=True)
    parser.add_argument("--batch", required=True)
    args = parser.parse_args()
    result = run(args.root, args.output, args.section, args.batch)
    print({"completed_supplement_chunks": len(result["chunks"]), "complete_campaign_archive": False})


if __name__ == "__main__":
    main()

"""Package the initial A evaluator record. This is not a blind reader bundle."""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

from ghostscale.validation.soundingline.v16.records import canonical, file_digest, read, write, now


def package(repo, destination):
    repo, destination = Path(repo).resolve(), Path(destination).resolve()
    sources = set()
    for packet in ("a-pilot-1", "a-screen-1"):
        root = repo/"results/v17"/packet
        lock = read(root/"LOCK.json")
        assert read(root/"VERIFICATION.json")["passed"]
        sources.update(lock["source_files"])
        sources.update(str((root/name).relative_to(repo)).replace("\\", "/")
                       for name in ("LOCK.json","INDEX.json","COMPARISONS.json","COMPLETION.json","VERIFICATION.json"))
        for chunk in read(root/"INDEX.json")["chunks"]:
            p = root/chunk["path"]
            assert p.resolve().is_relative_to(root.resolve())
            assert file_digest(p) == chunk["sha256"]
            sources.add(p.relative_to(repo).as_posix())
    sources.update(("ghostscale/__init__.py","ghostscale/validation/__init__.py",
                    "ghostscale/validation/soundingline/__init__.py",
                    "runners/verify_v17_packet.py","runners/package_v17_snapshot.py",
                    "results/v17/CAMPAIGN.json","results/v17/setup/ADMISSION.json",
                    "results/v17/setup/TEST_REPORT.json","results/v17/setup/DEVELOPMENT_FAILURES.json"))
    members = {}
    for relative in sorted(sources):
        path = repo/relative
        assert path.resolve().is_relative_to(repo)
        members[relative] = path.read_bytes()
    members["SNAPSHOT_README.md"] = (
        "# V17 initial A evaluator snapshot\n\n"
        "Contains evaluator truth. Do not give this archive to a blind reader.\n"
        "This is a limited A baseline slice, not a completed V17 study.\n\n"
        "From this extracted directory, with Python 3.10 or newer:\n\n"
        "    python -m runners.verify_v17_packet --root results/v17/a-screen-1\n\n"
        "The original verification receipt includes a timestamp, so this command\n"
        "refuses to overwrite it. To run the read-only check without writing a receipt:\n\n"
        "    python -c \"from runners.verify_v17_packet import verify; print(verify('results/v17/a-screen-1'))\"\n\n"
        "Re-run the construction fixture into a new directory:\n\n"
        "    python -m runners.run_v17 --stage fixture --root fresh-fixture\n\n"
        "Raw chunks, proper source hashes and original receipts are retained.\n"
        "No third-party library or Rust toolchain is needed for this baseline snapshot.\n"
    ).encode()
    inventory={name:{"sha256":hashlib.sha256(data).hexdigest(),"bytes":len(data)} for name,data in members.items()}
    members["SNAPSHOT_MANIFEST.json"]=canonical(inventory)+b"\n"
    destination.parent.mkdir(parents=True,exist_ok=True)
    with ZipFile(destination,"x",compression=ZIP_DEFLATED,compresslevel=9) as archive:
        for name,data in sorted(members.items()):
            archive.writestr(name,data)
    with ZipFile(destination) as archive:
        assert set(archive.namelist())==set(members)
        for name,data in members.items():
            assert archive.read(name)==data,name
    report={"schema":"v17.snapshot.1","created_at":now(),"archive_sha256":file_digest(destination),
            "archive_bytes":destination.stat().st_size,"members":len(members),"all_member_bytes_reread":True,
            "contains_evaluator_truth":True,"blind_reader_bundle":False,"campaign_complete":False,
            "scope":"complete initial pilot and A baseline screen raw chunks with frozen source closure",
            "packager_sha256":file_digest(Path(__file__)),"inventory":inventory}
    write(destination.with_suffix(".json"),report)
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    r=package(Path(__file__).resolve().parents[1],args.out)
    print(json.dumps({k:v for k,v in r.items() if k!="inventory"}))

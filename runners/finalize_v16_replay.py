"""Verify complete bounded replay coverage and produce a portable retained bundle."""
import argparse
from pathlib import Path
import os
import shutil
from ghostscale.validation.soundingline.v16.runtime import REPO
from ghostscale.validation.soundingline.v16.records import read, write, now, file_digest
from ghostscale.validation.soundingline.v16.record_integrity import source_locks
from ghostscale.validation.soundingline.v16.replay_plan import select_native, select_archive, lookup, item
from ghostscale.validation.soundingline.v16.replay_coverage import phase, catalogue, allocation, collect
from ghostscale.validation.soundingline.v16.raw_archive import manifest, create, identity
from runners.reaggregate_v16_packet import inventory


def endpoints(root, name):
    packet, completion, work = inventory(root, name)
    result = []
    for summary, design, n in work:
        for condition, index in [(design["conditions"][0], 0), (design["conditions"][-1], n-1)]:
            path, row = lookup(summary.parent, condition["id"], index)
            result.append(item(root, path, name, "native-card", condition, row["seed_components"]["constructors"]))
    return result


def development(root):
    selected = select_native(root)+select_archive(root)
    for path in sorted((root/"native-fixture-1/units").glob("*_points.json")):
        selected.append(item(root, path, "native-fixture-1", "native", None, 1))
    return selected


def run(root, output, archive_root, input_path):
    if output.exists() or not output.resolve().is_relative_to(root.resolve()):
        raise ValueError("replay handoff requires a new retained attempt inside the campaign")
    inputs = read(input_path)
    phases = {"development": development(root), "constructor": endpoints(root, "constructor-expansion-1"),
        "boundary": endpoints(root, "boundary-expansion-1"), "confirmation": endpoints(root, "confirmation-1")}
    if set(inputs["phases"]) != set(phases):
        raise ValueError("replay handoff requires exactly the four declared phases")
    checks = source_locks(root, REPO)
    files, phase_receipts = {}, {}
    for name, selected in phases.items():
        directory = (root/inputs["phases"][name]).resolve()
        if not directory.is_relative_to(root.resolve()):
            raise ValueError("replay phase directory escapes the campaign")
        for path, sha in phase(root, REPO, directory, selected).items():
            collect(files, path, sha)
        phase_receipts[name] = {"path": (directory/"RECEIPT.json").relative_to(root.resolve()).as_posix(), "sha256": file_digest(directory/"RECEIPT.json"), "checks": len(selected)}
    descriptive, catalogue_files = catalogue(root)
    for path, sha in catalogue_files.items():
        collect(files, path, sha)
    scope = allocation(phases, descriptive)
    # Package imports, original scientific locks, and all executed V16 tools ship
    # beside the selected source records. No environment is synchronized here.
    implementation = [*REPO.joinpath("ghostscale").rglob("*.py"), *REPO.joinpath("runners").glob("*v16*.py"),
        *root.joinpath("packets").glob("*.json"), REPO/"pyproject.toml", REPO/"uv.lock",
        REPO/"docs/versions/v16-acquired-craft/CODING_PACKAGE.md"]
    for path in implementation:
        collect(files, path.resolve(), file_digest(path))
    dependencies = {path.relative_to(REPO.resolve()).as_posix(): sha for path,sha in files.items()}
    write(output/"dependency_points.json", {"files": dependencies})
    bundle_plan = {**scope, "source_checks": checks, "phases": phase_receipts,
        "dependencies": (output/"dependency_points.json").relative_to(REPO).as_posix(),
        "dependencies_sha256": file_digest(output/"dependency_points.json"),
        "planned_at": now(), "descriptive_selection_qualification": "Catalogue cases were selected from outcomes; repeated or overlapping cases add no independent sample",
        "comparison": "Complete generated worlds, acquisition histories, guarded predictions, continuations, program execution and scores; original OS measurements are retained observations"}
    write(output/"PLAN.json", bundle_plan)
    instructions = ("# V16 bounded replay bundle\n\nExtract raw.zip into a new empty directory. Member paths recreate a small source checkout. "
        "Use a compatible isolated Python environment with the dependencies declared in pyproject.toml and uv.lock. "
        "The original environment must remain unchanged.\n\nFrom the extracted checkout, run:\n\n```text\npython -B -m runners.replay_v16_bundle --root results/v16 --plan "
        +(output/"PLAN.json").relative_to(REPO).as_posix()+" --output results/v16/portable-replay-attempt-1\n```\n\n"
        "The command checks every bundled dependency before regenerating the fixed cases. "
        "An existing output directory is refused. Timestamps, transport aliases and OS clock/RSS samples may differ; scientific state and scores must match. "
        "This bounded subset complements the complete raw archive and is not a claim that every campaign unit was replayed.\n")
    output.joinpath("README.md").write_text(instructions, encoding="utf-8", newline="\n")
    paths = [*files, output/"dependency_points.json", output/"PLAN.json", output/"README.md"]
    plan = manifest(REPO, paths, scope={"kind": "portable bounded whole-unit replay bundle", "distinct_units": scope["distinct_units"], "scientific_observations_added": 0})
    directory = archive_root/"portable-replay"/output.name
    archived = create(REPO, directory, plan)
    portable = output/"replay.zip"
    shutil.copy2(directory/"raw.zip", portable)
    if identity(portable) != archived["archive_identity"]:
        raise ValueError("versioned portable replay bundle differs from its verified archive")
    chunk = {"relative_archive": directory.relative_to(archive_root).as_posix()+"/raw.zip", "archive_identity": archived["archive_identity"],
        "plan_sha256": archived["plan_sha256"], "member_count": archived["verified_members"], "uncompressed_bytes": archived["uncompressed_bytes"]}
    write(archive_root/"RECEIPTS"/("portable-replay-"+output.name+".json"), {"execution_state": "completed", "instrument_state": "valid", "recorded_at": now(), "chunks": [chunk], "complete_campaign_archive": False})
    result = {"execution_state": "completed", "instrument_state": "valid", "recorded_at": now(), "process_id": os.getpid(),
        "whole_unit_replay": True, "all_selected_sources_and_replayed_bytes_rechecked": True,
        **{key:value for key,value in scope.items() if key != "selected"}, "phases": phase_receipts,
        "source_checks": checks, "plan_sha256": file_digest(output/"PLAN.json"), "portable_bundle": chunk,
        "versioned_replay_bundle": portable.relative_to(root).as_posix(),
        "dependency_files": len(dependencies), "campaign_complete": False,
        "qualification": "All fixed selected units were wholly replayed; the bundle's bytes were independently reread. Full raw archive coverage and documentary closeout remain separate proofs.",
        "sources": {path.relative_to(REPO).as_posix(): file_digest(path) for path in [Path(__file__).resolve(), REPO/"ghostscale/validation/soundingline/v16/replay_coverage.py", REPO/"runners/replay_v16_bundle.py"]}}
    write(output/"RECEIPT.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, default=REPO/"results/v16")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("existing replay handoff attempt is retained")
    try:
        result = run(args.root.resolve(), args.output.resolve(), args.archive_root.resolve(), args.inputs.resolve())
    except Exception as error:
        if args.output.resolve().is_relative_to(args.root.resolve()):
            write(args.output/"FAILURE.json", {"execution_state": "failed", "instrument_state": "unresolved", "recorded_at": now(), "error": repr(error)})
        raise
    print({key: result[key] for key in ["execution_state", "whole_unit_replay", "total_checks", "distinct_units"]})


if __name__ == "__main__":
    main()

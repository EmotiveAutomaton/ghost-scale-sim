"""Join bounded whole-unit replay phases without inflating distinct scientific cases."""
from .records import read, file_digest


def checked_file(base, name, expected):
    path = (base/name).resolve()
    if not path.is_relative_to(base.resolve()) or not path.is_file() or file_digest(path) != expected:
        raise ValueError("bounded replay dependency is missing, escaping or changed: "+name)
    return path


def collect(mapping, path, expected):
    if path in mapping and mapping[path] != expected:
        raise ValueError("bounded replay dependencies disagree on source bytes")
    mapping[path] = expected


def check_case(root, selected, receipt, replay_base):
    source = checked_file(root, selected["source_unit"], selected["source_unit_sha256"])
    if receipt.get("world_reader_and_score_match") is not True or not receipt.get("source_files") or not receipt.get("replay_files"):
        raise ValueError("bounded replay has no complete world/reader/score comparison")
    if receipt.get("selected_source", selected["source_unit"]) != selected["source_unit"]:
        raise ValueError("bounded replay compared a different selected source")
    if receipt["source_files"].get("units/"+source.name) != selected["source_unit_sha256"]:
        raise ValueError("bounded replay source unit is absent from its dependency inventory")
    files = {}
    for base, values in [(source.parent.parent, receipt["source_files"]), (replay_base, receipt["replay_files"])]:
        for name, expected in values.items():
            collect(files, checked_file(base, name, expected), expected)
    return files


def phase(root, repo, directory, expected):
    plan_path, receipt_path = directory/"PLAN.json", directory/"RECEIPT.json"
    plan, receipt = read(plan_path), read(receipt_path)
    if receipt.get("execution_state") != "completed" or receipt.get("instrument_state") != "valid" or receipt.get("selected_units_wholly_replayed") is not True or receipt.get("failed_comparisons"):
        raise ValueError("bounded replay phase is incomplete or failed")
    if receipt["plan_sha256"] != file_digest(plan_path) or plan["selected"] != expected:
        raise ValueError("bounded replay phase omits or changes frozen endpoint coverage")
    if len(expected) != receipt["n_units"] or len(expected) != len(receipt["checks"]) or len(expected) != plan["n_units"]:
        raise ValueError("bounded replay phase count differs")
    files = {plan_path.resolve(): file_digest(plan_path), receipt_path.resolve(): file_digest(receipt_path)}
    for name, sha in plan["replay_sources"].items():
        collect(files, checked_file(repo, name, sha), sha)
    for index, (selected, check) in enumerate(zip(expected, receipt["checks"])):
        path = directory/"checks"/f"case-{index:03d}.json"
        if read(path) != check:
            raise ValueError("bounded replay saved check differs from its completion")
        collect(files, path.resolve(), file_digest(path))
        for path, sha in check_case(root, selected, check, directory/"private"/f"case-{index:03d}").items():
            collect(files, path, sha)
    return files


def catalogue(root):
    directory = root/"case-catalogue-2"
    plan_path, completion_path = directory/"SELECTION.json", directory/"COMPLETION.json"
    plan, completion = read(plan_path), read(completion_path)
    if completion.get("execution_state") != "completed" or completion.get("instrument_state") != "valid" or completion.get("all_selected_units_wholly_replayed") is not True:
        raise ValueError("descriptive catalogue whole replay is incomplete")
    if file_digest(plan_path) != completion["selection_sha256"] or len(plan["selected"]) != completion["case_count"] or len(plan["selected"]) != len(completion["checks"]) or len(plan["selected"]) > 36:
        raise ValueError("descriptive catalogue replay selection/count differs")
    files = {plan_path.resolve(): file_digest(plan_path), completion_path.resolve(): file_digest(completion_path)}
    for index, (selected, reference) in enumerate(zip(plan["selected"], completion["checks"])):
        key = f"case-{index:03d}"
        if reference["case_id"] != key:
            raise ValueError("descriptive catalogue replay order differs")
        path = checked_file(directory, "checks/"+key+".json", reference["check_sha256"])
        collect(files, path, reference["check_sha256"])
        for path, sha in check_case(root, selected["replay"], read(path), directory/"private/replay"/key).items():
            collect(files, path, sha)
    return [row["replay"] for row in plan["selected"]], files


def allocation(phases, descriptive):
    if set(phases) != {"development", "constructor", "boundary", "confirmation"}:
        raise ValueError("final whole replay requires all four declared phases")
    for name, count in [("development", 68), ("constructor", 52), ("boundary", 34)]:
        if len(phases[name]) != count:
            raise ValueError("final whole replay endpoint coverage differs: "+name)
    if len(phases["confirmation"]) > 6 or len(descriptive) > 36:
        raise ValueError("final whole replay exceeds its frozen bound")
    native = [row for values in phases.values() for row in values]
    if len({row["source_unit"] for row in native}) != len(native):
        raise ValueError("native replay phases duplicate a scientific unit")
    unique = {row["source_unit"]: row for row in native}
    for row in descriptive:
        if row["source_unit"] in unique and unique[row["source_unit"]] != row:
            raise ValueError("overlapping descriptive replay changes the unit's scientific identity")
        unique[row["source_unit"]] = row
    return {"native_and_search_checks": len(native), "descriptive_checks": len(descriptive),
        "total_checks": len(native)+len(descriptive), "distinct_units": len(unique),
        "selected": list(unique.values()), "maximum_total_checks": 196,
        "scope": "Bounded mechanistic replay; descriptive overlaps add checks but no independent scientific observations"}

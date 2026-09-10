"""A finite source-bound explanatory index with full selected-unit replay.

Runs beside science as read-only verification of completed sources. It never
owns or writes live runner status and never reads the unfinished final expansion.
"""
from .records import read, write, digest, file_digest, now
from .runtime import REPO, PACKAGE, freeze
from .record_integrity import source_locks
from .aggregate_science import registered_designs
from .replay_plan import lookup, item
from .whole_replay import replay_item
from .reader_process import ReaderProcess
from .expansion_runner import EXTENSIONS
from .catalogue_cases import LENSES, labels, clean_failure

PACKET = "case-catalogue-1"
MAX_CASES = 64
INDICES = tuple(range(8))


def inventory(root):
    selected = {}
    for path in sorted((root/"packets").glob("*-scout-*.json")):
        if path.stem == "inquiry-scout-1":
            continue
        packet = read(path)
        spec = packet["identity"]["design"]
        for card, design in registered_designs(spec).items():
            base = root/path.stem/card if (root/path.stem/card/"units").exists() else root/path.stem
            selected[card] = {"card_id": card, "packet_id": path.stem, "base": base.relative_to(root).as_posix(),
                "design": design, "constructors": spec.get("constructors", 8)}
    expanded = read(root/"packets/constructor-expansion-1.json")
    for entry in expanded["identity"]["design"]["cards"]:
        card = entry["card_id"]
        selected[card] = {"card_id": card, "packet_id": "constructor-expansion-1",
            "base": "constructor-expansion-1/"+card, "design": entry["design"], "constructors": entry["constructors"]}
    if len(selected) != 30:
        raise ValueError("catalogue needs all thirty native cards with explicit source identity")
    for entry in selected.values():
        base = root/entry["base"]
        expanded_source = entry["packet_id"] == "constructor-expansion-1"
        entry["summary_name"] = "AGGREGATE.json" if expanded_source else "SUMMARY.json"
        entry["audit_name"] = "INDEPENDENT_AUDIT.json" if expanded_source else "REAGGREGATION.json"
        for name in ["COMPLETION.json", entry["audit_name"]]:
            receipt = read(base/name)
            if receipt.get("execution_state") != "completed" or receipt.get("instrument_state") != "valid":
                raise ValueError("catalogue source is not completed and independently valid")
    return [selected[card] for card in sorted(selected)]


def execute(root, heartbeat=None, *, resume=False):
    output = root/PACKET
    selection_path = output/"SELECTION.json"
    if resume and not (root/"packets"/(PACKET+".json")).exists():
        raise ValueError("catalogue resume requires its original packet")
    entries = inventory(root)
    source_locks(root, REPO)
    sources = {REPO/name for name in read(root/"packets/boundary-expansion-1.json")["identity"]["files"]}
    sources.update([PACKAGE/"catalogue_cases.py", PACKAGE/"catalogue_runner.py", REPO/"runners/catalogue_v16.py", REPO/"tests/test_v16_catalogue.py"])
    sources.update(PACKAGE/name for name in ["aggregate_science.py", "replay_plan.py", "whole_replay.py"])
    provenance = {}
    for entry in entries:
        for path in [root/"packets"/(entry["packet_id"]+".json"), *(root/entry["base"]/name for name in ["COMPLETION.json", entry["summary_name"], entry["audit_name"]])]:
            provenance[path.relative_to(root).as_posix()] = file_digest(path)
    control = root/"constructor-controls-1/COMPLETION.json"
    if read(control).get("instrument_state") != "valid":
        raise ValueError("catalogue expansion source controls are missing or invalid")
    provenance[control.relative_to(root).as_posix()] = file_digest(control)
    spec = {"source_hashes": provenance, "candidate_indices_per_condition": list(INDICES),
        "case_cap": MAX_CASES, "lens_names": LENSES, "selection": "First matching case per lens and first registered comparison failure or tie per card, stable card/condition/index order",
        "interpretation": "Descriptive indexing of previously completed native discoveries; not new explanatory search success or population inference",
        "excludes": ["unfinished boundary-expansion-1", "confirmation outcomes"],
        "source_design_hashes": {entry["card_id"]: digest(entry["design"]) for entry in entries}}
    packet = freeze(root, PACKET, sorted(sources), spec)
    if (output/"COMPLETION.json").exists():
        completion = read(output/"COMPLETION.json")
        if completion["packet_hash"] != packet["packet_hash"] or completion["selection_sha256"] != file_digest(selection_path):
            raise ValueError("completed catalogue selection changed")
        for check in completion["checks"]:
            if check["check_sha256"] != file_digest(output/"checks"/(check["case_id"]+".json")):
                raise ValueError("completed catalogue validation changed")
        return completion
    if selection_path.exists():
        selection = read(selection_path)
        if selection["packet_hash"] != packet["packet_hash"]:
            raise ValueError("catalogue cannot change its selected source packet")
    else:
        chosen, found, failures = {}, {}, {}
        examined = 0
        for entry in entries:
            base = root/entry["base"]
            summary = read(base/entry["summary_name"])
            for condition in entry["design"]["conditions"]:
                for index in INDICES:
                    path, row = lookup(base, condition["id"], index)
                    uid = row["unit_id"]
                    public = read(base/"public"/(uid+".json"))
                    private = read(base/"private"/(uid+".json"))
                    prediction = read(base/"predictions"/(uid+".json"))
                    evidence = labels(row, public, private, prediction)
                    hits = {key: value for key, value in evidence.items() if key not in found}
                    failure = clean_failure(row, summary["conditions"][condition["id"]]["contrasts"])
                    card = entry["card_id"]
                    if hits or (failure is not None and card not in failures):
                        key = path.relative_to(root).as_posix()
                        case = chosen.setdefault(key, {"replay": item(root, path, entry["packet_id"], "native-card", condition, entry["constructors"]),
                            "card_id": card, "lenses": {}, "failure": None})
                        case["lenses"].update(hits)
                        found.update({name: key for name in hits})
                        if failure is not None and card not in failures:
                            case["failure"] = failure
                            failures[card] = key
                    examined += 1
            print({"catalogue_scanned": entry["card_id"], "candidates": examined, "selected": len(chosen)}, flush=True)
        if len(chosen) > MAX_CASES:
            raise ValueError("catalogue allocation exceeded its frozen cap")
        selection = {"packet_hash": packet["packet_hash"], "selected": list(chosen.values()), "candidates_examined": examined,
            "lens_coverage": {name: {"state": "found" if name in found else "not found in bounded search", "source": found.get(name)} for name in LENSES},
            "failure_coverage": {entry["card_id"]: {"state": "found" if entry["card_id"] in failures else "not found in bounded search", "source": failures.get(entry["card_id"])} for entry in entries},
            "frozen_at": now(), "new_explanatory_case_count": 0,
            "B02_relation": "The six independently validated adaptive-search semantic cases and frozen follow-up remain separately accounted in archive-search-1"}
        write(selection_path, selection)
    checks = []
    with ReaderProcess(output/"reader", extensions=EXTENSIONS) as reader:
        for index, case in enumerate(selection["selected"]):
            key = f"case-{index:03d}"
            path = root/case["replay"]["source_unit"]
            row = read(path)
            base, uid = path.parent.parent, row["unit_id"]
            public, private, prediction = [read(base/folder/(uid+".json")) for folder in ["public", "private", "predictions"]]
            actual = labels(row, public, private, prediction)
            if any(actual.get(name) != value for name, value in case["lenses"].items()):
                raise ValueError("catalogue lens no longer matches the frozen raw case")
            check = replay_item(root, output/"private"/"replay"/key, case["replay"], reader)
            write(output/"public"/(key+".json"), {"case_id": key, "public_observation": public,
                "pre_reveal_reader_submission": prediction, "source_observation_sha256": file_digest(base/"public"/(uid+".json")),
                "access_contract": "Only the original public input and pre-reveal reader output; no evaluator truth or case label"})
            write(output/"private"/(key+"_points.json"), {"case_id": key, "actual_truth_and_continuation": private,
                "evaluated_arms_and_all_recorded_costs": row["arms"], "classification": case,
                "scope": "Outcome-selected descriptive example, not an independent confirmatory claim"})
            write(output/"checks"/(key+".json"), check)
            checks.append({"case_id": key, "check_sha256": file_digest(output/"checks"/(key+".json"))})
    result = {"execution_state": "completed", "instrument_state": "valid", "packet_hash": packet["packet_hash"],
        "completed_at": now(), "selection_sha256": file_digest(selection_path), "case_count": len(checks), "checks": checks,
        "all_selected_units_wholly_replayed": True, "reader_truth_separation": True,
        "lens_coverage": selection["lens_coverage"], "failure_coverage": selection["failure_coverage"],
        "scientific_scope": "Constructed-world mechanism examples; descriptive selection is not population inference",
        "full_campaign_closeout": False}
    write(output/"COMPLETION.json", result)
    return result

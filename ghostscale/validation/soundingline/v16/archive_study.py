"""B02 immutable candidate journals, frozen follow-up, and independent validation."""
import time
from .records import read, write, file_digest, now, digest
from .runtime import remaining_seconds
from .archive_design import DESIGN, CONDITIONS
from .archive_search import select, novelty, summarize
from .archive_cases import case_record
from .archive_reference import verify_case, verify_summary
from .selection import EXTENSION
from .selection_study import execute_unit
from .reader_process import ReaderProcess
from .resource_accounting import MeasuredReader
from .record_integrity import independent_units
from .archive_controls import validate_consumers


def candidate(base, condition, index, namespace, packet, reader, constructors):
    uid = digest([namespace, condition["id"], index])[:24]
    path = base/"private"/(uid+"-archive-case.json")
    if path.exists():
        result = read(path)
        row = read(base/"units"/(uid+"_points.json"))
        if row["packet_hash"] != packet["packet_hash"] or result["unit_sha256"] != file_digest(base/"units"/(uid+"_points.json")):
            raise ValueError("archive resume source changed")
        verify_case(base, row, result["case"])
        return result
    measured = MeasuredReader(reader)
    start, cpu = time.perf_counter(), time.process_time()
    row = execute_unit(base, condition, index, namespace=namespace, packet=packet, reader=measured,
                       constructors=constructors, scope="fixture" if namespace.endswith("/fixture") else "explanatory archive discovery")
    interpreted = case_record(base, row, measured)
    verified = verify_case(base, row, interpreted)
    result = {"unit_sha256": file_digest(base/"units"/(uid+"_points.json")), "case": interpreted, "independent_check": verified,
              "resources": {"requests": measured.samples, "parent_cpu_seconds": time.process_time()-cpu,
                            "wall_seconds": time.perf_counter()-start, "startup_included": False},
              "completed_at": now()}
    write(path, result)
    return result


def execute(root, output, heartbeat, packet, *, fixture=False):
    replicas, steps, followup_n = (2, 6, 4) if fixture else (DESIGN["search_replicates"], DESIGN["steps_per_replicate"], DESIGN["followup"]["makers_per_condition"])
    total = 2*replicas*steps+len(CONDITIONS)*followup_n
    search, followup, completed = [], [], 0
    with ReaderProcess(output/"public/reader", extensions=[EXTENSION]) as reader:
        for method in ["fixed", "adaptive"]:
            for replicate in range(replicas):
                history, seen = [], set()
                namespace = DESIGN["search_namespace"]+f"/{method}/replicate-{replicate}"+("/fixture" if fixture else "")
                base = output/"private/search"/method/str(replicate)
                for step in range(steps):
                    if remaining_seconds(root) <= 0:
                        return {"execution_state": "checkpointed", "reason": "immutable ceiling", "campaign_complete": False}
                    condition = select(method, replicate, history)
                    record = candidate(base, condition, step, namespace, packet, reader, steps)
                    case = record["case"]
                    new = novelty(case["cases"], seen)
                    journal = {"method": method, "replicate": replicate, "step": step, "condition": condition["id"],
                        "new_cases": new, "candidate_receipt": str((base/"private"/(case["unit_id"]+"-archive-case.json")).relative_to(output)).replace("\\", "/"),
                        "candidate_sha256": file_digest(base/"private"/(case["unit_id"]+"-archive-case.json")),
                        "physical_primitives": record["independent_check"]["physical_primitives"],
                        "separate_verification_primitives": record["independent_check"]["separate_verification_primitives"]}
                    write(output/"units"/f"{method}-{replicate}-{step}_points.json", journal)
                    history.append(journal); search.append(journal); seen.update(new)
                    completed += 1
                    heartbeat(completed_units=completed, planned_units=total, card_id="B02", archive_phase="search")
        # The first witnesses are committed before any follow-up maker is opened.
        write(output/"SEARCH_FREEZE.json", {"packet_hash": packet["packet_hash"], "candidate_journal_hash": digest(search),
            "first_witness_policy": "first observation per semantic ID in each method/replicate; no replacement using follow-up",
            "search_candidates": len(search)})
        namespace = DESIGN["followup"]["namespace"]+("/fixture" if fixture else "")
        base = output/"private/followup"
        for condition in CONDITIONS:
            for index in range(followup_n):
                if remaining_seconds(root) <= 0:
                    return {"execution_state": "checkpointed", "reason": "immutable ceiling", "campaign_complete": False}
                record = candidate(base, condition, index, namespace, packet, reader, followup_n)
                followup.append({"condition": condition["id"], "cases": record["case"]["cases"], "index": index})
                completed += 1
                heartbeat(completed_units=completed, planned_units=total, card_id="B02", archive_phase="frozen follow-up")
    raw_rows = [read(path) for folder in ["search", "followup"]
                for path in sorted((output/"private"/folder).glob("**/units/*_points.json"))]
    sampling = independent_units(raw_rows)
    if len(raw_rows) != total:
        raise ValueError("archive raw sample denominator differs")
    report = summarize(search, followup, 3 if fixture else DESIGN["followup"]["required_case_recurrences"])
    aggregate_check = verify_summary(search, followup, report, 3 if fixture else DESIGN["followup"]["required_case_recurrences"])
    if any(report["methods"][method]["total_physical_primitives"] != 113*replicas*steps for method in ["fixed", "adaptive"]):
        raise ValueError("archive policies received unequal physical budgets")
    write(output/"AGGREGATE.json", report)
    write(output/"INDEPENDENT_AGGREGATE.json", aggregate_check)
    write(output/"DEPENDENCE.json", {"sampling": sampling, "paired_or_nested": DESIGN["sampling"],
        "case_hits_are_not_independent_makers": True, "semantic_case_ids_exclude_unit_identity": True,
        "all_observed_and_rejected_production_independently_reexecuted": True})
    controls = validate_consumers(output, packet["packet_hash"], fixture=fixture)
    files = {str(path.relative_to(output)).replace("\\", "/"): file_digest(path)
             for folder in ["public", "private", "predictions", "units"] for path in sorted((output/folder).rglob("*"))
             if path.is_file() and path.suffix not in {".log", ".tmp", ".lock"}}
    write(output/"RAW_MANIFEST.json", {"files": files, "retention": "through verified final archive handoff"})
    result = {"card_id": "B02", "execution_state": "completed", "instrument_state": "valid", "evidence_scope": "fixture" if fixture else "archive discovery",
        "candidate_maker_records": len(search), "followup_maker_condition_records": len(followup),
        "archive_replicates_per_method": replicas, "physical_budget_per_method": 113*replicas*steps,
        "aggregate_sha256": file_digest(output/"AGGREGATE.json"), "raw_manifest_sha256": file_digest(output/"RAW_MANIFEST.json"),
        "consumer_attack_sha256": file_digest(output/"CONSUMER_ATTACKS.json"),
        "source_targets": "existing selection and process interpretation; additional native archive categories indexed separately",
        "confirmation_state": "not a confirmation", "campaign_complete": False, "completed_at": now()}
    write(output/"COMPLETION.json", result)
    return result

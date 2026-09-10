"""B01 finite read-only transfer: public inputs, separate evaluation, actual consumer."""
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import copy
import importlib.metadata
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import uuid
from .records import canonical, digest, read, write, now, file_digest
from .runtime import REPO, PACKAGE
from .transfer_consumer import FIELDS

MODULES = ("records", "world", "learning", "craft", "reconstruction", "opportunity",
           "self_monitor", "self_trajectory", "inquiry")
SELECTION = {"k01-scout-1": ("K01",), "reading-scout-1": ("P01", "P03"),
             "behavior-scout-1": ("O01", "O04", "S02", "S04", "S05"),
             "inquiry-scout-1": ("R01", "R03", "R05")}
TARGETS = {"craft": ["executed_reconstruction"], "reading": ["future_artifact", "executable_reconstruction"],
           "opportunity": ["changed_opportunity_response", "cause_uncertainty"],
           "self": ["controller", "original_goal", "adopted_goal", "repair"],
           "trajectory": ["resumed_controller", "original_goal", "adopted_goal", "repair"],
           "inquiry": ["paid_inquiry_action"], "inquiry-learning": ["updated_competence_state"],
           "inquiry-construction": ["heldout_executable_programs"],
           "inquiry-reading": ["unseen_maker_continuation"]}
DESIGN = {"card_id": "B01", "selection": SELECTION, "unit_selection": "seed index zero in every declared condition; all reader arms",
          "scope": "read-only export and replay of recorded public interaction branches; no new policy experiment",
          "ids": "random uuid4 aliases independent of condition, latent state and seed",
          "order": "random case/task aliases and shuffled independent cases; temporal order retained within interactions",
          "evaluator_join": "only after immutable predictions committed",
          "reader": "copied frozen reference functions; NumPy dependency explicit; no Ghost/Sounding installs",
          "known_answer_tolerance": 1e-12, "replay_tolerance": "exact JSON values for unchanged reader code",
          "private_access": "trusted imports then CPython file/process/network guard; not hostile-native OS isolation",
          "dependencies": ["X01", "X02", "X03", "X05", "X06"],
          "scientific_promotion": "pending full consumer-specific attack joins and confirmation",
          "complete_campaign": False}


def alias_fields(value, task_id, lineage_id):
    if isinstance(value, dict):
        return {key: task_id if key == "task_id" else lineage_id if key == "lineage_id"
                else alias_fields(item, task_id, lineage_id) for key, item in value.items()}
    if isinstance(value, list):
        return [alias_fields(item, task_id, lineage_id) for item in value]
    return value


def envelope(frame, task_id, lineage_id):
    observed = alias_fields(frame["public"], task_id, lineage_id)
    result = {"schema_version": "v16.transfer.1", "task_id": task_id, "lineage_id": lineage_id,
              "access_tier": observed.get("access_tier", "declared-public-interaction"),
              "final_artifact": observed.get("final_artifact", observed.get("artifacts", observed.get("artifact"))),
              "declared_context": {"operation": frame["kind"], "reader_options": frame["options"],
                                   "observation": observed},
              "permitted_prior_artifacts": observed.get("permitted_prior_artifacts", []),
              "permitted_query_descriptions": observed.get("permitted_query_descriptions", []),
              "query_costs": observed.get("query_costs", {"recorded_query_cost": observed.get("query_cost", 0)}),
              "target_request": {"targets": TARGETS[frame["kind"]], "request": observed.get("target_request")},
              "reader_action_budget": observed.get("reader_action_budget", "finite reference computation; retained counters")}
    assert set(result) == FIELDS
    return result


def stage9_evidence(public):
    """Actual Stage 9 evidence shape; no fabricated task/model/scorer identity."""
    operation = public["declared_context"]["operation"]
    support = range(16) if operation == "reading" else range(2) if operation in {"opportunity", "self", "trajectory"} else range(4) if operation == "inquiry-reading" else ()
    values = list(support)
    random.SystemRandom().shuffle(values)
    options = {uuid.uuid4().hex: json.dumps({"future_artifact": value}, separators=(",", ":")) for value in values}
    return {"prefix": "Predict or execute the declared synthetic task from this public evidence only.\n"
                      + canonical(public).decode() + "\nResponse:\n", "options": options}


def install_consumer(destination):
    files = {"consumer.py": PACKAGE/"transfer_consumer.py"}
    files.update({f"v16_reference/{name}.py": PACKAGE/f"{name}.py" for name in MODULES})
    for relative, source in files.items():
        path = destination/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.read_bytes() != source.read_bytes():
            raise ValueError("exported reference source changed")
        if not path.exists():
            shutil.copyfile(source, path)
    init = destination/"v16_reference/__init__.py"
    if not init.exists():
        init.write_bytes(b"")
    return {relative: file_digest(destination/relative) for relative in [*files, "v16_reference/__init__.py"]}


class Consumer:
    def __init__(self, directory, timeout=30):
        self.directory = Path(directory)
        self.timeout = timeout

    def __enter__(self):
        started = time.perf_counter()
        env = dict(os.environ, PYTHONPATH=str(self.directory), OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
        self.log = (self.directory/"consumer-stderr.log").open("ab")
        self.child = subprocess.Popen([sys.executable, "-s", "-B", "-u", "-m", "consumer"],
            cwd=self.directory, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        self.pool = ThreadPoolExecutor(max_workers=1)
        try:
            self.identity = self.receive()
            if not self.identity.get("ready"):
                raise ValueError("standalone reader did not start")
            self.startup_wall_seconds = time.perf_counter()-started
        except BaseException:
            self.close()
            raise
        return self

    def receive(self):
        try:
            line = self.pool.submit(self.child.stdout.readline).result(timeout=self.timeout)
        except TimeoutError:
            self.child.terminate()
            self.child.wait(timeout=5)
            raise RuntimeError("standalone consumer exceeded request time budget")
        if not line:
            raise RuntimeError("standalone consumer exited; inspect its retained stderr")
        return json.loads(line)

    def request(self, public):
        self.child.stdin.write(canonical(public)+b"\n")
        self.child.stdin.flush()
        return self.receive()

    def close(self):
        if self.child.poll() is None:
            try:
                self.child.stdin.write(b'{"operation":"shutdown"}\n')
                self.child.stdin.flush()
                self.child.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                self.child.terminate()
                self.child.wait(timeout=5)
        self.pool.shutdown(wait=True, cancel_futures=True)
        self.log.close()

    def __exit__(self, *error):
        self.close()


def export(root, destination, packet):
    # The original extractor is fixed source and reads only declared reader frames.
    from runners.replay_v16_readers import requests
    manifest_path = destination/"PUBLIC_MANIFEST.json"
    if manifest_path.exists():
        manifest = read(manifest_path)
        if manifest["packet_hash"] != packet["packet_hash"]:
            raise ValueError("transfer resume packet mismatch")
        return manifest
    sources = install_consumer(destination/"public/consumer")
    candidates = []
    source_packets = {}
    for packet_id, cards in SELECTION.items():
        source = read(root/"packets"/f"{packet_id}.json")
        for name, expected in source["identity"]["files"].items():
            if file_digest(REPO/name) != expected:
                raise ValueError("source packet changed before transfer")
        source_packets[packet_id] = source["packet_hash"]
        for path in sorted((root/packet_id).glob("**/units/*_points.json")):
            row = read(path)
            if row["card_id"] in cards and row["seed_components"]["index"] == 0:
                unit_root = path.parent.parent
                completion = read(unit_root/"COMPLETION.json")
                if completion["execution_state"] != "completed" or completion["instrument_state"] != "valid":
                    raise ValueError("unvalidated source capability cannot become admission fixture")
                candidates.append((path, row, list(requests(unit_root, row))))
    plan_path = destination/"private/EXPORT_PLAN.json"
    if plan_path.exists():
        plan = read(plan_path)
        if plan["packet_hash"] != packet["packet_hash"]:
            raise ValueError("partial export plan belongs to another packet")
        by_unit = {row["unit_id"]: (path, row, frames) for path, row, frames in candidates}
        candidates = [by_unit[item["unit_id"]] for item in plan["cases"]]
    else:
        random.SystemRandom().shuffle(candidates)
        plan = {"packet_hash": packet["packet_hash"], "cases": [
            {"unit_id": row["unit_id"], "case_id": uuid.uuid4().hex, "lineage_id": uuid.uuid4().hex,
             "task_ids": [uuid.uuid4().hex for _ in frames]}
            for path, row, frames in candidates], "exported_at": now()}
        write(plan_path, plan)
    public_files, stage9_files, evaluations, cases = {}, {}, {}, []
    for (path, row, frames), case_plan in zip(candidates, plan["cases"]):
        case_id, lineage_id = case_plan["case_id"], case_plan["lineage_id"]
        task_ids, expected = [], {}
        for frame, task_id in zip(frames, case_plan["task_ids"]):
            observed = envelope(frame, task_id, lineage_id)
            relative = f"public/observations/{task_id}.json"
            public_files[relative] = write(destination/relative, observed)
            # Stage9 all-option evidence or open generation. Missing model identity
            # is explicit in the interface report; this is not a launchable task lock.
            shaped_path = destination/f"public/stage9/{task_id}.json"
            shaped = read(shaped_path) if shaped_path.exists() else stage9_evidence(observed)
            stage9_files[f"public/stage9/{task_id}.json"] = write(shaped_path, shaped)
            task_ids.append(task_id)
            expected[task_id] = frame["result"]
        truth = read(path.parent.parent/"private"/f"{row['unit_id']}.json")
        evaluation = {"case_id": case_id, "lineage_id": lineage_id, "task_ids": task_ids,
                      "source": {"unit_path": str(path.relative_to(root)).replace("\\", "/"),
                                 "unit_sha256": file_digest(path), "packet_hash": row["packet_hash"],
                                 "card_id": row["card_id"], "condition": row["condition"],
                                 "seed_components": row["seed_components"]},
                      "true_production_record": truth,
                      "acquisition_record": truth.get("acquisition_record", truth.get("world", {}).get("acquisition")),
                      "original_goal": truth.get("actual_original_goal", truth.get("original_goal")),
                      "current_goal": truth.get("current_goal"), "controller_state": truth.get("controller"),
                      "actual_feasibility": truth.get("actual_feasibility"),
                      "maker_beliefs": truth.get("maker_beliefs"),
                      "considered_alternatives": truth.get("considered_alternatives"),
                      "hidden_continuations": truth.get("hidden_continuation", truth.get("hidden_future")),
                      "intervention_outcomes": truth.get("rounds", truth.get("frames")),
                      "equivalence_classes_for_declared_targets": truth.get("equivalence_classes"),
                      "field_semantics": "null means not separately defined in this family; complete original truth retained above",
                      "scoring_contract": {"prediction_replay": "exact", "source_outcomes": row["arms"],
                                           "no_new_scientific_effect_estimate": True},
                      "expected_predictions": expected}
        relative = f"private/{case_id}-evaluation.json"
        evaluations[relative] = write(destination/relative, evaluation)
        cases.append({"case_id": case_id, "lineage_id": lineage_id, "tasks_in_recorded_order": task_ids})
    manifest = {"schema_version": "v16.transfer-manifest.1", "packet_hash": packet["packet_hash"],
                "source_packets": source_packets, "consumer_sources": sources, "observations": public_files,
                "stage9_evidence": stage9_files,
                "cases": cases, "n_cases": len(cases), "n_tasks": len(public_files),
                "dependencies": {"numpy": importlib.metadata.version("numpy")},
                "scope": DESIGN["scope"], "exported_at": plan["exported_at"]}
    write(destination/"private/EVALUATION_MANIFEST.json", evaluations)
    write(manifest_path, manifest)
    return manifest


def consume(destination, heartbeat=lambda **kw: None):
    # No private evaluation data is read in this phase.
    manifest = read(destination/"PUBLIC_MANIFEST.json")
    began, cpu = time.perf_counter(), time.process_time()
    prediction_files = {}
    with Consumer(destination/"public/consumer") as consumer:
        if consumer.identity["sources"] != manifest["consumer_sources"]:
            raise ValueError("loaded standalone source differs from export")
        probe = consumer.request({"operation": "probe-private",
                                  "path": str((destination/"private/EVALUATION_MANIFEST.json").resolve())})
        if probe.get("ok") or not probe.get("error", "").startswith("PermissionError:"):
            raise ValueError("standalone consumer private access probe failed")
        for count, (relative, expected) in enumerate(manifest["observations"].items(), 1):
            path = destination/relative
            if file_digest(path) != expected:
                raise ValueError("public observation changed")
            public = read(path)
            target = f"predictions/{public['task_id']}.json"
            result = read(destination/target) if (destination/target).exists() else consumer.request(public)
            if not result.get("ok"):
                raise ValueError(result)
            if result["observation_sha256"] != digest(public):
                raise ValueError("retained prediction observation mismatch")
            prediction_files[target] = write(destination/target, result)
            if count % 50 == 0:
                heartbeat(completed_units=count, planned_units=manifest["n_tasks"])
        identity = consumer.identity
        startup = consumer.startup_wall_seconds
    receipt = {"prediction_files": prediction_files, "reader_identity": identity, "private_probe": probe,
               "committed_at": now(), "manifest_sha256": file_digest(destination/"PUBLIC_MANIFEST.json"),
               "wall_seconds": time.perf_counter()-began, "parent_cpu_seconds": time.process_time()-cpu,
               "startup_wall_seconds": startup,
               "reader_cpu_seconds": sum(read(destination/path)["runtime"]["reader_cpu_seconds"]
                                          for path in prediction_files),
               "resource_scope": "actual consumed requests; no retrospective claim about original scout resources"}
    path = destination/"PREDICTIONS_COMMITTED.json"
    if path.exists():
        old = read(path)
        if old["prediction_files"] != prediction_files or old["manifest_sha256"] != receipt["manifest_sha256"]:
            raise ValueError("resumed predictions differ")
        return old
    write(path, receipt)
    return receipt


def evaluate(destination):
    committed = read(destination/"PREDICTIONS_COMMITTED.json")
    manifest = read(destination/"PUBLIC_MANIFEST.json")
    if committed["manifest_sha256"] != file_digest(destination/"PUBLIC_MANIFEST.json"):
        raise ValueError("evaluation public manifest changed")
    for relative, expected in committed["prediction_files"].items():
        if file_digest(destination/relative) != expected:
            raise ValueError("prediction changed before evaluator join")
    cases, count = [], 0
    for relative, expected_hash in read(destination/"private/EVALUATION_MANIFEST.json").items():
        if file_digest(destination/relative) != expected_hash:
            raise ValueError("private evaluation changed")
        truth = read(destination/relative)
        for task_id, expected in truth["expected_predictions"].items():
            predicted = read(destination/"predictions"/f"{task_id}.json")
            public = read(destination/"public/observations"/f"{task_id}.json")
            if predicted["result"] != expected or predicted["observation_sha256"] != digest(public):
                raise ValueError("standalone prediction does not match validated source")
            count += 1
        cases.append({"case_id": truth["case_id"], "source": truth["source"],
                      "n_tasks": len(truth["task_ids"]), "exact_prediction_agreement": True})
    if count != manifest["n_tasks"]:
        raise ValueError("incomplete evaluator join")
    result = {"card_id": "B01", "execution_state": "completed", "instrument_state": "valid",
              "consumer_state": "standalone executed; private read denied; predictions committed before evaluator join",
              "n_cases": len(cases), "n_tasks": count, "cases": cases,
              "scientific_promotion": "pending consumer-specific attacks; no real-text admission",
              "real_text_launch": False, "campaign_complete": False, "evaluated_at": now()}
    path = destination/"COMPLETION.json"
    if path.exists():
        previous = read(path)
        result["evaluated_at"] = previous["evaluated_at"]
    write(path, result)
    return result

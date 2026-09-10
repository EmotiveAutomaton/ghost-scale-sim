"""Execute the exported consumer without importing Ghost in its child process."""
import copy
from itertools import product
import json
from math import exp
import uuid
import pytest
from ghostscale.validation.soundingline.v16.transfer import (
    Consumer, install_consumer, envelope, stage9_evidence, consume, evaluate)
from ghostscale.validation.soundingline.v16.records import read, write, digest


def wrapped(kind, public, **options):
    return envelope({"kind": kind, "public": public, "options": options},
                    uuid.uuid4().hex, uuid.uuid4().hex)


def reading_observation():
    return {"schema_version": "v16.public.2", "task_id": None, "lineage_id": None,
            "access_tier": "artifact-only", "final_artifact": 3,
            "declared_context": {"training_attempts": 0, "current_observation":
                {"artifact": 3, "target": 3, "feasible": list(range(8)), "prefix": []}},
            "permitted_prior_artifacts": [], "permitted_query_descriptions": [], "query_costs": {},
            "target_request": {"future_target": 12}, "reader_action_budget": 128}


def test_standalone_transfer_known_answer_and_private_read_denial(tmp_path):
    installed = install_consumer(tmp_path/"consumer")
    private = tmp_path/"evaluator.json"
    private.write_text('{"true_library": "secret"}')
    with Consumer(tmp_path/"consumer") as reader:
        assert reader.identity["sources"] == installed
        assert reader.identity["ghostscale_imports"] == []
        probe = reader.request({"operation": "probe-private", "path": str(private)})
        assert not probe["ok"] and probe["error"].startswith("PermissionError:")
        public = wrapped("reading", reading_observation(), strategy="maker")
        first = reader.request(public)
        assert first["ok"]
        # An independent scalar reference: with zero acquisition all routes use
        # primitive coding. Enumerate the complete finite support, including STOP.
        weights = [0.0]*16
        for length in range(4):
            for route in product(range(8), repeat=length):
                board = 0
                for command in route:
                    bit = 1 << (command % 4)
                    board = board | bit if command < 4 else board & ~bit
                weights[board] += exp(-(board ^ 12).bit_count()-0.3*length)
        expected = [mass/sum(weights) for mass in weights]
        assert first["result"]["future_probabilities"] == pytest.approx(expected, abs=1e-12)
        private.write_text('{"true_library": "changed", "seed": 922}')
        assert reader.request(public)["result"] == first["result"]
        impossible = copy.deepcopy(public)
        impossible["declared_context"]["observation"]["declared_context"]["current_observation"]["artifact"] = 15
        mismatch = reader.request(impossible)
        assert mismatch["ok"] and mismatch["result"]["model_mismatch"]
        contaminated = copy.deepcopy(public)
        contaminated["true_production_record"] = [0, 1]
        assert not reader.request(contaminated)["ok"]


def test_standalone_transfer_learning_changes_executed_competence_and_noise_does_not(tmp_path):
    install_consumer(tmp_path)
    prior = [0.9/24]*24+[0.1]
    mapping = (2, 0, 3, 1)
    with Consumer(tmp_path) as reader:
        initial = reader.request(wrapped("inquiry-construction", {"beliefs": [prior, prior]}))["result"]
        learned = reader.request(wrapped("inquiry-learning",
            {"beliefs": [prior, prior], "examples": [
                {"domain": 0, "command": command, "cell": mapping[command]} for command in range(3)],
             "forget_domains": []}))["result"]
        after = reader.request(wrapped("inquiry-construction", {"beliefs": learned["beliefs"]}))["result"]
        def score(result):
            outcomes = []
            for program in result["programs"]:
                if program["domain"] == 0:
                    artifact = 0
                    for command in program["commands"]:
                        artifact |= 1 << mapping[command]
                    outcomes.append(artifact == program["goal"])
            return sum(outcomes)/len(outcomes)
        assert score(initial) == 1/6 and score(after) == 1
        noise = reader.request(wrapped("inquiry-learning",
            {"beliefs": [prior, prior], "examples": [
                {"domain": 0, "command": 0, "cell": 0}, {"domain": 0, "command": 0, "cell": 1}],
             "forget_domains": []}))["result"]["beliefs"][0]
        assert noise[-1] == 1
        # Actual independent random physical outputs give two successes in 16.
        assert sum(((1 << a) | (1 << b)) == 3 for a, b in product(range(4), repeat=2))/16 == 0.125


def test_stage9_transfer_has_complete_opaque_options_and_exact_evidence_shape():
    public = wrapped("reading", reading_observation(), strategy="maker")
    evidence = stage9_evidence(public)
    assert set(evidence) == {"prefix", "options"}
    assert len(evidence["options"]) == 16
    assert all(len(key) == 32 for key in evidence["options"])
    assert {json.loads(value)["future_artifact"] for value in evidence["options"].values()} == set(range(16))
    reordered = dict(reversed(list(evidence["options"].items())))
    assert reordered == evidence["options"]  # the mapping survives order permutation
    assert canonical_task(public) in evidence["prefix"]
    assert set(stage9_evidence(wrapped("inquiry-learning",
        {"beliefs": [], "examples": [], "forget_domains": []}))["options"]) == set()


def canonical_task(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def test_transfer_evaluator_requires_committed_predictions_and_rejects_changed_results(tmp_path):
    public = wrapped("reading", reading_observation(), strategy="maker")
    sources = install_consumer(tmp_path/"public/consumer")
    relative = f"public/observations/{public['task_id']}.json"
    observation_hash = write(tmp_path/relative, public)
    write(tmp_path/"PUBLIC_MANIFEST.json", {
        "observations": {relative: observation_hash}, "consumer_sources": sources,
        "n_tasks": 1, "n_cases": 1})
    with pytest.raises(FileNotFoundError):
        evaluate(tmp_path)
    # Private data is deliberately absent while predictions are being committed;
    # the access probe must still be PermissionError, never a missing-file fallback.
    committed = consume(tmp_path)
    response = read(tmp_path/"predictions"/f"{public['task_id']}.json")
    evaluation = {"case_id": "case", "source": {"scope": "fixture"},
                  "task_ids": [public["task_id"]],
                  "expected_predictions": {public["task_id"]: response["result"]}}
    private_hash = write(tmp_path/"private/case-evaluation.json", evaluation)
    write(tmp_path/"private/EVALUATION_MANIFEST.json", {"private/case-evaluation.json": private_hash})
    assert evaluate(tmp_path)["instrument_state"] == "valid"
    assert consume(tmp_path)["prediction_files"] == committed["prediction_files"]
    response["result"]["future_probabilities"][0] = 0
    write(tmp_path/"predictions"/f"{public['task_id']}.json", response, immutable=False)
    with pytest.raises(ValueError, match="prediction changed"):
        evaluate(tmp_path)

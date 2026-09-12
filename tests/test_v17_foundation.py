"""Meaningful V17 admission: real physics, evidence boundaries, costs and CLI resume."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from ghostscale.validation.soundingline.v16.records import read, write
from ghostscale.validation.soundingline.v17 import packets
from ghostscale.validation.soundingline.v17.contracts import Costs, BudgetExhausted, forecast_scores, public_evidence
from ghostscale.validation.soundingline.v17.craft import METHODS, REGIMES, acquire, make_case, solve
from ghostscale.validation.soundingline.v17.programs import encode, execute, expand
from ghostscale.validation.soundingline.v17.validity import checks


@pytest.mark.parametrize("arity", [0, 1, 2])
def test_abstraction_expansion_matches_primitive_physics_and_cost(arity):
    args = [3, 7][:arity]
    body = {"op": "seq", "items": [
        {"op": "place", "cell": {"arg": 0} if arity else 3},
        {"op": "place", "cell": {"arg": 1} if arity == 2 else 7}]}
    library = [{"arity": arity, "body": body}]
    program = expand({"op": "call", "index": 0, "args": args}, library)
    for initial in (0, 8, 65535):
        assert execute(program, initial=initial) == execute([3, 7], initial=initial)
        assert execute(program, initial=initial)["primitive_cost"] == 2


@pytest.mark.parametrize("bad", [-1, 32, True, "place"])
def test_invalid_primitive_never_becomes_valid_encoding(bad):
    with pytest.raises(ValueError):
        encode([bad])
    assert not execute([bad])["legal"]


def test_illegal_argument_and_unknown_call_are_rejected():
    with pytest.raises(ValueError):
        expand({"op": "place", "cell": {"arg": 0}})
    with pytest.raises(ValueError):
        expand({"op": "call", "index": 4, "args": []}, [])


@pytest.mark.parametrize("probabilities", [None, {}, {"a": .8, "b": .8}, {"a": float("nan"), "b": 0},
                                          {"a": 1}, {"a": -.1, "b": 1.1}])
def test_missing_invalid_forecast_is_not_uniform(probabilities):
    with pytest.raises(ValueError):
        forecast_scores(probabilities, ["a", "b"], "a")


def test_proper_scores_keep_zero_truth_probability_explicit():
    assert forecast_scores({"a": 1, "b": 0}, ["a", "b"], "b") == {
        "log_loss_nats": None, "log_loss_infinite": True, "brier_score": 2}
    assert forecast_scores({"a": 1, "b": 0}, ["a", "b"], "a")["log_loss_nats"] == 0


def test_private_truth_and_process_cannot_enter_artifact_request():
    case = {"artifact": 4, "public_context": {"allowed": [1]}, "earlier_artifacts": [2],
            "recorded_process": [{"action": 2}], "true_goal": "SECRET", "private": "SECRET"}
    artifact = public_evidence(case, "artifact")
    assert set(artifact) == {"artifact", "context"}
    assert "SECRET" not in json.dumps(artifact)
    artifact["context"]["allowed"].append(9)
    assert case["public_context"]["allowed"] == [1]
    assert set(public_evidence(case, "artifact_collection")) == {"artifact", "context", "earlier_artifacts"}


@pytest.mark.parametrize("method", METHODS)
def test_real_charges_respect_budget_and_storage_cap(method):
    case = make_case("test-cost", 0, 0, "changed_constraint")
    for budget in (0, 8, 128, 2048):
        row = solve(case["public"], method, budget, memory_cap=16)
        cost = row["costs"]
        assert cost["search_total"] <= budget
        assert cost["definition_storage"] <= 16
        assert cost["hypothetical_execution"] == row["attempted_primitives"]
        if row["program"] is None:
            assert row["missing_output"] and not row["task_success"] and row["execution"] is None
        else:
            assert cost["actual_execution"] == len(row["execution"]["trace"])
    assert acquire(case["public"]["training"], method, 0)["costs"]["definition_storage"] == 0


def test_known_answer_gates():
    report = checks()
    assert report["passed"], report


@pytest.mark.parametrize("regime", REGIMES)
def test_withheld_target_reachable_and_not_in_acquisition(regime):
    for c in range(8):
        case = make_case("test-holdout", c, 0, regime)
        p = case["public"]
        assert p["target"] not in {t["target"] for t in p["training"]}
        witness = execute(case["private"]["realized_history"], initial=p["initial"], forbidden=p["forbidden"])
        assert witness["legal"] and witness["artifact"] == p["target"]


def cli(root, *extra):
    return subprocess.run([sys.executable, "-m", "runners.run_v17", "--stage", "fixture",
                           "--root", str(root), *extra], cwd=packets.REPO,
                          capture_output=True, text=True, timeout=90)


def test_literal_cli_resume_retains_rows_and_aggregate(tmp_path):
    whole, resumed = tmp_path/"whole", tmp_path/"resumed"
    assert cli(whole).returncode == 0
    assert cli(resumed, "--stop-after-chunks", "1").returncode == 0
    first = (resumed/"raw/chunk-00000_points.jsonl").read_bytes()
    assert not (resumed/"COMPLETION.json").exists()
    assert cli(resumed).returncode == 0
    assert (resumed/"raw/chunk-00000_points.jsonl").read_bytes() == first
    assert read(whole/"COMPARISONS.json") == read(resumed/"COMPARISONS.json")
    for chunk in read(whole/"INDEX.json")["chunks"]:
        assert (whole/chunk["path"]).read_bytes() == (resumed/chunk["path"]).read_bytes()
    # A corrupted retained chunk must fail, never regenerate or silently accept it.
    with (resumed/"raw/chunk-00000_points.jsonl").open("ab") as stream:
        stream.write(b"CORRUPTION")
    failure = cli(resumed)
    assert failure.returncode != 0 and "hash mismatch" in failure.stderr


def test_source_or_design_change_cannot_resume_frozen_packet(tmp_path):
    root = tmp_path/"run"
    assert cli(root, "--stop-after-chunks", "1").returncode == 0
    lock = read(root/"LOCK.json")
    lock["design"]["memory_cap"] += 1
    write(root/"LOCK.json", lock, immutable=False)
    failure = cli(root)
    assert failure.returncode != 0 and "frozen design or source changed" in failure.stderr


def test_crash_between_chunk_creation_and_index_is_replayed_without_replacement(tmp_path, monkeypatch):
    root = tmp_path/"run"
    original = packets.write
    def interrupted(path, value, **kwargs):
        if path.name == "INDEX.json":
            raise RuntimeError("simulated crash before index")
        return original(path, value, **kwargs)
    monkeypatch.setattr(packets, "write", interrupted)
    design = packets.specification("fixture")
    with pytest.raises(RuntimeError, match="simulated crash"):
        packets.run_packet(root, design)
    first = (root/"raw/chunk-00000_points.jsonl").read_bytes()
    monkeypatch.setattr(packets, "write", original)
    assert packets.run_packet(root, design)["state"] == "completed"
    assert (root/"raw/chunk-00000_points.jsonl").read_bytes() == first

def test_private_evaluator_field_in_construction_request_is_rejected():
    public = make_case("test-boundary", 0, 0, "familiar_combinations")["public"]
    public["private"] = {"true_history": [0, 1]}
    with pytest.raises(ValueError, match="unexpected public"):
        solve(public, "primitive_search", 2048)


def test_screen_refuses_unadmitted_consumer(tmp_path):
    result = subprocess.run([sys.executable, "-m", "runners.run_v17", "--stage", "screen",
                             "--root", str(tmp_path/"screen")],
                            cwd=packets.REPO, capture_output=True, text=True, timeout=10)
    assert result.returncode != 0
    assert "source-bound admission receipt" in result.stderr
    assert not (tmp_path/"screen").exists()

import pytest
from ghostscale.validation.soundingline.v16.attack_access_stable import alias, comparison, requests
from ghostscale.validation.soundingline.v16.reader_process import ReaderProcess
from ghostscale.validation.soundingline.v16.records import write
from ghostscale.validation.soundingline.v16.inquiry_stable import PRIOR

PREFIX = "ghostscale.validation.soundingline.v16.inquiry_stable:"


def test_amended_access_frames_retain_actual_reader_identity_and_hide_evaluator(tmp_path):
    row = {"unit_id": "unit", "card_id": "R03", "unit_kind": "inquiry"}
    write(tmp_path/"public/unit.json", {"task_id": "opaque"})
    first = {"kind": PREFIX+"learn_public", "public": {"examples": []}, "options": {}, "result": {"beliefs": []}}
    final = {"kind": PREFIX+"construct_public", "public": {"beliefs": []}, "options": {}, "result": {"programs": []}}
    write(tmp_path/"predictions/unit-initial.json", {"requests": [first]})
    write(tmp_path/"predictions/unit.json", {"requests": [final]})
    write(tmp_path/"private/unit.json", {"true_mapping": [2, 0, 3, 1], "seed": 77})
    result = list(requests(tmp_path, row))
    assert result == [first, final]
    assert all("true_mapping" not in frame["public"] for frame in result)


def test_amended_access_aliases_cache_and_private_read_are_actual(tmp_path):
    kind = PREFIX+"choose"
    public = {"schema_version": "v16.inquiry.1", "task_id": "fixture", "beliefs": [list(PRIOR), list(PRIOR)],
        "offers": [0, 1], "histories": [[], []], "familiarity": [4, 0], "pending_commands": [[], []],
        "feedback_batches": [2, 2], "future_weights": [0.5, 0.5], "opportunity_cost": 0.01,
        "remaining_interactions": 8, "tie_draw": 0.25, "committed_domain": None,
        "offer_probabilities": [[0.1, 0.2, 0.6, 0.1], [0.4, 0.3, 0.2, 0.1]]}
    private = tmp_path/"private.json"
    write(private, {"true_mapping": [0, 1, 2, 3]})
    with ReaderProcess(tmp_path/"reader", extensions=[kind]) as reader:
        original = reader.request(kind, public, policy="value-learning")
        comparison(reader.request(kind, alias(public), policy="value-learning"), original)
        reader.request(kind, public, policy="eig")
        write(private, {"true_mapping": [2, 3, 0, 1]}, immutable=False)
        comparison(reader.request(kind, public, policy="value-learning"), original)
        with pytest.raises(RuntimeError, match="PermissionError"):
            reader.request("_probe_forbidden_read", {"path": str(private)})
        with pytest.raises(ValueError, match="access/cache"):
            comparison(original, {**original, "scores": [999, -999]})

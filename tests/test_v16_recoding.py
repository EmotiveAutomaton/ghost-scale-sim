from itertools import product
import pytest
from ghostscale.validation.soundingline.v16.recoding import (
    macro_record, compile_macros, board, action, close, reading_observation,
    recognition_observation, inquiry_observation, inquiry_result_equivalent,
    pushed_probabilities, SWAP_MOTIFS)
from ghostscale.validation.soundingline.v16.reference import interpret
from ghostscale.validation.soundingline.v16.reconstruction import prepare, reader
from ghostscale.validation.soundingline.v16.reader_process import ReaderProcess
from ghostscale.validation.soundingline.v16.records import canonical
from ghostscale.validation.soundingline.v16.inquiry import PRIOR
from ghostscale.validation.soundingline.v16.graphic_reference import interpret as large_interpret
from ghostscale.validation.soundingline.v16.assembly_reference import interpret as assembly_interpret


def test_macro_spelling_and_boundaries_preserve_executed_trace_and_primitive_cost():
    examples = [([], []), ([0, 1, 4], [1]), ([0, 1, 2, 9], [2])]
    for program, cuts in examples:
        first = macro_record(program, cuts, "alpha")
        second = macro_record(program, [], "beta", True)
        assert first != second or not program
        assert compile_macros(first) == compile_macros(second) == program
    first = macro_record([0, 1, 4], [1], "alpha")
    second = macro_record([0, 1, 4], [], "beta")
    assert interpret(compile_macros(first)) == interpret(compile_macros(second))
    assert interpret(compile_macros(second))["primitive_cost"] == 3
    assert len(second["calls"]) == 1  # an intentionally broken macro-token cost
    assert len(second["calls"]) != interpret(compile_macros(second))["primitive_cost"]
    assembly = {"parents": [-1, 0, 0], "defaults": [0, 1, 0]}
    assert assembly_interpret(assembly, compile_macros(macro_record([0, 1, 2, 9], [2]))) == assembly_interpret(assembly, [0, 1, 2, 9])
    assert large_interpret(compile_macros(macro_record([0, 1, 5, 16, 9], [2, 4]))) == large_interpret([0, 1, 5, 16, 9])
    with pytest.raises(ValueError, match="unknown procedure"):
        compile_macros({"procedures": {}, "calls": ["missing"]})


def test_cell_relabeling_preserves_all_small_world_primitive_transitions():
    permutation = SWAP_MOTIFS
    for initial, command in product(range(16), range(8)):
        before = interpret([command], start=initial)
        after = interpret([action(command, permutation)], start=board(initial, permutation))
        assert board(before["artifact"], permutation) == after["artifact"]
        assert before["primitive_cost"] == after["primitive_cost"]
    large = tuple(reversed(range(16)))
    for initial in [0, 65535, *(1 << index for index in range(16))]:
        for command in range(32):
            before = large_interpret([command], initial=initial)
            after = large_interpret([action(command, large)], initial=board(initial, large))
            assert board(before["artifact"], large) == after["artifact"]


def test_exact_reading_predictions_follow_the_motif_preserving_world_relabeling():
    public, _, _ = prepare("X02-known-answer", "P03", {"id": "fixture", "prior_works": 4}, 0)
    translated = reading_observation(public)
    before, after = reader(canonical(public)), reader(canonical(translated))
    assert close(pushed_probabilities(before["future_probabilities"], SWAP_MOTIFS), after["future_probabilities"])
    assert close([before["history_probabilities"][index] for index in (0, 2, 1, 3)], after["history_probabilities"])


def test_command_map_public_relabeling_preserves_learning_and_execution_values(tmp_path):
    permutation = (2, 0, 3, 1)
    observations = [
        ("inquiry-learning", {"beliefs": [PRIOR, PRIOR], "examples": [
            {"domain": 0, "command": 0, "cell": 2}], "forget_domains": []}),
        ("inquiry-construction", {"beliefs": [PRIOR, PRIOR]}),
        ("inquiry-reading", {"belief": PRIOR, "artifact": 3})]
    with ReaderProcess(tmp_path) as worker:
        for kind, public in observations:
            translated = inquiry_observation(public, permutation)
            before = worker.request(kind, public)
            after = worker.request(kind, translated)
            assert inquiry_result_equivalent(kind, before, after, public, translated, permutation)


def test_recognition_uses_recoded_physical_relations_not_cell_spellings(tmp_path):
    import random
    public = {"schema_version": "v16.recognition.1", "task_id": "fixture",
              "world": {"permutation": list(range(16)), "style_reuse": 0.9, "core_reuse": 0.8},
              "references": [[{"artifact": 55, "first_action": 0}],
                             [{"artifact": 199, "first_action": 1}]],
              "anonymous": [{"artifact": 55, "first_action": None}]}
    kind = "ghostscale.validation.soundingline.v16.recognition:public_reader"
    with ReaderProcess(tmp_path, extensions=[kind]) as worker:
        original = worker.request(kind, public, method="craft")
        for seed in range(20):
            permutation = list(range(16))
            random.Random(seed).shuffle(permutation)
            changed = recognition_observation(public, permutation)
            assert close(original, worker.request(kind, changed, method="craft"))


def test_inquiry_recoding_retains_old_failure_and_validates_corrected_reader(tmp_path):
    public = {"schema_version": "v16.inquiry.1", "task_id": "fixture", "beliefs": [PRIOR, PRIOR],
              "offers": [0, 1], "histories": [[], []], "familiarity": [4, 0],
              "pending_commands": [[], []], "feedback_batches": [1, 1],
              "future_weights": [0.5, 0.5], "opportunity_cost": 0.01,
              "remaining_interactions": 8, "tie_draw": 0.25, "committed_domain": None,
              "offer_probabilities": [[0.25]*4, [0.25]*4]}
    from itertools import permutations
    partial = [1/6 if mapping[0] == 2 else 0.0 for mapping in permutations(range(4))]+[0.0]
    public["beliefs"] = [partial, PRIOR]
    public["feedback_batches"] = [2, 2]
    public["offer_probabilities"] = [[0.1, 0.2, 0.6, 0.1], [0.4, 0.3, 0.2, 0.1]]
    permutation = (2, 0, 3, 1)
    translated = inquiry_observation(public, permutation)
    assert translated["offer_probabilities"][0][permutation[2]] == 0.6
    stable_kind = "ghostscale.validation.soundingline.v16.inquiry_stable:choose"
    with ReaderProcess(tmp_path, extensions=[stable_kind]) as worker:
        for policy in ["recognition", "surprise", "eig", "signed-progress", "absolute-progress",
                       "uniform", "decline", "value-learning"]:
            before = worker.request("inquiry", public, policy=policy)
            after = worker.request("inquiry", translated, policy=policy)
            equivalent = inquiry_result_equivalent("inquiry", before, after, public, translated, permutation)
            if policy == "value-learning":
                assert not equivalent
                assert before["scores"][1] == pytest.approx(0.05)
                assert after["scores"][1] == pytest.approx(0.0725)
            else:
                assert equivalent
            corrected_before = worker.request(stable_kind, public, policy=policy)
            corrected_after = worker.request(stable_kind, translated, policy=policy)
            assert inquiry_result_equivalent("inquiry", corrected_before, corrected_after, public, translated, permutation)

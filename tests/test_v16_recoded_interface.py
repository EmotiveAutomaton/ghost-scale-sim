import copy
import pytest
from ghostscale.validation.soundingline.v16.recoded_interface import encode, decode
from ghostscale.validation.soundingline.v16.reader_process import ReaderProcess
from ghostscale.validation.soundingline.v16.records import canonical
from ghostscale.validation.soundingline.v16.inquiry_stable import PRIOR, choose, construct_public

EXTENSION = "ghostscale.validation.soundingline.v16.recoded_interface:public_reader"


def test_macro_interface_expands_boundaries_and_rejects_unknown_procedure():
    public = {"training": {"attempts": [[0, 1, 4], [1, 2]], "instructions": [[0, 1], [1, 2]]},
              "commands": 2, "prefix": [0, 1], "library": [[0, 1]], "beliefs": [list(PRIOR)]}
    first, second = encode(public, "alpha", False), encode(public, "beta", True)
    assert first != second
    decoded, a = decode(first)
    other, b = decode(second)
    assert decoded == other == public
    assert a["expanded_primitive_tokens"] == b["expanded_primitive_tokens"]
    assert a["macro_calls"] < b["macro_calls"]
    broken = copy.deepcopy(first)
    broken["prefix"]["_v16_macro"]["calls"][0] = "missing"
    with pytest.raises(ValueError, match="unknown procedure"):
        decode(broken)


def test_recoded_actual_consumer_preserves_public_predictions(tmp_path):
    public = {"beliefs": [list(PRIOR), list(PRIOR)]}
    kind = "ghostscale.validation.soundingline.v16.inquiry_stable:construct_public"
    expected = construct_public(canonical(public))
    with ReaderProcess(tmp_path, extensions=[EXTENSION]) as worker:
        for split in (False, True):
            result = worker.request(EXTENSION, encode(public, str(split), split), operation=kind, reader_options={})
            assert result["result"] == expected
        # Actual observed command sequences pass through changed spellings and
        # boundaries, then drive the existing acquisition and construction code.
        from ghostscale.validation.soundingline.v16.craft import prepare_unit, construct_public as craft
        source = prepare_unit({"id": "fixture", "training_attempts": 16, "search_primitive_budget": 32},
                              0, namespace="recoded-known-answer", constructors=1, evidence_scope="fixture")["public"]
        expected = craft(canonical(source))
        for split in (False, True):
            result = worker.request(EXTENSION, encode(source, str(split), split), operation="craft", reader_options={})
            assert result["result"] == expected
            assert result["representation"]["nonempty_program_fields"] > 0


def test_physical_recoding_uses_actual_predictions_and_detects_tampering(tmp_path):
    from ghostscale.validation.soundingline.v16.attack_recoding import compare_physical
    from ghostscale.validation.soundingline.v16.reconstruction import prepare
    public, _, _ = prepare("recoding-known-answer", "P01", {"id": "fixture", "prior_works": 2}, 0)
    with ReaderProcess(tmp_path) as worker:
        expected = worker.request("reading", public, strategy="maker")
        def call(kind, observed, options):
            return worker.request(kind, observed, **options)
        assert compare_physical("reading", public, expected, {"strategy": "maker"}, call)["passed"]
        changed = copy.deepcopy(expected)
        changed["future_probabilities"][0] += 0.1
        with pytest.raises(ValueError, match="prediction changed"):
            compare_physical("reading", public, changed, {"strategy": "maker"}, call)

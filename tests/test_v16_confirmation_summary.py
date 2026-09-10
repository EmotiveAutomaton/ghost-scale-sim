from copy import deepcopy
import pytest
from ghostscale.validation.soundingline.v16.confirmation_summary import summarize_subset
from ghostscale.validation.soundingline.v16.expansion_adapter import design, execute_unit, auditor
from ghostscale.validation.soundingline.v16.expansion_runner import EXTENSIONS
from ghostscale.validation.soundingline.v16.reader_process import ReaderProcess


@pytest.mark.parametrize("card", ["K01", "K03", "K04", "K05", "M01", "S02", "S04", "S05"])
def test_actual_native_subset_has_independent_physics_and_statistics(card, tmp_path):
    frozen = deepcopy(design(card))
    condition = frozen["conditions"][0]
    packet = {"packet_hash": "known-confirmation-subset", "identity": {"commission_hash": "fixture", "environment": {"scope": "known fixture"}}}
    base = tmp_path/card
    with ReaderProcess(tmp_path/"reader", extensions=EXTENSIONS) as reader:
        rows = [execute_unit(card, base, condition, index, namespace="known-confirmation-subset-"+card,
            packet=packet, reader=reader, constructors=2, scope="fixture") for index in range(2)]
    summary = summarize_subset(card, rows, [condition["id"]])
    result = auditor(card)(base, summary)
    assert result["instrument_state"] == "valid"
    assert summary["n_maker_packets"] == 2
    assert set(summary["conditions"]) == {condition["id"]}
    assert design(card) == frozen
    with pytest.raises(ValueError, match="regimes"):
        summarize_subset(card, rows, ["undeclared-condition"])

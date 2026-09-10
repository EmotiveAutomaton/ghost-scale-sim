from copy import deepcopy
import pytest
from ghostscale.validation.soundingline.v16.dependence import (
    source_bound_sample, duplicate_attacks, partition, cluster_fixture, hierarchical_control, sampling_groups)
from ghostscale.validation.soundingline.v16.records import digest


def test_dependence_rejects_renamed_histories_seeds_and_forged_source_identity():
    rows = cluster_fixture()
    record = duplicate_attacks(rows[0])
    assert record["instrument_state"] == "valid"
    assert len(record["attacks"]) == 4 and all(attack["actual_rejection"] for attack in record["attacks"])
    admitted = {row["unit_id"]: digest(row) for row in rows}
    changed = deepcopy(rows)
    changed[0]["nested_episodes"] += 1
    with pytest.raises(ValueError, match="immutable source"):
        source_bound_sample(changed, admitted)


def test_dependence_preserves_independent_coincidences_and_paired_condition_groups():
    rows = cluster_fixture()
    assert rows[0]["arms"] == rows[1]["arms"]
    source_bound_sample(rows, {row["unit_id"]: digest(row) for row in rows})
    additional = deepcopy(rows[0])
    additional.update(unit_id="same-maker-new-condition", condition="paired-task-condition")
    result = sampling_groups(rows+[additional])
    assert result["retained_units"] == 65
    assert len(result["conditions"]) == 2
    assert sorted(row["constructors"] for row in result["conditions"]) == [1, 8]


def test_actual_hierarchical_reducer_exposes_nested_false_precision():
    result = hierarchical_control()
    assert result["instrument_state"] == "valid", result
    assert result["valid_summary"]["n_makers"] == 64
    assert result["intentionally_broken_summary"]["n"] == 2048


def test_dependence_retains_every_physically_produced_rejected_candidate():
    import random
    from ghostscale.validation.soundingline.v16.graphic_reference import interpret
    from ghostscale.validation.soundingline.v16.selection import retained
    candidates = []
    for style in [1, 1, 1, 0]:
        program = [0, 1, 2]+([4, 5] if style == 0 else [6, 7])
        candidates.append({"program": program, "execution": interpret(program), "decoration": style})
    assert all(item["execution"]["legal"] for item in candidates)
    batch = {"candidates": candidates, "retained_indices": retained(candidates, "selected", 0, random.Random(1))}
    actual = partition(batch, 4)
    assert (actual["produced"], actual["retained"], actual["rejected"]) == (4, 1, 3)
    assert sum(item["execution"]["primitive_cost"] for item in candidates) == 20
    erased = {"candidates": [candidates[3]], "retained_indices": [0]}
    with pytest.raises(ValueError, match="discarded"):
        partition(erased, 4)
    doubled = {**batch, "retained_indices": [3, 3]}
    with pytest.raises(ValueError, match="partition"):
        partition(doubled, 4)


def test_dependence_follows_separately_retained_original_production(tmp_path):
    from ghostscale.validation.soundingline.v16.dependence import unit_batch_inventory
    from ghostscale.validation.soundingline.v16.records import write
    batch = {"candidates": [{"artifact": 1}, {"artifact": 2}], "retained_index": 0}
    mapping = {"private/unit-original.json": write(tmp_path/"private/unit-original.json", {"batches": [batch]*3}),
               "private/unit.json": write(tmp_path/"private/unit.json", {"future": batch})}
    write(tmp_path/"RAW_MANIFEST.json", {"files": mapping})
    inventory = unit_batch_inventory(tmp_path, {"unit_id": "unit"})
    assert len(inventory) == 4 and sum(item["rejected"] for item in inventory) == 4
    assert sum(item["path"].startswith("private/unit-original.json") for item in inventory) == 3
    write(tmp_path/"private/unit-extra.json", {"unexpected": batch})
    with pytest.raises(ValueError, match="raw manifest"):
        unit_batch_inventory(tmp_path, {"unit_id": "unit"})

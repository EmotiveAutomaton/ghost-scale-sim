import pytest
from ghostscale.validation.soundingline.v16.records import write, read, file_digest
from ghostscale.validation.soundingline.v16.replay_coverage import allocation, phase


def test_global_replay_requires_each_phase_and_counts_descriptive_overlap_once():
    phases = {name: [{"source_unit": name+"/"+str(i)} for i in range(n)] for name,n in
              [("development", 68), ("constructor", 52), ("boundary", 34), ("confirmation", 4)]}
    result = allocation(phases, [phases["constructor"][0], {"source_unit": "descriptive/new"}])
    assert result["total_checks"] == 160 and result["distinct_units"] == 159
    with pytest.raises(ValueError, match="four declared"):
        allocation({k:v for k,v in phases.items() if k != "boundary"}, [])
    phases["boundary"][0] = phases["constructor"][0]
    with pytest.raises(ValueError, match="duplicate"):
        allocation(phases, [])


def test_phase_checks_full_source_and_replay_dependencies(tmp_path):
    root, output = tmp_path/"results/v16", tmp_path/"results/v16/checks"
    source, replay = root/"native/units/u_points.json", output/"private/case-000/units/u_points.json"
    for path in [source, replay]:
        write(path, {"known_scientific_score": 1})
    selected = {"source_unit": "native/units/u_points.json", "source_unit_sha256": file_digest(source)}
    check = {"world_reader_and_score_match": True, "selected_source": selected["source_unit"],
        "source_files": {"units/u_points.json": file_digest(source)}, "replay_files": {"units/u_points.json": file_digest(replay)}}
    write(output/"PLAN.json", {"selected": [selected], "n_units": 1, "replay_sources": {}})
    write(output/"checks/case-000.json", check)
    receipt = {"execution_state": "completed", "instrument_state": "valid", "selected_units_wholly_replayed": True,
        "plan_sha256": file_digest(output/"PLAN.json"), "n_units": 1, "checks": [check]}
    write(output/"RECEIPT.json", receipt)
    assert len(phase(root, tmp_path, output, [selected])) == 5
    with pytest.raises(ValueError, match="endpoint coverage"):
        phase(root, tmp_path, output, [])
    write(replay, {"known_scientific_score": 2}, immutable=False)
    with pytest.raises(ValueError, match="changed"):
        phase(root, tmp_path, output, [selected])

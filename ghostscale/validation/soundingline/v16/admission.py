"""Explicit test and attack joins; missing records never imply admission."""
from pathlib import Path
from .records import read, file_digest
from .runtime import REPO

CONSUMERS = {
    "X01": ["K01", "K02", "P01", "P02", "P03", "O01", "S01", "R01", "M01", "V01", "B01"],
    "X02": ["K01", "K02", "P01", "P04", "M01", "B01"],
    "X03": ["P01", "P02", "P03", "P04", "O01", "M04", "V03", "B01"],
    "X04": ["K01", "K02", "K03", "K04", "K05", "P03", "O02", "O03", "S02",
            "R01", "R02", "R03", "R04", "R05", "M03", "B02"],
    "X05": ["S04", "S05", "O04", "M01", "M03", "V03", "B01"],
    "X06": ["K04", "P04", "O04", "M04", "V03", "B01"],
    "X07": ["K01", "K02", "K05", "P03", "M01", "M02", "M04", "V01", "V02", "B02", "B03", "B04"],
    "X08": ["K01", "R01", "R02", "R03", "R04", "R05", "B02", "B03", "B04"],
}

K01_REQUIRED = {
    "artifact_marginalization": "test_known_answer_gates_detect_their_broken_instruments",
    "executable_acquisition": "test_known_answer_gates_detect_their_broken_instruments",
    "serialized_access": "test_reader_rejects_evaluator_fields_and_impossible_artifact",
    "craft-cost": "test_acquisition_cost_access_and_estimand_controls",
    "estimand-identity": "test_acquisition_cost_access_and_estimand_controls",
    "restart-reaggregation": "test_interrupted_packet_resumes_without_duplicates_or_changed_clocks",
    "X01": "test_acquisition_cost_access_and_estimand_controls",
    "X02": "test_acquisition_cost_access_and_estimand_controls",
    "X04": "test_acquisition_cost_access_and_estimand_controls",
    "X07": "test_duplicate_maker_cannot_inflate_the_analysis",
    "X08": "test_interrupted_packet_resumes_without_duplicates_or_changed_clocks",
}


def craft_admission(root: Path):
    path = root / "ADMISSION_CHECKS.json"
    if not path.exists():
        raise ValueError("missing executable admission receipt")
    receipt = read(path)
    for relative, expected in receipt["source_hashes"].items():
        if file_digest(REPO / relative) != expected:
            raise ValueError(f"tested admission source changed: {relative}")
    joined = {}
    for dependency, test in K01_REQUIRED.items():
        if receipt["tests"].get(test) != "passed":
            raise ValueError(f"unresolved K01 dependency {dependency}: {test}")
        joined[dependency] = {"instrument_state": "valid", "test": test,
                              "receipt": "ADMISSION_CHECKS.json"}
    return joined

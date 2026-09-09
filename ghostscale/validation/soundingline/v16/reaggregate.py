"""Standalone audit of retained native rows; imports no scientific or primary reducer."""
from __future__ import annotations
from pathlib import Path
import hashlib
import json
import math


def regenerate(root: Path):
    rows = [json.loads(path.read_bytes()) for path in sorted((root / "units").glob("*_points.json"))]
    if not rows:
        raise ValueError("empty raw archive cannot support a result")
    identities = set()
    totals = {"n": 0, "success_count": 0, "legal_count": 0, "future_log_score_sum": 0.0,
              "ambiguous_history_count": 0}
    for row in rows:
        if row["unit_id"] in identities:
            raise ValueError("duplicate independent unit")
        identities.add(row["unit_id"])
        for directory, hash_key in [("predictions", "prediction_hash"), ("private", "truth_hash")]:
            file = root / directory / (row["unit_id"] + ".json")
            if hashlib.sha256(file.read_bytes()).hexdigest() != row[hash_key]:
                raise ValueError("referenced raw evidence changed")
        probabilities = row["prediction"]["future_probabilities"]
        if abs(sum(probabilities) - 1) > 1e-10 or any(value < 0 for value in probabilities):
            raise ValueError("invalid prediction distribution")
        # Independently execute the submitted reconstruction; no primary interpreter.
        board = [False] * 4
        legal = True
        for index, primitive in enumerate(row["prediction"]["reconstruction"]["program"]):
            if index >= 3 or type(primitive) is not int or not 0 <= primitive < 8:
                legal = False
                break
            board[primitive % 4] = primitive < 4
        artifact = sum((2 ** index) for index, occupied in enumerate(board) if occupied)
        success = legal and artifact == row["public"]["final_artifact"]
        log_score = math.log(probabilities[row["truth"]["future_artifact"]])
        if success != row["outcomes"]["success"] or legal != row["outcomes"]["legal"]:
            raise ValueError("primary execution score disagrees")
        if abs(log_score - row["outcomes"]["future_log_score"]) > 1e-10:
            raise ValueError("primary prediction score disagrees")
        totals["n"] += 1
        totals["success_count"] += success
        totals["legal_count"] += legal
        totals["future_log_score_sum"] += log_score
        totals["ambiguous_history_count"] += len(row["truth"]["equivalence_classes"]) > 1
    return totals

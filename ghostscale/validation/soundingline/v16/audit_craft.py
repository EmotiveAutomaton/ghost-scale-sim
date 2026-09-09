"""Independent raw-to-interval audit. No primary science or reducer imports."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import random


def audit(root: Path, expected: dict):
    rows = [json.loads(path.read_bytes()) for path in sorted((root / "units").glob("*_points.json"))]
    if not rows or len({row["unit_id"] for row in rows}) != len(rows):
        raise ValueError("missing or duplicated raw maker rows")
    values = {}
    arm_values = {}
    for row in rows:
        for folder, hash_field in [("public", "observation_hash"), ("private", "truth_hash"),
                                   ("predictions", "prediction_hash")]:
            path = root / folder / (row["unit_id"] + ".json")
            if hashlib.sha256(path.read_bytes()).hexdigest() != row[hash_field]:
                raise ValueError("raw hash mismatch")
        scores = {}
        for arm, record in row["arms"].items():
            success, legal, search = [], [], []
            for target, submission in zip(row["public"]["targets"], record["submissions"]):
                board = 0
                valid = not submission["search_timeout"]
                for offset, action in enumerate(submission["program"]):
                    if offset > 2 or type(action) is not int or not 0 <= action < 8:
                        valid = False
                        break
                    mask = 1 << (action % 4)
                    board = board | mask if action < 4 else board & ~mask
                legal.append(valid)
                success.append(valid and board == target)
                spent = sum(item["cost"] for item in submission["attempted_programs"])
                if spent != submission["search_primitives"] or spent > row["public"]["search_primitive_budget"]:
                    raise ValueError("primitive cost budget disagreement")
                search.append(spent)
            measured = {"success": sum(success)/len(success), "legal": sum(legal)/len(legal),
                        "search_cost": sum(search)/len(search)}
            if measured != record["outcomes"]:
                raise ValueError("primary outcomes differ from independent interpreter")
            scores[arm] = measured["success"]
            arm_values.setdefault((row["condition"], arm), []).append(
                [measured["success"], measured["legal"], measured["search_cost"],
                 record["costs"]["training_primitives"], record["costs"]["library_definition"]])
        for rival in ["pooled", "primitive"]:
            values.setdefault((row["condition"], rival), {}).setdefault(row["constructor_id"], []).append(
                scores["personal"] - scores[rival])
    checked = 0
    for (condition, rival), groups in values.items():
        primary = next(item for item in expected["conditions"][condition]["contrasts"]
                       if item["estimand"]["rival"] == rival)
        flattened = [x for group in groups.values() for x in group]
        mean = sum(flattened)/len(flattened)
        # Reconstruct original unit order for the frozen resampling contract.
        ordered = {}
        for row in sorted((row for row in rows if row["condition"] == condition),
                          key=lambda row: row["seed_components"]["index"]):
            ordered.setdefault(row["constructor_id"], []).append(
                row["arms"]["personal"]["outcomes"]["success"] - row["arms"][rival]["outcomes"]["success"])
        block_values = list(ordered.values())
        rng = random.Random(primary["bootstrap"]["seed"])
        estimates = []
        for replicate in range(primary["bootstrap"]["replicates"]):
            total, count = 0.0, 0
            for block_number in range(len(block_values)):
                block = block_values[rng.randrange(len(block_values))]
                for within in range(len(block)):
                    total += block[rng.randrange(len(block))]
                    count += 1
            estimates.append(total/count)
        estimates.sort()
        interval = [estimates[int(p*(len(estimates)-1))] for p in [0.025, 0.975]]
        if abs(mean-primary["mean"]) > 1e-10 or interval != primary["interval_95"]:
            raise ValueError("paired mean or hierarchical interval failed regeneration")
        sd = math.sqrt(sum((x-mean)**2 for x in flattened)/max(1,len(flattened)-1))
        if abs(sd-primary["paired_standard_deviation"]) > 1e-10:
            raise ValueError("paired standard deviation failed regeneration")
        if primary["n_makers"] != len(flattened) or primary["n_constructors"] != len(groups):
            raise ValueError("denominator disagreement")
        bar = primary["estimand"]["practical_bar"]
        criterion = "held" if interval[0] >= bar else "failed" if interval[1] < bar else "inconclusive"
        if primary["criterion_state"] != criterion:
            raise ValueError("criterion state disagreement")
        checked += 1
    fields = ["success_mean", "legal_mean", "search_primitives_mean",
              "training_primitives_mean", "definition_cost_mean"]
    for (condition, arm), packets in arm_values.items():
        for column, field in enumerate(fields):
            if abs(sum(packet[column] for packet in packets)/len(packets) -
                   expected["conditions"][condition]["arms"][arm][field]) > 1e-10:
                raise ValueError("arm aggregate disagreement")
    if expected["n_maker_packets"] != len(rows):
        raise ValueError("wrong total raw denominator")
    return {"execution_state": "completed", "instrument_state": "valid", "n_raw_units": len(rows),
            "reproduced_contrasts": checked, "all_reported_aggregates_reproduced": True,
            "full_rollout_replay": False, "method": "separate interpreter and hierarchical reducer"}

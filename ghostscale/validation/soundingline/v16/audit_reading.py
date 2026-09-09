"""Independent reading-packet execution, log scoring and hierarchical intervals."""
import hashlib
import json
import math
import random
from pathlib import Path


def audit(root: Path, summary: dict):
    rows = [json.loads(path.read_bytes()) for path in (root/"units").glob("*_points.json")]
    if not rows or len({row["unit_id"] for row in rows}) != len(rows):
        raise ValueError("empty or duplicate maker archive")
    rows.sort(key=lambda row:(row["condition"],row["seed_components"]["index"]))
    checked = 0
    for row in rows:
        references = {}
        for folder,hash_key in [("public","observation_hash"),("private","truth_hash"),
                                ("predictions","prediction_hash")]:
            payload = (root/folder/(row["unit_id"]+".json")).read_bytes()
            if hashlib.sha256(payload).hexdigest() != row[hash_key]:
                raise ValueError("raw reference hash mismatch")
            references[folder] = json.loads(payload)
        if references["public"] != row["public"] or references["private"] != row["private"]:
            raise ValueError("embedded observation or truth differs from separate raw file")
        future = references["private"]["hidden_continuations"]["artifact"]
        for name,arm in row["arms"].items():
            prediction = references["predictions"]["arms"][name]
            if prediction["future_probabilities"] != arm["future_probabilities"]:
                raise ValueError("scored prediction differs from pre-reveal submission")
            if prediction["reconstruction"] != arm["reconstruction"]:
                raise ValueError("scored program differs from submitted program")
            board = [0,0,0,0]
            legal = not prediction["reconstruction"]["search_timeout"]
            for step,action in enumerate(prediction["reconstruction"]["program"]):
                if step >= 3 or type(action) is not int or action not in range(8):
                    legal = False
                    break
                board[action%4] = int(action < 4)
            artifact = sum(value*(2**position) for position,value in enumerate(board))
            outcomes = {"future_log_score":math.log(prediction["future_probabilities"][future]),
                        "legal":float(legal),"success":float(legal and artifact == references["public"]["final_artifact"])}
            if outcomes != arm["outcomes"]:
                raise ValueError("independent reading score differs")
    for condition,reported in summary["conditions"].items():
        sample = [row for row in rows if row["condition"] == condition]
        if not sample:
            raise ValueError("missing condition")
        for name,metrics in reported["arms"].items():
            for field,value in metrics.items():
                actual = sum(row["arms"][name]["outcomes"][field] for row in sample)/len(sample)
                if abs(actual-value) > 1e-10:
                    raise ValueError("arm aggregate disagreement")
        ambiguity = sum(len(row["private"]["equivalence_classes"]) for row in sample)/len(sample)
        if ambiguity != reported["mean_route_collision_class_size"]:
            raise ValueError("ambiguity aggregate disagreement")
        for contrast in reported["contrasts"]:
            spec = contrast["estimand"]
            if spec["arm"] == spec["rival"]:
                raise ValueError("same-arm contrast")
            groups = {}
            for row in sample:
                difference = row["arms"][spec["arm"]]["outcomes"][spec["target"]] - row["arms"][spec["rival"]]["outcomes"][spec["target"]]
                groups.setdefault(row["constructor_id"],[]).append(difference)
            values = [value for group in groups.values() for value in group]
            mean = sum(values)/len(values)
            rng = random.Random(contrast["bootstrap"]["seed"])
            blocks = list(groups.values())
            estimates = []
            for iteration in range(contrast["bootstrap"]["replicates"]):
                total,count = 0.0,0
                for block_index in range(len(blocks)):
                    block = blocks[rng.randrange(len(blocks))]
                    for member in range(len(block)):
                        total += block[rng.randrange(len(block))]
                        count += 1
                estimates.append(total/count)
            estimates.sort()
            interval = [estimates[int(probability*(len(estimates)-1))] for probability in [0.025,0.975]]
            if abs(mean-contrast["mean"]) > 1e-10 or max(abs(a-b) for a,b in zip(interval,contrast["interval_95"])) > 1e-10:
                raise ValueError("independent hierarchical interval or mean disagreement")
            sd = math.sqrt(sum((value-mean)**2 for value in values)/max(1,len(values)-1))
            if abs(sd-contrast["paired_standard_deviation"]) > 1e-10:
                raise ValueError("standard deviation mismatch")
            if contrast["n_makers"] != len(values) or contrast["n_constructors"] != len(groups):
                raise ValueError("denominator mismatch")
            criterion = "held" if interval[0] >= spec["practical_bar"] else (
                "failed" if interval[1] < spec["practical_bar"] else "inconclusive")
            if criterion != contrast["criterion_state"]:
                raise ValueError("criterion mismatch")
            checked += 1
    if len(rows) != summary["n_maker_packets"]:
        raise ValueError("total denominator mismatch")
    return {"execution_state":"completed","instrument_state":"valid",
            "n_raw_units":len(rows),"reproduced_contrasts":checked,
            "all_reported_aggregates_reproduced":True,"full_rollout_replay":False}

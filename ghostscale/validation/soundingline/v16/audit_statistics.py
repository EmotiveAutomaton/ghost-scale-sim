"""Independent raw-outcome-to-summary audit. No imports from the primary reducer."""
import math
import random


def verify(rows,summary):
    if not rows or len({row["unit_id"] for row in rows}) != len(rows):
        raise ValueError("empty or duplicate independent maker rows")
    checked = 0
    for condition,reported in summary["conditions"].items():
        sample = sorted((row for row in rows if row["condition"]==condition),
                        key=lambda row:row["seed_components"]["index"])
        if not sample:
            raise ValueError("missing registered condition")
        for name,metrics in reported["arms"].items():
            for metric,value in metrics.items():
                average = sum(row["arms"][name]["outcomes"][metric] for row in sample)/len(sample)
                if abs(average-value)>1e-10:
                    raise ValueError(f"arm aggregate mismatch: {condition}/{name}/{metric}")
        for contrast in reported["contrasts"]:
            spec = contrast["estimand"]
            if spec["arm"]==spec["rival"]:
                raise ValueError("same-arm contrast")
            groups = {}
            for row in sample:
                difference = row["arms"][spec["arm"]]["outcomes"][spec["target"]] - row["arms"][spec["rival"]]["outcomes"][spec["target"]]
                groups.setdefault(row["constructor_id"],[]).append(difference)
            values = [value for block in groups.values() for value in block]
            mean = sum(values)/len(values)
            rng = random.Random(contrast["bootstrap"]["seed"])
            blocks = list(groups.values())
            estimates = []
            for iteration in range(contrast["bootstrap"]["replicates"]):
                total,count = 0.0,0
                for block_id in range(len(blocks)):
                    block = blocks[rng.randrange(len(blocks))]
                    for member in range(len(block)):
                        total += block[rng.randrange(len(block))]
                        count += 1
                estimates.append(total/count)
            estimates.sort()
            interval = [estimates[int(p*(len(estimates)-1))] for p in [0.025,0.975]]
            if abs(mean-contrast["mean"])>1e-10 or max(abs(a-b) for a,b in zip(interval,contrast["interval_95"]))>1e-10:
                raise ValueError("independent paired mean or hierarchical interval mismatch")
            sd = math.sqrt(sum((value-mean)**2 for value in values)/max(1,len(values)-1))
            if abs(sd-contrast["paired_standard_deviation"])>1e-10:
                raise ValueError("paired standard deviation mismatch")
            if contrast["n_makers"]!=len(values) or contrast["n_constructors"]!=len(groups):
                raise ValueError("sampling denominator mismatch")
            state = "held" if interval[0]>=spec["practical_bar"] else (
                "failed" if interval[1]<spec["practical_bar"] else "inconclusive")
            if state!=contrast["criterion_state"]:
                raise ValueError("criterion state mismatch")
            checked += 1
    if summary["n_maker_packets"]!=len(rows):
        raise ValueError("total unit denominator mismatch")
    return checked

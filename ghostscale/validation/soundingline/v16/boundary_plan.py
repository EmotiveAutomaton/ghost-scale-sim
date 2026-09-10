"""One finite final expansion for named uncertainty in unchanged contrasts."""
from copy import deepcopy
import math
from .records import read, write, file_digest, now


def uncertain_contrasts(entry, summary):
    design = entry["design"]
    if set(summary["conditions"]) != {row["id"] for row in design["conditions"]}:
        raise ValueError("boundary selection requires every original condition")
    selected = []
    for condition, result in sorted(summary["conditions"].items()):
        for contrast in result["contrasts"]:
            estimand = contrast["estimand"]
            if "primary" in design and tuple(estimand[key] for key in ["arm", "rival", "target", "units", "practical_bar"]) not in {tuple(item) for item in design["primary"]}:
                continue
            if "primary_estimands" in design and estimand["id"] not in design["primary_estimands"]:
                continue
            low, high = contrast["interval_95"]
            bar = estimand["practical_bar"]
            if not low <= high or not all(math.isfinite(value) for value in [low,high,contrast["mean"],bar]) or bar <= 0:
                raise ValueError("boundary selection requires a valid finite interval and practical bar")
            boundaries = [point for point in [-bar, bar] if low < point <= high]
            if low < 0 < high and high-low >= bar:
                boundaries.append(0)
            if boundaries:
                selected.append({"condition": condition, "estimand": estimand,
                    "mean": contrast["mean"], "interval_95": contrast["interval_95"],
                    "unresolved_boundaries": sorted(boundaries), "criterion_state": contrast["criterion_state"]})
    return selected


def choose(entries, summaries, closed):
    cards = []
    dispositions = []
    for original in entries:
        card = original["card_id"]
        uncertainty = uncertain_contrasts(original, summaries[card])
        if uncertainty:
            entry = deepcopy(original)
            entry.update(n_per_condition=1024, constructors=128, namespace="v16-boundary-expansion-1-"+card,
                         reason="Final fresh constructor allocation for retained uncertainty in the named unchanged contrasts",
                         uncertainty=uncertainty)
            cards.append(entry)
        dispositions.append({"card_id": card, "state": "one final expansion selected" if uncertainty else "exhausted at 256",
            "reason": "Named intervals straddle a positive/negative practical boundary, or zero with width at least the practical bar" if uncertainty else
                      "No registered primary contrast retains uncertainty under the stated finite boundary rule",
            "uncertain_contrasts": uncertainty, "new_n_per_condition": 1024 if uncertainty else 0})
    dispositions.extend({"card_id":row["card_id"], "state":"exhausted at scout", "reason":row["reason"], "new_n_per_condition":0} for row in closed)
    if len(dispositions) != 30 or len({row["card_id"] for row in dispositions}) != 30:
        raise ValueError("every native card needs one finite ladder disposition")
    return {"schema_version":"v16.final-boundary-plan.1", "cards":cards,
        "dispositions":sorted(dispositions,key=lambda row:row["card_id"]),
        "planned_maker_condition_records":sum(row["n_per_condition"]*len(row["design"]["conditions"]) for row in cards),
        "selection_rule":"All primary contrasts; positive/negative practical boundary crossing, or zero crossing with interval width at least the practical bar. Secondary contrasts never trigger expansion.",
        "sampling":"128 constructor redraws, eight independent acquired histories per constructor and condition; paired contexts are not independent makers",
        "condition_rule":"Retain every original condition of a selected card, including losses and nulls; unchanged arms, recipes, targets and bars",
        "maximum_further_expansions_after_this":0, "evidence_scope":"discovery final boundary expansion",
        "confirmation":"This is exploration, not confirmation; no final boundary can trigger a third expansion"}


def freeze_selection(root):
    path = root/"expansion-closure/SELECTION.json"
    if path.exists():
        saved = read(path)
        for name, expected in saved["source_hashes"].items():
            if file_digest(root/name) != expected:
                raise ValueError("boundary selection source changed")
        return saved
    source = root/"constructor-expansion-1"
    control_path = root/"constructor-controls-1/COMPLETION.json"
    completed, controls = read(source/"COMPLETION.json"), read(control_path)
    lock_path = root/"packets/constructor-expansion-1.json"
    lock = read(lock_path)
    spec = lock["identity"]["design"]
    if any(row.get("instrument_state") != "valid" or row.get("execution_state") != "completed" for row in [completed,controls]):
        raise ValueError("boundary selection requires completed source science and actual source controls")
    if controls["source_packet_hash"] != lock["packet_hash"] or controls["condition_controls"] != sum(len(row["design"]["conditions"]) for row in spec["cards"]):
        raise ValueError("boundary source-control coverage differs")
    paths = [source/"COMPLETION.json",control_path,lock_path]
    summaries = {}
    for entry in spec["cards"]:
        summary_path = source/entry["card_id"]/"AGGREGATE.json"
        receipt = read(source/entry["card_id"]/"COMPLETION.json")
        if file_digest(summary_path) != receipt["aggregate_sha256"]:
            raise ValueError("source boundary summary changed")
        summaries[entry["card_id"]] = read(summary_path)
        paths.extend([summary_path,source/entry["card_id"]/"COMPLETION.json"])
    result = choose(spec["cards"],summaries,spec["closed_at_scout"])
    result.update(recorded_at=now(),source_hashes={path.relative_to(root).as_posix():file_digest(path) for path in paths})
    write(path,result)
    return result

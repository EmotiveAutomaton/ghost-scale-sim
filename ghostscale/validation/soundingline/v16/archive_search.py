"""Finite fixed and adaptive schedules; semantic novelty never changes targets."""
import random
from .archive_design import CONDITIONS, CASE_IDS
from .records import seed_for


def select(method, replicate, history):
    step = len(history)
    if method == "fixed":
        return CONDITIONS[(step+replicate) % len(CONDITIONS)]
    if method != "adaptive":
        raise ValueError("unknown archive policy")
    if step < 3:
        return CONDITIONS[4*step]
    counts = {row["id"]: sum(item["condition"] == row["id"] for item in history) for row in CONDITIONS}
    if step % 4 == 3:
        candidates = CONDITIONS
    else:
        # The best prior novelty event remains the anchor until another event
        # improves it. An edit changes exactly one registered discrete axis.
        best = max(history, key=lambda row: (len(row["new_cases"]), -row["step"]))
        anchor = next(row for row in CONDITIONS if row["id"] == best["condition"])
        candidates = [row for row in CONDITIONS if sum(row[key] != anchor[key] for key in ["retention", "process", "rejected"]) == 1]
    least = min(counts[row["id"]] for row in candidates)
    candidates = [row for row in candidates if counts[row["id"]] == least]
    return random.Random(seed_for("v16-archive-policy-1", replicate, step)).choice(candidates)


def novelty(cases, seen):
    if set(cases) != set(CASE_IDS) or any(type(value) is not bool for value in cases.values()):
        raise ValueError("unregistered archive semantic target")
    return sorted(key for key, present in cases.items() if present and key not in seen)


def summarize(search, followup, required_recurrences):
    frequencies = {condition["id"]: {key: 0 for key in CASE_IDS} for condition in CONDITIONS}
    denominators = {condition["id"]: 0 for condition in CONDITIONS}
    for record in followup:
        condition = record["condition"]
        denominators[condition] += 1
        for key, present in record["cases"].items():
            frequencies[condition][key] += int(present)
    methods = {}
    for method in ["fixed", "adaptive"]:
        replicates = []
        for replicate in sorted({row["replicate"] for row in search if row["method"] == method}):
            rows = [row for row in search if row["method"] == method and row["replicate"] == replicate]
            witnesses = {}
            # Preserve the first candidate for each semantic identity; no swapping
            # to a successful witness after opening the follow-up set.
            for row in sorted(rows, key=lambda item: item["step"]):
                for key in row["new_cases"]:
                    if key in witnesses:
                        raise ValueError("semantic novelty counted twice")
                    condition = row["condition"]
                    witnesses[key] = {"candidate": row["candidate_receipt"], "condition": condition,
                        "followup_recurrences": frequencies[condition][key], "followup_makers": denominators[condition],
                        "validated": frequencies[condition][key] >= required_recurrences}
            replicates.append({"replicate": replicate, "candidates": len(rows),
                "physical_primitives": sum(row["physical_primitives"] for row in rows),
                "separate_verification_primitives": sum(row["separate_verification_primitives"] for row in rows),
                "distinct_provisional_cases": len(witnesses), "distinct_validated_cases": sum(item["validated"] for item in witnesses.values()),
                "witnesses": witnesses})
        methods[method] = {"replicates": replicates,
            "mean_distinct_validated_cases": sum(row["distinct_validated_cases"] for row in replicates)/len(replicates),
            "total_physical_primitives": sum(row["physical_primitives"] for row in replicates)}
    return {"methods": methods, "followup_frequencies": frequencies, "followup_denominators": denominators,
            "inference": "descriptive finite search comparison; recurrence filter and multiple case targets are not population confirmations"}

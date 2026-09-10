"""One declared primary analysis per frozen constructor/task packet."""
from .records import digest
from .confirmation_bounded import bounded_mean
from .confirmation_math import bounded_equivalence


def grouped_primary(claim, rows):
    n = claim["n_constructor_packets"]
    conditions = claim["condition_ids"]
    if not 2 <= n <= 4096 or len(conditions) != len(set(conditions)) or not conditions:
        raise ValueError("invalid frozen confirmation allocation")
    if n*claim["histories_per_constructor_packet"] > 4096:
        raise ValueError("confirmation exceeds its independent acquisition-history cap")
    if claim["total_fresh_acquisition_histories"] != n*claim["histories_per_constructor_packet"]:
        raise ValueError("confirmation primary-history denominator differs from its allocation")
    if not claim.get("namespace") or not claim.get("execution_packet_hash"):
        raise ValueError("confirmation requires its frozen fresh lineage and execution packet")
    if len(rows) != n*len(conditions) or len({row["unit_id"] for row in rows}) != len(rows):
        raise ValueError("confirmation omitted or duplicated a native condition record")
    groups = {}
    for row in rows:
        if row["lineage"] != claim["namespace"] or row["packet_hash"] != claim["execution_packet_hash"]:
            raise ValueError("confirmation data belong to another lineage or source packet")
        index = row["seed_components"]["index"]
        if row["card_id"] != claim["card_id"] or row["condition"] not in conditions or not 0 <= index < n:
            raise ValueError("confirmation native identity differs from its frozen claim")
        if row["seed_components"]["constructors"] != n:
            raise ValueError("confirmation reused constructors across independent packets")
        group = groups.setdefault(index, {})
        if row["condition"] in group:
            raise ValueError("confirmation duplicated a paired context")
        group[row["condition"]] = row
    packets = []
    for index in range(n):
        group = groups.get(index, {})
        if set(group) != set(conditions) or len({row["constructor_id"] for row in group.values()}) != 1:
            raise ValueError("confirmation contexts do not form their declared constructor cluster")
        differences = []
        for condition in conditions:
            row = group[condition]
            for spec in claim["estimands"]:
                differences.append(row["arms"][spec["arm"]]["outcomes"][spec["target"]]
                    - row["arms"][spec["rival"]]["outcomes"][spec["target"]])
        packets.append({"unit_id": digest([claim["card_id"], index, sorted(row["unit_id"] for row in group.values())]),
            "constructor_id": next(iter(group.values()))["constructor_id"], "differences": differences})
    if len({row["constructor_id"] for row in packets}) != n:
        raise ValueError("independent constructor labels repeat between claim packets")
    return packets


def analyze_primary(claim, rows):
    packets = grouped_primary(claim, rows)
    if claim["kind"] == "equivalence":
        if claim["card_id"] != "S02" or len(claim["condition_ids"]) != 2 or len(claim["estimands"]) != 2:
            raise ValueError("joint equivalence requires its declared two-context two-rival S02 packet")
        result = bounded_equivalence(packets, outcome_difference_bound=claim["external_difference_bound"],
            margin=claim["margin"], alpha=claim["alpha_planning"])
        p = result["exact_one_sided_p"]
        result["primary_estimand"] = "All four absolute paired mean repair differences are smaller than the declared margin"
        result["grouping_qualification"] = claim["grouping"]
    elif claim["kind"] == "capability":
        if len(claim["condition_ids"]) != 1 or len(claim["estimands"]) != 1:
            raise ValueError("capability requires one frozen condition and one primary comparison")
        rows_for_mean = [{"unit_id": row["unit_id"], "constructor_id": row["constructor_id"], "difference": row["differences"][0]} for row in packets]
        result = bounded_mean(rows_for_mean, null_mean=claim["estimands"][0]["practical_bar"], alpha=claim["alpha_planning"])
        p = result["valid_one_sided_p"]
        result["primary_estimand"] = claim["estimands"][0]
    else:
        raise ValueError("unknown frozen primary confirmation method")
    return {"instrument_state": "valid", "card_id": claim["card_id"], "kind": claim["kind"],
        "n_independent_constructor_packets": len(packets), "n_native_condition_records": len(rows),
        "total_fresh_acquisition_histories": claim["total_fresh_acquisition_histories"],
        "history_count_scope": "Primary maker histories in the declared constructor packet; auxiliary comparator training remains separately recorded and costed in raw units",
        "primary_p": p, "primary_analysis": result,
        "multiplicity_state": "Await the single frozen familywise ledger; this record does not add another claim",
        "source_regime_selected_through_discovery": True}

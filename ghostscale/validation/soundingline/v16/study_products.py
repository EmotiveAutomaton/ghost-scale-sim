"""Lossless report projections of completed discovery and frozen confirmation.

No scientific reducer is executed and no condition is pooled. These products
index existing estimates; independent numerical proofs remain separate inputs.
"""
from copy import deepcopy
import csv
import json
from pathlib import Path

from .records import read, write, file_digest, now
from .completion_guard import bound_receipt
from .expansion_adapter import CARDS, design

CAPABILITIES = {
    "executable construction": ["K01", "K03", "K04", "K05", "P01", "P04"],
    "historical correspondence and ambiguity": ["P01", "P02", "P03", "O01", "O03", "O04", "M01", "M02", "M04", "V01", "V02", "V03"],
    "future prediction and changed opportunity": ["K02", "P02", "P03", "P04", "O01", "O02", "O03", "O04", "M01", "M02", "M03", "M04"],
    "self-correction and resumption": ["S01", "S02", "S03", "S04", "S05"],
    "inquiry and subsequent competence": ["R01", "R02", "R03", "R04", "R05", "M03"],
}


def project_card(disposition, definition, summary):
    card = disposition["card_id"]
    if disposition.get("execution_state") != "completed" or disposition.get("instrument_state") != "valid":
        raise ValueError("study products require completed valid final sources")
    conditions = definition["conditions"]
    ids = [row["id"] for row in conditions]
    n = disposition["n_per_condition"]
    if (summary["card_id"] != card or len(ids) != len(set(ids)) or
            set(summary["conditions"]) != set(ids) or summary["n_maker_packets"] != n*len(ids)):
        raise ValueError("study projection changed the registered condition or maker inventory")
    comparisons, performances = [], []
    counts = {state: 0 for state in ["held", "failed", "inconclusive"]}
    access = {key: deepcopy(value) for key, value in definition.items()
              if "access" in key or "information" in key}
    for condition in conditions:
        values = summary["conditions"][condition["id"]]
        seen = set()
        for contrast in values["contrasts"]:
            estimand = contrast["estimand"]
            mandatory = {"id", "arm", "rival", "target", "units", "practical_bar", "numerator", "denominator", "aggregation", "interval_target"}
            if not mandatory <= set(estimand) or not contrast.get("bootstrap"):
                raise ValueError("study contrast lacks its estimand or uncertainty contract")
            identity = estimand["id"]
            if identity in seen or contrast["n_makers"] != n or contrast["criterion_state"] not in counts:
                raise ValueError("study contrast duplicated, changed its denominator or lost its criterion")
            seen.add(identity)
            counts[contrast["criterion_state"]] += 1
            comparisons.append({"card_id": card, "condition": condition["id"],
                "source": disposition["final_source"], "condition_factors": deepcopy(condition),
                "access_contract": access, **deepcopy(contrast)})
        for arm, metrics in values["arms"].items():
            performances.append({"card_id": card, "condition": condition["id"], "arm": arm,
                "source": disposition["final_source"], "condition_factors": deepcopy(condition),
                "n_makers": n, "access_contract": access, "recorded_metrics": deepcopy(metrics),
                "cost_scope": "Original metric-specific accounting; same-condition comparisons only. Physical invocation and OS cost supplements remain in source-control receipts."})
    if counts != disposition["criterion_counts"]:
        raise ValueError("study criterion accounting differs from the final ladder")
    return {"card_id": card, "execution_state": "completed", "instrument_state": "valid",
        "question": definition.get("question", definition.get("mechanism", card)),
        "final_source": disposition["final_source"], "n_per_condition": n,
        "condition_records": summary["n_maker_packets"], "criterion_counts": counts,
        "criterion_scope": "Each registered condition and comparator separately; a failed positive bar is not an equivalence test",
        "definition": deepcopy(definition), "source_qualifications": {key: deepcopy(value) for key, value in summary.items()
            if key not in {"conditions", "card_id", "n_maker_packets"}},
        "expansion_disposition": deepcopy(disposition), "comparisons": comparisons,
        "performance_cost": performances, "confirmed_claims": [],
        "confirmation_scope": "Only the explicitly joined frozen primary claim; no whole-card promotion"}


def confirmation_join(cards, selected, multiplicity, primaries):
    by_card = {row["card_id"]: row for row in cards}
    by_claim = {row["claim_id"]: row for row in selected}
    outcomes = multiplicity["claims"]
    if (len(by_claim) != len(selected) or len(selected) > 3 or
            len({row["card_id"] for row in selected}) != len(selected) or
            multiplicity["frozen_primary_count"] != len(selected) or
            len(outcomes) != len(selected) or {row["claim_id"] for row in outcomes} != set(by_claim) or
            set(primaries) != set(by_claim) or multiplicity["familywise_alpha"] != .05 or
            multiplicity["no_failed_claim_replaced"] is not True):
        raise ValueError("study confirmation join changed the frozen claim family")
    for outcome in outcomes:
        claim = by_claim[outcome["claim_id"]]
        primary = primaries[outcome["claim_id"]]
        if (claim["card_id"] not in by_card or primary["card_id"] != claim["card_id"] or
                outcome["card_id"] != claim["card_id"] or primary["primary_p"] != outcome["p"] or
                primary["n_independent_constructor_packets"] != claim["n_constructor_packets"] or
                primary["total_fresh_acquisition_histories"] != claim["total_fresh_acquisition_histories"]):
            raise ValueError("study confirmation identity, p-value or denominator changed")
        by_card[claim["card_id"]]["confirmed_claims"].append({"claim": deepcopy(claim),
            "multiplicity_outcome": deepcopy(outcome), "primary": deepcopy(primary),
            "scope": "This original condition/estimand only; confirmation failures are retained equally"})


def csv_file(path, rows, *, common_fields=()):
    common = {key: rows[0][key] for key in common_fields}
    if any(row[key] != value for row in rows for key, value in common.items()):
        raise ValueError("a shared table note would erase distinct row-level accounting")
    fields = sorted({key for row in rows for key in row}-set(common))
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True, separators=(",", ":"))
                if isinstance(value, (dict, list)) else value for key, value in row.items() if key not in common})
    return common


def run(root, output):
    if output.exists():
        raise ValueError("study products require a new retained output directory")
    sources = {}
    def load(name):
        path = root/name
        sources[name] = file_digest(path)
        return bound_receipt(root, name, sources[name])
    closed = load("expansion-closure/COMPLETION.json")
    if closed.get("execution_state") != "completed" or closed.get("instrument_state") != "valid" or not closed["all_thirty_dispositions_explicit"]:
        raise ValueError("study products await the completed finite discovery ladder")
    if len(closed["cards"]) != len(CARDS) or {row["card_id"] for row in closed["cards"]} != set(CARDS):
        raise ValueError("study products omitted or duplicated a native card")
    for name, expected in closed["source_hashes"].items():
        bound_receipt(root, name, expected)
        sources[name] = expected
    cards = []
    for disposition in sorted(closed["cards"], key=lambda row: row["card_id"]):
        base = root/disposition["final_source"]
        summary = base/("AGGREGATE.json" if (base/"AGGREGATE.json").exists() else "SUMMARY.json")
        name = summary.relative_to(root).as_posix()
        if name not in sources:
            raise ValueError("final ladder did not bind the report's scientific source")
        cards.append(project_card(disposition, design(disposition["card_id"]), load(name)))
    selection = load("confirmation-selection/COMPLETION.json")
    claims = load("confirmation-selection/CLAIMS.json")
    completed = load("confirmation/COMPLETION.json")
    multiplicity = load("confirmation/MULTIPLICITY.json")
    if any(row.get("execution_state") != "completed" or row.get("instrument_state") != "valid" for row in [selection, completed]):
        raise ValueError("study products await completed frozen confirmation, including empty or failed claims")
    if selection["claims_sha256"] != sources["confirmation-selection/CLAIMS.json"] or completed["claims"] != multiplicity["claims"]:
        raise ValueError("study confirmation source changed")
    primaries = {}
    for outcome in multiplicity["claims"]:
        name = "confirmation/"+outcome["card_id"]+"/PRIMARY.json"
        primaries[outcome["claim_id"]] = load(name)
        if sources[name] != outcome["primary_sha256"]:
            raise ValueError("study primary bytes changed after multiplicity was recorded")
    confirmation_join(cards, claims["selected"], multiplicity, primaries)
    output.mkdir(parents=True)
    write(output/"NATIVE_CARDS.json", {"cards": cards, "population_pooling": False,
        "scope": "Final discovery source per card; original failed lineages remain in the full archive and numerical audit"})
    write(output/"CAPABILITY_MAP.json", {"capabilities": CAPABILITIES,
        "interpretation": "Question and measurement index, not a claim that every indexed capability improved",
        "evidence": "NATIVE_CARDS.json supplies actual regimes, outcomes, access limits and primary confirmations"})
    write(output/"CONFIRMATION.json", {"selection": claims, "multiplicity": multiplicity,
        "primaries": primaries, "scope": "Frozen discovery-selected regimes on fresh constructor histories"})
    csv_file(output/"COMPARISONS_summary.csv", [row for card in cards for row in card["comparisons"]])
    cost_notes = csv_file(output/"PERFORMANCE_COST_summary.csv", [row for card in cards for row in card["performance_cost"]],
                          common_fields=("cost_scope",))
    write(output/"COVERAGE.json", {"cards": [{"card_id": row["card_id"], "definition": row["definition"],
        "source": row["final_source"], "source_qualifications": row["source_qualifications"],
        "n_per_condition": row["n_per_condition"], "expansion": row["expansion_disposition"]} for row in cards],
        "unsearched_regions": ["Continuous or open-ended program grammars", "Human or language-model readers",
            "Fitted-parameter transfer from the graphic world to assembly", "Unbounded external search or additional discovery expansions"],
        "conditional_scopes": ["Supplied-state resource benchmarks do not establish learned inference",
            "Recognition, inquiry choice and later competence are separate recorded targets",
            "Deterministic artifact/history probe bounds do not compare arbitrary information targets",
            "Observed absence of a practical advantage is not automatically equivalence"]})
    text = ["# Final V16 numerical index", "", "Generated from completed frozen records. Each row below is one card; conditions and comparators remain separate in the linked tables.", "",
        "| Card | Conditions | Makers per condition | Held comparisons | Failed positive criteria | Unresolved comparisons | Frozen confirmation claims |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for card in cards:
        counts = card["criterion_counts"]
        text.append(f"| {card['card_id']} | {len(card['definition']['conditions'])} | {card['n_per_condition']} | {counts['held']} | {counts['failed']} | {counts['inconclusive']} | {len(card['confirmed_claims'])} |")
    text += ["", "A failed positive criterion is not a scientific null or an equivalence result. Confirmation applies only to each frozen primary claim. No counts in this index are pooled independent sample sizes.", "",
        "Shared accounting note for every performance/cost CSV row: "+cost_notes["cost_scope"], "",
        "The complete per-row metadata, including this shared note, remains in NATIVE_CARDS.json. The CSV retains every condition, access contract, arm, denominator and recorded metric.", "",
        "[Native records and claim joins](NATIVE_CARDS.json), [capability index](CAPABILITY_MAP.json), [coverage and limits](COVERAGE.json), [complete comparisons](COMPARISONS_summary.csv), [performance and recorded costs](PERFORMANCE_COST_summary.csv), [confirmation](CONFIRMATION.json).", ""]
    (output/"README.md").write_text("\n".join(text), encoding="utf-8", newline="\n")
    result = {"execution_state": "completed", "instrument_state": "valid", "recorded_at": now(),
        "native_cards": len(cards), "registered_conditions": sum(len(row["definition"]["conditions"]) for row in cards),
        "registered_comparisons": sum(len(row["comparisons"]) for row in cards),
        "frozen_confirmation_claims": len(claims["selected"]), "input_hashes": sources,
        "output_hashes": {path.name: file_digest(path) for path in output.iterdir() if path.is_file()},
        "producer_sha256": file_digest(Path(__file__)), "full_aggregate_regeneration": False,
        "campaign_complete": False, "scope": "Lossless report projection; independent numerical and documentary proofs are separate"}
    write(output/"RECEIPT.json", result)
    return result

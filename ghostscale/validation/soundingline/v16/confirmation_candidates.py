"""Transparent finite candidate ranking before any confirmation data exist.

This module plans only. Actual selection must join the completed final ladder,
its source controls and admitted confirmation runner before freezing a packet.
"""
import math
from .confirmation_bounded import plan_power
from .confirmation_math import equivalence_power

BOUNDED_FRACTIONS = {"success", "original_goal_success", "adopted_goal_success",
    "original_goal_recovery", "identity_accuracy", "future_success"}


def priority(card, spec, *, kind="capability", mean=0., cost=0.):
    rival = spec["rival"]
    if rival in {"direct-table", "direct-mixture", "bayes-error", "direct-error"}:
        seriousness = 0
        rival_reason = "A complete same-evidence predictive or ordinary-error rival"
    elif rival in {"pooled", "generic", "matched-generic", "generic-options", "purpose-matched-generic"}:
        seriousness = 1
        rival_reason = "A trained generic construction or production rival"
    elif rival in {"primitive", "surface-only"}:
        seriousness = 2
        rival_reason = "A primitive construction or surface-only account"
    else:
        seriousness = 3
        rival_reason = "A reduced-evidence, no-reference, or weaker baseline"
    # Relevance refers to the commissioned question, not the observed estimate.
    relevance = 0 if card == "S02" else 1 if card.startswith(("S", "M", "O")) else 2
    scale = 1 if kind == "equivalence" else abs(mean)/spec["practical_bar"]
    return [0, seriousness, relevance, -scale, cost, card, spec["id"]], {
        "validity_and_target": "Requires completed valid native source and realized bounded target before entry",
        "serious_rival": rival_reason,
        "relevance": "Self-reconstruction versus complete error monitoring" if card == "S02" else
            "Intent, maker or considered-option inference" if relevance == 1 else "Acquired construction capability",
        "last_tiebreakers": "Practical size then recorded cost, followed by stable identifiers"}


def enumerate_candidates(entries):
    """Entries are already independently checked completed source summaries."""
    candidates, exclusions = [], []
    for entry in entries:
        card, design, summary = entry["card_id"], entry["design"], entry["summary"]
        if entry["constructors"] < 20:
            exclusions.append({"card_id": card, "reason": "Discovery remained below the miniature constructor coverage floor"})
            continue
        if design.get("family") == "critic":
            exclusions.append({"card_id": card, "reason": "Supplied-state resource benchmark does not realize learned historical inference"})
            continue
        for condition, values in sorted(summary["conditions"].items()):
            for contrast in values["contrasts"]:
                spec = contrast["estimand"]
                if "primary" in design and tuple(spec[k] for k in ["arm", "rival", "target", "units", "practical_bar"]) not in {tuple(row) for row in design["primary"]}:
                    continue
                if "primary_estimands" in design and spec["id"] not in design["primary_estimands"]:
                    continue
                reason = None
                if spec["target"] not in BOUNDED_FRACTIONS:
                    reason = "This bounded-mean confirmation instrument does not establish an external range for the registered target"
                elif contrast["criterion_state"] != "held":
                    reason = "The final discovery interval does not establish the named practical boundary"
                if reason:
                    exclusions.append({"card_id": card, "condition": condition, "estimand": spec["id"], "reason": reason})
                    continue
                order, rationale = priority(card, spec, mean=contrast["mean"], cost=entry["seconds_per_condition_record"])
                candidates.append({"card_id": card, "kind": "capability", "condition_ids": [condition],
                    "estimands": [spec], "source_base": entry["source_base"], "source_sha256": entry["source_sha256"],
                    "discovery_mean": contrast["mean"], "discovery_variance": contrast["paired_standard_deviation"]**2,
                    "discovery_makers": contrast["n_makers"], "discovery_constructors": contrast["n_constructors"],
                    "rank": order, "rationale": rationale, "histories_per_constructor_packet": 2 if card == "M01" else 1,
                    "seconds_per_condition_record": entry["seconds_per_condition_record"]})
        if card == "S02":
            all_contrasts = [contrast for value in summary["conditions"].values() for contrast in value["contrasts"]]
            exact_discovery_agreement = all(abs(row["mean"]) <= 1e-10 and row["paired_standard_deviation"] == 0 for row in all_contrasts)
            if exact_discovery_agreement:
                specs = {row["estimand"]["id"]: row["estimand"] for row in all_contrasts}
                first = next(iter(specs.values()))
                order, rationale = priority(card, first, kind="equivalence", cost=entry["seconds_per_condition_record"])
                candidates.append({"card_id": card, "kind": "equivalence", "condition_ids": sorted(summary["conditions"]),
                    "estimands": list(specs.values()), "source_base": entry["source_base"], "source_sha256": entry["source_sha256"],
                    "rank": order, "rationale": rationale, "histories_per_constructor_packet": len(summary["conditions"]),
                    "discovery_variance": 0, "external_difference_bound": 2., "margin": .05,
                    "bound_derivation": "Each actual reset has signed correct-minus-harmful repair in [-1,1]; every reader difference is in [-2,2]",
                    "grouping": "One fresh constructor, one distinct acquisition history per original memory condition; all paired reader differences form one joint event",
                    "seconds_per_condition_record": entry["seconds_per_condition_record"]})
    return sorted(candidates, key=lambda row: row["rank"]), exclusions


def allocate(candidates, *, maximum_claims=3):
    if not 0 <= maximum_claims <= 3:
        raise ValueError("confirmation reserve has at most three claims")
    selected, dispositions, used_cards = [], [], set()
    for candidate in sorted(candidates, key=lambda row: row["rank"]):
        if candidate["card_id"] in used_cards or len(selected) >= maximum_claims:
            dispositions.append({"candidate": candidate, "state": "not selected", "reason": "Higher-ranked distinct card packets filled the finite selection"})
            continue
        cap = 4096//candidate["histories_per_constructor_packet"]
        if candidate["kind"] == "equivalence":
            power = equivalence_power(outcome_difference_bound=2., margin=.05, maximum_n=cap)
            power["grouping_qualification"] = candidate["grouping"]
            n = power["n_independent_maker_packets"]
        else:
            bar = candidate["estimands"][0]["practical_bar"]
            alternative = 2*bar
            # Keep the recorded variance visible. Inflating a very small variance
            # permits the declared three-point planning law without asserting a
            # nonexistent zero-sample or normal-variance power result.
            variance = max(candidate["discovery_variance"], abs(alternative)-alternative**2+1e-12)
            try:
                power = plan_power(discovery_variance=variance, null_mean=bar, practical_increment=bar, maximum_n=cap)
            except ValueError as error:
                dispositions.append({"candidate": candidate, "state": "exploratory", "reason": "Unsupported planning moments: "+str(error)})
                continue
            power["original_discovery_variance"] = candidate["discovery_variance"]
            power["variance_inflation"] = variance-candidate["discovery_variance"]
            n = power["n_independent_makers"]
        if power is None or power["planning_state"] != "adequate":
            dispositions.append({"candidate": candidate, "state": "exploratory", "reason": "Insufficient actual-estimand power within the independent-history cap", "power": power})
            continue
        claim = {**candidate, "power": power, "n_constructor_packets": n,
            "total_fresh_acquisition_histories": n*candidate["histories_per_constructor_packet"],
            "alpha_planning": .05/3, "evidence_scope": "confirmation of a discovery-selected regime",
            "source_regime_selected_through_discovery": True, "replacement_after_confirmation_forbidden": True}
        selected.append(claim)
        used_cards.add(candidate["card_id"])
        dispositions.append({"candidate": candidate, "state": "selected after pre-data power planning", "power": power})
    return {"selected": selected, "candidate_dispositions": dispositions,
        "selection_rule": "Lexicographic validity/target, serious rival, relevance, practical size/cost; at most one packet per card and at most three cards",
        "familywise_alpha": .05, "planning_allocation": .05/3, "final_adjustment": "Holm across exactly the frozen primary claims",
        "confirmation_data_accessed": False}

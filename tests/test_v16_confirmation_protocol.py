from copy import deepcopy
import pytest
from ghostscale.validation.soundingline.v16.confirmation_protocol import grouped_primary, analyze_primary


def known_equivalence(n=320):
    specs = [{"arm": "self-model", "rival": name, "target": "net_repair"} for name in ["bayes-error", "direct-error"]]
    claim = {"card_id": "S02", "kind": "equivalence", "n_constructor_packets": n,
        "condition_ids": ["partial", "intact"], "histories_per_constructor_packet": 2,
        "estimands": specs, "external_difference_bound": 2., "margin": .05, "alpha_planning": .05/3,
        "total_fresh_acquisition_histories": 2*n, "grouping": "Two distinct acquired histories per independent constructor"}
    claim.update(namespace="fixture-confirmation-protocol", execution_packet_hash="fixture-protocol-source")
    rows = [{"card_id": "S02", "unit_id": f"{condition}-{index}", "condition": condition,
        "constructor_id": f"constructor-{index}", "seed_components": {"index": index, "constructors": n},
        "lineage": claim["namespace"], "packet_hash": claim["execution_packet_hash"],
        "arms": {name: {"outcomes": {"net_repair": 0.}} for name in ["self-model", "bayes-error", "direct-error"]}}
        for index in range(n) for condition in claim["condition_ids"]]
    return claim, rows


def test_contexts_are_a_single_joint_event_and_never_inflate_independent_n():
    claim, rows = known_equivalence()
    result = analyze_primary(claim, rows)
    assert result["n_independent_constructor_packets"] == 320
    assert result["n_native_condition_records"] == 640
    assert result["primary_analysis"]["paired_components"] == 4
    assert result["primary_analysis"]["equivalence_established_at_unadjusted_allocation"]
    # A rare failure in either context counts once per constructor, even when
    # both rivals disagree. Widespread failures must destroy equivalence.
    for row in rows[:40]:
        row["arms"]["self-model"]["outcomes"]["net_repair"] = 1.
    failed = analyze_primary(claim, rows)
    assert failed["primary_analysis"]["disagreement_events"] == 20
    assert not failed["primary_analysis"]["equivalence_established_at_unadjusted_allocation"]


def test_missing_contexts_reused_constructors_and_excess_histories_are_rejected():
    claim, rows = known_equivalence()
    with pytest.raises(ValueError, match="omitted"):
        grouped_primary(claim, rows[:-1])
    changed = deepcopy(rows)
    changed[2]["constructor_id"] = changed[0]["constructor_id"]
    with pytest.raises(ValueError, match="cluster"):
        grouped_primary(claim, changed)
    changed = deepcopy(rows)
    for row in changed:
        row["seed_components"]["constructors"] = 8
    with pytest.raises(ValueError, match="reused"):
        grouped_primary(claim, changed)
    changed = deepcopy(rows)
    changed[0]["lineage"] = "already-inspected-discovery"
    with pytest.raises(ValueError, match="lineage"):
        grouped_primary(claim, changed)
    changed = deepcopy(claim)
    changed["total_fresh_acquisition_histories"] = 1
    with pytest.raises(ValueError, match="denominator"):
        grouped_primary(changed, rows)
    claim["histories_per_constructor_packet"] = 20
    with pytest.raises(ValueError, match="cap"):
        grouped_primary(claim, rows)

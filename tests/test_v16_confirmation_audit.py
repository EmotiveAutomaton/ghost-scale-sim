from copy import deepcopy
import pytest
from ghostscale.validation.soundingline.v16.confirmation_audit import primary
from ghostscale.validation.soundingline.v16.confirmation_protocol import analyze_primary


def known(n, kind):
    conditions = ["one"] if kind == "capability" else ["one", "two"]
    specs = [{"arm": "reader", "rival": name, "target": "outcome", "practical_bar": .05}
        for name in (["direct"] if kind == "capability" else ["direct", "bayes"])]
    claim = {"card_id": "K01" if kind == "capability" else "S02", "kind": kind,
        "n_constructor_packets": n, "condition_ids": conditions, "estimands": specs,
        "namespace": "known-audit", "execution_packet_hash": "known-source", "alpha_planning": .05/3,
        "histories_per_constructor_packet": len(conditions), "total_fresh_acquisition_histories": n*len(conditions),
        "external_difference_bound": 2., "margin": .05, "grouping": "known independent contexts"}
    rows = [{"card_id": claim["card_id"], "unit_id": f"{index}-{condition}", "constructor_id": str(index),
        "condition": condition, "seed_components": {"index": index, "constructors": n},
        "lineage": claim["namespace"], "packet_hash": claim["execution_packet_hash"],
        "arms": {name: {"outcomes": {"outcome": int(name == "reader" and kind == "capability" and index%4 == 0)}}
            for name in ["reader", "direct", "bayes"]}} for index in range(n) for condition in conditions]
    return claim, rows


def test_analytic_inversion_reproduces_held_and_unresolved_means_and_detects_corruption():
    for n in [32, 1024]:
        claim, rows = known(n, "capability")
        result = analyze_primary(claim, rows)
        checked = primary(claim, rows, result)
        assert checked["instrument_state"] == "valid"
        assert result["primary_analysis"]["criterion_state"] == ("held" if n == 1024 else "not established")
        corrupted = deepcopy(result)
        corrupted["primary_analysis"]["paired_mean"] += .01
        with pytest.raises(ValueError, match="numerical"):
            primary(claim, rows, corrupted)


def test_independent_beta_binomial_calculation_reproduces_joint_null_and_failure():
    claim, rows = known(320, "equivalence")
    result = analyze_primary(claim, rows)
    assert primary(claim, rows, result)["recalculated"]["disagreement_events"] == 0
    for row in rows[:40]:
        row["arms"]["reader"]["outcomes"]["outcome"] = 1
    result = analyze_primary(claim, rows)
    assert primary(claim, rows, result)["recalculated"]["disagreement_events"] == 20
    assert not result["primary_analysis"]["equivalence_established_at_unadjusted_allocation"]

from ghostscale.validation.soundingline.v16.confirmation_candidates import enumerate_candidates, priority


def entry(card="K01", constructors=32, target="success", held=True, rival="pooled"):
    spec = {"id": card+"-primary", "arm": "personal", "rival": rival, "target": target,
            "units": "success_fraction", "practical_bar": .05}
    return {"card_id": card, "constructors": constructors,
        "design": {"primary": [(spec["arm"], rival, target, spec["units"], .05)]},
        "summary": {"conditions": {"condition": {"contrasts": [{"estimand": spec, "mean": .5,
            "paired_standard_deviation": .5, "n_makers": 256, "n_constructors": constructors,
            "criterion_state": "held" if held else "inconclusive"}]}}},
        "source_base": "source", "source_sha256": "source-hash", "seconds_per_condition_record": .1}


def test_large_estimate_cannot_bypass_validity_target_or_rival_priority():
    good = entry()
    candidates, excluded = enumerate_candidates([good, entry("K03", constructors=8),
        entry("P04", target="future_log_score"), entry("K04", held=False)])
    assert [row["card_id"] for row in candidates] == ["K01"]
    assert len(excluded) == 3
    spec = good["summary"]["conditions"]["condition"]["contrasts"][0]["estimand"]
    strong = priority("K01", {**spec, "rival": "direct-table"}, mean=.06)[0]
    weak = priority("K01", {**spec, "rival": "primitive"}, mean=.9)[0]
    assert strong < weak


def test_supplied_oracle_state_is_not_a_learned_reader_confirmation():
    supplied = entry("O02")
    supplied["design"]["family"] = "critic"
    candidates, excluded = enumerate_candidates([supplied])
    assert not candidates
    assert "Supplied-state" in excluded[0]["reason"]

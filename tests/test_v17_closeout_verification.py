"""Known answers and deliberate failures for independent V17 closeout."""
import copy
import math
import pytest
from runners.verify_v17_closeout import compare, unit_stats, paired, confirmed, holm


def row(method, success, score, missing=False):
    return dict(method=method, target="future", task_success=success,
                brier_score=score, missing_output=missing, invalid_program=False,
                costs=dict(cold_total=20, repeat_online=4), evidence_tier="artifact",
                log_loss_infinite=missing)


def test_raw_aggregation_keeps_missing_outputs_in_denominator():
    items = [dict(rows=[row("a", True, .125)]), dict(rows=[row("a", False, 1.125, True)])]
    expected = [dict(method="a", target="future", rows=2, valid=1,
                     apparatus_failures=0, score=.625, success=.5, cold=20., online=4.,
                     stop_regret=0., decision_regret=0., personal_goal_success=0.,
                     infinite=1, evidence_tiers=["artifact"])]
    compare(unit_stats(items), expected)
    for field, value in (("valid", 2), ("score", .125), ("infinite", 0)):
        corrupted = copy.deepcopy(expected)
        corrupted[0][field] = value
        with pytest.raises(ValueError):
            compare(unit_stats(items), corrupted)


def test_constructor_average_uses_paired_histories_and_direction():
    items = [dict(rows=[row("a", True, .1), row("b", False, .7)]),
             dict(rows=[row("a", False, .9), row("b", True, .1)])]
    design = dict(method="a", rival="b", target="future", metric="brier", low=-2, high=2)
    assert paired(items, design) == pytest.approx(-.1)
    assert paired(items, dict(design, metric="success")) == 0
    with pytest.raises(KeyError):
        paired([dict(rows=[row("a", True, .1)])], design)


def test_fixed_bound_and_family_known_answers():
    design = dict(id="known", constructors=100, low=-1, high=1, minimum_gain=.05)
    positive = confirmed([.3] * 100, design)
    assert positive["mean_gain"] == .3
    assert positive["p_value"] == pytest.approx(math.exp(-3.125))
    assert positive["lower_bound"] == pytest.approx(.3 - math.sqrt(2 * math.log(60) / 100))
    assert confirmed([.05] * 100, design)["p_value"] == 1
    assert confirmed([-.3] * 100, design)["p_value"] == 1
    with pytest.raises(ValueError, match="incomplete"):
        confirmed([.3] * 99, design)
    result = holm([dict(id="a", p_value=.01), dict(id="b", p_value=.03), dict(id="c", p_value=.2)])
    assert [r["holm_adjusted_p"] for r in result] == pytest.approx([.03, .06, .2])
    assert [r["holm_rejected"] for r in result] == [True, False, False]


def test_comparison_refuses_missing_fields_and_nan():
    with pytest.raises(ValueError):
        compare(dict(count=1), dict(count=1, valid=1))
    with pytest.raises(ValueError):
        compare(float("nan"), 0.)

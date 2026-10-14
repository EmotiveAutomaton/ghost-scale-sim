"""Known answers and adversarial controls precede V22 development observations."""
import itertools
import numpy as np
import pytest
from ghostscale.validation.soundingline.v22 import studies, credibility, homeostasis, discrete, attention, hierarchy
from ghostscale.validation.soundingline.v22.checker import INDEPENDENT
from ghostscale.validation.soundingline.v22.common import score, bern


@pytest.mark.parametrize('card', studies.CARDS)
@pytest.mark.parametrize('index', [0, 7, 35, 127])
def test_independent_law_marginal(card, index):
    public, truth = studies.sample(card, 'engineering-controls', 900+index, index)
    forecasts, _ = studies.evaluate(card, public, truth)
    assert np.allclose(forecasts['law-marginal'], INDEPENDENT[card](public), atol=1e-10)
    assert np.isfinite(studies.features(card, public)).all()


def test_duplicated_source_evidence_is_not_fresh():
    p, t = credibility.sample('engineering-controls', 901, 7)
    p.update(k=2, qh=[1., 1.], ch=[.9, .9], roots=[0, 0])
    single = dict(p, k=1, qh=[1.], ch=[.9], roots=[0])
    assert np.allclose(credibility.posterior(p)[0], credibility.posterior(single)[0])
    assert not np.allclose(credibility.posterior(p, unique=False)[0], credibility.posterior(single)[0])


def test_unrelated_source_history_is_not_current_source_evidence():
    p, _ = credibility.sample('engineering-controls', 901, 15)
    p['condition'] = 'unrelated'
    empty = dict(p, k=0, qh=[], ch=[], roots=[])
    assert np.allclose(credibility.posterior(p)[0], credibility.posterior(empty)[0])
    assert not np.allclose(credibility.posterior(p, allow_unrelated=False)[0], credibility.posterior(empty)[0])


def test_homeostasis_nonidentifying_choices_keep_prior():
    p, t = homeostasis.sample('engineering-controls', 901, 55)
    for row in p['history']+[p['current']]:
        row['effects'] = np.zeros((3, p['d'])).tolist()
        row['costs'] = [0., 0., 0.]
    pred, post = homeostasis.posterior(p)
    assert np.allclose(pred, 1/3)
    assert np.allclose(post, 1/len(post))
    assert np.allclose(homeostasis.posterior(p, 'effort')[1], post)


def test_no_signature_does_not_identify_goal_from_selection():
    p, t = discrete.sample_a2('engineering-controls', 901, 4)
    p['signature'] = False
    flipped = dict(p, selection=[1-v for v in p['selection']])
    assert np.allclose(discrete.infer_a2(p), discrete.infer_a2(flipped))


def test_c2_similarity_disclosure_is_information():
    p, t = discrete.sample_c2('engineering-controls', 901, 0)
    p.update(k=0, witnesses=[], records=[], coupled=False, preference_witness=-1, disclosed=False,
             share_lib=None, share_pref=None)
    assert np.isclose(discrete.infer_c2(p)[1, 1], .5, atol=1e-14)
    revealed = dict(p, disclosed=True, share_lib=True, share_pref=True)
    assert np.isclose(discrete.infer_c2(revealed)[1, p['own_pref']], 1., atol=1e-14)


def test_c2_record_shuffle_can_remove_preference_information():
    p, t = discrete.sample_c2('engineering-controls', 901, 0)
    p.update(k=32, witnesses=[0]*32, records=[1]*32, coupled=False, disclosed=False,
             share_lib=None, share_pref=None, unrelated=False)
    assert discrete.infer_c2(p)[1, 1] > .99
    assert np.isclose(discrete.infer_c2(dict(p, unrelated=True))[1, 1], .5)


def test_devaluation_alias_has_no_preintervention_route_signature():
    p, t = discrete.sample_b2('engineering-controls', 901, 2)
    assert p['condition'] == 'alias'
    assert np.isclose(discrete.infer_b2(p)[0, 1], .5)
    devalued = dict(p, devalued=True)
    assert discrete.infer_b2(devalued, only_route=0)[1, 1] < discrete.infer_b2(devalued, only_route=1)[1, 1]


def test_attention_is_prefix_causal_and_floor_has_cost():
    p, t = attention.sample('engineering-controls', 901, 5)
    p['prefix'] = 4
    other = dict(p, observations=p['observations'][:4]+[[1, 1, 1]]*12,
                 flags=p['flags'][:4]+[1]*12)
    for policy in attention.POLICIES:
        assert attention.trajectory(p, policy)[1] == attention.trajectory(other, policy)[1]
    p.update(prefix=16, flags=[1]*16, observations=[[1, 1, 1]]*16)
    _, path = attention.trajectory(p, 'withdraw-.7')
    assert path[-1]['fired'] and path[-1]['precision'][0] == .05


def test_tree_against_exhaustive_joint_enumeration():
    parents, _ = hierarchy.tree(2, depth=2)
    evidence = [-1, -1, -1, 1, 0, 1, 1]
    states = np.array(list(itertools.product((0, 1), repeat=7)))
    mass = []
    for s in states:
        p = .5
        for i in range(1, 7):
            p *= .8 if s[i] == s[parents[i]] else .2
        for i, obs in enumerate(evidence):
            if obs >= 0:
                p *= .9 if obs == s[i] else .1
        mass.append(p)
    mass = np.array(mass)
    posterior, logev = hierarchy.belief_propagation(parents, evidence, .8)
    assert np.allclose(posterior[:, 1], (mass/mass.sum())@states)
    assert np.isclose(np.exp(logev), mass.sum())


def test_tree_without_evidence_is_prior():
    parents, _ = hierarchy.tree(3)
    for mode in ('joint', 'top-down', 'bottom-up'):
        p, logev = hierarchy.belief_propagation(parents, [-1]*len(parents), .8, mode)
        assert np.allclose(p, .5)
        assert abs(logev) < 1e-12


def test_uncapped_loss_and_full_brier_keep_failures():
    values = score(np.array([[[1., 0.]]]), np.array([[1]]))
    assert np.isinf(values['log_loss'][0])
    assert values['brier'][0] == 2.


def test_binary_marginal_roundoff_is_not_a_probability_floor():
    assert np.array_equal(bern([1+2e-16, 0]), [[0, 1], [1, 0]])
    with pytest.raises(ValueError):
        bern([1.001])


def test_original_e56_scorer_retains_per_step_uniform_gate_identity():
    from ghostscale.validation.soundingline.v22.legacy_attention import e56_accounting
    p, truth = attention.sample('engineering-legacy', 902, 5)
    result = e56_accounting(p, truth)
    for name in ('process_uptake', 'goal_uptake', 'value_uptake'):
        assert np.isclose(result['early-guard'][name], .2*result['no-gate'][name])


def test_g5_physical_reuse_has_independent_execution_and_charged_budget():
    from ghostscale.validation.soundingline.v22.legacy_attention import physical
    from ghostscale.validation.soundingline.v22.checker import physical_execution
    result = physical('engineering-legacy', 902)
    assert len(result['baseline']['rows']) == 48
    assert len(result['withdrawal']) == 8
    for row in result['baseline']['rows']+result['withdrawal']:
        for output in row['outputs']:
            state, stopped, legal = physical_execution(row['world'], output['plan']['program'])
            assert state == output['execution']['state']
            assert stopped == output['execution']['successfully_stopped']
            assert legal == output['execution']['legal']
            assert output['charged_total'] <= 256

"""Known-answer controls for distinct, unadmitted V24 theory directions."""
import numpy as np

from ghostscale.validation.soundingline.v24 import context_convergence,disturbance,higher_goal_editing,needs_drives,play_uncertainty,setpoints


def test_needs_drives_alias_recovery_and_truth_separation():
    controls=needs_drives.controls()
    assert controls['baseline_observational_alias']
    assert controls['selective_satiation_known_answer']
    assert controls['public_consequence_evaluator_separation']
    public,truth=needs_drives.generate(needs_drives.settings()[5],'need_a','fixture',trials=96)
    assert truth=={'mechanism':'need_a'} and 'mechanism' not in public
    assert np.isclose(needs_drives.posterior(public).sum(),1)


def test_setpoints_null_restoration_equivalent_means_and_recovery():
    controls=setpoints.controls()
    assert controls['zero_decay_null_maintenance']
    assert controls['restoration_intervention_present']
    assert controls['equivalent_means_available']
    assert controls['planted_maintenance_recovered']
    public,truth=setpoints.generate(setpoints.settings()[-1],'intrinsic_act','fixture')
    assert truth=={'mechanism':'intrinsic_act'} and 'mechanism' not in public
    assert np.isclose(setpoints.posterior(public).sum(),1)


def test_play_uncertainty_controls_and_truth_separation():
    controls=play_uncertainty.controls()
    assert all(controls[name] for name in (
        'no_informative_feedback_leaves_solution_uniform','known_high_uncertainty_recovered',
        'urgent_need_suppresses_play','partner_label_shuffle_invariant'))
    public,truth=play_uncertainty.generate(play_uncertainty.settings()[0],'partner_information','fixture')
    assert 'mechanism' not in public and truth['mechanism']=='partner_information'


def test_higher_goal_editing_controls_and_uncensored_drafts():
    controls=higher_goal_editing.controls()
    assert all(controls[name] for name in (
        'no_conflict_alias','planted_higher_goal_recovered','style_only_recovered',
        'role_permutation_invariant'))
    public,truth=higher_goal_editing.generate(higher_goal_editing.settings()[-1],'higher_goal','fixture')
    assert public['uncensored_drafts'] and 'mechanism' not in public
    assert truth=={'mechanism':'higher_goal'}


def test_context_convergence_controls_and_omitted_support():
    controls=context_convergence.controls()
    assert all(controls.values())
    public,truth=context_convergence.generate('c','fixture',length=32)
    omitted=context_convergence.read(public,[.5,.5],support=('a','b'))
    assert truth=={'mechanism':'c'} and 'c' not in omitted['support'] and omitted['normalized']


def test_disturbance_controls_and_empty_support_is_explicit():
    controls=disturbance.controls()
    assert all(controls.values())
    public,truth=disturbance.generate(disturbance.settings()[0],'habit','fixture')
    assert truth=={'mechanism':'habit'} and 'mechanism' not in public
    assert disturbance.feasible_region(public)['literal_primary_available']

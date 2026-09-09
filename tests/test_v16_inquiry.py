import math
from itertools import product
import pytest
from ghostscale.validation.soundingline.v16.inquiry import (
    PRIOR,PERMUTATIONS,GOALS,update,construction,competence,expected_after,prediction,choose)
from ghostscale.validation.soundingline.v16.records import canonical


def test_real_feedback_changes_success_on_unseen_compositions():
    mapping=(2,0,3,1)
    def score(belief):
        count=0
        for goal in GOALS:
            commands=construction(belief,goal)
            artifact=0
            for command in commands:
                artifact|=1<<mapping[command]
            count+=artifact==goal
        return count/6
    learned=PRIOR
    assert score(learned)==1/6
    for command in range(3):
        learned=update(learned,command,mapping[command])
    assert score(learned)==1.0
    assert score(PRIOR)==1/6  # genuine memory loss changes executable capability


def test_noise_is_not_a_fixed_hidden_mapping_and_cannot_gain_competence():
    noise=update(update(PRIOR,0,0),0,1)
    assert noise[-1]==1
    assert prediction(noise,0)==[0.25]*4
    assert competence(noise)==0.125
    skill,uncertainty,cost=expected_after(noise,(0,1))
    assert skill==0.125 and uncertainty==0
    # Independently enumerate both random physical outputs.
    successes=sum(((1<<a)|(1<<b))==GOALS[0] for a,b in product(range(4),repeat=2))
    assert successes/16==0.125


def test_exact_learning_value_matches_independent_one_query_sum():
    skill,entropy,cost=expected_after(PRIOR,(0,))
    expected=sum(0.25*competence(update(PRIOR,0,cell)) for cell in range(4))
    assert abs(skill-expected)<1e-10
    assert abs(competence(PRIOR)-(0.9/6+0.1/8))<1e-10
    assert skill>competence(PRIOR)


def test_policy_has_no_evaluator_truth_input():
    public={"schema_version":"v16.inquiry.1","task_id":"opaque","beliefs":[PRIOR,PRIOR],
            "offers":[0,1],"histories":[[],[]],"familiarity":[4,0],
            "pending_commands":[[],[]],"feedback_batches":[1,1],"future_weights":[0.5,0.5],
            "opportunity_cost":0.01,"remaining_interactions":8,"tie_draw":0.25,
            "committed_domain":None,"offer_probabilities":[[0.25]*4]*2}
    decision=choose(canonical(public),"value-learning")
    public["task_id"]="different-hidden-seed"
    assert decision==choose(canonical(public),"value-learning")
    public["true_mapping"]=[0,1,2,3]
    with pytest.raises(ValueError,match="schema"):
        choose(canonical(public),"value-learning")

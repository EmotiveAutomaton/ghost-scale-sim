import math
from itertools import product
from ghostscale.validation.soundingline.v16.self_monitor import (
    TRAINING_ALIGNMENTS,controller_prior,generate,public_reader,evaluate)
from ghostscale.validation.soundingline.v16.records import canonical


def test_controller_prior_matches_exhaustive_training_actions():
    for goal,q in product(range(2),TRAINING_ALIGNMENTS):
        p = q if goal else 1-q
        expected = sum(math.prod(p if action else 1-p for action in history)
                       for history in product(range(2),repeat=8) if sum(history)>4)
        assert abs(controller_prior(goal,q)-expected) < 1e-10


def test_intact_memory_no_extra_control_information_and_partial_positive():
    full,private,_ = generate("self-fixture",{"id":"full","memory":"intact"},0)
    a,b = public_reader(canonical(full)),public_reader(canonical(full),strategy="memory-only")
    assert a["controller_probabilities"] == b["controller_probabilities"]
    assert a["controller_probabilities"][private["controller"]] == 1
    partial = {**full,"memory":{**full["memory"],"controller":None},"artifacts":[1]}
    a,b = public_reader(canonical(partial)),public_reader(canonical(partial),strategy="memory-only")
    assert a["controller_probabilities"] != b["controller_probabilities"]


def test_strong_direct_error_monitor_uses_same_evidence_and_cost():
    for memory in ["intact","partial","absent"]:
        for retarget in [False,True]:
            public,private,_ = generate("error-fixture",{"id":memory+str(retarget),
                                                       "memory":memory,"retarget":retarget},0)
            self_model = public_reader(canonical(public))
            direct = public_reader(canonical(public),strategy="bayes-error")
            assert self_model["repair"] == direct["repair"]
            assert self_model["adopted_goal"] == direct["adopted_goal"]
            assert self_model["costs"] == direct["costs"]
            assert max(abs(a-b) for a,b in zip(self_model["controller_probabilities"],
                                              direct["controller_probabilities"])) < 1e-10
            private["controller"] = 99
            public["task_id"] = "altered-seed-and-filename"
            assert self_model == public_reader(canonical(public))


def test_success_at_retarget_is_not_original_goal_recovery():
    public,private,_ = generate("retarget-fixture",{"id":"retarget","retarget":True},0)
    public["current_goal"] = 1-private["actual_original_goal"]
    prediction = public_reader(canonical(public),strategy="direct-completion")
    outcome = evaluate(private,prediction,reset_coin=0.9,future_coin=0.9)
    assert outcome["adopted_goal_success"] == 1
    assert outcome["original_goal_success"] == 0
    assert outcome["continued_original_goal"] == 0
    assert outcome["original_goal_recovery"] == 1

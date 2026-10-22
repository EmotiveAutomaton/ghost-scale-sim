"""Finite play-as-information construction for the V24 design card."""
import math

import numpy as np

from ..v23.common import normalize, rng

MECHANISMS = ("own_information", "partner_information", "pragmatic", "exploration_habit")


def settings():
    return [dict(own_uncertainty=own, partner_uncertainty=partner,
                 urgent_need=urgent, informative_feedback=feedback)
            for own in (.1, .9) for partner in (.1, .9)
            for urgent in (False, True) for feedback in (False, True)]


def probability(mechanism, row):
    suppression = 4. if row["urgent_need"] else 0.
    information = 1. if row["informative_feedback"] else 0.
    if mechanism == "own_information":
        utility = [0., 3 * row["own_uncertainty"] * information - suppression, -suppression]
    elif mechanism == "partner_information":
        utility = [0., -suppression, 3 * row["partner_uncertainty"] * information - suppression]
    elif mechanism == "pragmatic":
        utility = [2. if row["urgent_need"] else .5, -1., -1.]
    elif mechanism == "exploration_habit":
        utility = [0., .6 - suppression, .6 - suppression]
    else:
        raise ValueError("unknown play mechanism")
    return normalize(np.asarray(utility) * 2)


def generate(setting, mechanism, identity, *, trials=64):
    if setting not in settings() or mechanism not in MECHANISMS or trials < 1:
        raise ValueError("frozen play setting, mechanism, and positive trials required")
    random = rng("v24-play-uncertainty", identity, setting, mechanism)
    solution = int(random.integers(2));partner_skill = int(random.integers(2));history=[]
    for trial in range(trials):
        row = dict(trial=trial, **setting)
        p = probability(mechanism, row);row["action"] = int(random.choice(3, p=p))
        if not setting["informative_feedback"] or row["action"] == 0:
            row["feedback"] = None
        elif row["action"] == 1:
            row["feedback"] = solution if random.random() < .9 else 1 - solution
        else:
            row["feedback"] = partner_skill if random.random() < .9 else 1 - partner_skill
        row["partner_label"] = trial % 2
        history.append(row)
    return dict(identity=identity, setting=setting, history=history), dict(
        mechanism=mechanism, solution=solution, partner_skill=partner_skill)


def posterior(public):
    log_mass=np.zeros(len(MECHANISMS))
    for row in public["history"]:
        for index,mechanism in enumerate(MECHANISMS):
            log_mass[index]+=math.log(probability(mechanism,row)[row["action"]])
    return normalize(log_mass)


def solution_posterior(public):
    mass=np.ones(2)/2
    for row in public["history"]:
        if row["action"]==1 and row["feedback"] is not None:
            mass*=np.array([.9 if row["feedback"]==value else .1 for value in range(2)])
            mass/=mass.sum()
    return mass


def controls():
    informative=dict(own_uncertainty=.9,partner_uncertainty=.1,urgent_need=False,informative_feedback=True)
    planted,_=generate(informative,"own_information","v24-engineering-excluded-play",trials=128)
    no_feedback=dict(informative,informative_feedback=False)
    null,_=generate(no_feedback,"own_information","v24-engineering-excluded-play-null",trials=128)
    urgent=dict(informative,urgent_need=True)
    suppressed,_=generate(urgent,"own_information","v24-engineering-excluded-play-urgent",trials=128)
    shuffled=dict(planted,history=[dict(row,partner_label=1-row["partner_label"]) for row in planted["history"]])
    return dict(
        no_informative_feedback_leaves_solution_uniform=bool(np.allclose(solution_posterior(null),[.5,.5])),
        known_high_uncertainty_recovered=bool(np.argmax(posterior(planted))==0 and posterior(planted)[0]>.99),
        urgent_need_suppresses_play=sum(row["action"]!=0 for row in suppressed["history"])<16,
        partner_label_shuffle_invariant=bool(np.allclose(posterior(planted),posterior(shuffled),atol=0,rtol=0)),
        settings=len(settings()),mechanisms=len(MECHANISMS),
    )

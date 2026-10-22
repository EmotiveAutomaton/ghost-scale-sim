"""Finite controlled-variable construction with external disturbances."""
import math

import numpy as np

from ..v23.common import normalize, rng

MECHANISMS=("closed_loop","open_loop","habit")
ACTIONS=np.array([-1.,0.,1.])


def settings():
    return [dict(disturbance=disturbance,controllable=controllable,empty_feasible=empty)
            for disturbance in (0.,1.) for controllable in (False,True)
            for empty in (False,True)]


def probability(mechanism,row):
    effect=1. if row["controllable"] else 0.
    if mechanism=="closed_loop":utility=-4*np.abs(row["state"]+row["disturbance"]+effect*ACTIONS-row["target"])
    elif mechanism=="open_loop":utility=-4*np.abs(ACTIONS-row["planned_action"])
    elif mechanism=="habit":utility=-4*np.abs(ACTIONS-row["cue_action"])
    else:raise ValueError("unknown disturbance mechanism")
    return normalize(utility)


def generate(setting,mechanism,identity,*,steps=64):
    if setting not in settings() or mechanism not in MECHANISMS or steps<1:
        raise ValueError("frozen disturbance setting, mechanism, and positive steps required")
    random=rng("v24-disturbance",identity,setting,mechanism);state=0.;history=[]
    for step in range(steps):
        disturbance=setting["disturbance"]*(1 if step%2==0 else -1)
        target=2. if setting["empty_feasible"] and step%2 else 0.
        row=dict(step=step,state=state,disturbance=disturbance,target=target,
                 controllable=setting["controllable"],planned_action=0.,cue_action=float(step%3-1))
        p=probability(mechanism,row);index=int(random.choice(3,p=p));row["action"]=float(ACTIONS[index])
        state=state+disturbance+(row["action"] if setting["controllable"] else 0.)
        row["next_state"]=state;history.append(row)
    return dict(identity=identity,setting=setting,history=history),dict(mechanism=mechanism)


def posterior(public):
    log_mass=np.zeros(len(MECHANISMS))
    for row in public["history"]:
        action_index=int(np.where(ACTIONS==row["action"])[0][0])
        for index,mechanism in enumerate(MECHANISMS):
            log_mass[index]+=math.log(probability(mechanism,row)[action_index])
    return normalize(log_mass)


def feasible_region(public,tolerance=.25):
    if tolerance<=0:raise ValueError("positive tolerance required")
    lower,upper=-math.inf,math.inf
    for row in public["history"]:
        if not row["controllable"]:continue
        implied=row["state"]+row["disturbance"]+row["action"]
        target=row["target"]
        lower=max(lower,target-tolerance-implied);upper=min(upper,target+tolerance-implied)
    return dict(state="EMPTY_FEASIBLE_SET" if lower>upper else "FEASIBLE",
                interval=None if lower>upper else [lower,upper],literal_primary_available=lower<=upper)


def controls():
    zero=dict(disturbance=0.,controllable=True,empty_feasible=False)
    alias,_=generate(zero,"closed_loop","v24-engineering-excluded-disturbance-alias",steps=1)
    open_likelihood=probability("open_loop",alias["history"][0]);closed_likelihood=probability("closed_loop",alias["history"][0])
    planted_setting=dict(disturbance=1.,controllable=True,empty_feasible=False)
    planted,_=generate(planted_setting,"closed_loop","v24-engineering-excluded-disturbance-recovery",steps=128)
    uncontrollable,_=generate(dict(disturbance=1.,controllable=False,empty_feasible=False),"closed_loop",
                              "v24-engineering-excluded-disturbance-uncontrollable")
    contradictory=dict(planted,history=[dict(planted["history"][0],target=-2.),dict(planted["history"][0],target=2.)])
    return dict(
        zero_disturbance_alias=bool(np.allclose(open_likelihood,closed_likelihood,atol=0,rtol=0)),
        planted_compensatory_response=bool(np.argmax(posterior(planted))==0 and posterior(planted)[0]>.99),
        uncontrollable_state=not any(row["controllable"] for row in uncontrollable["history"]),
        empty_feasible_set=feasible_region(contradictory)["state"]=="EMPTY_FEASIBLE_SET",
    )

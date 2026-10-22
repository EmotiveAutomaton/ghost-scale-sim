"""Finite generator/editor hierarchy construction for the V24 design card."""
import math

import numpy as np

from ..v23.common import normalize, rng

MECHANISMS=("higher_goal","style_only","generator_only","global_edit_rate")


def settings():
    return [dict(conflict=conflict,higher_goal_observable=observable,role_permuted=permuted)
            for conflict in (False,True) for observable in (False,True)
            for permuted in (False,True)]


def edit_probability(mechanism,row):
    if mechanism=="higher_goal":
        return .9 if row["conflict"] and row["higher_goal_observable"] and row["goal_violation"] else .1
    if mechanism=="style_only":return .85 if row["style_violation"] else .1
    if mechanism=="generator_only":return .1
    if mechanism=="global_edit_rate":return .5
    raise ValueError("unknown editing mechanism")


def generate(setting,mechanism,identity,*,drafts=96):
    if setting not in settings() or mechanism not in MECHANISMS or drafts<1:
        raise ValueError("frozen editing setting, mechanism, and positive drafts required")
    random=rng("v24-higher-goal-editing",identity,setting,mechanism);history=[]
    for index in range(drafts):
        generator_preference=index%2;goal_violation=bool(index%3==0);style_violation=bool(index%4==0)
        row=dict(draft=index,generator_preference=generator_preference,
                 goal_violation=goal_violation,style_violation=style_violation,**setting)
        row["edited"]=bool(random.random()<edit_probability(mechanism,row))
        row["final_choice"]=1-generator_preference if row["edited"] and goal_violation else generator_preference
        row["generator_role"]="B" if setting["role_permuted"] else "A"
        row["editor_role"]="A" if setting["role_permuted"] else "B"
        history.append(row)
    return dict(identity=identity,setting=setting,uncensored_drafts=True,history=history),dict(mechanism=mechanism)


def posterior(public):
    log_mass=np.zeros(len(MECHANISMS))
    for row in public["history"]:
        for index,mechanism in enumerate(MECHANISMS):
            p=edit_probability(mechanism,row);log_mass[index]+=math.log(p if row["edited"] else 1-p)
    return normalize(log_mass)


def controls():
    alias=dict(conflict=False,higher_goal_observable=True,role_permuted=False)
    public,_=generate(alias,"higher_goal","v24-engineering-excluded-edit-alias")
    # With no conflict, higher-goal and generator-only assign the same likelihood.
    evidence=[]
    for mechanism in ("higher_goal","generator_only"):
        evidence.append(sum(math.log(edit_probability(mechanism,row) if row["edited"] else
                                     1-edit_probability(mechanism,row)) for row in public["history"]))
    planted_setting=dict(conflict=True,higher_goal_observable=True,role_permuted=False)
    planted,_=generate(planted_setting,"higher_goal","v24-engineering-excluded-edit-recovery",drafts=192)
    style,_=generate(planted_setting,"style_only","v24-engineering-excluded-edit-style",drafts=192)
    permuted=dict(planted,history=[dict(row,generator_role=row["editor_role"],editor_role=row["generator_role"])
                                  for row in planted["history"]])
    return dict(
        no_conflict_alias=bool(np.isclose(evidence[0],evidence[1],atol=1e-12)),
        planted_higher_goal_recovered=bool(np.argmax(posterior(planted))==0 and posterior(planted)[0]>.99),
        style_only_recovered=bool(np.argmax(posterior(style))==1 and posterior(style)[1]>.99),
        role_permutation_invariant=bool(np.allclose(posterior(planted),posterior(permuted),atol=0,rtol=0)),
        settings=len(settings()),mechanisms=len(MECHANISMS),
    )

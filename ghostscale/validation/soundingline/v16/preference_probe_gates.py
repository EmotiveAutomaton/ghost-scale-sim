"""Known indistinguishable histories and real probe separation controls."""
import copy
import json
import math
import random
from .assembly import World,execute
from .preference_probe_world import produce,aligned_models,CAUSES
from .preference_probe import public_reader,decision
from .preference_probe_reference import aligned,information,prediction,choice

def fixture():
    world=World()
    training=[{"date":date,"program":[0,6,1,9],"target":[1,0,-1],"feedback":True} for date in range(8)]
    base={"date":8,"initial":[1,0,-1],"required_part":2,"orientation_standard":[0,0,0],"price":0.05,
        "max_steps":6,"search_budget":256,"consideration":"all","retention":"random","produced_count":1}
    return {"schema_version":"v16.preference-probe.1","task_id":"known-probe","phase":1,"world":world.public(),
        "law":{"low":0.2,"high":0.8,"beta":8.0},"training":training,
        "history":[{"context":{**base,"date":date,"max_steps":1},"state":[1,0,-1]} for date in range(8)],
        "task_context":base,"query_fees":{"open":0.01,"time":0.015,"neutral":0.015,"all":0.02},"answer":None}

def run():
    public=fixture();routine=[0,6,1];old=[1,0,-1]
    histories=[];legal=[];found=[]
    for cause in CAUSES:
        events=[produce(public["world"],{**public["task_context"],"date":date},public["law"],cause,"history",routine,old,random.Random(90+date)) for date in range(8)]
        histories.append([{"context":event["context"],"state":event["execution"]["state"]} for event in events])
        legal.extend(event["execution"]["successfully_stopped"] for event in events)
    for query in ["open","time","neutral","all"]:
        native=aligned_models(public,query);independent=aligned(public,query)
        found.append(native[0]==independent[0] and native[2]==independent[2] and
            all(abs(a-b)<1e-12 for row,other in zip(native[1],independent[1]) for a,b in zip(row,other)))
    open_states,opened,_=aligned_models(public,"open")
    all_states,all_rows,_=aligned_models(public,"all")
    time_states,time_rows,_=aligned_models(public,"time")
    future_states,future_rows,_=aligned_models(public,"future")
    for policy in ["eig-cause","eig-future"]:
        first=decision(public,policy);other=choice(public,policy)
        found.append(first["query"]==other["query"] and abs(first["information_gain"]-other["information_gain"])<1e-12 and first["costs"]==other["costs"])
    visible={**copy.deepcopy(public),"phase":2,"answer":{"query":"all","state":[1,0,0]}}
    actual=public_reader(json.dumps(visible).encode(),"all")
    direct=public_reader(json.dumps(visible).encode(),"direct-all")
    independent=prediction(visible,"all")
    checks={
      "probe-physical":{"all_original_decisions_execute":all(legal),
        "time_grant_adds_actual_options_for_restricted_planner":sum(p>0 for p in time_rows[1])>sum(p>0 for p in opened[1]),
        "artifact_reproduction_executes":execute(World(),actual["reproduction_program"])["state"]==old},
      "probe-collisions":{"all_three_public_histories_identical":histories[0]==histories[1]==histories[2]==public["history"],
        "opening_alone_cannot_separate_profile_from_incentive":max(abs(a-b) for a,b in zip(opened[0],opened[2]))<1e-12,
        "all_probe_retains_capability_audience_ambiguity":all_rows[1]==all_rows[2] and actual["cause_posterior"][1]==actual["cause_posterior"][2],
        "future_can_share_behavior_across_distinct_causes":future_rows[1]==future_rows[2],
        "no_probe_perfect_cause_identification":actual["cause_entropy"]>0},
      "probe-information":{"independent_planner_and_mutual_information_agree":all(found),
        "cause_information_is_bounded":0<=information(all_rows)<=math.log(3)+1e-12,
        "future_information_is_bounded_by_cause_information":information(all_rows,future_rows)<=information(all_rows)+1e-12,
        "known_artifact_and_history_decline_inquiry":all(decision(public,policy)["query"] is None for policy in ["eig-artifact","eig-history"])},
      "probe-reference":{"same_information_direct_rival":actual==direct,
        "independent_probabilities":max(abs(a-b) for a,b in zip(actual["probabilities"],independent["probabilities"]))<1e-12,
        "independent_work_counts":actual["costs"]==independent["costs"]}}
    gates=[{"id":key,"checks":values,"instrument_state":"valid" if all(values.values()) else "failed","evidence_scope":"fixture"} for key,values in checks.items()]
    return {"gates":gates,"instrument_state":"valid" if all(row["instrument_state"]=="valid" for row in gates) else "failed"}

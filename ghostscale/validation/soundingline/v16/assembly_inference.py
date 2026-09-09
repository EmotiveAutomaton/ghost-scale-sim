"""Acquired W2 procedures and exact finite artifact-level predictive inference."""
from collections import defaultdict
from functools import lru_cache
import json
import math
import numpy as np
from .assembly import World,execute,legal_histories,fragments,plan
from .learning import encoding_cost


def training_recipes(world):
    return [(0,1,9),(0,6,1,9)]


def training_example(world,kind):
    intended=training_recipes(world)[kind if kind<2 else 0]
    target=execute(world,intended)["state"]
    program=intended if kind<2 else (9,0)  # actual failed post-stop action
    return {"program":list(program),"target":target}


def acquisition(world,*,rng,attempts,reliability,topic_probability):
    preferred=rng.randrange(2)
    records=[]
    for date in range(attempts):
        kind=preferred if rng.random()<topic_probability else 1-preferred
        if rng.random()>reliability:
            kind=2
        example=training_example(world,kind)
        records.append({"date":date,**example,"execution":execute(world,example["program"])})
    return {"records":records,"learned":fragments(records,world),"preferred_curriculum":preferred}


@lru_cache(maxsize=256)
def prior(world,attempts=8,reliability=0.9,topic_probability=0.8):
    weights=defaultdict(float)
    for a in range(attempts+1):
        for b in range(attempts-a+1):
            failed=attempts-a-b
            corpus=[training_example(world,kind) for kind,count in [(0,a),(1,b),(2,failed)] for _ in range(count)]
            library=tuple(tuple(fragment) for fragment in fragments(corpus,world)["library"])
            coefficient=math.factorial(attempts)/(math.factorial(a)*math.factorial(b)*math.factorial(failed))
            mass=0.0
            for preferred in range(2):
                p_a=reliability*(topic_probability if preferred==0 else 1-topic_probability)
                p_b=reliability-p_a
                mass+=0.5*coefficient*p_a**a*p_b**b*(1-reliability)**failed
            weights[library]+=mass
    libraries=tuple(sorted(weights))
    return libraries,tuple(weights[library] for library in libraries)


@lru_cache(maxsize=256)
def routes(world,max_steps):
    programs=tuple(legal_histories(world,max_steps))
    states=tuple(tuple(execute(world,program)["state"]) for program in programs)
    support=tuple(sorted(set(states)))
    indices=tuple(support.index(state) for state in states)
    return programs,states,support,indices


@lru_cache(maxsize=2048)
def kernel(world,library,target,max_steps=4,beta=1.0,length_cost=0.2):
    programs,states,support,indices=routes(world,max_steps)
    costs=np.array([encoding_cost(program,library) for program in programs],dtype=float)
    distance=np.array([sum(a!=b for a,b in zip(state,target)) for state in states],dtype=float)
    weights=np.exp(-beta*distance-length_cost*costs)
    weights/=weights.sum()
    artifacts=np.bincount(indices,weights=weights,minlength=len(support))
    return {"programs":programs,"support":support,"route_probabilities":weights,
            "artifact_probabilities":artifacts,"route_artifact_indices":indices}


def draw(world,library,target,*,rng,max_steps=4,beta=1.0,length_cost=0.2):
    distribution=kernel(world,tuple(map(tuple,library)),tuple(target),max_steps,beta,length_cost)
    index=rng.choices(range(len(distribution["programs"])),weights=distribution["route_probabilities"],k=1)[0]
    program=distribution["programs"][index]
    result=execute(world,program)
    return {"program":list(program),"execution":result,"artifact":result["artifact"],"state":result["state"]}


def decode(world,visible):
    if set(visible)!={"parts","relations"}:
        raise ValueError("assembly artifact schema violation")
    state=[-1]*3
    seen=set()
    for part,orientation in visible["parts"]:
        if type(part) is not int or type(orientation) is not int or part not in range(3) or orientation not in (0,1) or part in seen:
            raise ValueError("malformed assembly artifact part")
        seen.add(part)
        state[part]=orientation
    checked=execute(world,[],initial=tuple(state))
    if checked["artifact"]!=visible:
        raise ValueError("artifact relations disagree with public mechanics")
    return tuple(state)


PUBLIC_KEYS={"schema_version","task_id","world","training_contract","current","prior_works","future_goal",
             "max_steps","beta","length_cost","search_budget"}


def public_reader(payload:bytes,strategy="maker"):
    public=json.loads(payload)
    if set(public)!=PUBLIC_KEYS or public["schema_version"]!="v16.assembly-reader.1":
        raise ValueError("assembly reader schema violation")
    if strategy not in {"maker","generic","direct-table","primitive"}:
        raise ValueError("unknown assembly reader")
    world=World(tuple(public["world"]["parents"]),tuple(public["world"]["defaults"]))
    libraries,masses=prior(world,**public["training_contract"])
    current=public["current"]
    state=decode(world,current["artifact"])
    observations=[current] if strategy in {"generic","primitive"} else [*public["prior_works"],current]
    weights=np.array(masses,dtype=float)
    likelihood_evaluations=0
    for observation in observations:
        target=tuple(observation["goal"])
        visible=decode(world,observation["artifact"])
        likelihood=[]
        for library in libraries:
            distribution=kernel(world,library,target,public["max_steps"],public["beta"],public["length_cost"])
            likelihood.append(distribution["artifact_probabilities"][distribution["support"].index(visible)]
                              if visible in distribution["support"] else 0.0)
            likelihood_evaluations+=len(distribution["programs"])
        weights*=likelihood
        total=float(weights.sum())
        if total<=0:
            return {"model_mismatch":True,"reason":"artifact outside represented production support"}
        weights/=total
    future=[kernel(world,library,tuple(public["future_goal"]),public["max_steps"],public["beta"],public["length_cost"])
            for library in libraries]
    support=future[0]["support"]
    if strategy=="direct-table":
        # Separate joint-table contraction with the same public model/evidence.
        joint=np.array(masses,dtype=float)
        for observation in observations:
            evidence=decode(world,observation["artifact"])
            for index,library in enumerate(libraries):
                distribution=kernel(world,library,tuple(observation["goal"]),public["max_steps"],public["beta"],public["length_cost"])
                joint[index]*=distribution["artifact_probabilities"][distribution["support"].index(evidence)]
        table=np.array([item["artifact_probabilities"] for item in future])
        probabilities=joint@table/float(joint.sum())
    else:
        probabilities=sum(weight*item["artifact_probabilities"] for weight,item in zip(weights,future))
    chosen=libraries[int(np.argmax(weights))] if strategy!="primitive" else ()
    reconstruction=plan(world,state,library=chosen,budget=public["search_budget"],max_steps=public["max_steps"])
    collision=sum(candidate==state for candidate in routes(world,public["max_steps"])[1])
    return {"model_mismatch":False,"future_support":[list(item) for item in support],
            "future_probabilities":[float(value) for value in probabilities],
            "library_posterior":[float(value) for value in weights],"reconstruction":reconstruction,
            "history_collision_count":collision,"costs":{"likelihood_evaluations":likelihood_evaluations,
            "prior_work_queries":len(observations)-1,"search_successor_evaluations":reconstruction["successor_evaluations"]},
            "scope":"exact finite acquired-library model; observed execution need not identify historical route"}

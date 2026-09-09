"""K05: real allocation of executed acquisition attempts, with separate information."""
from collections import Counter
from itertools import combinations
import json
import random
from .records import seed_for
from .world import execute
from .craft import construct

DESIGN={"card_id":"K05","question":"Does allocated acquisition effort change later repertoire beyond instruction, feedback and opportunity exposure?",
        "conditions":[{"id":f"instruction-{instruction}-feedback-{feedback}-offers-{offers}",
                       "instruction":instruction,"feedback":feedback,"offers":offers}
                      for instruction in [False,True] for feedback in [False,True] for offers in [32,64]],
        "arms":["focal-effort","other-effort"],
        "primary":[("focal-effort","other-effort","focal_transfer","success_fraction",0.05)],
        "attention":"12/4 versus4/12 executed topic attempts, always16 total; allocation is recorded before outcomes",
        "exposure":"32/64 balanced topic-cue opportunities; unprocessed offers do not reveal full instructions or feedback",
        "instruction":"fallible suggested command pair; intended training target remains evaluator-private",
        "feedback":"optional accurate binary success; without it learner compresses executed repetition, never hidden successes",
        "learner":"one zero-arity2action fragment,>=3 eligible repeats; definition cost2",
        "skill_anchor":"fixed six two-cell targets at128 search; measured without excluding unequal-anchor makers",
        "transfer":"four untrained three-cell compositions at32 search, focal and foil reported separately",
        "scope":"equal broad two-cell anchor performance is not equal specialized competence or human general ability",
        "sample_rule":{"scout":64,"constructors":8,"expansion":256,"expansion_constructors":20,"final_expansion":1024},
        "dependencies":["attention-realization","attention-information","independent-attention-costs"],
        "adversaries":["X04","X07","X08"],"repair_budget":1,
        "continuation":"expand a named feedback/allocation or matched-anchor transfer boundary"}
EXTENSION="ghostscale.validation.soundingline.v16.attention_craft:public_read"
LEARNING_EXTENSION="ghostscale.validation.soundingline.v16.attention_craft:public_learn"
PRETEST=[sum(1<<cell for cell in pair) for pair in combinations(range(4),2)]


def learn_processed(processed):
    counts=Counter(tuple(item["program"]) for item in processed if item["feedback"] is not False)
    candidates=sorted((program for program,count in counts.items() if count>=3),key=lambda program:(-counts[program],program))
    return [list(candidates[0])] if candidates else []


def prepare(namespace,condition,index,constructors):
    constructor_id=f"constructor-{index%constructors:03d}"
    rng=random.Random(seed_for(namespace,"attention-constructor",constructor_id))
    permutation=list(range(4))
    rng.shuffle(permutation)
    teacher,student=rng.uniform(0.6,1.0),rng.uniform(0.7,0.95)
    motifs=[permutation[:2],permutation[2:]]
    topics=[0,1]*(condition["offers"]//2)
    random.Random(seed_for(namespace,index,"offer-order",condition["offers"])).shuffle(topics)
    offers=[]
    for offer,topic in enumerate(topics):
        rng=random.Random(seed_for(namespace,index,"offer",offer))
        instruction=list(motifs[topic])
        if rng.random()>teacher:
            instruction[rng.randrange(2)]=rng.randrange(8)
        offers.append({"offer":offer,"topic":topic,"instruction":instruction,
                       "intended_target":sum(1<<cell for cell in motifs[topic])})
    allocations={}
    for name,quotas in [("focal-effort",[12,4]),("other-effort",[4,12])]:
        for topic,quota in enumerate(quotas):
            topic_indices=[item["offer"] for item in offers if item["topic"]==topic][:quota]
            allocations.setdefault(name,[]).extend(topic_indices)
        allocations[name].sort()
    # Return the allocation before executing any chosen trial.
    return {"constructor_id":constructor_id,"constructor":{"permutation":permutation,
                "teacher_reliability":teacher,"student_reliability":student},"offers":offers,"allocations":allocations,
            "transfer_targets":[sum(1<<cell for cell in [*motifs[0],other]) for other in motifs[1]]+
                               [sum(1<<cell for cell in [*motifs[1],other]) for other in motifs[0]]}


def perform(prepared,condition,index,namespace):
    public={"schema_version":"v16.attention-craft.1","task_id":"pending transport",
            "offered_topics":[item["topic"] for item in prepared["offers"]],"allocations":prepared["allocations"],
            "instruction_access":condition["instruction"],"feedback_access":condition["feedback"],
            "processed":{},"pretest_targets":PRETEST,"transfer_targets":prepared["transfer_targets"],
            "pretest_budget":128,"transfer_budget":32}
    private_trials={}
    for name,allocation in prepared["allocations"].items():
        records=[]
        truth=[]
        for position in allocation:
            offer=prepared["offers"][position]
            rng=random.Random(seed_for(namespace,index,"attempt",position))
            proposed=list(offer["instruction"]) if condition["instruction"] else [rng.randrange(8),rng.randrange(8)]
            program=list(proposed)
            if rng.random()>prepared["constructor"]["student_reliability"]:
                program[rng.randrange(2)]=rng.randrange(8)
            result=execute(program)
            success=result.legal and result.artifact==offer["intended_target"]
            records.append({"offer":position,"topic":offer["topic"],
                "instruction":offer["instruction"] if condition["instruction"] else None,"proposed":proposed,
                "program":program,"artifact":result.artifact,"feedback":success if condition["feedback"] else None})
            truth.append({"offer":position,"intended_target":offer["intended_target"],"actual_success":success})
        public["processed"][name]=records
        private_trials[name]=truth
    return public,private_trials


def public_learn(payload:bytes):
    public=json.loads(payload)
    if set(public)!={"schema_version","task_id","offered_topics","allocations","instruction_access","feedback_access",
                    "processed"}:
        raise ValueError("attention public schema violation")
    if public["schema_version"]!="v16.attention-craft.1":
        raise ValueError("wrong attention schema")
    arms={}
    for name,records in public["processed"].items():
        for item in records:
            if set(item)!={"offer","topic","instruction","proposed","program","artifact","feedback"}:
                raise ValueError("unexpected acquisition observation field")
            if not public["feedback_access"] and item["feedback"] is not None:
                raise ValueError("unavailable feedback entered the learner")
        library=learn_processed(records)
        arms[name]={"library":library,
            "costs":{"training_primitives":sum(len(item["program"]) for item in records),
                     "instruction_queries":len(records)*int(public["instruction_access"]),
                     "feedback_queries":len(records)*int(public["feedback_access"]),"definition_cost":sum(map(len,library)),
                     "offered_opportunities":len(public["offered_topics"]),"processed_trials":len(records)}}
    return arms


def public_read(payload:bytes):
    public=json.loads(payload)
    if set(public)!={"schema_version","task_id","acquisition","pretest_targets","transfer_targets","pretest_budget","transfer_budget"}:
        raise ValueError("attention construction schema violation")
    if public["schema_version"]!="v16.attention-construction.1":
        raise ValueError("wrong attention construction schema")
    return {name:{**acquired,
            "pretest":[construct(target,acquired["library"],primitive_budget=public["pretest_budget"]) for target in public["pretest_targets"]],
            "transfer":[construct(target,acquired["library"],primitive_budget=public["transfer_budget"]) for target in public["transfer_targets"]]}
            for name,acquired in public["acquisition"].items()}

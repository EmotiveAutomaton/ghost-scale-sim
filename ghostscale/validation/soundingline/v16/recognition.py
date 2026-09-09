"""M01: independently acquired core order and visible decorative routines."""
from itertools import product
import json
import random
from .graphic_world import execute,learn
from .records import seed_for

METHODS=["craft","direct-table","surface-only","no-reference"]
EXTENSION="ghostscale.validation.soundingline.v16.recognition:public_reader"
DESIGN={
 "card_id":"M01",
 "question":"Does identifying a familiar maker improve prediction of production organization, or only visible style?",
 "mechanism":"independently acquired core-order and decorative routines recombine with current subject; two core orders collide at the final artifact",
 "strongest_rival":"direct joint predictive table using the same public finite production law and observations",
 "access_arms":"identical public references and anonymous artifacts/process probes; surface-only ignores process, no-reference ignores named references",
 "target_realization":"two routines per maker are acquired from fourteen executed trials; five primitive actions produce each new composition",
 "conditions":[{"id":f"{family}-{topic}-{surface}-dose-{dose}","family":family,"topic":topic,"surface":surface,"dose":dose}
   for family in ["learned-order","goal-indifferent"] for topic in ["same-topic","new-topic"]
   for surface in ["faithful","changed-purpose"] for dose in [0,1,3]],
 "arms":METHODS,
 "primary":[("craft","surface-only","future_core_log_score","nats_per_event",0.02),
            ("craft","direct-table","future_core_log_score","nats_per_event",0.02),
            ("craft","no-reference","identity_accuracy","success_fraction",0.05)],
 "secondary":["identity proper score","historical core-order score","future decoration score","acquired routine posterior",
              "training and primitive execution cost","reader likelihood operations","correction by evidence dose"],
 "generator_families":["W1 sixteen-cell bounded routine recombination; learned-order and goal-indifferent core control"],
 "paired_unit":"two independently trained reference makers, one anonymous source and paired predictions on a fresh source event",
 "sample_rule":{"scout":64,"constructors":8,"expansion":256,"expansion_constructors":20,"final_expansion":1024},
 "dependencies":["large-graphic-physics","large-graphic-acquisition","recognition-collision","recognition-data-use","independent-recognition-scoring"],
 "adversaries":["X01","X02","X03","X04","X05","X06","X07","X08"],"repair_budget":1,
 "continuation":"expand a named recognition/process separation or misleading-cue correction boundary; identification alone stays in its own ledger",
 "coverage_limit":"exact finite routine catalog on a larger board, not exhaustive inference over all thirty-two-to-the-sixth programs",
 "misleading_context":"changed-purpose reverses anonymous decorative tendency without disclosure; the reader's habitual-style channel is then misspecified",
 "cost_contract":"counted reference likelihood operations and executed primitives; logical operations do not establish equal total CPU cost"
}

def constructor(namespace,identifier,condition):
    rng=random.Random(seed_for(namespace,"constructor",identifier))
    permutation=list(range(16))
    rng.shuffle(permutation)
    return {"permutation":permutation,"training_reliability":rng.uniform(0.70,0.95),
            "style_reuse":rng.uniform(0.75,0.95),
            "core_reuse":rng.uniform(0.75,0.95) if condition["family"]=="learned-order" else 0.5}

def recipes(world,kind):
    p=world["permutation"]
    return [p[:2],list(reversed(p[:2]))] if kind=="core" else [p[4:6],p[6:8]]

def acquire(namespace,index,maker,world):
    acquired={}
    for kind in ["core","decoration"]:
        rng=random.Random(seed_for(namespace,"training",index,maker,kind))
        direction=rng.randrange(2)
        candidates=recipes(world,kind)
        training=[]
        for date in range(7):
            choice=direction if rng.random()<world["training_reliability"] else 1-direction
            program=candidates[choice]
            result=execute(program)
            training.append({"date":date,"program":program,"target":result["artifact"],
                             "feedback":result["legal"],"execution":result})
        compiled=learn(training)
        routine=compiled["library"][0]
        acquired[kind]={"training":training,"compiled":compiled,"routine":routine,
                        "choice":candidates.index(routine),"teaching_direction":direction}
    return acquired

def produce(world,acquired,rng,topic,*,changed=False):
    core=acquired["core"]["choice"]
    decoration=acquired["decoration"]["choice"]
    core=core if rng.random()<world["core_reuse"] else 1-core
    if changed:
        decoration=1-decoration
    decoration=decoration if rng.random()<world["style_reuse"] else 1-decoration
    core_program=recipes(world,"core")[core]
    decoration_program=recipes(world,"decoration")[decoration]
    p=world["permutation"]
    # Actual acquired definitions are expanded whenever that routine is reused.
    core_used=core_program==acquired["core"]["routine"]
    decoration_used=decoration_program==acquired["decoration"]["routine"]
    program=list(acquired["core"]["routine"] if core_used else core_program)+[p[2+topic]]+list(
        acquired["decoration"]["routine"] if decoration_used else decoration_program)
    return {"program":program,"execution":execute(program),"core_order":core,"decoration":decoration,
            "topic":topic,"core_routine_used":core_used,"decoration_routine_used":decoration_used}

def observe(production,*,process):
    return {"artifact":production["execution"]["artifact"],
            "first_action":production["program"][0] if process else None}

def decode(world,observation):
    if set(observation)!={"artifact","first_action"}:
        raise ValueError("undeclared recognition observation field")
    p=world["permutation"]
    board=observation["artifact"]
    if type(board) is not int or not 0<=board<65536:
        raise ValueError("invalid graphic artifact")
    occupied={i for i in range(16) if board&(1<<i)}
    styles=[set(p[4:6]),set(p[6:8])]
    matching=[style for style in [0,1] if any(occupied==set(p[:2])|{p[2+topic]}|styles[style] for topic in [0,1])]
    if len(matching)!=1:
        raise ValueError("artifact outside declared bounded production catalog")
    first=observation["first_action"]
    if first is not None and (type(first) is not int or first not in p[:2]):
        raise ValueError("invalid process prefix")
    return matching[0],None if first is None else p[:2].index(first)

def probability(observations,state,world,*,process=True):
    core,style=state
    likelihood=1.0
    operations=0
    for visible in observations:
        decoration,order=decode(world,visible)
        likelihood*=world["style_reuse"] if decoration==style else 1-world["style_reuse"]
        operations+=1
        if process and order is not None:
            likelihood*=world["core_reuse"] if order==core else 1-world["core_reuse"]
            operations+=1
    return likelihood,operations

def infer(public,method):
    world=public["world"]
    reference=public["references"] if method!="no-reference" else [[],[]]
    use_process=method!="surface-only"
    hypotheses=[]
    costs=0
    for ca,ea,cb,eb,identity in product([0,1],repeat=5):
        states=[(ca,ea),(cb,eb)]
        weight=1/32
        for who in [0,1]:
            likelihood,operations=probability(reference[who],states[who],world,process=use_process)
            weight*=likelihood
            costs+=operations
        likelihood,operations=probability(public["anonymous"],states[identity],world,process=use_process)
        weight*=likelihood
        costs+=operations
        hypotheses.append((states,identity,weight))
    evidence=sum(weight for _,_,weight in hypotheses)
    if evidence<=0:
        raise ValueError("recognition observation has zero model support")
    mass_identity=[0.0,0.0];mass_core=[0.0,0.0];mass_style=[0.0,0.0]
    future_core=[0.0,0.0];future_style=[0.0,0.0]
    for states,identity,weight in hypotheses:
        # The direct table accumulates unnormalized joint observable probabilities.
        mass=weight if method=="direct-table" else weight/evidence
        core,style=states[identity]
        mass_identity[identity]+=mass;mass_core[core]+=mass;mass_style[style]+=mass
        for result in [0,1]:
            future_core[result]+=mass*(world["core_reuse"] if result==core else 1-world["core_reuse"])
            future_style[result]+=mass*(world["style_reuse"] if result==style else 1-world["style_reuse"])
    if method=="direct-table":
        for values in [mass_identity,mass_core,mass_style,future_core,future_style]:
            values[:]=[value/evidence for value in values]
    current_order=decode(world,public["anonymous"][0])[1] if use_process else None
    historical=list(future_core) if current_order is None else [float(i==current_order) for i in [0,1]]
    return {"identity":mass_identity,"acquired_core":mass_core,"acquired_decoration":mass_style,
            "future_core":future_core,"future_decoration":future_style,"historical_core":historical,
            "identity_choice":0 if mass_identity[0]>=mass_identity[1]-1e-12 else 1,
            "costs":{"likelihood_terms":costs,"hypotheses":32,"prediction_terms":128}}

def public_reader(payload:bytes,method):
    public=json.loads(payload)
    if set(public)!={"schema_version","task_id","world","references","anonymous"} or public["schema_version"]!="v16.recognition.1":
        raise ValueError("recognition public schema violation")
    if method not in METHODS or set(public["world"])!={"permutation","style_reuse","core_reuse"}:
        raise ValueError("unregistered reader or world metadata")
    world=public["world"]
    if sorted(world["permutation"])!=list(range(16)) or any(type(i) is not int for i in world["permutation"]):
        raise ValueError("invalid public graphic interface")
    if any(not 0<world[name]<1 for name in ["style_reuse","core_reuse"]):
        raise ValueError("unsupported deterministic channel")
    if len(public["references"])!=2 or not public["anonymous"]:
        raise ValueError("two reference sources and an anonymous work are required")
    return infer(public,method)

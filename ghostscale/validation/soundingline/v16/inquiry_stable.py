"""R-family measurement amendment: stable numeric ties under output recoding.

The original inquiry.py is retained byte-for-byte. Command/program argmax and
zero-value abstention use an explicit binary64 tie resolution. Probability laws, world, query fees,
logical cost formulas, estimands and scientific criteria are unchanged.
"""
"""A competence-changing command-map learner and finite inquiry policies.

The learner sees executed command/output examples. It never receives the true
mapping or an evaluator's held-out correctness. Two-cell goal construction is
evaluated independently after practice.
"""
from functools import lru_cache
from itertools import permutations,combinations
import json
import math
import numpy as np

PERMUTATIONS=tuple(permutations(range(4)))
COMMAND_PAIRS=tuple(combinations(range(4),2))
GOALS=tuple(sum(1<<cell for cell in pair) for pair in combinations(range(4),2))
PRIOR=tuple([0.9/24]*24+[0.1])
LIKELIHOOD=np.zeros((4,4,25),dtype=float)
PAIR_SUCCESS=np.zeros((6,6,25),dtype=float)
for command in range(4):
    for cell in range(4):
        LIKELIHOOD[command,cell,:24]=[float(mapping[command]==cell) for mapping in PERMUTATIONS]
        LIKELIHOOD[command,cell,24]=0.25
for goal_index,goal in enumerate(GOALS):
    for pair_index,pair in enumerate(COMMAND_PAIRS):
        PAIR_SUCCESS[goal_index,pair_index,:24]=[
            float(sum(1<<mapping[command] for command in pair)==goal) for mapping in PERMUTATIONS]
        # In the irreducible-noise ecology each executed primitive has an
        # independently random physical output: two orders among 16 outcomes.
        PAIR_SUCCESS[goal_index,pair_index,24]=0.125


def update(belief,command,cell):
    if type(command) is not int or type(cell) is not int or command not in range(4) or cell not in range(4):
        raise ValueError("unsupported public training observation")
    weighted=np.asarray(belief)*LIKELIHOOD[command,cell]
    total=float(weighted.sum())
    if total<=0:
        raise ValueError("impossible command-map evidence")
    return tuple(float(x) for x in weighted/total)


def prediction(belief,command):
    return [float(np.dot(belief,LIKELIHOOD[command,cell])) for cell in range(4)]


TIE_RESOLUTION = 64*math.ulp(1.0)


def stable_max_index(values):
    if not len(values) or any(not math.isfinite(float(value)) for value in values):
        raise ValueError("finite nonempty action values required")
    maximum = max(values)
    return next(index for index, value in enumerate(values) if maximum-value <= TIE_RESOLUTION)


def command_for_goal(belief,target_cell):
    values = [float(np.dot(belief, LIKELIHOOD[command,target_cell])) for command in range(4)]
    return stable_max_index(values)


def construction(belief,goal):
    index=GOALS.index(goal)
    values=PAIR_SUCCESS[index]@np.asarray(belief)
    return list(COMMAND_PAIRS[stable_max_index(values)])


def competence(belief):
    values=PAIR_SUCCESS@np.asarray(belief)
    return float(np.max(values,axis=1).mean())


def entropy(belief):
    return -sum(p*math.log(p) for p in belief if p>0)


@lru_cache(maxsize=4096)
def expected_after(belief,commands):
    """Exact expected competence and entropy after a public sequence of queries."""
    if not commands:
        return competence(belief),entropy(belief),925
    expected_skill,expected_entropy,evaluations=0.0,0.0,100
    for cell,probability in enumerate(prediction(belief,commands[0])):
        if probability==0:
            continue
        posterior=update(belief,commands[0],cell)
        skill,uncertainty,spent=expected_after(posterior,commands[1:])
        expected_skill+=probability*skill
        expected_entropy+=probability*uncertainty
        evaluations+=25+spent
    return expected_skill,expected_entropy,evaluations


def experienced_progress(history):
    successes=[float(item["success"]) for item in history]
    if not successes:
        return 0.25  # frozen optimistic initialization, not hidden learning truth
    recent=sum(successes[-2:])/len(successes[-2:])
    older=sum(successes[-4:-2])/len(successes[-4:-2]) if len(successes)>2 else 0.25
    return recent-older


PUBLIC_KEYS={"schema_version","task_id","beliefs","offers","histories","familiarity",
             "pending_commands","feedback_batches","future_weights","opportunity_cost",
             "remaining_interactions","tie_draw","committed_domain","offer_probabilities"}


def choose(payload:bytes,policy):
    public=json.loads(payload)
    if set(public)!=PUBLIC_KEYS or public["schema_version"]!="v16.inquiry.1":
        raise ValueError("inquiry public schema violation")
    allowed={"recognition","surprise","eig","signed-progress","absolute-progress",
             "uniform","decline","value-learning"}
    if policy not in allowed:
        raise ValueError("unknown inquiry policy")
    if policy=="decline":
        return {"domain":None,"commands":None,"scores":[0.0,0.0],"logical_model_evaluations":0,
                "commit_next":False}
    if public["committed_domain"] is not None:
        domain=public["committed_domain"]
        return {"domain":domain,"commands":command_for_goal(public["beliefs"][domain],public["offers"][domain]),
                "scores":[0.0,0.0],"logical_model_evaluations":100,"commit_next":False}
    scores=[]
    commands=[]
    evaluations=200  # four target probabilities x 25 hypotheses, for each of two domains
    for domain in range(2):
        belief=tuple(public["beliefs"][domain])
        command=command_for_goal(belief,public["offers"][domain])
        commands.append(command)
        pending=tuple(public["pending_commands"][domain])
        batch=public["feedback_batches"][domain]
        next_commands=pending+(command,)
        immediate=len(next_commands)>=batch
        base_skill,base_entropy=0.0,0.0
        if policy in {"eig","recognition","value-learning"}:
            base_skill,base_entropy=competence(belief),entropy(belief)
            evaluations+=925
        if immediate and policy in {"eig","recognition","value-learning"}:
            after,uncertainty,spent=expected_after(belief,next_commands)
        else:
            after,uncertainty,spent=base_skill,base_entropy,0
        evaluations+=spent
        gain=after-base_skill
        information=base_entropy-uncertainty
        if policy=="surprise":
            # Experienced surprise uses only returned training feedback.
            history=public["histories"][domain]
            score=history[-1]["surprise"] if history else math.log(4)
        elif policy in {"signed-progress","absolute-progress"}:
            progress=experienced_progress(public["histories"][domain])
            score=abs(progress) if policy=="absolute-progress" else progress
        elif policy=="recognition":
            score=(1+public["familiarity"][domain])/5*max(0.0,information)
        elif policy=="eig":
            score=max(0.0,information)
        elif policy=="uniform":
            score=1.0
        else:
            if not immediate and public["remaining_interactions"]>=2:
                # Delay is a public delivery contract. The second command is chosen
                # before its unreturned first outcome; no hidden feedback is used.
                alternatives=[]
                for future_goal in range(4):
                    second=command_for_goal(belief,future_goal)
                    skill,_,extra=expected_after(belief,next_commands+(second,))
                    evaluations+=100+extra
                    alternatives.append((skill-base_skill)/2)
                gain=sum(value*weight for value,weight in
                         zip(alternatives,public["offer_probabilities"][domain]))
            score=public["future_weights"][domain]*max(0.0,gain)-public["opportunity_cost"]
        scores.append(float(score))
    if policy=="value-learning" and max(scores)<=TIE_RESOLUTION:
        domain=None
    else:
        maximum=max(scores)
        tied=[index for index,score in enumerate(scores) if abs(score-maximum)<1e-12]
        domain=tied[min(int(public["tie_draw"]*len(tied)),len(tied)-1)]
    return {"domain":domain,"commands":None if domain is None else commands[domain],
            "scores":scores,"logical_model_evaluations":evaluations,
            "commit_next":domain is not None and policy=="value-learning" and
                          len(public["pending_commands"][domain])+1<public["feedback_batches"][domain] and
                          public["remaining_interactions"]>=2}


def learn_public(payload:bytes):
    public=json.loads(payload)
    if set(public)!={"beliefs","examples","forget_domains"}:
        raise ValueError("public learning schema violation")
    beliefs=[tuple(belief) for belief in public["beliefs"]]
    for domain in public["forget_domains"]:
        beliefs[domain]=PRIOR
    for example in public["examples"]:
        if set(example)!={"domain","command","cell"}:
            raise ValueError("only returned command/output examples enter learning")
        domain=example["domain"]
        beliefs[domain]=update(beliefs[domain],example["command"],example["cell"])
    return {"beliefs":[list(belief) for belief in beliefs],
            "logical_model_evaluations":25*len(public["examples"])}


def construct_public(payload:bytes):
    public=json.loads(payload)
    if set(public)!={"beliefs"} or len(public["beliefs"])!=2:
        raise ValueError("public construction schema violation")
    return {"programs":[{"domain":domain,"goal":goal,"commands":construction(belief,goal)}
                        for domain,belief in enumerate(public["beliefs"]) for goal in GOALS],
            "logical_model_evaluations":2*6*6*25}


def read_maker_public(payload:bytes):
    public=json.loads(payload)
    if set(public)!={"belief","artifact"}:
        raise ValueError("new-maker public schema violation")
    return {"future_probabilities":future_maker_prediction(public["belief"],public["artifact"]),
            "logical_model_evaluations":50}


def future_maker_prediction(belief,artifact):
    """New makers learn one of two domain motifs; predict its first primitive."""
    mass=[0.0]*4
    for weight,mapping in zip(belief[:24],PERMUTATIONS):
        for pair in ((0,1),(2,3)):
            if (1<<mapping[pair[0]])|(1<<mapping[pair[1]])==artifact:
                mass[mapping[pair[0]]]+=weight/2
    occupied=artifact.bit_count()
    artifact_noise=1/16 if occupied==1 else 2/16 if occupied==2 else 0
    for cell in range(4):
        mass[cell]+=belief[24]*artifact_noise/4
    total=sum(mass)
    if total==0:
        raise ValueError("maker artifact outside the represented support")
    return [value/total for value in mass]

"""A extension: common counted search across graphic and assembly representations."""
from collections import Counter
import heapq
from itertools import product
import random
from ..v16 import assembly,graphic_world
from ..v16.records import digest,seed_for
from . import craft,stitch_adapter as stitch
from .contracts import Costs,BudgetExhausted
from .programs import execute as graphic_execute
METHODS=("primitive_search","repeated_fragments","episodic_adaptation","stitch_abstractions","abstractions_and_exceptions")
REGIMES=tuple(w+":"+r for w in ("graphic","assembly") for r in craft.REGIMES)
PILOT_BUDGETS=(8,32,128,512,2048,8192)


def execute(public,program,initial=None):
    state=public["initial"] if initial is None else initial
    if public["world_kind"]=="graphic":
        run=graphic_execute(program,initial=state,forbidden=public["forbidden"],max_steps=public["max_steps"])
        return dict(state=run["artifact"],legal=run["legal"],primitive_cost=run["primitive_cost"],trace=run["trace"])
    world=assembly.World(tuple(public["world"]["parents"]),tuple(public["world"]["defaults"]))
    current=tuple(state)
    stopped=False
    trace=[]
    for action in program:
        before=current
        if action in public["forbidden"]:
            after,stop,legal=current,stopped,False
        else:
            after,stop,legal=assembly.step(world,current,stopped,action)
        trace.append(dict(before=list(before),action=action,after=list(after),legal=legal,stopped=stop))
        if not legal: return dict(state=list(current),legal=False,primitive_cost=len(trace),trace=trace)
        current,stopped=after,stop
    return dict(state=list(current),legal=True,primitive_cost=len(trace),trace=trace)


def make_case(namespace,constructor_index,history_index,regime):
    world_kind,condition=regime.split(":")
    rng=random.Random(seed_for(namespace,constructor_index,history_index,"craft-extension"))
    if world_kind=="graphic":
        original=craft.make_case(namespace,constructor_index,history_index,condition)
        public=original["public"]
        cells=original["private"]["constructor"]["cells"]
        structure=constructor_index%3
        if structure:
            motifs=([cells[0],c] for c in cells[1:4]) if structure==1 else (
                [cells[0],cells[1],c] for c in cells[2:5])
            training=[]
            for motif in motifs:
                for _ in range(4):
                    program=list(motif)
                    if history_index%2: program.reverse()
                    training.append(dict(initial=0,program=program,target=graphic_execute(program)["artifact"],forbidden=[],feedback=True))
            public["training"]=training
            targets=cells[:4] if structure==1 else cells[:5]
            if condition=="new_combinations": targets=[cells[0],cells[3],cells[6]]
            public["target"]=sum(1<<c for c in targets)
        public.update(world_kind="graphic",world={},max_steps=6)
        structural="graphic-"+str(structure)
    else:
        world=assembly.constructor(namespace,constructor_index)
        target=list(world.defaults)
        initial=list(assembly.EMPTY)
        forbidden=[]
        if condition=="new_combinations": target[2]=1-target[2]
        if condition=="changed_constraint":
            initial[0]=world.defaults[0]
            forbidden=[0]
        training=[]
        # Legal partial assemblies, never the full held-out target.
        for program in assembly.legal_histories(world,max_steps=4):
            result=assembly.execute(world,program)
            if result["state"]==target or all(v>=0 for v in result["state"]): continue
            training.append(dict(initial=list(assembly.EMPTY),program=list(program[:-1]),target=result["state"],forbidden=[]))
        rng.shuffle(training)
        training=(training[:4]*4)[:16]
        order=list(range(9))
        rng.shuffle(order)
        public=dict(schema="v17.craft-extension.1",world_kind="assembly",world=world.public(),training=training,
            initial=initial,target=target,forbidden=forbidden,max_steps=8,action_order=order)
        structural="assembly-"+str(world.parents)
    rng.shuffle(public["training"])
    public["schema"]="v17.craft-extension.1"
    if history_index%2: public["action_order"].reverse()
    if any(t["target"]==public["target"] for t in public["training"]):
        raise ValueError("held-out acquisition target leaked")
    for example in public["training"]:
        example.setdefault("feedback",True)
    witness = [cell for cell in range(16) if (public["initial"] ^ public["target"]) & (1 << cell) and public["target"] & (1 << cell)] if world_kind=="graphic" else assembly.plan(world,public["target"],initial=public["initial"],budget=100000,max_steps=8)["program"]
    witness_run=execute(public,witness)
    if not witness_run["legal"] or witness_run["state"]!=public["target"]:
        raise ValueError("unreachable held-out construction")
    private=dict(maker_training=public["training"],current_goal=public["target"],
        earlier_goals=[t["target"] for t in public["training"]],actual_constraints=public["forbidden"],
        believed_constraints=public["forbidden"],considered_options=public["action_order"],
        realized_history=witness_run["trace"],contributor_role="single maker",recipient_state=None)
    return dict(case_id=digest([namespace,constructor_index,history_index,regime]),
        constructor_id=digest([namespace,constructor_index]),history_id=digest([namespace,constructor_index,history_index]),
        namespace=namespace,family="A2",regime=regime,public=public,private=private,
        public_problem_sha256=digest(public),private_construction_sha256=digest([public,structural]),
        structural_family=structural)


def acquire(public,method,memory_cap,learned):
    costs=Costs()
    training=public["training"]
    costs.charge("training_acquisition",sum(len(t["program"]) for t in training))
    definitions=[]
    fragments=[]
    episodes=[]
    if method=="repeated_fragments":
        if public["world_kind"]=="graphic":
            baseline=graphic_world.learn(training)
            fragments=baseline["library"]
            costs.charge("learning",baseline["processing_primitives"]+len(training))
        else:
            world=assembly.World(tuple(public["world"]["parents"]),tuple(public["world"]["defaults"]))
            baseline=assembly.fragments([dict(t,program=t["program"]+[9]) for t in training],world)
            fragments=baseline["library"]
            costs.charge("learning",baseline["training_primitives"]+len(training))
        retained=[]
        for fragment in fragments:
            size=2*len(fragment)+1
            if costs.values["definition_storage"]+size<=memory_cap:
                retained.append(fragment)
                costs.charge("definition_storage",size)
        fragments=retained
    if method in ("stitch_abstractions","abstractions_and_exceptions"):
        costs.charge("learning",learned["learning_operations"]+learned["input_tokens"])
        for definition in learned["accepted"]:
            if costs.values["definition_storage"]+definition["storage_tokens"]<=memory_cap:
                definitions.append(definition)
                costs.charge("definition_storage",definition["storage_tokens"])
            else:
                # Prefix retention preserves references to earlier definitions.
                break
    if method in ("episodic_adaptation","abstractions_and_exceptions"):
        seen=set()
        # Hybrid: train-only residual episodes, shortest first then acquisition order.
        # A residual is not exactly reproduced by a retained zero-arity procedure.
        closed={tuple(stitch.expand(d["body"],{x["name"]:(x["body"],x["arity"]) for x in definitions},
                  world=public["world_kind"])) for d in definitions if d["arity"]==0}
        for t in sorted(training,key=lambda t:len(t["program"])):
            costs.charge("learning",len(t["program"])+1)
            identity=digest(t)
            if identity in seen or (method=="abstractions_and_exceptions" and tuple(t["program"]) in closed): continue
            seen.add(identity)
            size=2*len(t["program"])+3+len(t["forbidden"])
            if costs.values["definition_storage"]+size<=memory_cap:
                episodes.append(t["program"])
                costs.charge("definition_storage",size)
    return dict(costs=costs.values,definitions=definitions,fragments=fragments,episodes=episodes,
                memory_cap=memory_cap)


def proposals(state,public,rep,costs):
    candidates=[]
    for action in public["action_order"]:
        costs.charge("proposal_generation")
        candidates.append([action])
    for fragment in rep["fragments"]:
        costs.charge("retrieval",len(fragment))
        costs.charge("proposal_generation")
        candidates.append(fragment)
    definitions={d["name"]:(d["body"],d["arity"]) for d in rep["definitions"]}
    domain=range(16 if public["world_kind"]=="graphic" else 3)
    for definition in rep["definitions"]:
        costs.charge("retrieval",definition["storage_tokens"])
        for args in product(domain,repeat=definition["arity"]):
            costs.charge("argument_binding",len(args)+1)
            costs.charge("proposal_generation")
            candidates.append(stitch.expand(definition["body"],definitions,args,public["world_kind"]))
    for episode in rep["episodes"]:
        costs.charge("retrieval",2*len(episode)+3)
        adapted=[]
        for action in episode:
            costs.charge("argument_binding")
            if action in public["forbidden"]: continue
            if public["world_kind"]=="graphic":
                cell=action%16
                desired=bool(public["target"]&(1<<cell))
                present=bool(state&(1<<cell))
                keep=present!=desired and (action<16)==desired
            else:
                part=action%3
                keep=(action<3 and state[part]<0 and public["target"][part]>=0) or (
                    3<=action<6 and state[part]>=0 and public["target"][part]<0) or (
                    action>=6 and state[part]>=0 and state[part]!=public["target"][part])
            if keep: adapted.append(action)
        costs.charge("proposal_generation")
        if adapted: candidates.append(adapted)
    rank={a:i for i,a in enumerate(public["action_order"])}
    costs.charge("selection",len(candidates))
    return sorted((p for p in candidates if p),key=lambda p:(rank[p[0]],len(p),p))


def distance(public,state):
    return (state^public["target"]).bit_count() if public["world_kind"]=="graphic" else assembly.mismatch(state,public["target"])


def solve(public,rep,budget):
    costs=Costs(dict(rep["costs"]),search_budget=budget)
    queue=[(distance(public,public["initial"]),0,public["initial"],[])]
    seen={digest(public["initial"])}
    serial=0
    selected=None
    attempted=failed=0
    try:
        while queue:
            costs.charge("selection")
            _,_,state,program=heapq.heappop(queue)
            if distance(public,state)==0:
                selected=program
                break
            for fragment in proposals(state,public,rep,costs):
                if len(program)+len(fragment) > public["max_steps"]-(public["world_kind"]=="assembly"):
                    failed+=1
                    continue
                current=state
                legal=True
                for action in fragment:
                    costs.charge("hypothetical_execution")
                    transition=execute(public,[action],current)
                    attempted+=1
                    current=transition["state"]
                    if not transition["legal"]:
                        failed+=1
                        legal=False
                        break
                costs.charge("selection")
                if not legal: continue
                proposal=program+fragment
                if distance(public,current)==0:
                    selected=proposal
                    break
                identity=digest(current)
                if identity not in seen:
                    seen.add(identity)
                    serial+=1
                    heapq.heappush(queue,(distance(public,current),serial,current,proposal))
            if selected is not None: break
    except BudgetExhausted:
        pass
    actual=None
    if selected is not None:
        if public["world_kind"]=="assembly": selected=selected+[9]
        actual=execute(public,selected)
        costs.charge("actual_execution",actual["primitive_cost"])
    return dict(program=selected,execution=actual,costs=costs.receipt(16),
        task_success=actual is not None and actual["legal"] and actual["state"]==public["target"],
        missing_output=selected is None,invalid_program=actual is not None and not actual["legal"],
        attempted_primitives=attempted,failed_candidates=failed)


def evaluate_case(case,design):
    public=case["public"]
    learned=stitch.learn(public["training"],public["world_kind"])
    rows=[]
    for memory_cap in design.get("memory_caps",[32]):
        for method in METHODS:
            rep=acquire(public,method,memory_cap,learned)
            for budget in design.get("budgets_by_world",{}).get(public["world_kind"],design["budgets"]):
                row=solve(public,rep,budget)
                rows.append(dict(row,method=method+"@"+str(memory_cap),target="construction_budget_"+str(budget),
                    budget=budget,memory_cap=memory_cap,evidence_tier="maker acquisition and supplied construction goal",
                    representation={k:v for k,v in rep.items() if k!="costs"},
                    stitch_learning=learned if method in ("stitch_abstractions","abstractions_and_exceptions") else None))
    return rows

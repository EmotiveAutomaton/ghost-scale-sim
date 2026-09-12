"""E: finite native maker hypotheses, explicit artifact tiers and unseen choices.

These new counterfactual fixtures use A-D operations in a declared finite family.
They are not a re-export of V16 cases or evidence of unique historical recovery.
"""
import copy
from itertools import product
import random
from ..v16 import assembly
from ..v16.records import digest,seed_for
from . import recipient as b,adaptive as c,revision as d,craft_extension as a
from .contracts import Costs,COST_KINDS,forecast_scores
from .programs import execute as graphic_execute
REGIMES=("A","B","C","D")
METHODS=("surface_continuation","episodic_continuation","inverse_maker")
TIERS=("artifact","artifact_collection","recorded_process")


def hypotheses(context):
    methods={"A":("primitive_search","repeated_fragments","episodic_adaptation"),
             "B":("physical","communicator","producer_editor"),
             "C":("cached","retrieval","inverse_3","inverse_full"),
             "D":("local_edit","dependency_aware")}
    return [dict(goal=goal,method=method) for goal,method in product(range(2),methods[context["family"]])]


def forward(context,hypothesis,phase):
    """Execute one possible maker; called with public context and a hypothesis only."""
    task=copy.deepcopy(context["tasks"][phase])
    family=context["family"]
    goal=hypothesis["goal"]
    if family=="A":
        cells=context["cells"]
        task["target"]=sum(1<<x for x in (cells[:3] if goal==0 else cells[1:4]))
        training=[]
        for pair in (cells[:2],cells[1:3],cells[2:4]):
            for _ in range(4):
                training.append(dict(initial=0,program=pair,target=sum(1<<x for x in pair),forbidden=[],feedback=True))
        task["training"]=training
        rep=a.acquire(task,hypothesis["method"],32,{})
        result=a.solve(task,rep,512)
        run=result["execution"]
        artifact=run["state"] if run is not None else task["initial"]
        choice=str(result["program"][0]) if result["program"] else "MISSING"
        process=dict(actions=run["trace"] if run else [],missing=result["missing_output"])
        return dict(artifact=artifact,choice=choice,process=process,costs=result["costs"])
    if family=="B":
        n=len(task["world"]["templates"])
        desired=[float(k==goal) for k in range(n)]
        # The convention intervention is public; its source was fixed independently.
        model=dict(order=list(range(n)) if phase<3 else list(reversed(range(n))),precision=1.2,prior=[1/n]*n)
        costs=Costs()
        opportunities=b.opportunities(task["artifact"],task["world"])
        probabilities=b.accepted_distribution(opportunities,task["world"],desired,[(model,1)],
            hypothesis["method"],"correct",costs)
        chosen=max(range(len(probabilities)),key=probabilities.__getitem__)
        candidate=opportunities[chosen]
        run=graphic_execute(candidate["program"],initial=task["artifact"])
        costs.charge("actual_execution",run["primitive_cost"])
        response=b.recipient(run["artifact"],task["world"],model,costs)
        return dict(artifact=run["artifact"],choice=candidate["id"],
            process=dict(actions=run["trace"],recipient_response=max(range(n),key=response.__getitem__)),
            costs=costs.receipt(16))
    if family=="C":
        task["task"]["shared_brief"]=[float(k==goal) for k in range(2)]
        result=c.option_prediction(task,hypothesis["method"])
        probabilities=result["recipient"]
        choice=max(probabilities,key=probabilities.get)
        # A forecast is rendered as an actually executed two-cell display.
        run=graphic_execute([context["cells"][int(choice)]])
        costs=Costs({k:result["costs"][k] for k in COST_KINDS})
        costs.charge("actual_execution",run["primitive_cost"])
        return dict(artifact=run["artifact"],choice=choice,
            process=dict(actions=run["trace"],represented_variables=result["represented_variables"],
                         evaluated_hypotheses=result["hypotheses"]),
            costs=costs.receipt(16))
    task["remembered_goal"]=goal
    result=d.predict(task,hypothesis["method"])
    program=task["candidates"][result["chosen"]]["program"]
    return dict(artifact=d.encode(result["execution"]["state"]),choice=str(program[0]),
        process=dict(actions=result["execution"]["trace"]),costs=result["costs"])


def make_case(namespace,constructor_index,history_index,regime):
    if regime not in REGIMES: raise ValueError("unknown observer family")
    rng=random.Random(seed_for(namespace,constructor_index,history_index,regime,"observer"))
    cells=rng.sample(range(16),6)
    if regime=="A":
        tasks=[]
        for phase in range(4):
            initial=0 if phase<3 else 1<<cells[0]
            order=list(range(32)); rng.shuffle(order)
            tasks.append(dict(schema="v17.craft-extension.1",world_kind="graphic",world={},initial=initial,
                forbidden=[] if phase<3 else [cells[0]],max_steps=6,action_order=order))
        support=[str(k) for k in range(32)]+["MISSING"]
    elif regime=="B":
        templates=[sum(1<<x for x in cells[:2]),sum(1<<x for x in cells[2:4])]
        world=dict(templates=templates,visible=sum(1<<x for x in cells))
        tasks=[dict(world=world,artifact=0 if phase%2==0 else 1<<cells[0]) for phase in range(4)]
        support=[x["id"] for x in b.opportunities(tasks[3]["artifact"],world)]
    elif regime=="C":
        original=c.make_case(namespace+"-decoder",constructor_index*2,history_index,"misleading_old_cue")["public"]
        original["task"].pop("shared_brief")
        tasks=[]
        for phase,prefix in enumerate((1,4,8,12)):
            task=copy.deepcopy(original)
            task["task"]["calibration"]=task["task"]["calibration"][:prefix]
            task["resource_context"]["has_current_access"]=phase<2
            tasks.append(task)
        support=["0","1"]
    else:
        original=d.make_case(namespace+"-assembly",constructor_index,history_index,"stable")["public"]
        original.pop("remembered_goal")
        tasks=[]
        for phase in range(4):
            task=copy.deepcopy(original)
            if phase==3:
                task["edit_price"]=.03
            else:
                task["edit_price"]=(.03,.15,.4)[phase]
            tasks.append(task)
        world=assembly.World(tuple(original["world"]["parents"]),tuple(original["world"]["defaults"]))
        support=[str(action) for action in assembly.ACTIONS if assembly.step(world,tuple(original["initial"]),False,action)[2]]
    context=dict(family=regime,cells=cells,tasks=tasks,
        known_family="finite native maker-policy alternatives; actual method and goal withheld",
        intervention="new tool constraint" if regime=="A" else "recipient convention reverses" if regime=="B"
            else "new calibration observations and lost access" if regime=="C" else "changed cost of further revision")
    possible=hypotheses(context)
    true_index=rng.randrange(len(possible))
    actual=[forward(context,possible[true_index],phase) for phase in range(4)]
    if actual[3]["choice"] not in support:
        raise ValueError("future native action outside declared answer support")
    public=dict(schema="v17.observer.1",target_kind="unseen_native_choice",answer_support=support,query_cost=0,
        context=context,artifact=actual[2]["artifact"],earlier_artifacts=[x["artifact"] for x in actual[:2]],
        recorded_process=actual[2]["process"])
    private=dict(true_hypothesis_index=true_index,true_hypothesis=possible[true_index],
        future_choice=actual[3]["choice"],realized_history=actual,maker_training="generated within each declared candidate maker",
        current_goal=possible[true_index]["goal"],earlier_goals=[possible[true_index]["goal"]]*3,
        actual_constraints=context["intervention"],believed_constraints=context["intervention"],
        considered_options=support,contributor_role=regime,recipient_state="inside realized native hypotheses")
    return dict(case_id=digest([namespace,constructor_index,history_index,regime]),
        constructor_id=digest([namespace,constructor_index]),history_id=digest([namespace,constructor_index,history_index]),
        namespace=namespace,family="E",regime=regime,public=public,private=private,
        public_problem_sha256=digest(public),private_construction_sha256=digest([context,possible[true_index]]),
        structural_family="finite-native-"+regime)


def project(public,tier):
    if tier not in TIERS: raise ValueError("unknown observer tier")
    result={k:copy.deepcopy(public[k]) for k in ("schema","target_kind","answer_support","query_cost","context","artifact")}
    result["evidence_tier"]=tier
    if tier!="artifact":
        result["earlier_artifacts"]=copy.deepcopy(public["earlier_artifacts"])
    if tier=="recorded_process":
        result["recorded_process"]=copy.deepcopy(public["recorded_process"])
    return result


def predict(request,method):
    required={"schema","target_kind","answer_support","query_cost","context","artifact","evidence_tier"}
    tier=request["evidence_tier"]
    if tier!="artifact": required.add("earlier_artifacts")
    if tier=="recorded_process": required.add("recorded_process")
    if set(request)!=required or request["schema"]!="v17.observer.1":
        raise ValueError("observer explicit evidence schema violation")
    if method not in METHODS: raise ValueError("unknown observer method")
    support=request["answer_support"]
    costs=Costs()
    forecasts=[]
    weights=[]
    compatible=[]
    for hypothesis in hypotheses(request["context"]):
        # The cheaper comparators assume one policy per candidate goal.
        if method!="inverse_maker" and hypothesis["method"]!=hypotheses(request["context"])[0]["method"]:
            continue
        current=forward(request["context"],hypothesis,2)
        future=forward(request["context"],hypothesis,3)
        for result in (current,future):
            for kind in COST_KINDS: costs.charge("hypothetical_execution" if kind in ("actual_execution","training_acquisition") else kind,result["costs"][kind])
        mismatch=(current["artifact"]^request["artifact"]).bit_count()
        matches=current["artifact"]==request["artifact"]
        if tier!="artifact" and method!="surface_continuation":
            for phase,observed in enumerate(request["earlier_artifacts"]):
                past=forward(request["context"],hypothesis,phase)
                for kind in COST_KINDS: costs.charge("hypothetical_execution" if kind in ("actual_execution","training_acquisition") else kind,past["costs"][kind])
                mismatch+=(past["artifact"]^observed).bit_count()
                matches=matches and past["artifact"]==observed
        if tier=="recorded_process" and method=="inverse_maker":
            matches=matches and current["process"]==request["recorded_process"]
        weights.append(float(matches) if method=="inverse_maker" else 1/(1+mismatch))
        forecasts.append(future["choice"])
        compatible.append(matches)
        costs.charge("selection")
    if not sum(weights): raise ValueError("declared maker family cannot explain retained evidence")
    weights=b.normalize(weights)
    probabilities={label:0.0 for label in support}
    for label,weight in zip(forecasts,weights):
        probabilities[label]+=weight
    return dict(probabilities=probabilities,costs=costs.receipt(16),
        compatible_histories=sum(compatible),represented_hypotheses=len(weights),
        represented_variables=["possible_goal","native_maker_policy"] if method=="inverse_maker" else ["artifact_similarity"])


def evaluate_case(case,design=None):
    rows=[]
    for tier in TIERS:
        request=project(case["public"],tier)
        for method in METHODS:
            result=predict(request,method)
            p=result["probabilities"]
            truth=case["private"]["future_choice"]
            rows.append(dict(method=method,target="unseen_choice_"+tier,probabilities=p,
                **forecast_scores(p,request["answer_support"],truth),costs=result["costs"],evidence_tier=tier,
                compatible_histories=result["compatible_histories"],represented_hypotheses=result["represented_hypotheses"],
                represented_variables=result["represented_variables"],missing_output=False,invalid_program=False,
                task_success=max(p,key=p.get)==truth,history_unique=result["compatible_histories"]==1))
    return rows

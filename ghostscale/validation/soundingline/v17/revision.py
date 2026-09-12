"""D: actual assembly dependencies, paid feedback, reconsideration and stopping."""
import random
from ..v16 import assembly
from ..v16.records import digest, seed_for
from .contracts import Costs,forecast_scores
from . import recipient as b
REGIMES=("stable","revised","stale","uninformative_feedback","misleading_feedback")
METHODS=("local_edit","dependency_aware","internal_reconsideration","recipient_feedback","complete_error_monitor","stop_always")


def encode(state):
    return sum(1 << (2*k+value) for k,value in enumerate(state) if value>=0)


def candidates(world,initial,max_steps=5):
    found=[]
    def visit(state,program):
        stopped=assembly.execute(world,program+[assembly.STOP],initial)
        found.append(dict(id=str(len(found)),program=program+[assembly.STOP],
                          state=stopped["state"],primitive_cost=stopped["primitive_cost"]))
        if len(program)>=max_steps-1:
            return
        for action in assembly.ACTIONS[:-1]:
            after,_,legal=assembly.step(world,state,False,action)
            if legal:
                visit(after,program+[action])
    visit(tuple(initial),[])
    return found


def recipient_world(targets):
    return dict(templates=[encode(target) for target in targets],visible=63)


def candidate_values(public,goal,costs,local=False):
    values=[]
    world=recipient_world(public["goal_options"])
    model=dict(order=[0,1],precision=1.2,prior=[.5,.5])
    for candidate in public["candidates"]:
        costs.charge("proposal_generation")
        costs.charge("hypothetical_execution",candidate["primitive_cost"])
        state=candidate["state"]
        target=public["goal_options"][goal]
        if local:
            value=float(state[0]==target[0])
        else:
            response=b.recipient(encode(state),world,model,costs)
            physical=1-assembly.mismatch(state,target)/3
            value=.5*physical+.5*response[goal]
        costs.charge("selection")
        values.append(value-public["edit_price"]*max(0,candidate["primitive_cost"]-1))
    return values


def make_case(namespace,constructor_index,history_index,regime):
    if regime not in REGIMES:
        raise ValueError("unknown revision regime")
    world=assembly.constructor(namespace,constructor_index)
    initial=list(world.defaults)
    first=list(initial)
    second=list(initial)
    second[0]=1-second[0]
    # Some histories start solved, others require a dependency-bearing change.
    initial_goal=history_index%2
    actual_goal=1-initial_goal if regime in ("revised","stale") else initial_goal
    remembered=actual_goal if regime=="revised" else initial_goal
    if history_index%4==3:
        initial[2]=1-initial[2]
    public=dict(schema="v17.revision.1",world=world.public(),initial=initial,
        goal_options=[first,second],remembered_goal=remembered,
        purpose_revision_announced=regime=="revised",candidates=candidates(world,initial),
        edit_price=(.03,.08,.2,.4)[history_index%4],feedback_budget=1,
        answer_support=[],query_contract="one scored recipient satisfaction observation, paid at one complete internal-pass cost")
    public["answer_support"]=[c["id"] for c in public["candidates"]]
    truth_cost=Costs()
    truth_values=candidate_values(public,actual_goal,truth_cost)
    probabilities=b.softmax(truth_values,5)
    rng=random.Random(seed_for(namespace,constructor_index,history_index,regime))
    truth=b.draw(probabilities,rng)
    private=dict(maker_training=[],earlier_goals=[initial_goal],current_goal=actual_goal,
        actual_constraints=world.public(),believed_constraints=world.public(),considered_options=public["candidates"],
        realized_history=assembly.execute(world,[0,1,2]+([8] if history_index%4==3 else []))["trace"],contributor_role="self revision",
        recipient_state=dict(goal=actual_goal,precision=1.2),true_values=truth_values,
        benchmark_edit=str(truth),benchmark_distribution=probabilities,
        feedback_condition="uninformative" if regime=="uninformative_feedback" else "misleading" if regime=="misleading_feedback" else "correct",
        target_generation_costs=truth_cost.receipt())
    return dict(case_id=digest([namespace,constructor_index,history_index,regime]),
        constructor_id=digest([namespace,constructor_index]),history_id=digest([namespace,constructor_index,history_index]),
        namespace=namespace,family="D",regime=regime,public=public,private=private,
        public_problem_sha256=digest(public),private_construction_sha256=digest([public,actual_goal]),
        structural_family=str(world.parents))


def predict(public,method,*,feedback=None,supplied_goal=None):
    required={"schema","world","initial","goal_options","remembered_goal","purpose_revision_announced",
        "candidates","edit_price","feedback_budget","answer_support","query_contract"}
    if set(public)!=required or public["schema"]!="v17.revision.1":
        raise ValueError("revision public schema violation")
    if method not in METHODS:
        raise ValueError("unknown revision method")
    costs=Costs()
    world=assembly.World(tuple(public["world"]["parents"]),tuple(public["world"]["defaults"]))
    for candidate in public["candidates"]:
        check=assembly.execute(world,candidate["program"],public["initial"])
        if not check["legal"] or not check["successfully_stopped"] or check["state"]!=candidate["state"]:
            raise ValueError("invalid offered assembly execution")
    goal=public["remembered_goal"]
    if method=="complete_error_monitor":
        if supplied_goal not in (0,1):
            raise ValueError("complete monitor requires explicit true-goal assistance")
        goal=supplied_goal
    if method=="stop_always":
        probabilities=[1.0]+[0.0]*(len(public["candidates"])-1)
        values=None
        costs.charge("selection")
    else:
        values=candidate_values(public,goal,costs,method=="local_edit")
        if method in ("internal_reconsideration","recipient_feedback"):
            # The hypothetical model's second pass investigates the opposite goal.
            before=sum(costs.values.values())
            alternative=candidate_values(public,1-goal,costs)
            pass_cost=sum(costs.values.values())-before
            probe=max(range(len(values)),key=lambda k:abs(values[k]-alternative[k]))
            costs.charge("selection",len(values))
            if method=="recipient_feedback":
                if feedback is None or public["feedback_budget"]<1:
                    raise ValueError("feedback arm requires an available query")
                observation=feedback(probe)
                costs.charge("feedback_query",pass_cost)
                if observation is not None and abs(observation-alternative[probe]) < abs(observation-values[probe]):
                    values=alternative
            else:
                # Matched internal work actually recomputes the remembered-purpose
                # values; no unobserved external evidence enters reconsideration.
                values=candidate_values(public,goal,costs)
        probabilities=b.softmax(values,5)
    chosen=max(range(len(probabilities)),key=probabilities.__getitem__)
    result=assembly.execute(world,public["candidates"][chosen]["program"],public["initial"])
    costs.charge("actual_execution",result["primitive_cost"])
    return dict(probabilities=dict(zip(public["answer_support"],probabilities)),chosen=chosen,
        execution=result,costs=costs.receipt(16),values=values,
        represented_variables=["local_orientation"] if method=="local_edit" else ["dependency_graph","remembered_goal","recipient_effect"])


def evaluate_case(case,design=None):
    public,private=case["public"],case["private"]
    rows=[]
    for method in METHODS:
        queries=[]
        def feedback(index):
            queries.append(index)
            condition=private["feedback_condition"]
            if condition=="uninformative":
                return None
            goal=1-private["current_goal"] if condition=="misleading" else private["current_goal"]
            return candidate_values(public,goal,Costs())[index]
        result=predict(public,method,feedback=feedback if method=="recipient_feedback" else None,
            supplied_goal=private["current_goal"] if method=="complete_error_monitor" else None)
        if len(queries)>public["feedback_budget"]:
            raise ValueError("feedback allowance exceeded")
        execution=result["execution"]
        chosen=result["chosen"]
        actual_goal=public["goal_options"][private["current_goal"]]
        benchmark=private["true_values"]
        regret=max(benchmark)-benchmark[0]
        damage=sum(initial>=0 and final<0 for initial,final in zip(public["initial"],execution["state"]))
        rows.append(dict(method=method,target="next_revision",probabilities=result["probabilities"],
            **forecast_scores(result["probabilities"],public["answer_support"],private["benchmark_edit"]),
            costs=result["costs"],evidence_tier="own remembered purpose and dependency graph; true goal only for complete monitor",
            represented_variables=result["represented_variables"],program=public["candidates"][chosen]["program"],
            execution=execution,task_success=execution["legal"] and execution["state"]==actual_goal,
            missing_output=False,invalid_program=not execution["legal"],collateral_parts_removed=damage,
            dependency_violations=sum(not step["legal"] for step in execution["trace"]),
            inappropriate_stop=chosen==0 and regret>1e-12,stop_regret=regret if chosen==0 else 0.0,
            decision_regret=max(benchmark)-benchmark[chosen],expected_value_of_further_operations=regret,
            feedback_queries=queries,feedback_condition=private["feedback_condition"]))
    return rows

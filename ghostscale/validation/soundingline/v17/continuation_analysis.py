"""Independent row checks and bounded fixed-sample confirmation for V17."""
from collections import defaultdict
import math
from .contracts import COST_KINDS, SEARCH_KINDS
from ..v16.records import digest
from runners.verify_v17_screen import physics, d_values

def near(a,b):
    if not math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10): raise ValueError("independent arithmetic mismatch")

def verify_item(item):
    case=item["case"];public=case["public"];private=case["private"];family=case["family"]
    if digest(public)!=case["public_problem_sha256"]: raise ValueError("public projection hash mismatch")
    dvalues=d_values(public,private["current_goal"]) if family=="D" else None
    for row in item["rows"]:
        costs=row["costs"]
        if any(type(costs[k]) is not int or costs[k]<0 for k in COST_KINDS): raise ValueError("invalid counted cost")
        setup=sum(costs[k] for k in COST_KINDS[:3])
        near(costs["cold_total"],sum(costs[k] for k in COST_KINDS))
        near(costs["repeat_online"],costs["cold_total"]-setup)
        near(costs["amortized_total"],costs["repeat_online"]+setup/costs["deployments"])
        near(costs["search_total"],sum(costs[k] for k in SEARCH_KINDS))
        if family in ("A2","AP"):
            if row["program"] is None:
                if row["task_success"] or not row["missing_output"]: raise ValueError("missing construction counted as success")
            else:
                state,legal,cost=physics(public,row["program"])
                if row["task_success"]!=(legal and state==public["target"]): raise ValueError("construction physics mismatch")
                near(cost,costs["actual_execution"])
                if len(row["program"])>public["max_steps"]: raise ValueError("length allowance exceeded")
            rep=row["representation"]
            storage=sum(x["storage_tokens"] for x in rep["definitions"])+sum(2*len(x)+1 for x in rep["fragments"])+sum(2*len(x)+3 for x in rep["episodes"])
            near(storage,costs["definition_storage"])
            if storage>row["memory_cap"] or costs["search_total"]>row["budget"]: raise ValueError("memory/search allowance exceeded")
            continue
        target=row["target"]
        if family=="E": truth=private["future_choice"];support=public["answer_support"]
        elif family=="D": truth=private["benchmark_edit"];support=public["answer_support"]
        elif family=="F":
            truth=private["partner_future"] if target=="partner_prediction" else private["own_outcomes"][row["chosen_personal_option"]]
            support=public["answer_support"]
        else:
            truth=private["accepted_edit"] if target=="accepted_edit" else private["recipient_response"]
            support=public["answer_support"] if target=="accepted_edit" else [str(k) for k in range(len(row["probabilities"]))]
        p=row["probabilities"]
        if set(p)!=set(support) or truth not in p or any(not math.isfinite(v) or not 0<=v<=1 for v in p.values()): raise ValueError("forecast support/normalization failure")
        near(sum(p.values()),1)
        near(row["brier_score"],sum((p[k]-(k==truth))**2 for k in support))
        if row["log_loss_infinite"]!=(p[truth]==0): raise ValueError("infinite score flag mismatch")
        if p[truth]: near(row["log_loss_nats"],-math.log(p[truth]))
        elif row["log_loss_nats"] is not None: raise ValueError("zero probability concealed")
        if family=="D":
            state,legal,cost=physics(public,row["program"])
            if state!=row["execution"]["state"] or row["task_success"]!=(legal and state==public["goal_options"][private["current_goal"]]): raise ValueError("revision physics mismatch")
            near(cost,costs["actual_execution"])
            chosen=next(i for i,x in enumerate(public["candidates"]) if x["program"]==row["program"])
            near(row["decision_regret"],max(dvalues)-dvalues[chosen])
            if private["realized_history"][-1]["after"]!=public["initial"]: raise ValueError("creation trace mismatch")
        elif row["task_success"]!=(max(support,key=p.get)==truth): raise ValueError("decision tie-break mismatch")
    if family=="BA":
        for candidate in public["candidates"]:
            state,legal,_=physics(dict(world=public["assembly_world"],initial=public["initial"]),candidate["program"])
            if not legal or state!=candidate["state"]: raise ValueError("assembly candidate mismatch")
        if private["realized_history"][-1]["after"]!=public["initial"]: raise ValueError("assembly creation trace mismatch")
    return True

def unit_statistics(items):
    groups=defaultdict(list)
    for item in items:
        verify_item(item)
        for row in item["rows"]: groups[(row["method"],row["target"])].append(row)
    result=[]
    for (method,target),rows in sorted(groups.items()):
        losses=[r.get("brier_score",float(not r["task_success"])) for r in rows]
        result.append(dict(method=method,target=target,apparatus_failures=sum(bool(r.get("apparatus_failure")) for r in rows),rows=len(rows),valid=sum(not r["missing_output"] and not r["invalid_program"] for r in rows),
            score=sum(losses)/len(losses),success=sum(r["task_success"] for r in rows)/len(rows),
            cold=sum(r["costs"]["cold_total"] for r in rows)/len(rows),online=sum(r["costs"]["repeat_online"] for r in rows)/len(rows),
            stop_regret=sum(r.get("stop_regret",0) for r in rows)/len(rows),
            decision_regret=sum(r.get("decision_regret",0) for r in rows)/len(rows),
            personal_goal_success=sum(r.get("personal_goal_success") or False for r in rows)/len(rows),
            infinite=sum(r.get("log_loss_infinite",False) for r in rows),evidence_tiers=sorted({r["evidence_tier"] for r in rows})))
    return result

def paired_value(items,contrast):
    values=[]
    for item in items:
        selected={}
        for row in item["rows"]:
            if row["target"]==contrast["target"] and row["method"] in (contrast["method"],contrast["rival"]):
                selected[row["method"]]=row
        if len(selected)!=2: raise ValueError("frozen comparison lacks one paired output")
        x,y=selected[contrast["method"]],selected[contrast["rival"]]
        values.append((float(x["task_success"])-float(y["task_success"])) if contrast["metric"]=="success" else y["brier_score"]-x["brier_score"])
    value=sum(values)/len(values)
    if not contrast["low"]<=value<=contrast["high"]: raise ValueError("externally bounded estimand violated")
    return value

def confirmation(values,contrast):
    n=len(values)
    if n!=contrast["constructors"]: raise ValueError("confirmation requires complete fixed sample")
    mean=sum(values)/n
    variance=sum((x-mean)**2 for x in values)/(n-1) if n>1 else 0.
    span=contrast["high"]-contrast["low"]
    # Hoeffding's fixed-sample one-sided bound. No distribution/normality fit.
    gap=max(0.,mean-contrast["minimum_gain"])
    p=math.exp(-2*n*gap*gap/(span*span))
    radius=span*math.sqrt(math.log(3/.05)/(2*n))
    return dict(id=contrast["id"],n_constructor_clusters=n,mean_gain=mean,between_constructor_variance=variance,
        minimum_gain=contrast["minimum_gain"],p_value=p,simultaneous_lower_bound=max(contrast["low"],mean-radius),
        planned_radius=contrast["planned_radius"],method="fixed-sample Hoeffding; paired constructor averages; Holm family at most three",
        direction=contrast["direction"],claim_status="awaiting family multiplicity decision")

def holm(entries,alpha=.05):
    ordered=sorted(entries,key=lambda x:(x["p_value"],x["id"]))
    stopped=False;previous=0.
    for rank,row in enumerate(ordered):
        multiplier=len(ordered)-rank
        adjusted=max(previous,min(1.,multiplier*row["p_value"]));previous=adjusted
        rejected=not stopped and row["p_value"]<=alpha/multiplier
        stopped=stopped or not rejected
        row.update(holm_adjusted_p=adjusted,holm_rejected=rejected,
            claim_status="confirmed within this constructed family" if rejected else "not confirmed")
    return sorted(ordered,key=lambda x:x["id"])

def group_surface(stats):
    groups={}
    for row in stats:
        key=(row["method"],row["target"])
        g=groups.setdefault(key,dict(n=0,mean=0.,m2=0.,attempted=0,valid=0,success=0.,cold=0.,online=0.,
            stop=0.,regret=0.,personal=0.,infinite=0,apparatus=0,tiers=set()))
        g["n"]+=1;n=g["n"];delta=row["score"]-g["mean"]
        g["mean"]+=delta/n;g["m2"]+=delta*(row["score"]-g["mean"])
        g["attempted"]+=row["rows"];g["valid"]+=row["valid"]
        for dest,src in (("success","success"),("cold","cold"),("online","online"),("stop","stop_regret"),
                         ("regret","decision_regret"),("personal","personal_goal_success")):g[dest]+=row[src]
        g["apparatus"]+=row.get("apparatus_failures",0)
        g["infinite"]+=row["infinite"];g["tiers"].update(row["evidence_tiers"])
    result=[]
    for (method,target),g in sorted(groups.items()):
        n=g["n"];mean=g["mean"];variance=max(0.,g["m2"]/(n-1)) if n>1 else 0.
        result.append(dict(method=method,target=target,constructor_clusters=n,attempted=g["attempted"],
            valid=g["valid"],apparatus_failure_rows=g["apparatus"],claim_status="apparatus failure; not interpretable as method quality" if g["apparatus"] else "descriptive",mean_brier_or_failure=mean,between_constructor_variance=variance,
            descriptive_normal_95_interval=[mean-1.96*math.sqrt(variance/n),mean+1.96*math.sqrt(variance/n)],
            interval_status="descriptive at retained prefix; not confirmation or equivalence",
            accuracy_or_success=g["success"]/n,mean_cold_cost=g["cold"]/n,mean_repeat_cost=g["online"]/n,
            mean_stop_regret=g["stop"]/n,mean_decision_regret=g["regret"]/n,
            mean_personal_goal_success=g["personal"]/n,infinite_log_loss_count=g["infinite"],
            evidence_tiers=sorted(g["tiers"])))
    return result

"""Independent V17 row physics, proper-score arithmetic and aggregate means.

Does not import a V17 scientific consumer or call an old campaign stage.
Bootstrap intervals and the correctness of the generative assumptions are outside
this receipt; source-bound validity and separate replay cover different questions.
"""
import argparse
from collections import defaultdict
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
KINDS=("training_acquisition","learning","definition_storage","retrieval","proposal_generation",
       "argument_binding","hypothetical_execution","actual_execution","selection","feedback_query")
SEARCH=("retrieval","proposal_generation","argument_binding","hypothetical_execution","selection")

def read(path): return json.loads(Path(path).read_bytes())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def close(actual,expected):
    if not math.isclose(actual,expected,rel_tol=1e-11,abs_tol=1e-11):
        raise ValueError(f"independent arithmetic mismatch: {actual} versus {expected}")

def physics(public,program):
    graphic=public.get("world_kind")=="graphic"
    state=public["initial"] if graphic else list(public["initial"])
    stopped=False
    used=0
    for action in program:
        used+=1
        if type(action) is not int or action in public.get("forbidden",[]): return state,False,used
        if graphic:
            if not 0<=action<32: return state,False,used
            state=state|(1<<action) if action<16 else state&~(1<<(action-16))
        else:
            if stopped or not 0<=action<=9: return state,False,used
            if action==9:
                stopped=True
                continue
            part=action%3
            parents=public["world"]["parents"]
            if action<3:
                if state[part]>=0 or (parents[part]>=0 and state[parents[part]]<0): return state,False,used
                state[part]=public["world"]["defaults"][part]
            else:
                dependent=any(parent==part and state[child]>=0 for child,parent in enumerate(parents))
                if state[part]<0 or dependent: return state,False,used
                state[part]=-1 if action<6 else 1-state[part]
    return state,True,used

def d_values(public,goal):
    def encoded(state): return sum(1<<(2*k+v) for k,v in enumerate(state) if v>=0)
    templates=[encoded(t) for t in public["goal_options"]]
    result=[]
    for candidate in public["candidates"]:
        state,legal,cost=physics(public,candidate["program"])
        if not legal or state!=candidate["state"]: raise ValueError("independent candidate physics mismatch")
        board=encoded(state)
        weights=[math.exp(-1.2*((board^template)&63).bit_count()) for template in templates]
        response=weights[goal]/sum(weights)
        physical=1-sum(x!=y for x,y in zip(state,public["goal_options"][goal]))/3
        result.append(.5*physical+.5*response-public["edit_price"]*(cost-1))
    return result

def verify(root):
    root=Path(root)
    lock,index,completion=(read(root/name) for name in ("LOCK.json","INDEX.json","COMPLETION.json"))
    for name,key in (("LOCK.json","lock_sha256"),("INDEX.json","index_sha256"),("COMPARISONS.json","summary_sha256")):
        if sha(root/name)!=completion[key]: raise ValueError("completion input hash mismatch")
    groups=defaultdict(list)
    total=0
    total_rows=0
    identity=defaultdict(set)
    for chunk in index["chunks"]:
        path=root/chunk["path"]
        if sha(path)!=chunk["sha256"]: raise ValueError("chunk hash mismatch")
        items=[json.loads(line) for line in path.read_bytes().splitlines()]
        if len(items)!=chunk["cases"] or sum(len(x["rows"]) for x in items)!=chunk["rows"]: raise ValueError("chunk count mismatch")
        for item in items:
            case=item["case"]
            family=lock["design"]["family"]
            public,private=case["public"],case["private"]
            total+=1
            for key in ("case_id","constructor_id","history_id","public_problem_sha256","private_construction_sha256","structural_family"):
                identity[key].add(case[key])
            expected_d=d_values(public,private["current_goal"]) if family=="D" else None
            for row in item["rows"]:
                total_rows+=1
                costs=row["costs"]
                if any(type(costs[k]) is not int or costs[k]<0 for k in KINDS): raise ValueError("invalid counted cost")
                close(costs["cold_total"],sum(costs[k] for k in KINDS))
                setup=sum(costs[k] for k in KINDS[:3])
                close(costs["repeat_online"],costs["cold_total"]-setup)
                close(costs["amortized_total"],costs["repeat_online"]+setup/costs["deployments"])
                close(costs["search_total"],sum(costs[k] for k in SEARCH))
                if family=="A2":
                    if row["program"] is None:
                        if not row["missing_output"] or row["task_success"]: raise ValueError("missing construction treated as success")
                    else:
                        state,legal,used=physics(public,row["program"])
                        if legal!=(not row["invalid_program"]) or row["task_success"]!=(legal and state==public["target"]):
                            raise ValueError("independent construction success mismatch")
                        if used!=costs["actual_execution"]: raise ValueError("executed primitive cost mismatch")
                        if len(row["program"])>public["max_steps"]: raise ValueError("program exceeded step allowance")
                    if costs["hypothetical_execution"]!=row["attempted_primitives"]: raise ValueError("counted search differs from actual attempts")
                    close(costs["training_acquisition"],sum(len(t["program"]) for t in public["training"]))
                    rep=row["representation"]
                    storage=sum(d["storage_tokens"] for d in rep["definitions"])+sum(2*len(f)+1 for f in rep["fragments"])+sum(2*len(ep)+3 for ep in rep["episodes"])
                    close(costs["definition_storage"],storage)
                    if storage>row["memory_cap"] or costs["search_total"]>row["budget"]: raise ValueError("memory/search allowance exceeded")
                else:
                    truth=private["future_choice"] if family=="E" else private["benchmark_edit"] if family=="D" else private["accepted_edit"] if row["target"]=="accepted_edit" else private["recipient_response"]
                    p=row["probabilities"]
                    if truth not in p or any(not math.isfinite(v) or v<0 or v>1 for v in p.values()): raise ValueError("invalid forecast support")
                    close(sum(p.values()),1)
                    close(row["brier_score"],sum((v-(label==truth))**2 for label,v in p.items()))
                    if row["log_loss_infinite"]!=(p[truth]==0): raise ValueError("infinite log-loss flag mismatch")
                    if p[truth]: close(row["log_loss_nats"],-math.log(p[truth]))
                    elif row["log_loss_nats"] is not None: raise ValueError("zero mass concealed")
                    if family=="D":
                        state,legal,used=physics(public,row["program"])
                        if state!=row["execution"]["state"] or used!=costs["actual_execution"]: raise ValueError("revision execution mismatch")
                        if row["task_success"]!=(legal and state==public["goal_options"][private["current_goal"]]): raise ValueError("revision success mismatch")
                        chosen=next(i for i,c in enumerate(public["candidates"]) if c["program"]==row["program"])
                        close(row["decision_regret"],max(expected_d)-expected_d[chosen])
                        close(row["stop_regret"],max(expected_d)-expected_d[0] if chosen==0 else 0)
                    else:
                        # Canonical JSON sorts object keys; decisions used the declared support order.
                        support=public["answer_support"] if family=="E" or row["target"]=="accepted_edit" else [str(i) for i in range(len(p))]
                        if row["task_success"]!=(max(support,key=p.get)==truth): raise ValueError("forecast accuracy mismatch")
                groups[(case["regime"],row["method"],row.get("target","construction"))].append((case,row))
    summary=read(root/"COMPARISONS.json")
    for entry in summary["comparison_table"]:
        items=groups[(entry["regime"],entry["method"],entry["target"])]
        rows=[row for _,row in items]
        clusters=defaultdict(list)
        for case,row in items:
            clusters[case["constructor_id"]].append(Fraction(row.get("brier_score",float(not row["task_success"]))))
        mean=sum(sum(xs)/len(xs) for xs in clusters.values())/len(clusters)
        close(entry["mean_brier_or_failure"],float(mean))
        close(entry["accuracy_or_success"],sum(r["task_success"] for r in rows)/len(rows))
        close(entry["mean_cold_cost"],sum(r["costs"]["cold_total"] for r in rows)/len(rows))
        close(entry["mean_repeat_cost"],sum(r["costs"]["repeat_online"] for r in rows)/len(rows))
        if entry["attempted"]!=len(rows) or entry["valid"]!=sum(not r["missing_output"] and not r["invalid_program"] for r in rows): raise ValueError("aggregate denominator mismatch")
        if entry["constructor_clusters"]!=len(clusters): raise ValueError("constructor denominator mismatch")
        if entry["unique_private_constructions"]!=len({c["private_construction_sha256"] for c,_ in items}): raise ValueError("private uniqueness mismatch")
    if summary["uniqueness"]!={k:len(v) for k,v in identity.items()}: raise ValueError("overall uniqueness mismatch")
    if total!=completion["cases"] or total_rows!=completion["rows"]: raise ValueError("completion count mismatch")
    return dict(schema="v17.independent-verification.1",passed=True,cases=total,rows=total_rows,
        aggregate_groups=len(summary["comparison_table"]),input_completion_sha256=sha(root/"COMPLETION.json"),
        verifier_sha256=sha(Path(__file__)),scope="all proper scores/cost identities/means; independent A2 and D physics",
        excluded="bootstrap intervals, generative assumptions, full hypothesis-search cost reimplementation")

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    result=verify(args.root)
    if args.output:
        payload=json.dumps(result,sort_keys=True,separators=(",",":")).encode()+b"\n"
        if args.output.exists() and args.output.read_bytes()!=payload: raise ValueError("verification receipt differs")
        args.output.parent.mkdir(parents=True,exist_ok=True)
        if not args.output.exists(): args.output.write_bytes(payload)
    print(json.dumps(result))
if __name__=="__main__": main()

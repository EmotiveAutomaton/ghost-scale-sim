"""Independent adaptation learning, checking, execution and aggregate verification."""
from collections import Counter
from itertools import product
import hashlib
import json
import math
from .reference import interpret,token_cost
from .audit_statistics import verify


def learned(record):
    counts=Counter()
    for program,target in zip(record["attempts"],record["targets"]):
        result=interpret(program)
        if result["legal"] and result["artifact"]==target:
            for offset in range(len(program)-1):
                counts[tuple(program[offset:offset+2])]+=1
    library=[]
    for fragment in sorted(counts,key=lambda item:(-counts[item],item)):
        before=sum(token_cost(program,library) for program in record["attempts"])
        after=sum(token_cost(program,library+[fragment]) for program in record["attempts"])+len(fragment)
        if before-after>=1:
            library.append(fragment)
        if len(library)==2:
            break
    return library


def audit_unit(root,row):
    sources={}
    for folder,field in [("public","public_hash"),("private","private_hash"),("predictions","prediction_hash")]:
        payload=(root/folder/f"{row['unit_id']}.json").read_bytes()
        if hashlib.sha256(payload).hexdigest()!=row[field]:
            raise ValueError("purpose raw hash mismatch")
        sources[folder]=json.loads(payload)
    public=sources["public"]
    old,new=learned(public["old_training"]),learned(public["new_training"])
    retained=[]
    checking=[]
    for fragment in old:
        state=0
        harmful=False
        for action in fragment:
            result=interpret([action],start=state)
            before=(state^public["target"]).bit_count()
            after=(result["artifact"]^public["target"]).bit_count()
            checking.append({"before":state,"action":action,"after":result["artifact"],"goal_error_before":before,
                             "goal_error_after":after,"primitive_cost":result["primitive_cost"]})
            harmful |= not result["legal"] or after>before
            state=result["artifact"]
        if not harmful:
            retained.append(fragment)
    values=[]
    for length in range(4):
        for program in product(range(8),repeat=length):
            result=interpret(program)["artifact"]
            values.append((-(result^public["old_target"]).bit_count(),-(result^public["target"]).bit_count()))
    means=[sum(value[i] for value in values)/len(values) for i in range(2)]
    numerator=sum((a-means[0])*(b-means[1]) for a,b in values)
    denominator=math.sqrt(sum((a-means[0])**2 for a,b in values)*sum((b-means[1])**2 for a,b in values))
    alignment=sources["private"]["purpose_alignment"]
    if alignment["reference_programs"]!=585 or abs(alignment["reward_correlation"]-numerator/denominator)>1e-10:
        raise ValueError("purpose relation reference mismatch")
    for name,prediction in sources["predictions"]["arms"].items():
        library={"continued":old,"inhibited":retained,"relearned":new,"primitive":[]}[name]
        if prediction["library"]!=[list(fragment) for fragment in library]:
            raise ValueError("active acquired library mismatch")
        if prediction["checking"]!=(checking if name=="inhibited" else []):
            raise ValueError("actual checking trace mismatch")
        submission=prediction["submission"]
        for attempt in submission["attempted_programs"]:
            program=[]
            for token in attempt["tokens"]:
                program.extend([token] if isinstance(token,int) else library[int(token[1:])])
            execution=interpret(program)
            if (execution["artifact"],execution["legal"],execution["primitive_cost"])!=(attempt["artifact"],attempt["legal"],attempt["cost"]):
                raise ValueError("purpose search execution or cost mismatch")
        spent=sum(attempt["cost"] for attempt in submission["attempted_programs"])
        if spent!=submission["search_primitives"] or spent>public["budget"]:
            raise ValueError("purpose search budget mismatch")
        execution=interpret(submission["program"])
        if row["arms"][name]["execution"]!=execution:
            raise ValueError("purpose final execution mismatch")
        success=execution["legal"] and not submission["search_timeout"] and execution["artifact"]==public["target"]
        check_cost=sum(item["primitive_cost"] for item in checking) if name=="inhibited" else 0
        train_cost=sum(len(trace) for record in [public["old_training"],public["new_training"]] for trace in record["attempts"])
        definition=(sum(map(len,old)) if name!="primitive" else 0)+(sum(map(len,new)) if name=="relearned" else 0)
        expected={"success":float(success),"legal":float(execution["legal"]),"search_cost":float(spent),
                  "checking_cost":float(check_cost),"total_search_check_cost":float(spent+check_cost),
                  "training_primitives":float(train_cost),"definition_cost":float(definition),
                  "execution_primitives":float(execution["primitive_cost"])}
        if row["arms"][name]["outcomes"]!=expected:
            raise ValueError("independent purpose outcome or cost mismatch")


def audit(root,summary):
    rows=[json.loads(path.read_bytes()) for path in sorted((root/"units").glob("*_points.json"))]
    for row in rows:
        audit_unit(root,row)
    count=verify(rows,summary)
    return {"execution_state":"completed","instrument_state":"valid","n_raw_units":len(rows),
            "reproduced_contrasts":count,"all_reported_aggregates_reproduced":True,"full_rollout_replay":False}

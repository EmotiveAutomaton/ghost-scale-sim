"""Finite opportunity/self scouts with explicit instrument-local dependencies."""
from .runtime import PACKAGE,freeze,remaining_seconds
from .records import read,write,file_digest
from .behavior_designs import DESIGNS
from .behavior_study import execute_unit,summarize
from .opportunity_gates import run as opportunity_gates
from .self_gates import run as self_gates
from .self_trajectory import gates as trajectory_gates
from .audit_behavior import audit

MODULES=["world.py","reference.py","records.py","estimands.py","opportunity.py",
         "opportunity_gates.py","self_monitor.py","self_gates.py","self_trajectory.py",
         "behavior_designs.py","behavior_study.py","audit_statistics.py","audit_behavior.py","behavior_runner.py"]


def execute(root,heartbeat,*,resume=False):
    packet_id="behavior-scout-1"
    if resume and not (root/"packets"/f"{packet_id}.json").exists():
        raise ValueError("resume never prepares a behavioral packet")
    packet=freeze(root,packet_id,[PACKAGE/name for name in MODULES],
                  {"cards":DESIGNS,"evidence_scope":"discovery","n_per_condition":64,"constructors":8,
                   "seed_namespace":"v16-behavior-discovery-1","trajectory_rounds":3})
    output_root=root/packet_id
    gate_path=output_root/"GATES.json"
    if resume and not gate_path.exists():
        raise ValueError("interrupted before admission; explicit discovery can finish setup")
    if gate_path.exists():
        controls=read(gate_path)
    else:
        controls={"opportunity":opportunity_gates(),"self":self_gates(),
                  "trajectory":trajectory_gates()}
        write(gate_path,controls)
    dependencies={}
    for family in ["opportunity","self"]:
        for gate in controls[family]["gates"]:
            dependencies[gate["id"]]=gate["instrument_state"]
    dependencies[controls["trajectory"]["id"]]=controls["trajectory"]["instrument_state"]
    total=sum(64*len(design["conditions"]) for design in DESIGNS.values())
    completed=0
    blocked=[]
    for card,design in DESIGNS.items():
        output=output_root/card
        missing=[dependency for dependency in design["dependencies"] if dependencies.get(dependency)!="valid"]
        if missing:
            write(output/"COMPLETION.json",{"execution_state":"blocked","instrument_state":"failed",
                                            "criterion_state":"untested","missing_dependencies":missing,
                                            "warrant":"INSTRUMENT FAILED","evidence_scope":"discovery"})
            blocked.append(card)
            continue
        rows=[]
        for condition in design["conditions"]:
            for index in range(64):
                if remaining_seconds(root)<=0:
                    return {"execution_state":"checkpointed","completed_units":completed,
                            "planned_units":total,"reason":"immutable elapsed ceiling"}
                row=execute_unit(output,card,condition,index,namespace=f"v16-behavior-discovery-1-{card}",
                                 packet=packet)
                rows.append(row)
                completed+=1
                heartbeat(packet_id=packet_id,card_id=card,condition=condition["id"],
                          completed_units=completed,planned_units=total,last_unit=row["unit_id"])
        summary=summarize(card,rows)
        receipt=audit(output,summary)
        summary["independent_reaggregation"]="valid"
        write(output/"REAGGREGATION.json",receipt)
        write(output/"SUMMARY.json",summary)
        manifest={str(path.relative_to(output)).replace("\\","/"):file_digest(path)
                  for directory in ["public","private","predictions","units"]
                  for path in sorted((output/directory).glob("*.json"))}
        write(output/"RAW_MANIFEST.json",{"files":manifest,"archive_location":"this packet directory",
                                         "retention":"retain through final verified archival handoff"})
        write(output/"COMPLETION.json",{"execution_state":"completed","instrument_state":"valid",
                                        "criterion_state":"see typed per-condition contrasts",
                                        "evidence_scope":"discovery","n_maker_packets":len(rows),
                                        "warrant":"DESCRIPTIVE ONLY","pursuit":"OPENED",
                                        "dependency_joins":{d:dependencies[d] for d in design["dependencies"]},
                                        "expansion_state":"pending","confirmation_state":"untested"})
        dependencies[card+".instrument"]="valid"
    return {"execution_state":"completed","packet_id":packet_id,"n_maker_packets":completed,
            "blocked_cards":blocked,"independent_reaggregation":"valid for completed cards","campaign_complete":False}

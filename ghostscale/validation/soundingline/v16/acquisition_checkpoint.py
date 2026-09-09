"""Operational K01 continuation across additions to the unlocked CLI.

The original admission receipt included the command-line entrypoint. Scientific
packet files remain immutable; an extended entrypoint is verified by current
runtime tests instead of being mistaken for an estimator amendment.
"""
from .runtime import REPO,PACKAGE,freeze,remaining_seconds
from .records import read,write,file_digest
from .admission import K01_REQUIRED
from .craft import DESIGN
from .craft_gates import run as gates
from .study import unit,summarize,raw_manifest,ESTIMANDS
from .audit_craft import audit

MODULES=["world.py","reference.py","learning.py","records.py","craft.py","estimands.py",
         "study.py","craft_gates.py","audit_craft.py","acquisition_runner.py","admission.py"]


def admission(root):
    receipt=read(root/"ADMISSION_CHECKS.json")
    for module in MODULES:
        relative=str((PACKAGE/module).relative_to(REPO)).replace("\\","/")
        if receipt["source_hashes"].get(relative)!=file_digest(PACKAGE/module):
            raise ValueError("scientific admission source differs: "+relative)
    for test_name in K01_REQUIRED.values():
        if receipt["tests"].get(test_name)!="passed":
            raise ValueError("missing admitted K01 control: "+test_name)
    return {dependency:{"instrument_state":"valid","test":test,"receipt":"ADMISSION_CHECKS.json"}
            for dependency,test in K01_REQUIRED.items()}


def execute(root,heartbeat,*,resume=False):
    dependencies=admission(root)
    packet_id="k01-scout-1"
    lock_path=root/"packets"/f"{packet_id}.json"
    if resume and not lock_path.exists():
        raise ValueError("resume never prepares an acquisition packet")
    if lock_path.exists():
        packet=read(lock_path)
        for relative,expected in packet["identity"]["files"].items():
            if file_digest(REPO/relative)!=expected:
                raise ValueError("frozen acquisition source changed: "+relative)
    else:
        design={**DESIGN,"scope":"discovery","n_per_condition":64,"constructors":8,
                "estimands":[estimator.record() for estimator in ESTIMANDS],
                "dependency_joins":dependencies}
        packet=freeze(root,packet_id,[PACKAGE/name for name in MODULES],design)
    output=root/packet_id
    if (output/"COMPLETION.json").exists():
        completion=read(output/"COMPLETION.json")
        if completion["execution_state"]!="completed" or completion["instrument_state"]!="valid":
            raise ValueError("existing acquisition packet is not successfully completed")
        if file_digest(output/"SUMMARY.json")!=completion["summary_sha256"]:
            raise ValueError("completed acquisition summary changed")
        return {"execution_state":"completed","packet_id":packet_id,"already_completed":True,
                "n_maker_packets":completion["n_maker_packets"],"campaign_complete":False,
                "scientific_source_lock":"verified","independent_reaggregation":"previously completed"}
    gate_path=output/"GATES.json"
    if resume and not gate_path.exists():
        raise ValueError("resume has no completed acquisition admission")
    controls=read(gate_path) if gate_path.exists() else gates()
    write(gate_path,controls)
    if controls["instrument_state"]!="valid":
        raise ValueError("acquisition gates failed")
    design=packet["identity"]["design"]
    rows=[]
    for condition in design["conditions"]:
        for index in range(design["n_per_condition"]):
            if remaining_seconds(root)<=0:
                return {"execution_state":"checkpointed","reason":"immutable elapsed ceiling",
                        "completed_units":len(rows),"campaign_complete":False}
            rows.append(unit(output,condition,index,packet_hash=packet["packet_hash"],
                             namespace=design["seed_namespace"],constructors=design["constructors"],
                             evidence_scope="discovery"))
            heartbeat(packet_id=packet_id,card_id="K01",condition=condition["id"],
                      completed_units=len(rows),planned_units=192,last_unit=rows[-1]["unit_id"])
    summary=summarize(rows)
    receipt=audit(output,summary)
    summary["independent_reaggregation"]="valid"
    write(output/"REAGGREGATION.json",receipt)
    write(output/"RAW_MANIFEST.json",raw_manifest(output))
    write(output/"SUMMARY.json",summary)
    write(output/"COMPLETION.json",{
        "execution_state":"completed","instrument_state":"valid",
        "criterion_state":"see each paired contrast","evidence_scope":"discovery",
        "warrant":"DESCRIPTIVE ONLY","pursuit":"OPENED",
        "packet_hash":packet["packet_hash"],"summary_sha256":file_digest(output/"SUMMARY.json"),
        "n_maker_packets":len(rows),"n_conditions":3,"constructors_per_condition":8,
        "expansion_state":"planned; assess the preregistered capability or boundary rule",
        "confirmation_state":"untested","closeout_state":"incomplete"})
    return {"execution_state":"completed","packet_id":packet_id,"n_maker_packets":len(rows),
            "independent_reaggregation":"valid","campaign_complete":False}

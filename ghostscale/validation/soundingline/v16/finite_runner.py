"""Single-card packet adapter: immutable tests/source/design and finite unit loops."""
from .runtime import REPO,PACKAGE,freeze,remaining_seconds
from .records import read,write,file_digest
from .reader_process import ReaderProcess


def execute(root,heartbeat,module,*,resume=False):
    packet_id=module.PACKET_ID
    design=module.DESIGN
    if resume and not (root/"packets"/f"{packet_id}.json").exists():
        raise ValueError("resume never prepares a finite packet")
    files=[PACKAGE/name for name in module.MODULES]+[REPO/"runners/v16_reader_worker.py"]
    admission=read(root/module.SETUP/"ADMISSION.json")
    if any(admission["tests"].get(test)!="passed" for test in module.REQUIRED_TESTS):
        raise ValueError("required executed tests are missing or failed")
    for path in files:
        relative=str(path.relative_to(REPO)).replace("\\","/")
        if admission["source_hashes"].get(relative)!=file_digest(path):
            raise ValueError("finite packet source differs from actual test receipt")
    packet=freeze(root,packet_id,files,{"card":design,"n_per_condition":64,"constructors":8,
                  "seed_namespace":module.NAMESPACE,"scope":"discovery"})
    output=root/packet_id
    gate_path=output/"GATES.json"
    if resume and not gate_path.exists():
        raise ValueError("resume requires prior admission")
    gates=read(gate_path) if gate_path.exists() else module.gates()
    write(gate_path,gates)
    if gates["instrument_state"]!="valid":
        raise ValueError("finite packet admission failed; preserve controls")
    rows=[]
    total=64*len(design["conditions"])
    with ReaderProcess(output/"public/reader-workspace",extensions=module.EXTENSIONS) as reader:
        for condition in design["conditions"]:
            for index in range(64):
                if remaining_seconds(root)<=0:
                    return {"execution_state":"checkpointed","completed_units":len(rows),
                            "planned_units":total,"reason":"immutable ceiling","campaign_complete":False}
                row=module.execute_unit(output,condition,index,namespace=module.NAMESPACE,packet=packet,reader=reader)
                rows.append(row)
                heartbeat(packet_id=packet_id,card_id=design["card_id"],condition=condition["id"],completed_units=len(rows),
                          planned_units=total,last_unit=row["unit_id"],reader_pid=reader.identity["pid"])
    summary=module.summarize(rows)
    receipt=module.audit(output,summary)
    summary["independent_reaggregation"]="valid"
    write(output/"REAGGREGATION.json",receipt)
    write(output/"SUMMARY.json",summary)
    manifest={str(path.relative_to(output)).replace("\\","/"):file_digest(path)
              for directory in ["public","private","predictions","units"]
              for path in sorted((output/directory).glob("*.json"))}
    write(output/"RAW_MANIFEST.json",{"files":manifest,"archive_location":"this packet directory",
                                     "retention":"retain through final verified archival handoff"})
    write(output/"COMPLETION.json",{"card_id":design["card_id"],"execution_state":"completed","instrument_state":"valid",
        "criterion_state":"see typed per-condition contrasts","evidence_scope":"discovery","n_maker_packets":len(rows),
        "warrant":"DESCRIPTIVE ONLY","pursuit":"OPENED","dependencies":{name:"valid" for name in design["dependencies"]},
        "expansion_state":"pending","confirmation_state":"untested","reader_process":"separate public-only worker"})
    return {"execution_state":"completed","packet_id":packet_id,"n_maker_packets":len(rows),
            "independent_reaggregation":"valid","campaign_complete":False}

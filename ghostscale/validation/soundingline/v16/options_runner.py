"""Locally admitted finite K04 option scout."""
from .runtime import REPO,PACKAGE,freeze,remaining_seconds
from .records import read,write,file_digest
from .options_study import DESIGN,EXTENSION,execute_unit,summarize
from .options_gates import run
from .audit_options import audit
from .reader_process import ReaderProcess

MODULES=["world.py","reference.py","learning.py","craft.py","records.py","estimands.py",
         "options.py","options_study.py","options_gates.py","audit_statistics.py","audit_options.py",
         "options_runner.py","reader_process.py"]
REQUIRED_TESTS=["test_options_use_only_observed_edges_and_terminate",
    "test_empty_and_disconnected_observation_graph_are_not_complete_worlds",
    "test_degenerate_eigenspace_extrema_are_basis_and_sign_invariant",
    "test_option_controls_and_independent_execution_costs"]


def execute(root,heartbeat,*,resume=False):
    packet_id="options-scout-1"
    if resume and not (root/"packets"/f"{packet_id}.json").exists():
        raise ValueError("resume never prepares an option packet")
    files=[PACKAGE/name for name in MODULES]+[REPO/"runners/v16_reader_worker.py"]
    admission=read(root/"options-setup/ADMISSION.json")
    if any(admission["tests"].get(test)!="passed" for test in REQUIRED_TESTS):
        raise ValueError("option required executed tests are missing or failed")
    for path in files:
        relative=str(path.relative_to(REPO)).replace("\\","/")
        if admission["source_hashes"].get(relative)!=file_digest(path):
            raise ValueError("option source differs from actual test receipt")
    packet=freeze(root,packet_id,files,{"card":DESIGN,"n_per_condition":64,"constructors":8,
                  "seed_namespace":"v16-options-discovery-1","scope":"discovery"})
    output=root/packet_id
    gate_path=output/"GATES.json"
    if resume and not gate_path.exists():
        raise ValueError("resume requires prior admission")
    gates=read(gate_path) if gate_path.exists() else run()
    write(gate_path,gates)
    if gates["instrument_state"]!="valid":
        raise ValueError("option admission failed; preserve controls")
    rows=[]
    with ReaderProcess(output/"public/reader-workspace",extensions=[EXTENSION]) as reader:
        for condition in DESIGN["conditions"]:
            for index in range(64):
                if remaining_seconds(root)<=0:
                    return {"execution_state":"checkpointed","completed_units":len(rows),
                            "planned_units":192,"reason":"immutable ceiling","campaign_complete":False}
                row=execute_unit(output,condition,index,namespace="v16-options-discovery-1",packet=packet,reader=reader)
                rows.append(row)
                heartbeat(packet_id=packet_id,card_id="K04",condition=condition["id"],completed_units=len(rows),
                          planned_units=192,last_unit=row["unit_id"],reader_pid=reader.identity["pid"])
    summary=summarize(rows)
    receipt=audit(output,summary)
    summary["independent_reaggregation"]="valid"
    write(output/"REAGGREGATION.json",receipt)
    write(output/"SUMMARY.json",summary)
    manifest={str(path.relative_to(output)).replace("\\","/"):file_digest(path)
              for directory in ["public","private","predictions","units"]
              for path in sorted((output/directory).glob("*.json"))}
    write(output/"RAW_MANIFEST.json",{"files":manifest,"archive_location":"this packet directory",
                                     "retention":"retain through final verified archival handoff"})
    write(output/"COMPLETION.json",{"card_id":"K04","execution_state":"completed","instrument_state":"valid",
        "criterion_state":"see typed per-condition contrasts","evidence_scope":"discovery","n_maker_packets":len(rows),
        "warrant":"DESCRIPTIVE ONLY","pursuit":"OPENED","dependencies":{name:"valid" for name in DESIGN["dependencies"]},
        "expansion_state":"pending","confirmation_state":"untested","reader_process":"separate public-only worker"})
    return {"execution_state":"completed","packet_id":packet_id,"n_maker_packets":len(rows),
            "independent_reaggregation":"valid","campaign_complete":False}

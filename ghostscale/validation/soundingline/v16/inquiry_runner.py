"""Finite R01-R05 scout admission and single-supervisor execution."""
from .runtime import REPO,PACKAGE,freeze,remaining_seconds
from .records import read,write,file_digest
from .inquiry_designs import DESIGNS
from .inquiry_study import execute_unit,summarize
from .inquiry_gates import run
from .audit_inquiry import audit
from .reader_process import ReaderProcess

MODULES=["world.py","learning.py","records.py","estimands.py","inquiry.py","inquiry_world.py",
         "inquiry_designs.py","inquiry_study.py","inquiry_gates.py","audit_statistics.py",
         "audit_inquiry.py","reader_process.py","inquiry_runner.py"]
REQUIRED_TESTS=["test_real_feedback_changes_success_on_unseen_compositions",
                "test_noise_is_not_a_fixed_hidden_mapping_and_cannot_gain_competence",
                "test_exact_learning_value_matches_independent_one_query_sum",
                "test_policy_has_no_evaluator_truth_input","test_admission_known_answers",
                "test_delayed_feedback_skill_loss_and_stop_are_actual_episodes",
                "test_equal_examples_and_zero_practice_do_not_invent_enactment_benefit",
                "test_separate_reader_matches_known_prediction_and_cannot_open_private_record"]


def execute(root,heartbeat,*,resume=False):
    packet_id="inquiry-scout-1"
    if resume and not (root/"packets"/f"{packet_id}.json").exists():
        raise ValueError("resume never prepares an inquiry packet")
    files=[PACKAGE/name for name in MODULES]+[REPO/"runners/v16_reader_worker.py"]
    admission=read(root/"inquiry-setup/ADMISSION.json")
    if any(admission["tests"].get(test)!="passed" for test in REQUIRED_TESTS):
        raise ValueError("inquiry required executed tests are missing or failed")
    for path in files:
        relative=str(path.relative_to(REPO)).replace("\\","/")
        if admission["source_hashes"].get(relative)!=file_digest(path):
            raise ValueError("inquiry source differs from actual test receipt")
    packet=freeze(root,packet_id,files,{"cards":DESIGNS,"n_per_condition":64,"constructors":8,
                  "seed_namespace":"v16-inquiry-discovery-1","horizon":8,"scope":"discovery"})
    output_root=root/packet_id
    gate_path=output_root/"GATES.json"
    if resume and not gate_path.exists():
        raise ValueError("resume requires prior admission")
    gates=read(gate_path) if gate_path.exists() else run()
    write(gate_path,gates)
    if gates["instrument_state"]!="valid":
        raise ValueError("inquiry gate failed; preserve controls")
    total=sum(64*len(design["conditions"]) for design in DESIGNS.values())
    completed=0
    with ReaderProcess(output_root/"public/reader-workspace") as reader:
        for card,design in DESIGNS.items():
            output=output_root/card
            rows=[]
            for condition in design["conditions"]:
                for index in range(64):
                    if remaining_seconds(root)<=0:
                        return {"execution_state":"checkpointed","completed_units":completed,
                                "planned_units":total,"reason":"immutable ceiling","campaign_complete":False}
                    row=execute_unit(output,card,condition,index,namespace=f"v16-inquiry-discovery-1-{card}",
                                     packet=packet,reader=reader)
                    rows.append(row)
                    completed+=1
                    heartbeat(packet_id=packet_id,card_id=card,condition=condition["id"],completed_units=completed,
                              planned_units=total,last_unit=row["unit_id"],reader_pid=reader.identity["pid"])
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
                "criterion_state":"see typed per-condition contrasts","evidence_scope":"discovery",
                "n_maker_packets":len(rows),"warrant":"DESCRIPTIVE ONLY","pursuit":"OPENED",
                "dependencies":{name:"valid" for name in design["dependencies"]},
                "expansion_state":"pending","confirmation_state":"untested","reader_process":"separate public-only worker"})
    return {"execution_state":"completed","packet_id":packet_id,"n_maker_packets":completed,
            "cards":list(DESIGNS),"independent_reaggregation":"valid","campaign_complete":False}

"""Finite reader scouts; existing packet resume never prepares a new packet."""
from .runtime import PACKAGE, freeze, remaining_seconds
from .records import read, write, file_digest
from .reading_study import DESIGNS, execute_unit, summarize
from .reading_gates import run as gates
from .audit_reading import audit

MODULES = ["world.py","reference.py","learning.py","craft.py","records.py","estimands.py",
           "reconstruction.py","reading_study.py","reading_gates.py","audit_reading.py","reading_runner.py"]


def execute(root,heartbeat,*,resume=False):
    packet_id = "reading-scout-1"
    if resume and not (root/"packets"/f"{packet_id}.json").exists():
        raise ValueError("resume requires the already frozen reading packet")
    packet = freeze(root,packet_id,[PACKAGE/name for name in MODULES],
                    {"cards":DESIGNS,"evidence_scope":"discovery","n_per_condition":64,"constructors":8,
                     "seed_namespace":"v16-reading-discovery-1"})
    root_output = root/packet_id
    gate_path = root_output/"GATES.json"
    if resume and not gate_path.exists():
        raise ValueError("packet was interrupted before admission; explicit discovery can finish setup")
    receipt = read(gate_path) if gate_path.exists() else gates()
    write(gate_path,receipt)
    if receipt["instrument_state"] != "valid":
        raise ValueError("reading instrument failed its known-answer gates")
    total = sum(len(design["conditions"])*64 for design in DESIGNS.values())
    completed = 0
    for card,design in DESIGNS.items():
        output = root_output/card
        rows = []
        for condition in design["conditions"]:
            for index in range(64):
                if remaining_seconds(root) <= 0:
                    return {"execution_state":"checkpointed","reason":"immutable elapsed ceiling",
                            "completed_units":completed,"planned_units":total}
                row = execute_unit(output,card,condition,index,
                                   namespace=f"v16-reading-discovery-1-{card}",
                                   packet_hash=packet["packet_hash"])
                rows.append(row)
                completed += 1
                heartbeat(packet_id=packet_id,card_id=card,condition=condition["id"],
                          completed_units=completed,planned_units=total,last_unit=row["unit_id"])
        summary = summarize(card,rows)
        audited = audit(output,summary)
        summary["independent_reaggregation"] = "valid"
        write(output/"REAGGREGATION.json",audited)
        write(output/"SUMMARY.json",summary)
        raw = {str(path.relative_to(output)).replace("\\","/"):file_digest(path)
               for folder in ["public","private","predictions","units"]
               for path in sorted((output/folder).glob("*.json"))}
        write(output/"RAW_MANIFEST.json",{"files":raw,"archive_location":"this packet directory",
                                         "retention":"retain through final verified archival handoff"})
        write(output/"COMPLETION.json",{"execution_state":"completed","instrument_state":"valid",
                                        "criterion_state":"see paired contrasts","evidence_scope":"discovery",
                                        "n_maker_packets":len(rows),"warrant":"DESCRIPTIVE ONLY",
                                        "pursuit":"OPENED","expansion_state":"pending","confirmation_state":"untested"})
    return {"execution_state":"completed","packet_id":packet_id,"n_maker_packets":completed,
            "independent_reaggregation":"valid","campaign_complete":False}

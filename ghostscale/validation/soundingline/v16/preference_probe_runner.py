"""V03 instrument admission, explicit dependency joins and measured reader work."""
import sys
import time
from .preference_probe import DESIGN,EXTENSION
from .preference_probe_study import execute_unit as scientific_unit,summarize
from .preference_probe_gates import run as gates
from .audit_preference_probe import audit
from .tradeoffs_runner import MODULES as BASE_MODULES,dependencies
from .finite_runner import execute as finite_execute
from .resource_accounting import MeasuredReader
from .records import read,write,file_digest

PACKET_ID="preference-probe-scout-1"
SETUP="preference-probe-setup"
NAMESPACE="v16-preference-probe-discovery-1"
EXTENSIONS=[EXTENSION]
MODULES=BASE_MODULES+["preference_probe_world.py","preference_probe.py","preference_probe_reference.py",
    "preference_probe_study.py","preference_probe_gates.py","audit_preference_probe.py","preference_probe_runner.py"]
REQUIRED_TESTS=["test_preference_probes_preserve_collisions_and_execute_real_interventions",
    "test_preference_probe_information_uses_possible_answers_and_public_bytes_only",
    "test_preference_probe_paid_joins_future_and_all_scores_regenerate",
    "test_tradeoff_physics_null_and_selection_have_known_answers",
    "test_tradeoff_dates_identify_a_trajectory_beyond_undated_scores",
    "test_owned_reader_resource_measurement_preserves_public_predictions"]

def execute_unit(root,condition,index,*,reader,**kwargs):
    measured=MeasuredReader(reader);started=time.perf_counter();cpu=time.process_time()
    row=scientific_unit(root,condition,index,reader=measured,**kwargs)
    if measured.samples:
        write(root/"private"/f'{row["unit_id"]}-resources.json',{"unit_id":row["unit_id"],
            "scope":"observed unit execution, separate from selected logical operation counts",
            "parent_cpu_seconds":time.process_time()-cpu,"wall_seconds":time.perf_counter()-started,
            "requests":measured.samples,"startup_cost_included":False})
    return row

def execute(root,heartbeat,*,resume=False):
    dependencies(root,PACKET_ID)
    joins=[]
    for name in ["tradeoffs","trajectory"]:
        path=root/(name+"-scout-1")/"COMPLETION.json"
        record=read(path)
        if record["execution_state"]!="completed" or record["instrument_state"]!="valid":
            raise ValueError("conditional probe predecessor invalid")
        joins.append({"receipt":str(path.relative_to(root)).replace("\\","/"),"sha256":file_digest(path),"instrument_state":"valid"})
    write(root/PACKET_ID/"PROFILE_DEPENDENCIES.json",{"joins":joins,"scientific_gain_required":False})
    return finite_execute(root,heartbeat,sys.modules[__name__],resume=resume)

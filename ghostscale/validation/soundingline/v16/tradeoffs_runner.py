"""V01 conditional tradeoff packet and immutable capability dependency joins."""
import sys
import time
from .tradeoffs import design,EXTENSION
from .tradeoffs_study import execute_unit as scientific_unit,summarize
from .tradeoffs_gates import run as gates
from .audit_tradeoffs import audit
from .finite_runner import execute as finite_execute
from .resource_accounting import MeasuredReader
from .records import write,read,file_digest

DESIGN=design("V01")
PACKET_ID="tradeoffs-scout-1"
SETUP="tradeoffs-setup"
NAMESPACE="v16-tradeoffs-discovery-1"
EXTENSIONS=[EXTENSION]
MODULES=["assembly.py","assembly_reference.py","dependency_monitor.py","tradeoffs_world.py","tradeoffs.py",
    "tradeoffs_reference.py","tradeoffs_study.py","tradeoffs_gates.py","audit_tradeoffs.py","tradeoffs_runner.py",
    "audit_graphic_maker.py","records.py","estimands.py","audit_statistics.py","finite_runner.py","reader_process.py","resource_accounting.py"]
REQUIRED_TESTS=["test_tradeoff_physics_null_and_selection_have_known_answers",
    "test_tradeoff_public_reader_matches_independent_and_denies_hidden_fields",
    "test_tradeoff_actual_dated_choices_rejections_and_all_scores_regenerate",
    "test_tradeoff_dates_identify_a_trajectory_beyond_undated_scores",
    "test_owned_reader_resource_measurement_preserves_public_predictions"]

def dependencies(root,packet_id):
    paths=["reading-scout-1/P03/COMPLETION.json"]+[
        f"behavior-scout-1/{card}/COMPLETION.json" for card in ["O01","O02","O03","O04"]]+[
        f"{name}-scout-1/COMPLETION.json" for name in ["recognition","selection","audience","multi-actor"]]
    joins=[]
    for relative in paths:
        path=root/relative
        if not path.exists():
            raise ValueError(f"tradeoff eligibility dependency missing: {relative}")
        record=read(path)
        if record.get("execution_state")!="completed" or record.get("instrument_state")!="valid":
            raise ValueError(f"tradeoff eligibility dependency invalid: {relative}")
        joins.append({"receipt":relative,"sha256":file_digest(path),"instrument_state":"valid"})
    write(root/packet_id/"DEPENDENCIES.json",{"eligibility":"conditional constructor-defined tradeoffs only",
        "joins":joins,"scientific_gain_required":False,"human_value_inference":False})

def execute_unit(root,condition,index,*,reader,**kwargs):
    measured=MeasuredReader(reader);started=time.perf_counter();cpu=time.process_time()
    row=scientific_unit(root,condition,index,reader=measured,**kwargs)
    if measured.samples:
        write(root/"private"/f'{row["unit_id"]}-resources.json',{"unit_id":row["unit_id"],
            "scope":"observed unit execution, separate from scientific primitive-work counters",
            "parent_cpu_seconds":time.process_time()-cpu,"wall_seconds":time.perf_counter()-started,
            "requests":measured.samples,"startup_cost_included":False})
    return row

def execute(root,heartbeat,*,resume=False):
    dependencies(root,PACKET_ID)
    return finite_execute(root,heartbeat,sys.modules[__name__],resume=resume)

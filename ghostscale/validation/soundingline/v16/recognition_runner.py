"""M01 frozen recognition packet, with retained observed process resources."""
import sys
import time
from .recognition import DESIGN,EXTENSION
from .recognition_study import execute_unit as scientific_unit,summarize
from .recognition_gates import run as gates
from .audit_recognition import audit
from .finite_runner import execute as finite_execute
from .resource_accounting import MeasuredReader
from .records import write

PACKET_ID="recognition-scout-1"
SETUP="recognition-setup"
NAMESPACE="v16-recognition-discovery-1"
EXTENSIONS=[EXTENSION]
MODULES=["graphic_world.py","graphic_reference.py","recognition.py","recognition_study.py","recognition_gates.py",
         "audit_recognition.py","recognition_runner.py","records.py","estimands.py","audit_statistics.py",
         "finite_runner.py","reader_process.py","resource_accounting.py"]
REQUIRED_TESTS=["test_larger_graphic_world_independent_physics_and_boundaries",
 "test_large_graphic_acquisition_changes_new_composition_at_counted_budget",
 "test_large_graphic_behavior_recoding_does_not_identify_order",
 "test_acquired_core_style_separation_and_exact_prior_controls",
 "test_recognition_public_boundary_and_independent_prediction",
 "test_larger_maker_production_and_scores_independently_regenerate",
 "test_owned_reader_resource_measurement_preserves_public_predictions"]

def execute_unit(root,condition,index,*,reader,**kwargs):
    measured=MeasuredReader(reader)
    started=time.perf_counter();cpu=time.process_time()
    row=scientific_unit(root,condition,index,reader=measured,**kwargs)
    if measured.samples:
        write(root/"private"/f'{row["unit_id"]}-resources.json',{
            "unit_id":row["unit_id"],"scope":"observed unit execution, not a replacement for frozen scientific cost estimands",
            "parent_cpu_seconds":time.process_time()-cpu,"wall_seconds":time.perf_counter()-started,
            "requests":measured.samples,"startup_cost_included":False})
    return row

def execute(root,heartbeat,*,resume=False):
    return finite_execute(root,heartbeat,sys.modules[__name__],resume=resume)


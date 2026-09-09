"""M02 immutable production-selection packet with observed process resources."""
import sys
import time
from .selection import DESIGN,EXTENSION
from .selection_study import execute_unit as scientific_unit,summarize
from .selection_gates import run as gates
from .audit_selection import audit
from .finite_runner import execute as finite_execute
from .resource_accounting import MeasuredReader
from .records import write

PACKET_ID="selection-scout-1"
SETUP="selection-setup"
NAMESPACE="v16-selection-discovery-1"
EXTENSIONS=[EXTENSION]
MODULES=["graphic_world.py","graphic_reference.py","recognition.py","recognition_gates.py",
         "selection.py","selection_study.py","selection_gates.py","audit_selection.py","audit_graphic_maker.py",
         "selection_runner.py","records.py","estimands.py","audit_statistics.py",
         "finite_runner.py","reader_process.py","resource_accounting.py"]
REQUIRED_TESTS=["test_larger_graphic_world_independent_physics_and_boundaries",
 "test_large_graphic_acquisition_changes_new_composition_at_counted_budget",
 "test_large_graphic_behavior_recoding_does_not_identify_order",
 "test_actual_selection_law_and_paid_evidence_controls",
 "test_selected_batch_likelihood_matches_independent_enumeration",
 "test_selection_retains_all_production_and_scores_fresh_outcomes",
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


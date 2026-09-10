"""M03 immutable audience/context packet with observed process resources."""
import sys
import time
from .audience import DESIGN,EXTENSION
from .audience_study import execute_unit as scientific_unit,summarize
from .audience_gates import run as gates
from .audit_audience import audit
from .finite_runner import execute as finite_execute
from .resource_accounting import MeasuredReader
from .records import write

PACKET_ID="audience-scout-1"
SETUP="audience-setup"
NAMESPACE="v16-audience-discovery-1"
EXTENSIONS=[EXTENSION]
MODULES=["graphic_world.py","graphic_reference.py","recognition.py","recognition_gates.py",
         "selection.py","selection_study.py","selection_gates.py","audit_selection.py","audit_graphic_maker.py",
         "audience.py","audience_study.py","audience_gates.py","audit_audience.py","audience_runner.py",
         "records.py","estimands.py","audit_statistics.py","finite_runner.py","reader_process.py","resource_accounting.py"]
REQUIRED_TESTS=["test_audience_changes_execution_without_rewriting_known_history",
 "test_audience_reader_requires_paid_answers_and_matches_independent_joint",
 "test_audience_queries_old_history_and_new_work_independently_regenerate",
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

"""M04 immutable multiple-role packet with actual resource observations."""
import sys
import time
from .multi_actor import DESIGN,EXTENSION
from .multi_actor_study import execute_unit as scientific_unit,summarize
from .multi_actor_gates import run as gates
from .audit_multi_actor import audit
from .finite_runner import execute as finite_execute
from .resource_accounting import MeasuredReader
from .records import write

PACKET_ID="multi-actor-scout-1"
SETUP="multi-actor-setup"
NAMESPACE="v16-multi-actor-discovery-1"
EXTENSIONS=[EXTENSION]
MODULES=["graphic_world.py","graphic_reference.py","recognition.py","selection.py","audit_selection.py","audit_graphic_maker.py",
         "multi_actor_world.py","multi_actor.py","multi_actor_study.py","multi_actor_gates.py","multi_actor_reference.py",
         "audit_multi_actor.py","multi_actor_runner.py","records.py","estimands.py","audit_statistics.py",
         "finite_runner.py","reader_process.py","resource_accounting.py"]
REQUIRED_TESTS=["test_multi_actor_real_collisions_and_role_probes",
 "test_role_posterior_matches_independent_factorization_and_guards_paid_views",
 "test_multi_actor_all_candidate_and_role_outcomes_regenerate",
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

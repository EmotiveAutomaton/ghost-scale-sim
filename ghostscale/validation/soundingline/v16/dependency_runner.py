"""S03 frozen acquired-routine mismatch packet."""
import sys
from .dependency_monitor import DESIGN,EXTENSION
from .dependency_study import execute_unit,summarize
from .dependency_gates import run as gates
from .audit_dependency import audit
from .finite_runner import execute as finite_execute

PACKET_ID="dependency-scout-1"
SETUP="dependency-setup"
NAMESPACE="v16-dependency-discovery-1"
EXTENSIONS=[EXTENSION]
MODULES=["assembly.py","assembly_reference.py","records.py","estimands.py","audit_statistics.py",
         "dependency_monitor.py","dependency_study.py","dependency_gates.py","audit_dependency.py",
         "dependency_runner.py","finite_runner.py","reader_process.py"]
REQUIRED_TESTS=["test_actual_dependency_and_no_false_alarm_controls",
                "test_paid_inspection_and_committed_repairs_regenerate"]

def execute(root,heartbeat,*,resume=False):
    return finite_execute(root,heartbeat,sys.modules[__name__],resume=resume)


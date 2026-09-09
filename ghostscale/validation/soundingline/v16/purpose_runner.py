"""K03 frozen scientific packet adapter."""
import sys
from .purpose_craft import DESIGN,EXTENSION
from .purpose_study import execute_unit,summarize
from .purpose_gates import run as gates
from .audit_purpose import audit
from .finite_runner import execute as finite_execute

PACKET_ID="purpose-scout-1"
SETUP="purpose-setup"
NAMESPACE="v16-purpose-discovery-1"
EXTENSIONS=[EXTENSION]
MODULES=["world.py","reference.py","learning.py","craft.py","records.py","estimands.py",
         "purpose_craft.py","purpose_study.py","purpose_gates.py","audit_statistics.py","audit_purpose.py",
         "purpose_runner.py","finite_runner.py","reader_process.py"]
REQUIRED_TESTS=["test_purpose_relation_and_inhibition_are_measured_before_discovery",
               "test_adaptation_executes_and_independent_checker_detects_cost_tampering"]


def execute(root,heartbeat,*,resume=False):
    return finite_execute(root,heartbeat,sys.modules[__name__],resume=resume)

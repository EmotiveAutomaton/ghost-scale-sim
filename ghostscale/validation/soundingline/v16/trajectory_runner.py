"""V02 dated inherited-craft and redirected-purpose continuation packet."""
import sys
from .tradeoffs import design,EXTENSION
from .tradeoffs_runner import MODULES as BASE_MODULES,REQUIRED_TESTS,dependencies,execute_unit as measured_unit
from .tradeoffs_study import summarize as summarize_science
from .tradeoffs_gates import run as gates
from .audit_tradeoffs import audit
from .finite_runner import execute as finite_execute

DESIGN=design("V02")
PACKET_ID="trajectory-scout-1"
SETUP="trajectory-setup"
NAMESPACE="v16-tradeoff-trajectory-discovery-1"
EXTENSIONS=[EXTENSION]
MODULES=BASE_MODULES+["trajectory_runner.py"]

def execute_unit(root,condition,index,**kwargs):
    return measured_unit(root,condition,index,card="V02",**kwargs)

def summarize(rows):
    return summarize_science(rows,card="V02")

def execute(root,heartbeat,*,resume=False):
    dependencies(root,PACKET_ID)
    return finite_execute(root,heartbeat,sys.modules[__name__],resume=resume)

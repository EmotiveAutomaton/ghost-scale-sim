"""K05 frozen scientific packet adapter."""
import sys
from .attention_craft import DESIGN,EXTENSION,LEARNING_EXTENSION
from .attention_study import execute_unit,summarize
from .attention_gates import run as gates
from .audit_attention import audit
from .finite_runner import execute as finite_execute

PACKET_ID="attention-scout-1"
SETUP="attention-setup"
NAMESPACE="v16-attention-discovery-1"
EXTENSIONS=[EXTENSION,LEARNING_EXTENSION]
MODULES=["world.py","reference.py","learning.py","craft.py","records.py","estimands.py",
         "attention_craft.py","attention_study.py","attention_gates.py","audit_statistics.py","audit_attention.py",
         "attention_runner.py","finite_runner.py","reader_process.py"]
REQUIRED_TESTS=["test_attention_attempts_and_information_controls",
               "test_acquisition_is_committed_before_task_reveal_and_costs_regenerate"]


def execute(root,heartbeat,*,resume=False):
    return finite_execute(root,heartbeat,sys.modules[__name__],resume=resume)

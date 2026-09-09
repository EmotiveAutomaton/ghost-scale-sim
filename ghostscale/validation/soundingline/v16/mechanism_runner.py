"""P04 frozen cross-mechanism and independent-physics packet adapter."""
import sys
from .mechanism_study import DESIGN,EXTENSION,execute_unit,summarize
from .mechanism_gates import run as gates
from .audit_mechanism import audit
from .finite_runner import execute as finite_execute

PACKET_ID="mechanism-scout-1"
SETUP="mechanism-setup"
NAMESPACE="v16-mechanism-discovery-1"
EXTENSIONS=[EXTENSION]
MODULES=["world.py","reference.py","learning.py","craft.py","records.py","estimands.py","reconstruction.py",
         "assembly.py","assembly_reference.py","assembly_gates.py","assembly_inference.py","options.py",
         "mechanism_models.py","mechanism_reader.py","mechanism_study.py","mechanism_gates.py","audit_statistics.py",
         "audit_purpose.py","audit_mechanism.py","mechanism_runner.py","finite_runner.py","reader_process.py"]
REQUIRED_TESTS=["test_independent_assembly_physics_and_actual_revision_controls",
    "test_bounded_assembly_planner_reports_timeout_and_unreachable_separately",
    "test_assembly_extension_runs_through_public_only_worker",
    "test_assembly_acquisition_prior_matches_independent_ordered_training_enumeration",
    "test_assembly_artifact_likelihood_matches_independent_scalar_programs",
    "test_assembly_reader_uses_history_without_claiming_unique_route",
    "test_habit_prior_retains_order_and_matches_ordered_acquisition",
    "test_habit_is_actual_prior_execution_not_a_different_outcome_label",
    "test_bounded_and_habit_artifact_distributions_are_real_executions",
    "test_option_generator_learns_from_public_executed_generic_exploration",
    "test_same_information_joint_mixture_matches_and_uses_evidence",
    "test_off_model_reverse_direction_is_an_explicit_failure_not_uniform",
    "test_mechanism_reader_public_boundary_and_unknown_truth_fields",
    "test_distinct_maker_mechanisms_and_independent_physics_are_retained",
    "test_all_mechanism_admission_controls_pass"]


def execute(root,heartbeat,*,resume=False):
    return finite_execute(root,heartbeat,sys.modules[__name__],resume=resume)

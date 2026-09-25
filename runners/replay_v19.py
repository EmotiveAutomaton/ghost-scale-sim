"""Portable replay driver; run against an extracted, matching SOURCE.zip."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import argparse
from datetime import datetime,timezone
from pathlib import Path
import sys
import time


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--packet',type=Path,required=True);parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--retained-root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cpu-limit',type=float,default=7200)
    parser.add_argument('--inputs',type=Path,help='retained successor inputs with hashes pinned in the plan')
    parser.add_argument('--torch-python',type=Path,help='existing matching Torch interpreter for a tiny-reader replay; no installation')
    args=parser.parse_args();sys.path.insert(0,str(args.source.resolve()))
    from ghostscale.validation.soundingline.v18_3.io import read,write,file_digest
    from ghostscale.validation.soundingline.v18_4.priority import below_normal
    from ghostscale.validation.soundingline.v19.runtime import fingerprint,REPO
    below_normal();start=time.process_time();plan=read(args.packet/'PLAN.json')
    if REPO.resolve()!=args.source.resolve():raise ValueError('matching extracted source was not imported')
    if plan['environment']!=fingerprint():raise ValueError('replay environment differs')
    if any(file_digest(args.source/n)!=h for n,h in plan['sources'].items()):raise ValueError('replay source differs')
    args.output.mkdir(parents=True,exist_ok=False)
    if plan['design'].get('input_files'):
        import shutil
        if args.inputs is None:raise ValueError('this successor requires --inputs')
        for name,h in plan['design']['input_files'].items():
            if file_digest(args.inputs/name)!=h:raise ValueError('successor input differs')
            dest=args.output/'inputs'/name;dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(args.inputs/name,dest)
    child_cpu=0.
    def pulse(child_cpu_seconds=None,**details):
        nonlocal child_cpu
        if child_cpu_seconds is not None:child_cpu=max(child_cpu,float(child_cpu_seconds))
        left=args.cpu_limit-(time.process_time()-start)-child_cpu
        if left<=0:raise TimeoutError('replay CPU budget')
        return left
    kind=plan['design']['handler']
    if kind=='certify':from ghostscale.validation.soundingline.v19.retained import certify as handler
    elif kind=='packet-zero':from ghostscale.validation.soundingline.v19.retained import packet_zero as handler
    elif kind=='local-world':from ghostscale.validation.soundingline.v19.local_world import admission as handler
    elif kind=='state-error':from ghostscale.validation.soundingline.v19.state_error import run as handler
    elif kind=='forecast-collision':from ghostscale.validation.soundingline.v19.forecast_collision import run as handler
    elif kind=='forecast-collision-review':from ghostscale.validation.soundingline.v19.forecast_collision_review import run as handler
    elif kind=='joint-forecast-partition':from ghostscale.validation.soundingline.v19.joint_forecast_partition import run as handler
    elif kind=='joint-partition-review':from ghostscale.validation.soundingline.v19.joint_partition_review import run as handler
    elif kind=='retrospective-quotient':from ghostscale.validation.soundingline.v19.retrospective_quotient import run as handler
    elif kind=='retrospective-review':from ghostscale.validation.soundingline.v19.retrospective_review import run as handler
    elif kind=='reachable-retrospective':from ghostscale.validation.soundingline.v19.reachable_retrospective import run as handler
    elif kind=='retrospective-sufficient':from ghostscale.validation.soundingline.v19.retrospective_sufficient import run as handler
    elif kind=='retrospective-source':from ghostscale.validation.soundingline.v19.retrospective_source import run as handler
    elif kind=='sufficient-review':from ghostscale.validation.soundingline.v19.sufficient_review import run as handler
    elif kind=='source-review':from ghostscale.validation.soundingline.v19.source_review import run as handler
    elif kind=='source-interval':from ghostscale.validation.soundingline.v19.source_interval import run as handler
    elif kind=='interval-review':from ghostscale.validation.soundingline.v19.interval_review import run as handler
    elif kind=='source-prior':from ghostscale.validation.soundingline.v19.source_prior import run as handler
    elif kind=='balanced-review':from ghostscale.validation.soundingline.v19.balanced_review import run as handler
    elif kind=='time-retention-review':from ghostscale.validation.soundingline.v19.time_retention_review import run as handler
    elif kind=='retention-mass-priority':from ghostscale.validation.soundingline.v19.retention_mass_priority import run as handler
    elif kind=='retention-source-prior':from ghostscale.validation.soundingline.v19.retention_source_prior import run as handler
    elif kind=='retention-horizon-review':from ghostscale.validation.soundingline.v19.retention_horizon_review import run as handler
    elif kind=='retention-query-review':from ghostscale.validation.soundingline.v19.retention_query_review import run as handler
    elif kind=='byte-retention-review':from ghostscale.validation.soundingline.v19.byte_retention_review import run as handler
    elif kind=='aggregate-report-review':from ghostscale.validation.soundingline.v19.aggregate_report_review import run as handler
    elif kind=='aggregate-prior-basis':from ghostscale.validation.soundingline.v19.aggregate_prior_basis import run as handler
    elif kind=='aggregate-context-state':from ghostscale.validation.soundingline.v19.aggregate_context_state import run as handler
    elif kind=='aggregate-report-state':from ghostscale.validation.soundingline.v19.aggregate_report_state import run as handler
    elif kind=='law-rounding-envelope':from ghostscale.validation.soundingline.v19.law_rounding_envelope import run as handler
    elif kind=='law-likelihood-envelope':from ghostscale.validation.soundingline.v19.law_likelihood_envelope import run as handler
    elif kind=='schedule-likelihood-envelope':from ghostscale.validation.soundingline.v19.schedule_likelihood_envelope import run as handler
    elif kind=='context-precision-allocation':from ghostscale.validation.soundingline.v19.context_precision_allocation import run as handler
    elif kind=='robust-context-precision':from ghostscale.validation.soundingline.v19.robust_context_precision import run as handler
    elif kind=='context-precision-regret':from ghostscale.validation.soundingline.v19.context_precision_regret import run as handler
    elif kind=='robust-source-mass':from ghostscale.validation.soundingline.v19.robust_source_mass import run as handler
    elif kind=='source-prior-regret':from ghostscale.validation.soundingline.v19.source_prior_regret import run as handler
    elif kind=='randomized-source-storage':from ghostscale.validation.soundingline.v19.randomized_source_storage import run as handler
    elif kind=='randomized-source-regret':from ghostscale.validation.soundingline.v19.randomized_source_regret import run as handler
    elif kind=='prior-information':
        from ghostscale.validation.soundingline.v19.prior_information import run as handler
    elif kind=='realized-risk-storage':from ghostscale.validation.soundingline.v19.realized_risk_storage import run as handler
    elif kind=='source-mass-frontier':from ghostscale.validation.soundingline.v19.source_mass_frontier import run as handler
    elif kind=='shared-law-precision':from ghostscale.validation.soundingline.v19.shared_law_precision import run as handler
    elif kind=='aggregate-precision':from ghostscale.validation.soundingline.v19.aggregate_precision import run as handler
    elif kind=='byte-retention':from ghostscale.validation.soundingline.v19.byte_retention import run as handler
    elif kind=='announced-query-retention':from ghostscale.validation.soundingline.v19.announced_query_retention import run as handler
    elif kind=='retention-query':from ghostscale.validation.soundingline.v19.retention_query import run as handler
    elif kind=='retention-horizon':from ghostscale.validation.soundingline.v19.retention_horizon import run as handler
    elif kind=='time-retention':from ghostscale.validation.soundingline.v19.time_retention import run as handler
    elif kind=='balanced-retention':from ghostscale.validation.soundingline.v19.balanced_retention import run as handler
    elif kind=='retrospective-forgetting':from ghostscale.validation.soundingline.v19.retrospective_forgetting import run as handler
    elif kind=='report-context':from ghostscale.validation.soundingline.v19.report_context import run as handler
    elif kind=='report-coarsening':from ghostscale.validation.soundingline.v19.report_coarsening import run as handler
    elif kind=='forgetting-review':from ghostscale.validation.soundingline.v19.forgetting_review import run as handler
    elif kind=='context-review':from ghostscale.validation.soundingline.v19.context_review import run as handler
    elif kind=='coarsening-review':from ghostscale.validation.soundingline.v19.coarsening_review import run as handler
    elif kind=='prior-review':from ghostscale.validation.soundingline.v19.prior_review import run as handler
    elif kind=='source-disclosure':from ghostscale.validation.soundingline.v19.source_disclosure import run as handler
    elif kind=='source-identity':from ghostscale.validation.soundingline.v19.source_identity import run as handler
    elif kind=='identity-review':from ghostscale.validation.soundingline.v19.identity_review import run as handler
    elif kind in ('readout-scout','local-primary'):from ghostscale.validation.soundingline.v19.readouts import run as handler
    elif kind in ('frame','jointness'):from ghostscale.validation.soundingline.v19.local_alternatives import run as handler
    elif kind=='probability-head':from ghostscale.validation.soundingline.v19.probability_head import run as handler
    elif kind=='process-sufficiency':from ghostscale.validation.soundingline.v19.process_sufficiency import run as handler
    elif kind=='probability-convergence':from ghostscale.validation.soundingline.v19.probability_convergence import run as handler
    elif kind=='convergence-review':from ghostscale.validation.soundingline.v19.convergence_review import run as handler
    elif kind=='recursive-update':from ghostscale.validation.soundingline.v19.recursive_update import run as handler
    elif kind=='roll-in':from ghostscale.validation.soundingline.v19.roll_in import run as handler
    elif kind=='task-assessment':from ghostscale.validation.soundingline.v19.task_assessment import run as handler
    elif kind=='bounded-reset':from ghostscale.validation.soundingline.v19.bounded_reset import run as handler
    elif kind=='persistent-practice':from ghostscale.validation.soundingline.v19.practice import run as handler
    elif kind=='tiny-reader':from ghostscale.validation.soundingline.v19.tiny_reader import run as handler
    elif kind=='purchase-calibration':from ghostscale.validation.soundingline.v19.purchase_calibration import run as handler
    elif kind=='purchase-evaluation':from ghostscale.validation.soundingline.v19.purchase_evaluation import run as handler
    elif kind=='purchase-review':from ghostscale.validation.soundingline.v19.purchase_review import run as handler
    elif kind=='portfolio':from ghostscale.validation.soundingline.v19.portfolio import run as handler
    elif kind=='portfolio-review':from ghostscale.validation.soundingline.v19.portfolio_review import run as handler
    elif kind=='interchange-fixtures':from ghostscale.validation.soundingline.v19.interchange_fixtures import run as handler
    elif kind=='interchange-alignment':from ghostscale.validation.soundingline.v19.interchange_alignment import run as handler
    elif kind=='cue-correction':from ghostscale.validation.soundingline.v19.cue_correction import run as handler
    elif kind=='missing-tool':from ghostscale.validation.soundingline.v19.missing_tool import run as handler
    elif kind=='forward-rollout':from ghostscale.validation.soundingline.v19.rollout import run as handler
    elif kind=='joint-readout':from ghostscale.validation.soundingline.v19.joint_readout import run as handler
    elif kind=='joint-support':from ghostscale.validation.soundingline.v19.joint_support import run as handler
    elif kind=='joint-support-review':from ghostscale.validation.soundingline.v19.joint_support_review import run as handler
    elif kind=='joint-uncertainty':from ghostscale.validation.soundingline.v19.joint_uncertainty import run as handler
    elif kind=='joint-uncertainty-review':from ghostscale.validation.soundingline.v19.joint_uncertainty_review import run as handler
    elif kind=='goal-decision':from ghostscale.validation.soundingline.v19.goal_decision import run as handler
    elif kind=='goal-abstention':from ghostscale.validation.soundingline.v19.goal_abstention import run as handler
    elif kind=='goal-calibration':from ghostscale.validation.soundingline.v19.goal_calibration import run as handler
    elif kind=='goal-calibration-review':from ghostscale.validation.soundingline.v19.goal_calibration_review import run as handler
    elif kind=='goal-operation-calibration-review':from ghostscale.validation.soundingline.v19.goal_operation_calibration_review import run as handler
    elif kind=='goal-purpose-calibration-review':from ghostscale.validation.soundingline.v19.goal_purpose_calibration_review import run as handler
    elif kind=='goal-bin-resolution-review':from ghostscale.validation.soundingline.v19.goal_bin_resolution_review import run as handler
    elif kind=='goal-squared-decomposition-review':from ghostscale.validation.soundingline.v19.goal_squared_decomposition_review import run as handler
    elif kind=='goal-absolute-envelope-review':from ghostscale.validation.soundingline.v19.goal_absolute_envelope_review import run as handler
    elif kind=='goal-absolute-envelope':from ghostscale.validation.soundingline.v19.goal_absolute_envelope import run as handler
    elif kind=='goal-squared-decomposition':from ghostscale.validation.soundingline.v19.goal_squared_decomposition import run as handler
    elif kind=='goal-bin-resolution':from ghostscale.validation.soundingline.v19.goal_bin_resolution import run as handler
    elif kind=='goal-purpose-calibration':from ghostscale.validation.soundingline.v19.goal_purpose_calibration import run as handler
    elif kind=='goal-artifact-calibration':from ghostscale.validation.soundingline.v19.goal_artifact_calibration import run as handler
    elif kind=='goal-artifact-calibration-review':from ghostscale.validation.soundingline.v19.goal_artifact_calibration_review import run as handler
    elif kind=='goal-operation-calibration':from ghostscale.validation.soundingline.v19.goal_operation_calibration import run as handler
    elif kind=='goal-class-reliability':from ghostscale.validation.soundingline.v19.goal_class_reliability import run as handler
    elif kind=='goal-class-reliability-review':from ghostscale.validation.soundingline.v19.goal_class_reliability_review import run as handler
    elif kind=='goal-coarsening':from ghostscale.validation.soundingline.v19.goal_coarsening import run as handler
    elif kind=='goal-coarsening-review':from ghostscale.validation.soundingline.v19.goal_coarsening_review import run as handler
    elif kind=='goal-abstention-review':from ghostscale.validation.soundingline.v19.goal_abstention_review import run as handler
    elif kind=='goal-decision-review':from ghostscale.validation.soundingline.v19.goal_decision_review import run as handler
    elif kind=='goal-mass-review':from ghostscale.validation.soundingline.v19.goal_mass_review import run as handler
    elif kind=='goal-mass-control':from ghostscale.validation.soundingline.v19.goal_mass_control import run as handler
    elif kind=='temporal-mass-control':from ghostscale.validation.soundingline.v19.temporal_mass_control import run as handler
    elif kind=='witnessed-goal-review':from ghostscale.validation.soundingline.v19.witnessed_goal_review import run as handler
    elif kind=='witnessed-goal-factorization':from ghostscale.validation.soundingline.v19.witnessed_goal_factorization import run as handler
    elif kind=='within-step-factorization':from ghostscale.validation.soundingline.v19.within_step_factorization import run as handler
    elif kind=='temporal-factorization-review':from ghostscale.validation.soundingline.v19.temporal_factorization_review import run as handler
    elif kind=='temporal-factorization':from ghostscale.validation.soundingline.v19.temporal_factorization import run as handler
    elif kind=='joint-factorization':from ghostscale.validation.soundingline.v19.joint_factorization import run as handler
    elif kind=='joint-factorization-review':from ghostscale.validation.soundingline.v19.joint_factorization_review import run as handler
    elif kind=='joint-review':from ghostscale.validation.soundingline.v19.joint_review import run as handler
    elif kind=='crossed-rules':from ghostscale.validation.soundingline.v19.crossed_rules import run as handler
    elif kind=='crossed-review':from ghostscale.validation.soundingline.v19.crossed_review import run as handler
    elif kind=='forward-support':from ghostscale.validation.soundingline.v19.forward_support import run as handler
    elif kind=='support-transfer':from ghostscale.validation.soundingline.v19.support_transfer import run as handler
    elif kind=='transfer-review':from ghostscale.validation.soundingline.v19.transfer_review import run as handler
    elif kind=='support-review':from ghostscale.validation.soundingline.v19.support_review import run as handler
    elif kind=='routine-revision':from ghostscale.validation.soundingline.v19.routine_revision import run as handler
    elif kind=='routine-review':from ghostscale.validation.soundingline.v19.routine_review import run as handler
    elif kind=='transient-filter':from ghostscale.validation.soundingline.v19.transient_filter import run as handler
    elif kind=='transient-review':from ghostscale.validation.soundingline.v19.transient_review import run as handler
    elif kind=='unknown-review':from ghostscale.validation.soundingline.v19.unknown_review import run as handler
    elif kind=='unknown-change':from ghostscale.validation.soundingline.v19.unknown_change import run as handler
    elif kind=='timing-review':from ghostscale.validation.soundingline.v19.timing_review import run as handler
    elif kind=='change-timing':from ghostscale.validation.soundingline.v19.change_timing import run as handler
    elif kind=='omission-review':from ghostscale.validation.soundingline.v19.omission_review import run as handler
    elif kind=='filter-cost':from ghostscale.validation.soundingline.v19.filter_cost import run as handler
    elif kind=='provenance-tempering':from ghostscale.validation.soundingline.v19.provenance_tempering import run as handler
    elif kind=='tempering-review':from ghostscale.validation.soundingline.v19.tempering_review import run as handler
    elif kind=='mixture-compression':from ghostscale.validation.soundingline.v19.mixture_compression import run as handler
    elif kind=='compression-review':from ghostscale.validation.soundingline.v19.compression_review import run as handler
    elif kind=='marginal-transition':from ghostscale.validation.soundingline.v19.marginal_transition import run as handler
    elif kind=='future-quotient':from ghostscale.validation.soundingline.v19.future_quotient import run as handler
    elif kind=='primitive-feedback':from ghostscale.validation.soundingline.v19.primitive_feedback import run as handler
    elif kind=='pooled-replacement':from ghostscale.validation.soundingline.v19.pooled_replacement import run as handler
    elif kind=='primitive-pooling':from ghostscale.validation.soundingline.v19.primitive_pooling import run as handler
    elif kind=='replacement-review':from ghostscale.validation.soundingline.v19.replacement_review import run as handler
    elif kind=='pooling-review':from ghostscale.validation.soundingline.v19.pooling_review import run as handler
    elif kind=='noisy-disclosure':from ghostscale.validation.soundingline.v19.noisy_disclosure import run as handler
    elif kind=='noisy-review':from ghostscale.validation.soundingline.v19.noisy_review import run as handler
    elif kind=='repeated-disclosure':from ghostscale.validation.soundingline.v19.repeated_disclosure import run as handler
    elif kind=='repeated-review':from ghostscale.validation.soundingline.v19.repeated_review import run as handler
    elif kind=='joint-reply':from ghostscale.validation.soundingline.v19.joint_reply import run as handler
    elif kind=='reliability-mismatch':from ghostscale.validation.soundingline.v19.reliability_mismatch import run as handler
    elif kind=='mismatch-review':from ghostscale.validation.soundingline.v19.mismatch_review import run as handler
    elif kind=='uncertain-reliability':from ghostscale.validation.soundingline.v19.uncertain_reliability import run as handler
    elif kind=='uncertainty-review':from ghostscale.validation.soundingline.v19.uncertainty_review import run as handler
    elif kind=='robust-reply':from ghostscale.validation.soundingline.v19.robust_reply import run as handler
    elif kind=='robust-review':from ghostscale.validation.soundingline.v19.robust_review import run as handler
    elif kind=='continuous-reliability':from ghostscale.validation.soundingline.v19.continuous_reliability import run as handler
    elif kind=='continuous-review':from ghostscale.validation.soundingline.v19.continuous_review import run as handler
    elif kind=='joint-reply-review':from ghostscale.validation.soundingline.v19.joint_reply_review import run as handler
    elif kind=='optional-review':from ghostscale.validation.soundingline.v19.optional_review import run as handler
    elif kind=='optional-disclosure':from ghostscale.validation.soundingline.v19.optional_disclosure import run as handler
    elif kind=='disclosure-review':from ghostscale.validation.soundingline.v19.disclosure_review import run as handler
    elif kind=='metadata-disclosure':from ghostscale.validation.soundingline.v19.metadata_disclosure import run as handler
    elif kind=='input-privilege':from ghostscale.validation.soundingline.v19.input_privilege import run as handler
    elif kind=='input-privilege-review':from ghostscale.validation.soundingline.v19.input_privilege_review import run as handler
    elif kind=='feedback-review':from ghostscale.validation.soundingline.v19.feedback_review import run as handler
    elif kind=='quotient-review':from ghostscale.validation.soundingline.v19.quotient_review import run as handler
    elif kind=='source-omission':from ghostscale.validation.soundingline.v19.source_omission import run as handler
    elif kind=='provenance-restoration':from ghostscale.validation.soundingline.v19.provenance_restoration import run as handler
    elif kind=='restoration-review':from ghostscale.validation.soundingline.v19.restoration_review import run as handler
    elif kind=='support-mix':from ghostscale.validation.soundingline.v19.support_mix import run as handler
    elif kind=='mix-review':from ghostscale.validation.soundingline.v19.mix_review import run as handler
    elif kind=='rollout-transfer':from ghostscale.validation.soundingline.v19.rollout_transfer import run as handler
    else:raise ValueError('unsupported replay handler')
    effective=dict(plan,design=dict(plan['design'],retained_root=str(args.retained_root.resolve())))
    if kind=='tiny-reader':
        if args.torch_python is None or not args.torch_python.is_file():raise ValueError('tiny replay requires --torch-python')
        from datetime import timedelta
        campaign=args.output/'replay-campaign';campaign.mkdir()
        settings=args.output/'inputs/shared/TINY_SETTINGS.json'
        if file_digest(settings)!=plan['design']['settings_receipt_sha256']:raise ValueError('settings receipt differs')
        shutil.copyfile(settings,campaign/'TINY_SETTINGS.json')
        write(campaign/'TINY_TRAINING_CONFIG.json',dict(python=str(args.torch_python.resolve()),python_sha256=file_digest(args.torch_python)))
        # Preserve the scientific config's deadline byte identity for forecasts/fit identity.
        write(campaign/'ACCEPTANCE.json',dict(deadline=plan['design']['deadline']))
        if datetime.now(timezone.utc)>=datetime.fromisoformat(plan['design']['deadline']):raise ValueError('campaign cutoff elapsed; later reproduction needs a separately authorized replay plan')
        effective['campaign']=campaign
    summary=handler(args.output,effective,pulse);write(args.output/'SUMMARY.json',summary)
    expected=read(args.packet/'COMPLETE.json')['files']
    if (args.packet/'EVALUATOR_MANIFEST.json').exists():expected.update(read(args.packet/'EVALUATOR_MANIFEST.json')['files'])
    for name,sha in expected.items():
        if file_digest(args.output/name)!=sha:raise ValueError('replay differs: '+name)
    write(args.output/'REPLAY.json',dict(passed=True,files=expected,cpu_seconds=time.process_time()-start,child_cpu_seconds=child_cpu))
    print('V19 portable replay passed:',len(expected),'output identities')


if __name__=='__main__':main()

"""Short V19 handlers using the existing ownership, evidence and accounting tools."""
from datetime import datetime, timezone
from pathlib import Path
import os
import time
import uuid
import zipfile
from ..v16.runtime import local_owner
from ..v18_3.io import read, write, file_digest, now
from ..v18_4.runtime import fingerprint as inherited_fingerprint
from ..v18_4.priority import below_normal

REPO=Path(__file__).resolve().parents[4]


def fingerprint():
    import scipy
    return dict(inherited_fingerprint(),scipy=scipy.__version__)


def source_files():
    from ..v18_4.runtime import source_files as inherited_sources
    names=inherited_sources()
    for version in ('v16','v18_3','v18_4','v19'):
        names += [p.relative_to(REPO).as_posix() for p in (REPO/'ghostscale/validation/soundingline'/version).glob('*.py')]
    names += ['runners/run_v19.py','runners/replay_v19.py','runners/watch_v19.py','runners/watch_v18_4.py','runners/launch_background.py']
    names += [p.relative_to(REPO).as_posix() for p in (REPO/'tests').glob('test_v19*.py')]
    names += [p.relative_to(REPO).as_posix() for p in (REPO/'docs/versions/v19-local-maker').glob('*') if p.is_file()]
    return sorted(set(names))


def freeze(source,root,design,admission):
    files={name:file_digest(REPO/name) for name in source_files()}
    if not admission.get('passed') or admission['sources']!=files:
        raise ValueError('current source admission required')
    root.mkdir(parents=True,exist_ok=False);source.mkdir(parents=True,exist_ok=False)
    for name in files:
        p=source/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((REPO/name).read_bytes())
    with zipfile.ZipFile(root/'SOURCE.zip','x',zipfile.ZIP_DEFLATED) as z:
        for name in files:z.write(REPO/name,name)
    plan=dict(schema='v19.plan.1',design=design,sources=files,environment=fingerprint(),admission=admission,
        source_archive_sha256=file_digest(root/'SOURCE.zip'),scope='constructed method; no human generalization')
    write(root/'PLAN.json',plan)
    return plan


def consumed(campaign):
    previous=[read(p) for p in campaign.glob('attempts/*.json')]
    if any(a['state']=='running' for a in previous):raise ValueError('unreconciled running CPU attempt')
    return sum(max(a['cpu_seconds'],a.get('native_cpu_seconds',0),a.get('uncertainty_cpu_seconds',0))+a.get('child_cpu_seconds',0) for a in previous)


def card_consumed(campaign,design,packet):
    card=design.get('accounting_card',packet)
    return design.get('setup_charge_seconds',0)+sum(
        max(a['cpu_seconds'],a.get('native_cpu_seconds',0),a.get('uncertainty_cpu_seconds',0))+a.get('child_cpu_seconds',0)
        for a in (read(p) for p in campaign.glob('attempts/*.json'))
        if a.get('accounting_card',a['packet'])==card)


def run(root,campaign):
    with local_owner(campaign/'scientific-worker-owner'),local_owner(root):
        plan=read(root/'PLAN.json');acceptance=read(campaign/'ACCEPTANCE.json')
        if plan['environment']!=fingerprint():raise ValueError('environment fingerprint changed')
        if any(file_digest(REPO/n)!=v for n,v in plan['sources'].items()):raise ValueError('frozen source changed')
        if file_digest(root/'SOURCE.zip')!=plan['source_archive_sha256']:raise ValueError('archive changed')
        if (root/'COMPLETE.json').exists():
            receipt=read(root/'COMPLETE.json')
            if receipt['plan_sha256']!=file_digest(root/'PLAN.json'):raise ValueError('complete identity changed')
            if any(file_digest(root/n)!=h for n,h in {**receipt['files'],**receipt.get('execution_measurements',{})}.items()):raise ValueError('complete output changed')
            return 'complete'
        below_normal()
        previous=consumed(campaign)
        prior_card=card_consumed(campaign,plan['design'],root.name)
        attempt=uuid.uuid4().hex;started=time.monotonic();cpu=time.process_time();child_cpu=0.
        def emit(state,**details):
            write(root/'STATUS.json',dict(pid=os.getpid(),parent_pid=os.getppid(),attempt=attempt,heartbeat=now(),state=state,**details),immutable=False)
            write(campaign/'attempts'/f'{attempt}.json',dict(packet=root.name,state=state,cpu_seconds=time.process_time()-cpu,
                accounting_card=plan['design'].get('accounting_card',root.name),
                child_cpu_seconds=child_cpu,wall_seconds=time.monotonic()-started),immutable=False)
        def pulse(child_cpu_seconds=None,**details):
            nonlocal child_cpu
            if child_cpu_seconds is not None:child_cpu=max(child_cpu,float(child_cpu_seconds))
            used=time.process_time()-cpu+child_cpu
            cap=acceptance['cumulative_cpu_ceiling_seconds'] if plan['design'].get('reserve_eligible') else acceptance['exploratory_cpu_ceiling_seconds']
            if ((campaign/'STOP').exists() or datetime.now(timezone.utc)>=datetime.fromisoformat(acceptance['deadline'])
                or previous+used>=cap or prior_card+used>=plan['design']['cpu_cap_seconds']):
                raise TimeoutError('V19 resource cutoff; partial outputs remain incomplete')
            emit('running',**details)
            return min(cap-previous-used,plan['design']['cpu_cap_seconds']-prior_card-used)
        emit('running')
        try:
            effective=dict(plan,design=dict(plan['design']))
            effective['design']['retained_root']=read(campaign/'INPUTS.json')['retained_root']
            if plan['design']['handler']=='certify':
                from .retained import certify as handler
            elif plan['design']['handler']=='packet-zero':
                from .retained import packet_zero as handler
            elif plan['design']['handler']=='local-world':
                from .local_world import admission as handler
            elif plan['design']['handler']=='state-error':
                from .state_error import run as handler
            elif plan['design']['handler'] in ('readout-scout','local-primary'):
                from .readouts import run as handler
            elif plan['design']['handler'] in ('frame','jointness'):
                from .local_alternatives import run as handler
            elif plan['design']['handler']=='probability-head':
                from .probability_head import run as handler
            elif plan['design']['handler']=='process-sufficiency':
                from .process_sufficiency import run as handler
            elif plan['design']['handler']=='probability-convergence':
                from .probability_convergence import run as handler
            elif plan['design']['handler']=='recursive-update':
                from .recursive_update import run as handler
            elif plan['design']['handler']=='convergence-review':
                from .convergence_review import run as handler
            elif plan['design']['handler']=='roll-in':
                from .roll_in import run as handler
            elif plan['design']['handler']=='task-assessment':
                from .task_assessment import run as handler
            elif plan['design']['handler']=='bounded-reset':
                from .bounded_reset import run as handler
            elif plan['design']['handler']=='persistent-practice':
                from .practice import run as handler
            elif plan['design']['handler']=='tiny-reader':
                from .tiny_reader import run as handler
                effective['campaign']=campaign
            elif plan['design']['handler']=='purchase-calibration':
                from .purchase_calibration import run as handler
            elif plan['design']['handler']=='purchase-evaluation':
                from .purchase_evaluation import run as handler
            elif plan['design']['handler']=='purchase-review':
                from .purchase_review import run as handler
            elif plan['design']['handler']=='portfolio':
                from .portfolio import run as handler
            elif plan['design']['handler']=='portfolio-review':
                from .portfolio_review import run as handler
            elif plan['design']['handler']=='interchange-fixtures':
                from .interchange_fixtures import run as handler
            elif plan['design']['handler']=='interchange-alignment':
                from .interchange_alignment import run as handler
            elif plan['design']['handler']=='cue-correction':
                from .cue_correction import run as handler
            elif plan['design']['handler']=='missing-tool':
                from .missing_tool import run as handler
            elif plan['design']['handler']=='forward-rollout':
                from .rollout import run as handler
            elif plan['design']['handler']=='joint-readout':
                from .joint_readout import run as handler
            elif plan['design']['handler']=='joint-support':
                from .joint_support import run as handler
            elif plan['design']['handler']=='joint-support-review':
                from .joint_support_review import run as handler
            elif plan['design']['handler']=='joint-uncertainty':
                from .joint_uncertainty import run as handler
            elif plan['design']['handler']=='joint-uncertainty-review':
                from .joint_uncertainty_review import run as handler
            elif plan['design']['handler']=='witnessed-goal-factorization':
                from .witnessed_goal_factorization import run as handler
            elif plan['design']['handler']=='within-step-factorization':
                from .within_step_factorization import run as handler
            elif plan['design']['handler']=='temporal-factorization-review':
                from .temporal_factorization_review import run as handler
            elif plan['design']['handler']=='temporal-factorization':
                from .temporal_factorization import run as handler
            elif plan['design']['handler']=='joint-factorization':
                from .joint_factorization import run as handler
            elif plan['design']['handler']=='joint-factorization-review':
                from .joint_factorization_review import run as handler
            elif plan['design']['handler']=='joint-review':
                from .joint_review import run as handler
            elif plan['design']['handler']=='crossed-rules':
                from .crossed_rules import run as handler
            elif plan['design']['handler']=='crossed-review':
                from .crossed_review import run as handler
            elif plan['design']['handler']=='forward-support':
                from .forward_support import run as handler
            elif plan['design']['handler']=='support-transfer':
                from .support_transfer import run as handler
            elif plan['design']['handler']=='transfer-review':
                from .transfer_review import run as handler
            elif plan['design']['handler']=='support-review':
                from .support_review import run as handler
            elif plan['design']['handler']=='routine-revision':
                from .routine_revision import run as handler
            elif plan['design']['handler']=='routine-review':
                from .routine_review import run as handler
            elif plan['design']['handler']=='transient-filter':
                from .transient_filter import run as handler
            elif plan['design']['handler']=='transient-review':
                from .transient_review import run as handler
            elif plan['design']['handler']=='unknown-review':
                from .unknown_review import run as handler
            elif plan['design']['handler']=='unknown-change':
                from .unknown_change import run as handler
            elif plan['design']['handler']=='timing-review':
                from .timing_review import run as handler
            elif plan['design']['handler']=='change-timing':
                from .change_timing import run as handler
            elif plan['design']['handler']=='omission-review':
                from .omission_review import run as handler
            elif plan['design']['handler']=='filter-cost':
                from .filter_cost import run as handler
            elif plan['design']['handler']=='provenance-tempering':
                from .provenance_tempering import run as handler
            elif plan['design']['handler']=='tempering-review':
                from .tempering_review import run as handler
            elif plan['design']['handler']=='mixture-compression':
                from .mixture_compression import run as handler
            elif plan['design']['handler']=='compression-review':
                from .compression_review import run as handler
            elif plan['design']['handler']=='marginal-transition':
                from .marginal_transition import run as handler
            elif plan['design']['handler']=='future-quotient':
                from .future_quotient import run as handler
            elif plan['design']['handler']=='primitive-feedback':
                from .primitive_feedback import run as handler
            elif plan['design']['handler']=='pooled-replacement':
                from .pooled_replacement import run as handler
            elif plan['design']['handler']=='primitive-pooling':
                from .primitive_pooling import run as handler
            elif plan['design']['handler']=='replacement-review':
                from .replacement_review import run as handler
            elif plan['design']['handler']=='pooling-review':
                from .pooling_review import run as handler
            elif plan['design']['handler']=='noisy-disclosure':
                from .noisy_disclosure import run as handler
            elif plan['design']['handler']=='noisy-review':
                from .noisy_review import run as handler
            elif plan['design']['handler']=='repeated-disclosure':
                from .repeated_disclosure import run as handler
            elif plan['design']['handler']=='repeated-review':
                from .repeated_review import run as handler
            elif plan['design']['handler']=='joint-reply':
                from .joint_reply import run as handler
            elif plan['design']['handler']=='reliability-mismatch':
                from .reliability_mismatch import run as handler
            elif plan['design']['handler']=='mismatch-review':
                from .mismatch_review import run as handler
            elif plan['design']['handler']=='uncertain-reliability':
                from .uncertain_reliability import run as handler
            elif plan['design']['handler']=='uncertainty-review':
                from .uncertainty_review import run as handler
            elif plan['design']['handler']=='robust-reply':
                from .robust_reply import run as handler
            elif plan['design']['handler']=='robust-review':
                from .robust_review import run as handler
            elif plan['design']['handler']=='continuous-reliability':
                from .continuous_reliability import run as handler
            elif plan['design']['handler']=='continuous-review':
                from .continuous_review import run as handler
            elif plan['design']['handler']=='joint-reply-review':
                from .joint_reply_review import run as handler
            elif plan['design']['handler']=='optional-review':
                from .optional_review import run as handler
            elif plan['design']['handler']=='optional-disclosure':
                from .optional_disclosure import run as handler
            elif plan['design']['handler']=='disclosure-review':
                from .disclosure_review import run as handler
            elif plan['design']['handler']=='metadata-disclosure':
                from .metadata_disclosure import run as handler
            elif plan['design']['handler']=='input-privilege':
                from .input_privilege import run as handler
            elif plan['design']['handler']=='input-privilege-review':
                from .input_privilege_review import run as handler
            elif plan['design']['handler']=='feedback-review':
                from .feedback_review import run as handler
            elif plan['design']['handler']=='quotient-review':
                from .quotient_review import run as handler
            elif plan['design']['handler']=='source-omission':
                from .source_omission import run as handler
            elif plan['design']['handler']=='provenance-restoration':
                from .provenance_restoration import run as handler
            elif plan['design']['handler']=='restoration-review':
                from .restoration_review import run as handler
            elif plan['design']['handler']=='support-mix':
                from .support_mix import run as handler
            elif plan['design']['handler']=='mix-review':
                from .mix_review import run as handler
            elif plan['design']['handler']=='rollout-transfer':
                from .rollout_transfer import run as handler
            elif plan['design']['handler']=='validation-suite':
                from .validation_suite import run as handler
                effective['campaign']=campaign
            else:raise ValueError('handler not implemented/admitted')
            summary=handler(root,effective,pulse)
            pulse(phase='final-validation')
            if not all(summary['controls'].values()):raise ValueError('instrument controls failed')
            write(root/'SUMMARY.json',summary)
            files={p.relative_to(root).as_posix():file_digest(p) for p in root.rglob('*')
                if p.is_file() and p.suffix in ('.json','.gz','.npz') and p.name not in ('PLAN.json','STATUS.json','COMPLETE.json','CURRENT.json','CHILD_ACCOUNTING.json')}
            measurements={p.name:file_digest(p) for p in root.glob('TIMING.jsonl')}
            if plan['design']['handler']=='tiny-reader':
                measurements.update({p.relative_to(root).as_posix():file_digest(p) for p in root.rglob('*')
                    if p.is_file() and (p.suffix=='.pt' or p.name in ('CURRENT.json','CHILD_ACCOUNTING.json'))})
            write(root/'COMPLETE.json',dict(plan_sha256=file_digest(root/'PLAN.json'),files=files,
                execution_measurements=measurements,completed_at=now(),validity='scoped checks passed'))
            emit('complete');return 'complete'
        except BaseException as exc:
            emit('resource_cutoff' if isinstance(exc,TimeoutError) else 'failed',error=repr(exc));raise

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
    elif kind=='joint-review':from ghostscale.validation.soundingline.v19.joint_review import run as handler
    elif kind=='crossed-rules':from ghostscale.validation.soundingline.v19.crossed_rules import run as handler
    elif kind=='crossed-review':from ghostscale.validation.soundingline.v19.crossed_review import run as handler
    elif kind=='forward-support':from ghostscale.validation.soundingline.v19.forward_support import run as handler
    elif kind=='support-review':from ghostscale.validation.soundingline.v19.support_review import run as handler
    elif kind=='routine-revision':from ghostscale.validation.soundingline.v19.routine_revision import run as handler
    elif kind=='routine-review':from ghostscale.validation.soundingline.v19.routine_review import run as handler
    elif kind=='transient-filter':from ghostscale.validation.soundingline.v19.transient_filter import run as handler
    elif kind=='transient-review':from ghostscale.validation.soundingline.v19.transient_review import run as handler
    elif kind=='unknown-change':from ghostscale.validation.soundingline.v19.unknown_change import run as handler
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

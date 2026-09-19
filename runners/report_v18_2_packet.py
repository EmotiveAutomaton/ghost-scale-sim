"""File completed V18.2 packets without publishing machine-local operating files."""
import argparse
import copy
import gzip
import json
from pathlib import Path
import shutil
import zipfile
import numpy as np
from ghostscale.validation.soundingline.v16.records import read,write,file_digest,now
from ghostscale.validation.soundingline.v18_2.runtime import load
from ghostscale.validation.soundingline.v18_2 import model as m,verify as v


def units(root):
    return [u for p in sorted((root/'blocks').glob('*.json')) for u in load(root,p.stem)]


def make_examples(campaign):
    result={}
    for unit in units(campaign/'g1-long'):
        if unit['case']['truth']['initial_state'][0]:
            selected=[r for r in unit['rows'] if r['method']=='persistent' and r['probe']==0 and r['condition'] in ('goal-0-preference-0','goal-1-preference-0')]
            result['skill_across_goal_change']=dict(case=unit['case'],rows=selected,selection='first acquired-skill lineage; evaluator-selected illustration')
            break
    unit=units(campaign/'g2')[0]
    result['false_belief_correction']=dict(case=unit['case'],rows=[r for r in unit['rows'] if r['method']=='persistent' and r['probe']==0 and r['condition'] in ('false-belief','reader-only-correction','maker-correction')],selection='first lineage, fixed conditions')
    world=m.world('test',0);a=(1,1,0,0);b=(1,1,1,0);history_context=m.context(0,0)
    assert np.allclose(m.policy(world,a,history_context),m.policy(world,b,history_context))
    distinguishing=m.context(None,0)
    assert not np.allclose(m.policy(world,a,distinguishing),m.policy(world,b,distinguishing))
    result['unidentifiable_pair']=dict(world=world,evaluator_states=[a,b],history_context=history_context,
        history_action_distributions=[m.policy(world,s,history_context).tolist() for s in (a,b)],
        distinguishing_context=distinguishing,future_distributions=[m.artifacts(m.policy(world,s,distinguishing)).tolist() for s in (a,b)],
        selection='known-answer counterexample fixture; not a performance sample')
    result['failed_context_explanation']=dict(state='not found under declared illustration rule')
    for unit in units(campaign/'g3'):
        if unit['case']['truth']['future_state'][0]==0:continue
        wrong=next(r for r in unit['rows'] if r['method']=='wrong-context' and r['probe']==0)
        expanded=next(r for r in unit['rows'] if r['method']=='bounded-expansion' and r['probe']==0)
        if wrong['scores']['expected_log_loss']>expanded['scores']['expected_log_loss']+1e-8:
            result['failed_context_explanation']=dict(case=unit['case'],rows=[wrong,expanded],selection='first false no-skill card with worse future risk; outcome-selected illustration')
            break
    result['selective_transfer']=dict(state='not found under declared illustration rule')
    for unit in units(campaign/'g5-assembly'):
        selected=[r for r in unit['rows'] if r['condition']=='heldout-chain-dependent' and r['method']=='procedure-weight-1.0-apply-1' and r['scores']['task_transfer']==1 and r['scores']['unwanted_goal_execution']==0]
        if selected:
            result['selective_transfer']=dict(case_id=unit['case_id'],row=selected[0],selection='first successful own-goal transfer without unwanted goal execution; outcome-selected')
            break
    original={u['case']['case_id']:u for u in units(campaign/'learned')}
    for unit in units(campaign/'learned-aligned'):
        previous=original[unit['case']['case_id']]
        old=next(r for r in previous['rows'] if r['method']=='flat' and r['tier']=='process-history' and r['probe']==0)
        new=next(r for r in unit['rows'] if r['method']=='flat' and r['tier']=='process-history' and r['probe']==0)
        if new['scores']['expected_log_loss']<old['scores']['expected_log_loss']:
            result['representation_change']=dict(case=unit['case'],original=old,aligned=new,
                selection='first improved retained lineage after invertible public-role alignment; exposed illustration')
            break
    return dict(evaluator_truth_present=True,not_blind_reader_input=True,categories=result)


def paired(root,left,right,condition='base',metric='action_accuracy'):
    differences=[]
    for unit in units(root):
        if unit['case'].get('split')!='test':continue
        values={}
        for row in unit['rows']:
            if row['tier']=='process-history' and row.get('condition','base')==condition and row['method'] in (left,right) and metric in row.get('scores',{}):
                values.setdefault(row['method'],[]).append(row['scores'][metric])
        if all(method in values for method in (left,right)):
            differences.append(float(np.mean(values[left])-np.mean(values[right])))
    return dict(left=left,right=right,metric=metric,condition=condition,left_minus_right=v.interval(differences))


def report(campaign,public):
    ledger={};total_rows=0;all_lineages=set();eligible_lineages=set();verification_checks=0;replay_rows=0
    for plan_path in sorted(campaign.glob('*/PLAN.json')):
        root=plan_path.parent;name=root.name
        complete=read(root/'COMPLETE.json');verification=read(root/'VERIFICATION.json');assert verification['passed']
        assert complete['plan_sha256']==file_digest(plan_path)
        out=public/name;out.mkdir(parents=True,exist_ok=True)
        for filename in ('PLAN.json','COMPLETE.json','SUMMARY.json','AUDITED_SUMMARY.json','VERIFICATION.json','SOURCE.zip','FIT.json','GEOMETRY.json','SUPERSEDED.json','split.npz','flat.npz','GEOMETRY_REPAIR.json','GEOMETRY_REPAIR_SOURCE.py','REPLAY.json'):
            if (root/filename).exists():shutil.copyfile(root/filename,out/filename)
        with zipfile.ZipFile(out/'raw.zip','w',zipfile.ZIP_DEFLATED) as archive:
            for folder in ('raw','blocks','attempts'):
                for path in sorted((root/folder).glob('*')):archive.write(path,path.relative_to(root).as_posix())
            for path in sorted(root.glob('*_points.json.gz')):archive.write(path,path.name)
        current=read(root/('AUDITED_SUMMARY.json' if (root/'AUDITED_SUMMARY.json').exists() else 'SUMMARY.json'))
        if (root/'GEOMETRY_REPAIR.json').exists():
            correction=read(root/'GEOMETRY_REPAIR.json');current['cells'].update(correction['cells'])
            current['geometry_revision']='GEOMETRY_REPAIR.json overrides original rank-limited geometry rows; no new lineages'
        withheld=(root/'SUPERSEDED.json').exists();current['scientific_eligibility']='withheld; see SUPERSEDED.json' if withheld else 'discovery'
        write(out/'CURRENT_SUMMARY.json',current,immutable=False)
        original=read(root/'SUMMARY.json');rows=original.get('rows',verification.get('rows',0));total_rows+=rows
        test_ids=set()
        for unit in units(root):
            case=unit.get('case',unit)
            if case.get('split','test')=='test':test_ids.add(case['case_id'])
        all_lineages|=test_ids
        if not withheld:eligible_lineages|=test_ids
        verification_checks+=verification.get('independent_execution_checks',verification.get('independent_executions',0))
        replay_rows+=verification.get('replayed_rows',verification.get('replay_rows',0))
        ledger[name]=dict(state='withheld_original' if withheld else 'verified',rows=rows,test_lineages=len(test_ids),
            plan_sha256=file_digest(root/'PLAN.json'),verification_sha256=file_digest(root/'VERIFICATION.json'),
            current_summary_sha256=file_digest(out/'CURRENT_SUMMARY.json'),raw_archive_sha256=file_digest(out/'raw.zip'))
    direct=public/'direct-controls';direct.mkdir(exist_ok=True)
    shutil.copyfile(campaign/'direct-controls/SUMMARY.json',direct/'SUMMARY.json')
    with zipfile.ZipFile(direct/'raw.zip','w',zipfile.ZIP_DEFLATED) as archive:
        archive.write(campaign/'direct-controls/DIRECT_points.json.gz','DIRECT_points.json.gz')
    examples=make_examples(campaign);write(public/'WORKED_EXAMPLES.json',examples,immutable=False)
    contrasts=dict(hidden_current_state=paired(campaign/'g0-uncertain','persistent','direct'),
        policy_change=paired(campaign/'g1','adaptive','persistent','goal-0-preference-1','expected_log_loss'))
    write(public/'CONTRASTS.json',contrasts,immutable=False)
    cpu=sum(read(p)['cpu_seconds'] for p in campaign.glob('*/attempts/*.json'))
    final=dict(schema='v18.2.closeout.1',at=now(),state='finite_queue_complete',packets=len(ledger),
        verified_packets=sum(x['state']=='verified' for x in ledger.values()),withheld_original_packets=sum(x['state']=='withheld_original' for x in ledger.values()),
        recorded_comparison_and_diagnostic_rows=total_rows,distinct_test_lineages_including_withheld=len(all_lineages),
        distinct_eligible_test_lineages=len(eligible_lineages),independent_execution_rechecks=verification_checks,
        bounded_replay_rows=replay_rows,g6_exact_replayed_replicates=8192,direct_control_executions=512,
        measured_worker_and_audit_cpu_seconds=cpu,cpu_ceiling_seconds=64800,
        report_start=read(campaign/'ACCEPTANCE.json')['report_start'],delivery_deadline=read(campaign/'ACCEPTANCE.json')['delivery_deadline'],
        stopping_reason='all implemented distinct branch/follow-on contrasts completed; no occupancy padding, no added confirmation',
        independent_unit='maker/world lineage within a declared miniature; shared lineages across fit seeds or repairs counted once',
        scope='discovery constructed mechanisms and method checks; miniature — architecture untested',batches=ledger)
    write(public/'CLOSEOUT.json',final,immutable=False)
    write(public/'BRANCH_LEDGER.json',dict(active=False,updated_at=final['at'],batches=ledger,closeout='CLOSEOUT.json'),immutable=False)
    return {k:v for k,v in final.items() if k!='batches'}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',type=Path,required=True);parser.add_argument('--public',type=Path,required=True)
    a=parser.parse_args();print(json.dumps(report(a.campaign,a.public),sort_keys=True))

"""Local, traceable result landing. Publication and recipient delivery are separate."""
from pathlib import Path
from datetime import datetime,timezone
import math,json
import numpy as np
from .common import read,write,digest,canonical
from . import registry,contract

def primary_cells(summary):
    family=summary['family'];answer=[]
    for c in summary['cells']:
        s=summary['settings'][c['setting']];l=c['labels'];m=c['metric'];keep=False
        if family=='H1':keep=s.get('overlap')=='partial' and l==dict(reader='contrasts',history=4) and m=='primary'
        elif family=='H2':keep=s.get('d')==.6 and s.get('q')==.2 and l.get('reader')=='contrasts' and m=='primary'
        elif family=='H3':keep=s.get('r')==.6 and l.get('reader')=='contrasts' and m=='primary'
        elif family=='E1':keep=l.get('truth')=='B' and l.get('rival')=='A' and l.get('reader')=='threshold' and m in ('records_to_one_nat','not_within_256')
        elif family=='E2':keep=s.get('overlap')==4 and l.get('reader')=='contrasts' and l.get('history')==8 and m=='primary'
        elif family=='E3':keep=s.get('truth')=='cached' and l.get('reader')=='contrasts' and l.get('history')==32 and m=='primary'
        elif family=='U1':keep=s.get('s')==.5 and l.get('truth')==0 and m=='excess_capture'
        elif family=='R1':keep=l.get('reader') in ('remove','satiate','switch') and l.get('history')==1 and m=='primary'
        elif family=='R4':keep=l==dict(reader='contrasts',history=1024) and m=='primary'
        elif family=='R13':keep=l.get('reader')=='threshold' and l.get('truth') in (1,2,3) and m in ('records_to_one_nat','not_within_128')
        if keep:answer.append(dict(configuration=s,**c))
    return answer

def decisions(summary):
    f=summary['family'];cells=primary_cells(summary);out={}
    if f=='H1':
        groups={}
        for c in summary['cells']:
            if c['metric']=='log_gain':groups.setdefault((c['setting'],c['labels']['reader']),[]).append(c)
        for (setting,reader),rows in groups.items():out[str(setting)+':'+reader]=next((c['labels']['history'] for c in sorted(rows,key=lambda x:x['labels']['history']) if c['mean'] is not None and c['mean']>=1),None)
    if f=='E2':
        for k in sorted({c['configuration']['kappa'] for c in cells}):
            rows=sorted((c for c in cells if c['configuration']['kappa']==k),key=lambda c:c['labels']['distance'])
            out[str(k)]=next((c['labels']['distance'] for c in rows if c['mean'] is not None and c['mean']<0),None)
    if f in ('E1','R13'):
        for c in cells:
            if c['metric']!='records_to_one_nat':continue
            values=sorted(math.inf if v is None else v for v in c['whole_world_values']);median=float(np.median(values))
            out[canonical(dict(configuration=c['configuration'],labels=c['labels'])).decode()]=dict(censored_median=None if math.isinf(median) else median,not_reached=sum(math.isinf(x) for x in values),blocks=len(values))
    if f=='U1':
        for tau in sorted({c['configuration']['tau'] for c in cells}):
            rows=sorted((c for c in cells if c['configuration']['tau']==tau),key=lambda c:c['labels']['history'])
            out[str(tau)]=next((c['labels']['history'] for c in rows if c['mean'] is not None and abs(c['mean'])<=.05),None)
    return out

def status_export(campaign,repo):
    from runners.run_v26 import charges
    packets=[]
    for p in sorted((campaign/'packets').glob('*/PLAN.json')):
        plan=read(p);s=read(p.parent/'STATUS.json');packets.append(dict(packet=p.parent.name,family=plan['family'],kind=plan['kind'],scientific_kind=plan.get('scientific_kind'),state=s['state'],phase=s.get('phase'),cap_seconds=plan['cpu_cap_seconds']))
    q=read(campaign/'QUEUE_STATUS.json') if (campaign/'QUEUE_STATUS.json').exists() else {}
    summary=dict(at=datetime.now(timezone.utc).isoformat(),state=q.get('state','BUILDING'),gear=2,workers_maximum=2,active=q.get('active',[]),packets=packets,pools_charged_seconds=charges(campaign),pool_ceilings_seconds=contract.POOLS,fully_built=(campaign/'BUILD_COMPLETE.json').exists(),science_open=contract.SCIENCE_OPEN,new_starts_close=contract.NEW_STARTS_CLOSE,drain=contract.DRAIN,publication='local records; remote publication independently verified')
    write(repo/'results/v26/CURRENT_STATUS.json',summary,immutable=False)
    return summary

def land_local(campaign,packet,repo):
    plan=read(packet/'PLAN.json');family=plan['family'];summary=read(packet/'original/SUMMARY.json');complete=read(packet/'COMPLETE.json');out=repo/'results/v26'/packet.name;out.mkdir(parents=True,exist_ok=True)
    if (packet/'LOCAL_LANDING.json').exists():return
    for key,name in [('plan_sha256','PLAN.json'),('summary_sha256','original/SUMMARY.json'),('review_sha256','INDEPENDENT_REVIEW.json'),('replay_sha256','REPLAY.json')]:
        if complete[key]!=digest(packet/name):raise ValueError('landing binding mismatch')
    for name,source in [('SUMMARY.json','original/SUMMARY.json'),('COMPLETE.json','COMPLETE.json'),('INDEPENDENT_REVIEW.json','INDEPENDENT_REVIEW.json'),('REPLAY.json','REPLAY.json')]:write(out/name,read(packet/source))
    primary=primary_cells(summary);decision=decisions(summary);routes=registry.ROUTES[family];claim=f'G-V26-{family}-1' if plan['kind']=='study' else f'G-V26-{family}T{plan["texture_variant"]+1}-1'
    values=[c['mean'] for c in primary if c['mean'] is not None]
    wording=f"What does {registry.NAMES[family].replace('_',' ').lower()} identify in the declared constructed worlds? The verified {plan['kind']} retains {len(summary['settings'])} settings and {summary['independent_blocks_per_setting']} independent whole-world blocks per setting. "
    wording+=(f"The registered primary estimates range from {min(values):.6g} to {max(values):.6g}; their units and strata are listed in the report." if values else 'The registered primary is unavailable in this variant; the full descriptive curve remains recorded.')
    wording+=' This is descriptive constructed-method evidence, not confirmation or human evidence. Miniature, architecture untested.'
    report_name=f'{family}_REPORT.md' if plan['kind']=='study' else f'{family}_TEXTURE_{plan["texture_variant"]+1:03d}.md';relative='docs/versions/v26-top-and-ends/'+report_name
    lines=['# '+registry.NAMES[family].replace('_',' ').title(),'','Corrections first: earlier V25 findings and retained failures remain unchanged; this is a separate V26 card.','',wording,'',f'Claim `{claim}`.','',
        'The table lists each registered primary by its frozen setting and observation stratum. Estimates use nats for score contrasts, records for time-to-evidence, probabilities for capture/unresolved shares, and distance bins for crossover decisions. Missing crossings remain censored; cases are averaged within worlds before uncertainty.','',
        '| Setting | Observation stratum | Measure | Mean | Available blocks |','|---|---|---|---:|---:|']
    for c in primary:lines.append('| '+json.dumps(c['configuration'],sort_keys=True)+' | '+json.dumps(c['labels'],sort_keys=True)+' | '+c['metric']+' | '+str(c['mean'])+' | '+str(c['available_blocks'])+' |')
    lines+=['','Derived records-needed/crossover decisions (null means not reached within the sampled grid):','', '```json',json.dumps(decision,indent=2),'```','',
        'Design pre-mortem: '+json.dumps(read(campaign/'designs'/(family+'.json')),sort_keys=True),
        '', 'By construction and modelling limits: '+registry.CHOICES[family],
        '', 'Full original/replay files are byte-identical. The independent review uses a separate scalar posterior/decimal aggregation route; some likelihood kernels and the generative law are shared. Known-answer tests check the model equations, not real-world validity. No universal ceiling is inferred from a supplied-library reference.',
        '',routes[0],routes[1],'',f'[All estimates](../../../results/v26/{packet.name}/SUMMARY.json). [Verification](../../../results/v26/{packet.name}/INDEPENDENT_REVIEW.json).','']
    text='\n'.join(lines);(repo/relative).write_text(text,encoding='utf-8')
    if plan['kind']=='study':
        exchange_name=f'GHOST_V26_{family}_STUDY.md'
    else:
        exchange_name=f'GHOST_V26_{family}_TEXTURE_{plan["texture_variant"]+1:03d}.md'
    exchange=repo/'docs/exchange'/exchange_name;exchange.write_text(text,encoding='utf-8')
    for target in ('FINDINGS.md','docs/theory/READING_INTENT.md'):
        p=repo/target;s=p.read_text(encoding='utf-8');idx=s.find('\n');entry='\n## '+claim+'\n\n'+wording+'\n\n';p.write_text(s[:idx+1]+entry+s[idx+1:],encoding='utf-8')
    claims=read(repo/'docs/theory/CLAIMS.json')
    if any(c['id']==claim for c in claims['claims']):raise ValueError('duplicate claim landing')
    claims['claims'].append(dict(id=claim,allowed_wording=wording,claim_kind='method',architecture='miniature, architecture untested',assumed_choice_model=registry.CHOICES[family],reader='Declared exact finite-library readers',estimate=primary,control_status='passes',source_file=relative,source_line=5,source_sha256=digest(repo/relative),supersedes=None,superseded_by=None,correction_class=None))
    write(repo/'docs/theory/CLAIMS.json',claims,immutable=False)
    write(packet/'LOCAL_LANDING.json',dict(at=datetime.now(timezone.utc).isoformat(),claim=claim,report=relative,report_sha256=digest(repo/relative),published=False,recipient_copy_pending=True))

def texture_decision(campaign,packet):
    plan=read(packet/'PLAN.json');f=plan['family'];state=read(campaign/'texture'/(f+'.json'));parent=read(campaign/'packets'/state['primary']/'original/SUMMARY.json');current=read(packet/'original/SUMMARY.json')
    baseline=registry.primary_setting(f);matching=[i for i,s in enumerate(parent['settings']) if all(s.get(k)==v for k,v in baseline.items())]
    # Stop on comparable scored estimates, not wall time, counters or fitted cost.
    metrics={'primary','log_gain','half_brier','accuracy','truth_probability','misread_per_pause','replacement_gain','total_gain','boundary_gain','mismatch_gain','mismatch_minus_flat','usage_minus_exact_loss','generator_accuracy','cue_map_mse','capture','excess_capture','detection','not_within_16','not_within_128','not_within_256'}
    base={canonical(dict(labels=c['labels'],metric=c['metric'])).decode():c['mean'] for c in parent['cells'] if c['setting'] in matching and c['metric'] in metrics}
    differences=[];missing=False
    for c in current['cells']:
        if c['metric'] not in metrics:continue
        k=canonical(dict(labels=c['labels'],metric=c['metric'])).decode()
        if k not in base or c['mean'] is None or base[k] is None:missing=True;continue
        differences.append(abs(c['mean']-base[k]))
    stable=bool(differences) and not missing and max(differences)<=.05
    # Records-needed and crossover variants have their declared ordinal bar;
    # incomparability never increments the no-change stopping counter.
    if f in ('E1','E2','U1'):
        stable=False
        a,b=decisions(parent),decisions(current)
        if f=='E2' and a and b:
            stable=all(str(k) in a and a[str(k)] is not None and v is not None and abs(a[str(k)]-v)<=.25 for k,v in b.items())
        if f=='U1' and a and b:
            levels=[1,2,4,8,16,32,64];stable=all(k in a and a[k] in levels and v in levels and abs(levels.index(a[k])-levels.index(v))<=1 for k,v in b.items())
        if f=='E1':
            def medians(summary,settings):
                result={}
                for c in primary_cells(summary):
                    if c['setting'] not in settings or c['metric']!='records_to_one_nat':continue
                    key=canonical(c['labels']).decode();vals=[math.inf if x is None else x for x in c['whole_world_values']];result[key]=(float(np.median(vals)),sum(math.isinf(x) for x in vals)/len(vals))
                return result
            aa,bb=medians(parent,matching),medians(current,[0])
            stable=bool(aa) and set(aa)==set(bb) and all((x[0]==bb[k][0] or (math.isfinite(x[0]) and math.isfinite(bb[k][0]) and abs(x[0]-bb[k][0])<=1)) and abs(x[1]-bb[k][1])<=.05 for k,x in aa.items())
    state['consecutive']=state['consecutive']+1 if stable else 0;state['completed_variants'].append(dict(index=plan['texture_variant'],no_practical_change=stable,largest_comparable_difference=max(differences) if differences else None,incomparable=missing))
    state['state']='STOP_FIVE_NO_CHANGE' if state['consecutive']>=5 else 'GRID_EXHAUSTED' if plan['texture_variant']+1>=len(registry.texture_variants(f)) else 'OPEN'
    write(campaign/'texture'/(f+'.json'),state,immutable=False);return state

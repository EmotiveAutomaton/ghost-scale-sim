"""One claim per verified logical card, with full scientific write-through."""
import json,math
from pathlib import Path
from . import registry,earlier
from .common import read,write,digest,bar_verdict


def bar(f,config,row):
    m=row['metric'];l=row['labels'];reader=l.get('reader')
    if f=='K1' and m=='timing_ratio' and config['truth']=='L':return .5,'at_most',True
    if f=='K2' and m=='primary' and config['phi']==0 and config['tournament']==.5:return .5,'at_least',True
    if f=='K2' and m=='false_dislike_excess' and config['phi']==0 and config['tournament']==.8:return .05,'at_least',True
    if f=='K3' and m=='overshoot_ratio' and config['beta']>0:return 1.5,'at_least',True
    if f=='K3' and m=='dynamic_records' and config['dynamic'] in ('S','P-') and config['probe']>0:return 128,'at_most',False
    if f=='K3' and m=='band_records' and config['width']==4:return 128,'at_most',True
    if f=='K4' and m=='policy_outcome_records' and config['mix']=='designed' and reader=='all':return 128,'at_most',True
    if f=='K5' and m=='RT_minus_RR' and l.get('source')=='Mc' and l.get('truth')==0:return .05,'outside',True
    if f=='K5' and m=='outsider_minus_insider':return .05,'outside',False
    if f=='K5' and m=='RT_context_loss' and l.get('source')=='Mc' and l.get('truth')==0 and config['context_switch']:return .05,'at_least',True
    if f=='K6' and m=='primary':return .5,'at_least',False
    if f=='K6' and m=='confirmation_excess':return .02,'at_least',False
    if f=='K7' and m=='marker_ratio' and config['law']=='M1' and config['fake']==0:return 1.5,'at_least',False
    if f=='K7' and m=='misread_gain' and config['law'] in ('M2','M3') and config['fake']==0:return -.05,'below',False
    return None,None,False


def annotate(summary,bounds=()):
    rows=[]
    for row in summary['cells']:
        replacement=next((b for b in bounds if b['setting']==row['setting'] and b['metric']==row['metric'] and b['labels']==row['labels']),None)
        if replacement:row=row|replacement
        if summary['family']=='K7' and row['metric']=='log_score':row=row|dict(metric='true_goal_probability')
        b,d,fixed=bar(summary['family'],summary['settings'][row['setting']],row)
        verdict='descriptive'
        if b is not None:
            if d=='outside':
                lo,hi=row['interval95']
                verdict='unresolved at this sample size' if lo is None or hi is None else 'met' if hi<-b or lo>b else 'not met' if lo>=-b and hi<=b else 'unresolved at this sample size'
            else:
                adjusted=dict(row)
                adjusted['interval95']=[-math.inf if row.get('lower_negative_infinity') else row['interval95'][0],-math.inf if row.get('upper_negative_infinity') else row['interval95'][1]]
                verdict=bar_verdict(adjusted,b,d)
            if summary['family']=='K6' and row['metric']=='primary':
                den=next(r for r in summary['cells'] if r['setting']==row['setting'] and r['metric']=='joint_gain')
                lo,hi=den['interval95']
                if lo is None or hi is None or lo<=0<=hi:verdict='unresolved at this sample size'
            if row['statistic']=='median_ratio' and (row['estimate_censored'] or row['lower_censored'] or row['upper_censored']) and replacement is None:
                verdict='unresolved at this sample size'
        rows.append(row|dict(bar=b,direction=d,verdict=verdict,packet_by_construction_or_scale_only=fixed))
    return rows


def assessment(rows):
    v=[r['verdict'] for r in rows if r['bar'] is not None]
    return 'not met' if 'not met' in v else 'met' if v and all(x=='met' for x in v) else 'unresolved at this sample size'


def land(campaign,packet,repo):
    if (packet/'LOCAL_LANDING.json').exists():return
    from .common import verify_complete
    verify_complete(packet);p=read(packet/'PLAN.json')
    if p['scientific_kind']=='probe':return
    # Setting partitions are execution units, not scientific cards.  Only the
    # verified logical aggregate receives a verdict, claim or exchange note.
    if p.get('split_part'):return
    from . import censoring
    summary=read(packet/'original/SUMMARY.json');bounds=censoring.retained(packet)
    if bounds:
        receipt=dict(raw_index_sha256=digest(packet/'original/RAW_INDEX.json'),bounds=bounds,independent_median_arithmetic=True,original_scientific_bytes_unchanged=True)
        write(packet/'CENSOR_BOUNDS.json',receipt);write(repo/'results/v27'/packet.name/'CENSOR_BOUNDS.json',receipt)
    rows=annotate(summary,bounds);card=read(campaign/'cards'/(p['card']+'.json'))
    claim='G-V27-'+p['card']+'-1';verdict=assessment(rows);f=p['family']
    relative='docs/versions/v27-motives-and-set-points/reports/'+p['card']+'.md'
    question=card.get('question',registry.QUESTIONS[f])
    wording=question+f" The {p['lane']} card retains {len(p['settings'])} settings and {p['blocks']} independent whole-world blocks per setting. The declared bar conjunction is {verdict}. Constructed-method evidence; miniature, architecture untested."
    if card.get('scope_note'):wording+=' '+card['scope_note']
    text=['# '+p['card'],'',wording,'','Claim `'+claim+'`. No human-mechanism or engine validation follows.',
        '', 'The table gives each setting, reader/source stratum, measure, estimate, pointwise 95% whole-block interval and bar disposition. Censored records-needed values are retained as beyond the observed horizon; unavailable intervals are not a null.',
        '', '| Setting | Stratum | Measure | Estimate | 95% interval | Bar | Verdict | Scale only |','|---:|---|---|---:|---|---|---|---|']
    for row in rows:
        def show(v,c=False):return 'beyond horizon' if c else 'unavailable' if v is None else f'{v:.6g}'
        if row.get('estimate_negative_infinity'):row=row|dict(estimate=-math.inf)
        if row.get('lower_negative_infinity') or row.get('upper_negative_infinity'):row=row|dict(interval95=[-math.inf if row.get('lower_negative_infinity') else row['interval95'][0],-math.inf if row.get('upper_negative_infinity') else row['interval95'][1]])
        lo,hi=row['interval95'];estimate=show(row['estimate'],row['estimate_censored'])
        if 'identified_estimate_bounds' in row:estimate='['+show(row['identified_estimate_bounds'][0])+', '+('unbounded' if row['upper_bound_unbounded'] else show(row['identified_estimate_bounds'][1]))+']'
        text.append('| '+str(row['setting'])+' | '+json.dumps(row['labels'],sort_keys=True)+' | '+row['metric'].replace('_',' ')+' | '+estimate+' | '+show(lo,row['lower_censored'])+' to '+('unbounded' if row.get('upper_interval_unbounded') else show(hi,row['upper_censored']))+' | '+str(row['bar'])+' '+str(row['direction'])+' | '+row['verdict']+' | '+str(row['packet_by_construction_or_scale_only'])+' |')
    protocol=read(campaign/'protocols'/(p['card']+'.json'))
    text+=['','Modelling choices are provisional: '+protocol['modelling_choices'],
        '', 'The packet lists some finite-sample expectations as by construction. Those rows remain scale-only; their empirical bar outcomes are not logical guarantees.',
        '', 'Case: '+('out-of-case descriptive map; it cannot upgrade its primary' if protocol['out_of_case'] else 'excluded case check passed')+'.',
        '', 'Complete raw replay and independent reconstruction passed. Shared generators and some numerical kernels remain a limitation. Intervals are pointwise descriptive and are not multiplicity-adjusted; adaptive follow-ups do not convert them to confirmation.',
        '',*registry.ROUTES[f],'',*earlier.prose(f),
        '', '[Protocol](../protocols/'+p['card']+'.json). [Complete measures](../../../../results/v27/'+packet.name+'/SUMMARY.json).','']
    report=repo/relative;report.parent.mkdir(parents=True,exist_ok=True);report.write_text('\n'.join(text),encoding='utf-8',newline='\n')
    exchange=repo/'docs/exchange'/('GHOST_'+claim.replace('-','_')+'.md')
    exchange.write_text('# '+claim+'\n\n'+wording+'\n\n'+'\n\n'.join(registry.ROUTES[f])+'\n\n[Report](../versions/v27-motives-and-set-points/reports/'+p['card']+'.md).\n',encoding='utf-8')
    for name,heading,entry in [
        ('FINDINGS.md','## '+claim,'\n## '+claim+'\n\n'+wording+'\n\n[Report]('+relative+').\n'),
        ('docs/theory/READING_INTENT.md','| '+claim+' |','| '+claim+' | '+question+' | QUALIFIED. '+wording+' [Report](../versions/v27-motives-and-set-points/reports/'+p['card']+'.md) |\n')]:
        path=repo/name;content=path.read_text(encoding='utf-8')
        if heading in content:continue
        if name=='FINDINGS.md':i=content.index('\n');content=content[:i+1]+entry+content[i+1:]
        else:
            title='## V27 motives and set points'
            if title not in content:
                i=content.index('\n');content=content[:i+1]+'\n'+title+'\n\n| Claim | Question | Status and evidence |\n|---|---|---|\n\n**What these add up to.** These constructed comparisons identify distinctions only within their declared generators and reader libraries. Scale-only verdicts do not validate the curator\'s human mechanisms.\n'+content[i+1:]
            i=content.index('|---|---|---|',content.index(title))+len('|---|---|---|\n');content=content[:i]+entry+content[i:]
            after='**What these add up to.** These constructed comparisons identify distinctions only within their declared generators and reader libraries. Scale-only verdicts do not validate the curator\'s human mechanisms.'
            import re
            content=re.sub(re.escape(after)+r'(?: Latest recorded card: [^\n]*)?',after+' Latest recorded card: '+claim+'; its declared bar conjunction is '+verdict+'.',content,count=1)
        path.write_text(content,encoding='utf-8',newline='\n')
    ledger=repo/'docs/theory/CLAIMS.json';claims=read(ledger)
    if any(x['id']==claim for x in claims['claims']):raise ValueError('claim exists without landing receipt')
    claims['claims'].append(dict(id=claim,allowed_wording=wording,claim_kind='method',architecture='miniature, architecture untested',
        assumed_choice_model=protocol['modelling_choices'],reader='Declared exact and approximate readers; approximation and nonconvergence retained',
        estimate=[row for row in rows if row['bar'] is not None],
        estimate_detail=dict(source_file='results/v27/'+packet.name+'/SUMMARY.json',sha256=digest(packet/'original/SUMMARY.json'),
            scope='All descriptive estimates and uncertainty are preserved in the complete summary and report; the ledger carries declared-bar rows.'),
        control_status='passes',source_file=relative,source_line=3,source_sha256=digest(report),
        supersedes=None,superseded_by=None,correction_class=None))
    ledger.write_text(json.dumps(claims,indent=2,sort_keys=True,ensure_ascii=False)+'\n',encoding='utf-8',newline='\n')
    write(packet/'LOCAL_LANDING.json',dict(claim=claim,report=relative,report_sha256=digest(report),verdict=verdict,
        unresolved=any(r['bar'] is not None and r['verdict']=='unresolved at this sample size' for r in rows),published=False,recipient_delivery='pending local copies'))

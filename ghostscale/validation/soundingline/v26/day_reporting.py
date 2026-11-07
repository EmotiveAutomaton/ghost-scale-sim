"""Idempotent theory-day reports, scoped claims and complete documentary landing."""
import json
from pathlib import Path
from . import day
from .common import read,write,digest
from .day_common import bar_verdict

EXPECTED=dict(G1='Pre-checks: hidden-stock c-versus-d sustained medians roughly 16–38 records; set-point posteriors around 0.42–0.55 without stocks versus near 1 with stocks. The 0.05 posterior-probability advantage arm was not met or unresolved.',
    G2='Pre-checks: frame-blind 19 records under equal type priors, versus about 5.5 with frames. Registered frame necessity is expected not met; these expectations are not mathematical guarantees.',
    G3='Exact registered con capture difference approximately −0.0897 and honest closure 8.99%. Contamination at theta 0.7 was expected to cross near delta 0.07, with 0.05 unresolved and 0.1 positive.',
    G4='Pre-checks: about 12 offers and about 199–222 records for A9 at one offer per 16 records; records saved about 29.5, with about 58% saving nothing. This finite ladder does not test all prices.',
    G5='Pre-checks: SP medians 47–58 records, LV 52.5–82.5; off-grid placement near 79% among case-passing makers. Both bars may be unresolved at this sample size.')

def featured(summary):
    f=summary['family'];rows=[]
    for row in summary['cells']:
        s=summary['settings'][row['setting']];l=row['labels'];m=row['metric'];bar=direction=None;fixed=False;keep=False
        if f=='G1':
            keep=m.startswith('setpoint_') and m.endswith('_256') or m=='arm' or (m in ('against_c','against_d') and l.get('truth') in ('c','d'))
            if m.startswith('setpoint_'):bar,direction=.8,'at_least'
            elif m=='arm' and l.get('truth')=='c':bar,direction=.05,'at_least'
            elif m.startswith('against_') and l.get('reader')=='both':bar,direction=128,'at_most'
        elif f=='G2':
            keep=m in ('records','records_ratio','identity_64','as_A_64','L_over_P_64')
            if l.get('truth') in ('P','Ps'):
                if m=='records' and l.get('reader')=='frame-blind':bar,direction=64,'above'
                if m=='records_ratio':bar,direction=2,'above'
            fixed=s['cue']==.15 and s['specialists']==.25 and s.get('pi') is None
        elif f=='G3':
            keep=(m in ('D_minus_K_capture','closed') and s['delta']==0 and l.get('source') in ('Mc','H')) or (m=='D_minus_K_logscore' and l.get('source') in ('M','H'))
            if m=='D_minus_K_capture' and l.get('source')=='Mc' and l.get('truth')==0:bar,direction=-.05,'at_most';fixed=True
            if m=='closed' and l.get('source')=='H':bar,direction=.05,'below';fixed=True
            if m=='D_minus_K_logscore' and l.get('source')=='M' and l.get('truth')==0 and s['delta']>0:bar,direction=0.,'above'
        elif f=='G4':
            keep=(l.get('truth')=='A9' and m in ('offers_needed','class_records','records_saved','nothing_saved')) or m in ('off_target_share','lock_delay')
            if l.get('truth')=='A9' and l.get('reader')=='offers' and m=='class_records':bar,direction=256,'at_most';fixed=True
            if m=='lock_delay':fixed=True
        elif f=='G5':
            keep=m in ('records','conditional_placement','placement','fixed_conditional_placement','fixed_sp_class','large_tv_share','time_split')
            if m=='records' and not s['offgrid'] and l.get('truth') in ('SP','LV'):bar,direction=64,'at_most'
            if m=='conditional_placement' and s['offgrid']:bar,direction=.75,'at_least'
        if keep:rows.append(row|dict(bar=bar,direction=direction,verdict=bar_verdict(row,bar,direction) if bar is not None else 'descriptive',fixed_by_definitions_or_case_expectation=fixed))
    return rows

def show(value,censored=False):return 'beyond observed horizon' if censored else 'unavailable' if value is None else f'{value:.6g}'

def conjunction(rows):
    verdicts=[r['verdict'] for r in rows]
    return 'not met' if 'not met' in verdicts else 'met' if verdicts and all(v=='met' for v in verdicts) else 'unresolved at this sample size'

def primary_readout(summary,rows):
    f=summary['family'];selected=[]
    for row in rows:
        s=summary['settings'][row['setting']];l=row['labels'];m=row['metric']
        take=(f=='G1' and l.get('reader')=='both' and m in ('against_c','against_d')) or (f=='G2' and l.get('truth') in ('P','Ps') and ((m=='records' and l.get('reader')=='frame-blind') or m=='records_ratio')) or (f=='G3' and s['theta']==.7 and s['delta']==0 and ((m=='D_minus_K_capture' and l.get('source')=='Mc' and l.get('truth')==0) or (m=='closed' and l.get('source')=='H'))) or (f=='G4' and s['rho']==1/16 and not s['tau_known'] and l.get('truth')=='A9' and l.get('reader')=='offers' and m in ('offers_needed','class_records')) or (f=='G5' and not s['offgrid'] and m=='records' and l.get('truth') in ('SP','LV'))
        if take:selected.append(row)
    labels=dict(against_c='records to distinguish balance-relative from direction-relative firing',against_d='records to distinguish direction-relative from balance-relative firing',records='records needed',records_ratio='frame-blind to frame-aware records ratio',D_minus_K_capture='closed minus open capture',closed='honest-source closure share',offers_needed='offers needed',class_records='records needed')
    detail=[]
    for r in selected:
        lo,hi=r['interval95'];tag=', '.join(str(r['labels'][k]) for k in ('truth','source') if k in r['labels'])
        detail.append(f"{labels.get(r['metric'],r['metric'])} ({tag}): {show(r['estimate'],r['estimate_censored'])}, 95% interval {show(lo,r['lower_censored'])} to {show(hi,r['upper_censored'])}")
    barrows=[r for r in selected if r['bar'] is not None]
    assessment=('Exact con-protection bar: '+conjunction([r for r in barrows if r['metric']=='D_minus_K_capture'])+'; separate honest-source closure arm: '+conjunction([r for r in barrows if r['metric']=='closed']) if f=='G3' else 'Primary bar conjunction: '+conjunction(barrows))
    headline=assessment+'. '+('; '.join(detail)+'.' if detail else 'No matching primary stratum in this variant.')
    if f=='G3':
        crossing=sorted([(summary['settings'][r['setting']]['delta'],r['verdict']) for r in rows if r['metric']=='D_minus_K_logscore' and r['labels'].get('source')=='M' and r['labels'].get('truth')==0 and summary['settings'][r['setting']]['theta']==.7 and summary['settings'][r['setting']]['delta']>0])
        positive=[d for d,v in crossing if v=='met'];uncertain=[d for d,v in crossing if v=='unresolved at this sample size']
        lower=[d for d in uncertain if positive and d<min(positive)]
        grid_answer='above 0.1' if not positive else 'at or below 0.01' if min(positive)==.01 else 'between '+str(max(lower))+' and '+str(min(positive)) if lower else str(min(positive))
        headline+=' Contamination grid answer: '+grid_answer+'. Smallest tested contamination with an interval wholly above zero: '+(str(min(positive)) if positive else 'none')+'; unresolved tested levels: '+(', '.join(map(str,uncertain)) or 'none')+'. This is a discrete grid result, not an interpolated threshold.'
    return headline

def exact_references(summary):
    if summary['family']!='G3':return []
    from .day_closure import exact
    rows=[]
    for r in summary['cells']:
        s=summary['settings'][r['setting']];l=r['labels'];m=r['metric']
        if s['delta']!=0 or l.get('source') not in ('H','Mc') or m not in ('D_minus_K_capture','closed'):continue
        value=exact(l['source'],l['truth'],s['theta'],s['records'],s['rho'],s['switch'],s['overpush'],s['expanded'],s['pull_away'])[m]
        lo,hi=r['interval95'];rows.append(dict(setting=r['setting'],labels=l,metric=m,exact=value,estimate=r['estimate'],interval95=[lo,hi],exact_inside_pointwise_interval=lo is not None and hi is not None and lo<=value<=hi))
    return rows

def land(campaign,packet,repo):
    if (packet/'LOCAL_LANDING.json').exists():return
    from runners.run_v26_followups import verify_complete
    verify_complete(packet);p=read(packet/'PLAN.json')
    if p['scientific_kind']=='probe':return
    summary=read(packet/'original/SUMMARY.json');protocol=read(campaign/'protocols'/p['protocol_file']);rows=featured(summary);f=p['family'];claim='G-V26-DAY-'+p['card']+'-1'
    relative='docs/versions/v26-top-and-ends/'+day.REL+'/'+p['card']+'_REPORT.md';report=repo/relative
    outcomes={v:sum(row['verdict']==v for row in rows) for v in ('met','not met','unresolved at this sample size')}
    wording=(day.QUESTIONS[f]+' The '+('registered study' if p['lane']=='primary' else 'separate '+('out-of-case map' if protocol['out_of_case'] else 'texture variant'))+
        f" retains {len(summary['settings'])} settings with sixteen independent whole-world blocks per setting. "+primary_readout(summary,rows)+' Constructed-method evidence; miniature, architecture untested.')
    lines=['# '+p['card']+' — '+protocol['cell'],'',wording,'','Claim `'+claim+'`. No human mechanism or engine display is validated.','',
        'This table identifies the setting index, observation stratum, measure, point estimate, pointwise 95% whole-block interval, registered bar and its disposition. Records-needed estimates are censored; unavailable limits are not imputed to the horizon. Asterisks mark results fixed by definitions or expected from the case construction, rather than independent tests of the theory.','',
        '| Setting | Stratum | Measure | Estimate | 95% interval | Bar | Assessment |','|---:|---|---|---:|---|---|---|']
    for row in rows:
        lo,hi=row['interval95'];interval=show(lo,row['lower_censored'])+' to '+show(hi,row['upper_censored']);bar='—' if row['bar'] is None else row['direction'].replace('_',' ')+' '+str(row['bar']);mark=' *' if row['fixed_by_definitions_or_case_expectation'] else ''
        lines.append('| '+str(row['setting'])+' | '+json.dumps(row['labels'],sort_keys=True)+' | '+row['metric'].replace('_',' ')+mark+' | '+show(row['estimate'],row['estimate_censored'])+' | '+interval+' | '+bar+' | '+row['verdict']+' |')
    lines+=['','Pre-check scale, not saved scientific results: '+EXPECTED[f],'',
        'For conjunctive primaries, every required component must be met; one not-met component makes the conjunction not met, otherwise it remains unresolved. No headline verdict is read from a point estimate. The one-nat sustained margin is finite-horizon descriptive evidence, not an error-controlled or anytime-valid decision.',
        '', 'By construction (retained from the prospective card):','',protocol['by_construction'],'',
        'Declared modelling choices and assumed choice model: '+protocol['modelling_choices'],'',
        'Case classification: '+('out-of-case descriptive map; its failed check is retained and cannot change the primary' if protocol['out_of_case'] else 'passed the excluded case checks')+'.',
        '', 'Complete original/replay raw bytes match. Independent inference, complete block-bootstrap reaggregation and separate decimal mean aggregation passed. Some likelihood kernels and the declared generative model are shared; this validates finite calculations, not model realism. Raw makers remain nested within independent blocks. Intervals are pointwise and descriptive, not multiplicity-adjusted.',
        '',*day.ROUTES[f],'','Recipient delivery awaits the owner’s uptake; this report does not claim a new cross-repository delivery.',
        '',f'[Protocol](protocols/{p["card"]}.json). [All measures and frozen settings](../../../../results/v26/{day.REL}/{packet.name}/SUMMARY.json). [Independent review](../../../../results/v26/{day.REL}/{packet.name}/INDEPENDENT_REVIEW.json).','']
    references=exact_references(summary)
    if references:
        target=repo/'results/v26'/day.REL/packet.name/'EXACT_REFERENCES.json'
        write(target,dict(comparisons=references,interpretation='Pointwise simulation intervals may miss an exact expectation by chance. Differences are diagnostic, not a fitted acceptance rule.'))
        lines+=['Exact source-gate expectations are retained beside simulation estimates in [EXACT_REFERENCES.json](../../../../results/v26/'+day.REL+'/'+packet.name+'/EXACT_REFERENCES.json). A pointwise interval miss is not automatically an instrument failure; compare size, standard error and independent reconstruction before diagnosing a repair.','']
    text='\n'.join(lines)
    if report.exists() and report.read_text(encoding='utf-8')!=text:raise ValueError('report exists with different bytes')
    report.write_text(text,encoding='utf-8',newline='\n')
    exchange=repo/'docs/exchange'/('GHOST_'+claim.replace('-','_')+'.md');exchange.write_text('# '+claim+'\n\n'+wording+'\n\n'+'\n\n'.join(day.ROUTES[f])+'\n\nDelivery awaits owner uptake. [Report](../'+relative.removeprefix('docs/')+').\n',encoding='utf-8',newline='\n')
    path=repo/'FINDINGS.md';content=path.read_text(encoding='utf-8')
    if '## '+claim not in content:
        i=content.find('\n');path.write_text(content[:i+1]+'\n## '+claim+'\n\n'+wording+'\n\n[Report]('+relative+').\n'+content[i+1:],encoding='utf-8',newline='\n')
    path=repo/'docs/theory/READING_INTENT.md';content=path.read_text(encoding='utf-8');heading='## V26 theory-day relevance checks, November 6'
    if heading not in content:
        i=content.find('\n');content=content[:i+1]+'\n'+heading+'\n\n| Claim | Question | Status and evidence |\n|---|---|---|\n\n**What these add up to.** These separately commissioned constructed tests examine identifiable distinctions under stated reader and choice assumptions. Fixed-by-design outcomes and descriptive variants do not corroborate human mechanisms.\n'+content[i+1:]
    if '| '+claim+' |' not in content:
        anchor=content.index('|---|---|---|',content.index(heading))+len('|---|---|---|');content=content[:anchor]+'\n| '+claim+' | '+day.QUESTIONS[f]+' | QUALIFIED constructed method. '+wording+' [Report](../'+relative.removeprefix('docs/')+') |'+content[anchor:]
        path.write_text(content,encoding='utf-8',newline='\n')
    claims=read(repo/'docs/theory/CLAIMS.json');row=dict(id=claim,allowed_wording=wording,claim_kind='method',architecture='miniature, architecture untested',assumed_choice_model=protocol['modelling_choices'],reader='Declared finite exact readers; pointwise whole-block intervals',estimate=rows,control_status='passes',source_file=relative,source_line=3,source_sha256=digest(report),supersedes=None,superseded_by=None,correction_class=None)
    old=next((x for x in claims['claims'] if x['id']==claim),None)
    if old is not None and old!=row:raise ValueError('claim changed without correction')
    if old is None:
        claims['claims'].append(row)
        (repo/'docs/theory/CLAIMS.json').write_text(json.dumps(claims,indent=2,sort_keys=True,ensure_ascii=False)+'\n',encoding='utf-8',newline='\n')
    write(packet/'LOCAL_LANDING.json',dict(claim=claim,report=relative,report_sha256=digest(report),published=False,recipient_delivery='awaits owner uptake'))

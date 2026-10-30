"""Land completed texture studies from bound results; never launch science or Git."""
import argparse,copy,math,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from ghostscale.validation.soundingline.v23.common import read,write,digest
from runners.prepare_v25_texture import CARDS

REPO=Path(__file__).resolve().parents[1]


def statistics(values):
    available=[float(x) for x in values if x is not None]
    mean=math.fsum(available)/len(available) if available else None
    se=float(np.std(available,ddof=1)/math.sqrt(len(available))) if len(available)>1 else None
    return dict(mean=mean,standard_error=se,available_worlds=len(available),unavailable_worlds=len(values)-len(available),
        descriptive_95_percent_interval=[mean-1.96*se,mean+1.96*se] if se is not None else None)


def selected(summary,setting,metric,reader,history):
    return [c for c in summary['cells'] if c['setting']==setting and c['metric']==metric and c['labels'].get('reader')==reader and c['labels'].get('history')==history]


def values(summary,setting,metric,reader,history):
    cells=selected(summary,setting,metric,reader,history)
    if not cells:raise ValueError('missing frozen reporting cell')
    rows=list(zip(*(c['whole_world_values'] for c in cells),strict=True))
    return [math.fsum(row)/len(row) if all(x is not None for x in row) else None for row in rows]


def difference(a,b):return [x-y if x is not None and y is not None else None for x,y in zip(a,b,strict=True)]
def number(value):return 'unavailable' if value is None else f'{value:.6f}'
def span(values):
    xs=[v for v in values if v is not None]
    return f'{min(xs):.6f} to {max(xs):.6f}' if xs else 'unavailable'


def land(campaign,card):
    started=time.process_time();campaign=Path(campaign);identity=CARDS[card]['identity']
    candidates=[p.parent for p in (campaign/'packets').glob('*/COMPLETE.json') if read(p.parent/'PLAN.json').get('science_identity')==identity]
    if len(candidates)!=1:raise ValueError('exactly one verified packet for this scientific identity required')
    packet=candidates[0];out=REPO/'results/v25'/card
    if out.exists():
        result=read(out/'RESULT.json')
        if result['complete_sha256']!=digest(packet/'COMPLETE.json'):raise ValueError('existing result binding differs')
        return result
    plan=read(packet/'PLAN.json');complete=read(packet/'COMPLETE.json')
    if read(packet/'STATUS.json')['state']!='COMPLETE' or plan.get('texture_phase')!='study':raise ValueError('complete scientific study required')
    for key,name in [('plan_sha256','PLAN.json'),('summary_sha256','original/SUMMARY.json'),('replay_sha256','REPLAY.json'),('review_sha256','INDEPENDENT_REVIEW.json')]:
        if complete[key]!=digest(packet/name):raise ValueError('result binding changed')
    if not read(packet/'original/CONTROLS.json')['passed'] or read(packet/'REPLAY.json')['state']!='passed':raise ValueError('controls/replay failed')
    review=read(packet/'INDEPENDENT_REVIEW.json')
    if review['state']!='INDEPENDENT_RECONSTRUCTION_PASSED':raise ValueError('numerical reconstruction missing')
    summary=read(packet/'original/SUMMARY.json');rows=[]
    for si,setting in enumerate(plan['settings']):
        row=dict(setting=si,configuration=setting)
        if card=='W4F1':
            base=values(summary,si,'detection_ge_two','individual',128)
            for reader in ('estimated','oracle'):
                row[reader+'_gain']=statistics(difference(values(summary,si,'detection_ge_two',reader,128),base))
                row[reader+'_exact_mean_false']=statistics(values(summary,si,'exact_mean_false_detection',reader,128))
            row['individual_exact_mean_false']=statistics(values(summary,si,'exact_mean_false_detection','individual',128))
        elif card=='W6F2':row['gain_per_outcome_block']=statistics(values(summary,si,'nats_per_outcome_block','pooled_event_rate',32))
        elif card=='W7F1':
            for n in (8,32):
                row[f'gain_per_violation_{n}']=statistics(values(summary,si,'nats_per_violation','pooled_event_rate',n))
                row[f'total_gain_{n}']=statistics(values(summary,si,'primary','contrasts',n))
        else:
            for reader in ('neither','pleasure','distress','both'):
                row[reader]=dict(log_odds=statistics(values(summary,si,'primary',reader,32)),unresolved_fraction=statistics(values(summary,si,'not_within_32',reader,'threshold')),false_confidence=statistics(values(summary,si,'false_confident',reader,32)))
        rows.append(row)
    if card=='W4F1':
        finding=f"Does estimating a population prior from more other makers improve deviation detection? Across population sizes 8/32/64/256 and concentrations 3/30, estimated-prior minus individual detection at 128 records ranges from {span([r['estimated_gain']['mean'] for r in rows if not r['configuration']['placebo']])}; the oracle-prior contrast ranges from {span([r['oracle_gain']['mean'] for r in rows if not r['configuration']['placebo']])}. Each reader is calibrated to 5% on a separate near-mean panel; exact-mean false rates remain separate. This is a constructed-method diagnostic of estimation and shrinkage, not a human population claim."
        table='Population size and concentration identify each setting; entries are paired detection differences at 128 records.\n\n| Size | Concentration | Placebo | Estimated minus individual | Oracle minus individual |\n|---|---|---|---|---|\n'
        for row in rows:
            s=row['configuration'];table+=f"| {s['population_size']} | {s['alpha']} | {s['placebo']} | {number(row['estimated_gain']['mean'])} | {number(row['oracle_gain']['mean'])} |\n"
    elif card=='W6F2':
        finding='Does the small logged-signature benefit survive a larger fresh replication and changing signal strength? At signal scales '+', '.join(str(r['configuration']['kappa']) for r in rows)+', the mean true-ends gain per observed outcome block at 32 records is '+', '.join(number(r['gain_per_outcome_block']['mean']) for r in rows)+' nats, respectively, against the descriptive .02 margin. Each setting has 64 independent fresh worlds; signal scale is not an assumed linear information multiplier. This is a constructed-method dose map, not powered confirmation or evidence about human emotion.'
        table='Each row gives signal scale, mean information gain per outcome block and its whole-world standard error at 32 records.\n\n| Scale | Mean nats/block | Standard error | Available worlds |\n|---|---|---|---|\n'
        for row in rows:
            a=row['gain_per_outcome_block'];table+=f"| {row['configuration']['kappa']} | {number(a['mean'])} | {number(a['standard_error'])} | {a['available_worlds']} |\n"
    elif card=='W7F1':
        finding=f"Does attention mainly help before final outputs reveal the goal, and does unrelated attention obscure it? Across the three habit distributions and unrelated-attention probabilities 0/.1/.3/.6, correct-attention gains per violation range from {span([r['gain_per_violation_8']['mean'] for r in rows])} nats at 8 units and {span([r['gain_per_violation_32']['mean'] for r in rows])} at 32. Total gains and information per attention event are retained separately. This is a constructed-method finite-history/noise map; final outputs still have distinct exact goal supports, so no new exact alias is claimed."
        table='Habit identifies the native proposal distribution; nuisance probability is unrelated attention. Gains are mean whole-world nats per violation.\n\n| Habit | Nuisance probability | Gain at 8 units | Gain at 32 units |\n|---|---|---|---|\n'
        for row in rows:
            s=row['configuration'];table+=f"| {s['habit']} | {s['epsilon']} | {number(row['gain_per_violation_8']['mean'])} | {number(row['gain_per_violation_32']['mean'])} |\n"
    else:
        same=[r for r in rows if r['configuration']['habit_affect']==r['configuration']['end_distress']]
        assert all(abs(r['distress']['log_odds']['mean'])<1e-10 for r in same)
        finding=f"Which observed affect channel separates a valued end from a cached habit? When their blocked-distress flags agree, distress alone leaves the exact end/habit alias intact through 32 trained-range choices. Across all four flag combinations and noise SDs .25/.5/1/2, the unresolved fraction after 32 records ranges from {span([r['pleasure']['unresolved_fraction']['mean'] for r in rows])} with pleasure alone and {span([r['both']['unresolved_fraction']['mean'] for r in rows])} with both channels. All readers see paired observations. This constructed-method result localizes identification to the supplied signal laws; pleasure is planted as an end-only signal, not validated as a real psychological mechanism."
        table='Flags specify habit/end distress; SD is affect measurement noise. Entries are fractions not reaching 1 nat of correct end/habit evidence by 32 records, averaged within whole worlds.\n\n| Habit distress | End distress | SD | Neither | Pleasure | Distress | Both |\n|---|---|---|---|---|---|---|\n'
        for row in rows:
            s=row['configuration'];table+=f"| {s['habit_affect']} | {s['end_distress']} | {s['noise_sd']} | "+' | '.join(number(row[name]['unresolved_fraction']['mean']) for name in ('neither','pleasure','distress','both'))+' |\n'
    finding+=' Miniature, architecture untested.'
    out.mkdir();public_summary=copy.deepcopy(summary)
    for cell in public_summary['cells']:cell.pop('whole_world_values',None)
    write(out/'SUMMARY.json',public_summary);write(out/'RAW_INDEX.json',read(packet/'original/RAW_INDEX.json'))
    write(out/'ANALYSIS.json',dict(card=card,rows=rows,uncertainty='Descriptive 1.96-standard-error intervals across whole worlds, not simultaneous or confirmatory; missing worlds remain counted.'))
    claimid=f'G-V25-{card}-1';at=datetime.now(timezone.utc).isoformat()
    result=dict(at=at,state='VERIFIED_EXPLORATORY_FOLLOWUP',card=card,family=plan['family'],packet=packet.name,claim_id=claimid,generation=1,worlds=review['worlds'],confirmation=False,complete_sha256=digest(packet/'COMPLETE.json'),plan_sha256=digest(packet/'PLAN.json'),source_archive_sha256=plan['source_archive_sha256'],replay_sha256=digest(packet/'REPLAY.json'),review_sha256=digest(packet/'INDEPENDENT_REVIEW.json'),cpu_seconds=read(campaign/'accounting'/(packet.name+'.json'))['cpu_seconds'])
    write(out/'RESULT.json',result)
    text=f'# {card}: exploratory texture\n\nCorrections first: earlier W4 export and N16 stress-map corrections remain; no prior scientific value or criterion changed.\n\n{finding}\n\nClaim `{claimid}`. {result["worlds"]} fresh whole worlds; complete byte replay, independent numerical reconstruction and controls passed.\n\n{table}\nThe full curves and descriptive uncertainty are in [machine results](../../../results/v25/{card}/). [Prospective protocol](TEXTURE_PROTOCOL_2026-10-29.md). This is a single generation from the primary, with no descendant automatically commissioned. Independent arithmetic shares the declared generative model; neither model realism nor human transfer is validated.\n'
    report=REPO/f'docs/versions/v25-pieces-and-ends/{card}_REPORT.md';report.write_text(text,encoding='utf-8',newline='\n')
    exchange=REPO/f'docs/exchange/GHOST_V25_{card}_2026-10-29.md'
    exchange.write_text(text.replace('../../../results/','../../results/').replace('(TEXTURE_PROTOCOL_2026-10-29.md)','(../versions/v25-pieces-and-ends/TEXTURE_PROTOCOL_2026-10-29.md)'),encoding='utf-8',newline='\n')
    ledger=read(REPO/'docs/theory/CLAIMS.json')
    if any(c['id']==claimid for c in ledger['claims']):raise ValueError('claim already exists')
    ledger['claims'].append(dict(id=claimid,allowed_wording=finding,estimate=rows,control_status='passes',claim_kind='method',architecture='miniature, architecture untested',assumed_choice_model={'W4F1':'Finite softmax choice library; leave-one-out population prior; known observation law','W6F2':'Native value generator with heavy overlap, .1 outcome-block probability and known logged signature law','W7F1':'Native revision/attention law with known habit and unrelated-attention probabilities','W12F1':'Native beta 3/.02 lapse, trained-range choices; known Gaussian affect law and flags'}[card],reader='Declared finite posterior and nuisance model; exact independent arithmetic',source_file=report.relative_to(REPO).as_posix(),source_line=5,source_sha256=digest(report),supersedes=None,superseded_by=None,correction_class=None));ledger['updated']=at
    write(REPO/'docs/theory/CLAIMS.json',ledger,immutable=False);write(REPO/'docs/exchange/v25-receiving-packet/CLAIMS.json',ledger,immutable=False)
    path=REPO/'FINDINGS.md';body=path.read_text(encoding='utf-8');head,tail=body.split('\n',1);path.write_text(head+f'\n\n## {card}, October 29\n\n{finding} Claim `{claimid}`. [Report](docs/versions/v25-pieces-and-ends/{card}_REPORT.md).\n'+tail,encoding='utf-8',newline='\n')
    path=REPO/'docs/theory/READING_INTENT.md';body=path.read_text(encoding='utf-8');anchor='|---|---|---|---|';offset=body.index(anchor)+len(anchor)
    body=body[:offset]+f'\n| {claimid} | {CARDS[card]["reason"]} | **EXPLORATORY constructed method (run)** | {finding} [Report](../versions/v25-pieces-and-ends/{card}_REPORT.md) |'+body[offset:]
    lines=body.splitlines();start=next(i for i,s in enumerate(lines) if claimid in s)
    end=next(i for i in range(start+1,len(lines)) if lines[i] and not lines[i].startswith('|'))
    lines.insert(end,f'The October 29 primary-derived texture adds {card}: '+finding+' It does not revise or confirm the parent.\n')
    path.write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    write(campaign/'accounting'/f'{card.lower()}-texture-landing.json',dict(pool='protected',kind='verification',cpu_seconds=time.process_time()-started,state='COMPLETE'))
    print(card,finding,flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--campaign',type=Path,required=True);parser.add_argument('--card',choices=CARDS,required=True)
    args=parser.parse_args();land(args.campaign,args.card)

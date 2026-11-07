"""Prospectively bounded expansion; no scientific model selection or retries."""
import math
from . import registry,contract,reporting
from .common import read,write,digest,verify_complete


def terminal_card(c,cid):
    matches=[p.parent for p in (c/'packets').glob('*/PLAN.json') if read(p)['card']==cid and read(p)['kind']!='probe']
    return matches[0] if len(matches)==1 and (matches[0]/'LOCAL_LANDING.json').exists() else None


def practical(row):
    m=row['metric']
    if 'records' in m:return 1.
    if 'brier' in m:return .02
    if m in ('primary','K_gap'):return .5
    if 'ratio' in m or 'share' in m:return .05
    return .05


def texture_change(primary,variant):
    a=read(primary/'original/SUMMARY.json');b=read(variant/'original/SUMMARY.json');lookup={}
    for row in a['cells']:lookup[(row['setting'],row['metric'],str(sorted(row['labels'].items())))]=row
    changes=[];unknown=False
    for row in b['cells']:
        if row['metric'] in ('peak_states',):continue
        old=lookup.get((row['setting'],row['metric'],str(sorted(row['labels'].items()))))
        if old is None:unknown=True;continue
        if row['estimate'] is None or old['estimate'] is None:unknown=True;continue
        if abs(row['estimate']-old['estimate'])>practical(row):changes.append(dict(metric=row['metric'],setting=row['setting'],difference=row['estimate']-old['estimate'],bar=practical(row)))
    return dict(changed=bool(changes),unresolved_comparison=unknown,changes=changes,
                counts_toward_stop=not changes and not unknown)


def eligible_second_generation(c,primary):
    landed=read(primary/'LOCAL_LANDING.json')
    if landed['verdict']=='met':return True
    p=read(primary/'PLAN.json');summary=read(primary/'original/SUMMARY.json');rows=reporting.annotate(summary)
    checks=read(c/'designs'/(p['card']+'.json')).get('excluded_rival_prechecks',[])
    for row in rows:
        if row['bar'] is None or row['estimate'] is None:continue
        for check in checks:
            if check['setting']!=row['setting']:continue
            for earlier in check['measures']:
                if earlier['metric']!=row['metric'] or earlier['labels']!=row['labels'] or earlier['statistic']!='mean':continue
                values=earlier['values'];mean=sum(values)/len(values)
                if mean*row['estimate']<0:return True
    return False


def register_descendants(c,parent,build_card,source):
    p=read(parent/'PLAN.json');cid=p['card'];card=read(c/'cards'/(cid+'.json'));landing=read(parent/'LOCAL_LANDING.json')
    marker=c/'expansions'/(cid+'.json')
    if marker.exists():return
    primary=terminal_card(c,p['family']);eligible=eligible_second_generation(c,primary) if primary else False
    generation=card.get('generation',0);registered=[]
    if card['lane']=='study' or (card['lane']=='followup' and generation==1 and eligible):
        for n,(axis,change) in enumerate(registry.expansion_menu(p['family']),1):
            if generation==1:
                # New generation crosses a distinct axis with the parent's
                # frozen perturbation; omit a duplicate instead of refilling.
                changed=[s|change for s in card['settings']]
                if changed==card['settings']:continue
            else:changed=[s|change for s in card['settings']]
            child=card|dict(id=cid+'-F'+str(n),lane='followup',generation=generation+1,settings=changed,
                primary_positive_or_sign_reversal=eligible,parent_complete_sha256=digest(parent/'COMPLETE.json'),
                parent_card=cid,axis=axis,order=20000+1000*contract.ORDER.index(p['family'])+generation*100+n)
            write(c/'cards'/(child['id']+'.json'),child);build_card(c,child,source);registered.append(child['id'])
    if card['lane'] in ('study','texture') and landing['unresolved']:
        child=card|dict(id=cid+'-P2',lane='precision',blocks=2*card['blocks'],parent_blocks=card['blocks'],
            parent_complete_sha256=digest(parent/'COMPLETE.json'),parent_card=cid,order=10000+card['order'])
        write(c/'cards'/(child['id']+'.json'),child);build_card(c,child,source);registered.append(child['id'])
    write(marker,dict(parent_complete_sha256=digest(parent/'COMPLETE.json'),registered=registered,
        generation_two_eligible=eligible,precision_once_only=True))


def advance(c,prepare,build_card,source):
    """Call between bounded dispatches. Sources and excluded checks precede data."""
    for f in contract.ORDER:
        primary=terminal_card(c,f)
        if primary is None:continue
        register_descendants(c,primary,build_card,source)
        cards=sorted([read(p) for p in (c/'cards').glob('*.json') if read(p)['family']==f],key=lambda x:x['order'])
        streak=0;stopped=False;next_texture=None;records=[]
        for card in (x for x in cards if x['lane']=='texture'):
            cid=card['id'];done=terminal_card(c,cid)
            if done:
                proto=read(c/'protocols'/(cid+'.json'))
                comparison=texture_change(primary,done)
                if not proto['out_of_case']:streak=streak+1 if comparison['counts_toward_stop'] else 0
                records.append(dict(card=cid,streak=streak,out_of_case=proto['out_of_case'],**comparison))
                register_descendants(c,done,build_card,source)
                if streak>=7:stopped=True;break
                continue
            if not stopped:next_texture=cid
            break
        write(c/'texture_lanes'/(f+'.json'),dict(primary=f,consecutive_unchanged=streak,stopped=stopped,completed=records,next=next_texture),immutable=False)
        eligible=[x for x in cards if x['lane'] in ('followup','precision')]
        if next_texture:eligible=[read(c/'cards'/(next_texture+'.json'))]+eligible
        for card in eligible:
            cid=card['id'];done=terminal_card(c,cid)
            if done:
                if card['lane']=='followup':register_descendants(c,done,build_card,source)
                continue
            if not (c/'protocols'/(cid+'.json')).exists():continue
            if any(read(p)['card']==cid for p in (c/'packets').glob('*/PLAN.json')):continue
            if read(c/'dispositions'/(cid+'.json'))['state']=='BUILT':prepare(c,cid)

"""Native twin observation adapters; truth is never an input to this module.

These streaming engineering bundles preserve public observations. Combining
them into empirically matched whole-testbed graphs remains a separate gate.
"""
from copy import deepcopy
from datetime import timedelta
import json
from . import ledger
from .common import canonical


def emit(v,public,schema,identity):
    e=ledger.Emission(schema,identity);family=v['family']
    if family not in ('T1','T2','T3','T4'):raise ValueError('unknown twin')
    header={k:value for k,value in public.items() if k not in ('history','query')}
    e.add('ContextEvent',event_id=ledger._identity(identity,'header'),session_id=e.session,
        t=ledger.BASE_TIME,kind='other',payload=dict(channel='twin-header',public=header))
    actor_ids={public.get('maker',0):e.maker};sessions={}
    def session(actor,number):
        if actor not in actor_ids:
            actor_ids[actor]=ledger._identity(identity,'actor',actor)
            e.add('Maker',maker_id=actor_ids[actor],source_dataset='ghost-v23-synthetic',identity='synthetic',display_label='Synthetic participant')
            e.rows['Artifact'][0]['maker_ids'].append(actor_ids[actor])
        key=(actor,number)
        if key not in sessions:
            sessions[key]=ledger._identity(identity,'session',actor,number)
            e.add('Session',session_id=sessions[key],maker_id=actor_ids[actor],artifact_id=e.artifact,
                t_start=ledger.BASE_TIME,t_end=ledger.BASE_TIME+timedelta(days=1))
        e.maker=actor_ids[actor];e.session=sessions[key]
    for i,row in enumerate(public['history']):
        actor=row.get('actor',row.get('maker',public.get('maker',0)))
        session(actor,row.get('session',row.get('game',row.get('article',0))))
        if family=='T1':
            observed={k:value for k,value in row.items() if k not in ('action','corrected','latency_bin','latency_s','keystrokes')}
            decision=e.decision('history',i,list(range(4)),row['action'],observed=observed)
            decision['cost'].update(latency_s=row['latency_s'],keystrokes=row['keystrokes'])
            outcome={k:row[k] for k in ('corrected','latency_bin')}
            e.add('ContextEvent',event_id=ledger._identity(identity,'outcome',i),session_id=e.session,
                t=ledger.BASE_TIME+timedelta(seconds=e.counter),kind='other',payload=dict(channel='twin-outcome',index=i,observed=outcome))
            if row['corrected']:
                e.add('Correction',correction_id=ledger._identity(identity,'correction',i),maker_id=e.maker,
                    decision_id=decision['decision_id'],before_ref=e.content([0]*4),after_ref=e.content(row['edit_delta']),
                    t=ledger.BASE_TIME+timedelta(seconds=e.counter))
        elif family=='T2':
            options=[dict(value=a,style=b) for a,b in zip(row['values'],row['style_features'],strict=True)]
            decision=e.decision('history',i,options,row['taken'],observed={k:x for k,x in row.items() if k!='taken'})
            decision['cost']['time_left_s']=row['clock_s']
        elif family=='T4':
            e.decision('history',i,list(range(3)),row['rating'],observed={k:x for k,x in row.items() if k!='rating'})
        else:
            # The diff is observed; the latent intent selecting a noisy delta is
            # deliberately not supplied as the chosen discrete alternative.
            e.decision('history',i,row['alternatives'],None,observed=row)
            revision=e.add('Revision',revision_id=ledger._identity(identity,'revision',i),artifact_id=e.artifact,
                maker_id=e.maker,t=ledger.BASE_TIME+timedelta(seconds=e.counter),content_ref=e.content(row['diff']))
            e.rows['Artifact'][0]['revisions'].append(revision['revision_id'])
            if row['summary'] is not None:
                goal=e.add('StatedGoal',goal_id=ledger._identity(identity,'summary',i),maker_id=e.maker,
                    scope='revision',scope_id=revision['revision_id'],text='synthetic-purpose-'+str(row['summary']),
                    source='edit_summary',t=ledger.BASE_TIME+timedelta(seconds=e.counter))
    session(public.get('maker',0),0)
    if family=='T1':
        for sid in sorted(set(sessions.values())):
            goal=e.add('StatedGoal',goal_id=ledger._identity(identity,'prompt',sid),maker_id=actor_ids[public['maker']],
                scope='session',scope_id=sid,text=canonical(public['statement']).decode(),source='prompt',t=ledger.BASE_TIME)
            next(s for s in e.rows['Session'] if s['session_id']==sid)['stated_goal_ids'].append(goal['goal_id'])
    if family=='T4':
        for i,value in enumerate(public['reason']):
            e.add('StatedPreference',pref_id=ledger._identity(identity,'reason',i),maker_id=e.maker,
                attribute='synthetic-recipe-'+str(i),value=value,scale=[],source='self_description',t=ledger.BASE_TIME)
    if 'query' in public:
        count=len(public['query']['values']) if family=='T2' else 4 if family=='T1' else 3
        e.decision('query',0,list(range(count)),None,observed=public['query'])
    return e.finish()


def decode(v,bundle,schema):
    ledger.validate(bundle,schema);family=v['family']
    headers=[r['payload']['public'] for r in bundle['rows']['ContextEvent'] if r['payload'].get('channel')=='twin-header']
    if len(headers)!=1:raise ValueError('one twin header required')
    public=deepcopy(headers[0]);history=[]
    outcomes={r['payload']['index']:r['payload']['observed'] for r in bundle['rows']['ContextEvent'] if r['payload'].get('channel')=='twin-outcome'}
    for decision in sorted(bundle['rows']['Decision'],key=lambda r:(r['context']['channel'],r['context']['index'])):
        context=decision['context'];row=deepcopy(context['observed'])
        if context['channel']=='query':public['query']=row;continue
        chosen=next((i for i,o in enumerate(decision['options']) if o['option_id']==decision['taken']),None)
        if family=='T1':
            row.update(action=chosen,latency_s=decision['cost']['latency_s'],keystrokes=decision['cost']['keystrokes'],**outcomes[context['index']])
        elif family=='T2':row['taken']=chosen
        elif family=='T4':row['rating']=chosen
        history.append(row)
    public['history']=history
    return public

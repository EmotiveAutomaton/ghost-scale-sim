"""Fixed-sample constructed-world check of S15-PROFILE-FEATURE-V1.

The sibling's actual function ASTs execute unchanged. No human data, hidden
preferences, future choices or evaluation summaries enter their feature builder.
"""
import ast
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import re
import time
import numpy as np
from scipy.stats import t
from .common import read, write, digest, rng

ARMS = ('baseline', 'own', 'donor')
CONDITIONS = ('planted', 'uninformative', 'constant')
ACTIONS = ('accept', 'edit', 'dismiss', 'ignore')
PROFILE_HASH = '33c28838f69882147ec21814ce007481a6a0487437321fe76035b9869a58c39a'
TASK_HASH = 'b98ce49c83fcc127210c54bbccab2b0a899930a3799b7903e38e81806c99d2c9'
IMPLEMENTATION = 'f8207f2b65e12885f1493555bab2792b1050495ea7231324f9d0760e94626fca'
SCHEMA = 'ab5c4f63f66693da9b3be0cf33598e1a244b25f6b7ad0538ef2c92ee7a9c49c7'
SL_PLAN = '96d000c78504054516a14bd6d5b9a88e45dbb89d460d57c82743efe75d0f41a0'


def configuration():
    return dict(study='S15-PROFILE-FEATURE-V1', worlds=20, writers=8, rounds=4,
        conditions=list(CONDITIONS), seed_namespace='ghost-v23-s15-profile-research-v1',
        task_dimensions=9, profile_dimensions=32, option_count=4, feature_columns=164,
        signal_probability_range=[.70,.95], profile_noise_words=[0,1,2],
        matched_donor='fixed random writer derangement, equal opportunity/history length',
        severity='independent world draws: signal strength, type/action mapping, hash collisions, nuisance words, task context, resolution delays and writer order',
        fitting='exact sibling predict_all; canonical Backbone unchanged; resolved strictly before target opens',
        score='mean paired half-Brier gain per independent world; all chronological targets included',
        interval='two-sided Student t 95% interval across 20 independent world means, descriptive finite generator support',
        positive_rule='both planted mean gain lower bounds > 0.005 against baseline and donor',
        null_rule='both own-minus-comparator gain upper bounds <= 0.005 in uninformative and constant conditions',
        structural_rule='constant own/donor predictions equal <=1e-12; future-label prefix and option-row-equivariance differences <=1e-7; exact strict train-ID sets; finite normalized probabilities',
        margin=.005, probability_tolerance=1e-7, future_prefix=12,
        structural_worlds=list(range(20)), option_control_condition='planted',
        pilot_seed_namespace='ghost-v23-s15-profile-cost-only-v2',
        pilot_excluded=True, no_adaptive_stopping=True, no_empirical_or_values_claim=True)


def verify_configuration(plan):
    if plan['configuration'] != configuration():
        raise ValueError('frozen profile research configuration differs')
    if plan['sounding_line_plan_sha256'] != SL_PLAN:
        raise ValueError('wrong Sounding Line comparison')
    if plan['implementation_hash'] != IMPLEMENTATION or plan['ledger_schema_hash'] != SCHEMA:
        raise ValueError('wrong canonical engine')
    if plan['cpu_cap_seconds'] > 7200 or plan.get('protected'):
        raise ValueError('profile validation cannot consume protected allocation')


def exact_functions(plan):
    from tompathy_engine.models.backbone import Backbone, Choice
    from tompathy_engine.interface import implementation_hash, schema_hash
    if implementation_hash() != IMPLEMENTATION or schema_hash() != SCHEMA:
        raise ValueError('loaded engine differs from requested engine')
    namespace = dict(np=np, re=re, hashlib=hashlib, datetime=datetime,
        timezone=timezone, ARMS=ARMS, ACTIONS=ACTIONS,
        engine=lambda: (Backbone, Choice, IMPLEMENTATION, SCHEMA))
    selected = {}
    for key, expected, names in (
        ('profile_source', PROFILE_HASH, ('profile_vector','choice_features','predict_all')),
        ('task_source', TASK_HASH, ('features',))):
        path=Path(plan[key])
        if digest(path) != expected: raise ValueError('sibling source changed')
        source=path.read_text(encoding='utf-8')
        tree=ast.parse(source)
        nodes=[node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        if {node.name for node in nodes} != set(names): raise ValueError('missing exact function')
        selected.update({node.name:hashlib.sha256(ast.get_source_segment(source,node).encode()).hexdigest() for node in nodes})
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
    return namespace, selected


def construct(world, condition, functions, *, pilot=False):
    cfg=configuration()
    random=rng(cfg['pilot_seed_namespace'] if pilot else cfg['seed_namespace'],world)
    writers=cfg['writers']; types=np.tile(np.arange(4),2); random.shuffle(types)
    action_map=random.permutation(4)
    strength=float(random.uniform(*cfg['signal_probability_range']))
    words=[f'preference{int(random.integers(0,1000000))}' for _ in range(4)]
    noise_count=int(random.choice(cfg['profile_noise_words']))
    profiles=[[words[types[w]]]+[f'context{int(random.integers(0,1000000))}' for _ in range(noise_count)] for w in range(writers)]
    donor=random.permutation(writers)
    while np.any(donor==np.arange(writers)): donor=random.permutation(writers)
    null_types=random.integers(4,size=writers)
    if condition=='uninformative':
        profiles=[[f'independent{int(random.integers(0,1000000))}' for _ in range(1+noise_count)] for w in range(writers)]
    elif condition=='constant': profiles=[['constant shared description'] for _ in range(writers)]
    elif condition!='planted': raise ValueError('unknown condition')
    vectors=[functions['profile_vector'](p) for p in profiles]
    records=[]; labels=[]; truth=[]; order=[]
    for round_i in range(cfg['rounds']):
        order.extend((round_i,int(w)) for w in random.permutation(writers))
    for index,(round_i,w) in enumerate(order):
        opening=1700000000000+index*2000
        # Some resolutions coincide with or follow later openings, testing <.
        resolution=opening+int(random.choice([500,2000,4500]))
        task=dict(document='constructed context '+('shared '*int(random.integers(1,8))),
            menu=['shared option '+('text '*int(random.integers(1,12))) for _ in range(int(random.integers(2,7)))],
            decision_index=round_i+1)
        visible=[r for r in records if r['writer']==f'w{w}' and r['resolved_at']<opening]
        if visible: task['previous_handling']=ACTIONS[labels[int(visible[-1]['key'][1:])]['action']]
        features=functions['features'](task,[],False)
        # World truth does not enter the task/profile functions. Null conditions
        # preserve predictable personal actions but remove informative profiles.
        preferred=int(action_map[types[w] if condition=='planted' else null_types[w]])
        probabilities=np.full(4,(1-strength)/3); probabilities[preferred]=strength
        y=int(random.choice(4,p=probabilities))
        row=dict(key=f'k{index}',writer=f'w{w}',prompt=f'p{round_i%3}',
            open_at=opening,resolved_at=resolution,
            profile_cutoff=1699999999999,features={
              'baseline':functions['choice_features'](features,np.zeros(32)).tolist(),
              'own':functions['choice_features'](features,vectors[w]).tolist(),
              'donor':functions['choice_features'](features,vectors[int(donor[w])]).tolist()})
        records.append(row); labels.append(dict(key=row['key'],action=y))
        truth.append(dict(key=row['key'],preferred_action=preferred,probabilities=probabilities.tolist()))
    return records,labels,dict(world=world,condition=condition,signal_probability=strength,
        type_to_action=action_map.tolist(),profile_words=profiles,donors=donor.tolist(),
        reader_profile_cutoff=1699999999999,targets=truth)


def validate_predictions(records, predictions):
    if len(records)!=len(predictions): raise ValueError('prediction coverage')
    for row,prediction in zip(records,predictions):
        if prediction['key'] != row['key']: raise ValueError('prediction identity')
        expected=sorted([r for r in records if r['resolved_at']<row['open_at']],key=lambda r:(r['open_at'],r['key']))
        for arm in ARMS:
            out=prediction['arms'][arm]; p=np.asarray(out['probabilities'])
            if out['training_keys'] != [r['key'] for r in expected]: raise ValueError('strict chronological fit violation')
            if p.shape!=(4,) or not np.isfinite(p).all() or np.any(p<0) or abs(p.sum()-1)>1e-10:
                raise ValueError('invalid probabilities')


def summarize(points, controls):
    result={}; passed=[]
    for condition in CONDITIONS:
        rows=[row for row in points if row['condition']==condition]
        result[condition]={}
        for comparator in ('baseline','donor'):
            values=np.asarray([row['gain_'+comparator] for row in rows])
            if len(values)<2:
                result[condition][comparator]=dict(mean=float(values.mean()) if len(values) else None,lower=None,upper=None,n=len(values));continue
            mean=float(values.mean()); radius=float(t.ppf(.975,len(values)-1)*values.std(ddof=1)/np.sqrt(len(values)))
            result[condition][comparator]=dict(mean=mean,lower=mean-radius,upper=mean+radius,n=len(values))
            passed.append(mean-radius>.005 if condition=='planted' else mean+radius<=.005)
    structural=bool(controls) and all(c['future_max_difference']<=1e-7 and c['option_max_difference']<=1e-7 and c['constant_max_difference']<=1e-12 for c in controls)
    complete=len(points)==configuration()['worlds']*3 and len(controls)==configuration()['worlds']
    return dict(status='PASS' if complete and all(passed) and structural else 'INCONCLUSIVE' if not complete else 'FAIL',
        constructed_world=True,constructed_software_fixture=not complete,
        results=result,structural_controls_pass=structural,complete_fixed_design=complete,
        worlds=len(controls),limitation='Finite constructed profile utility only; no human, semantic-value, calibration or broad architecture claim.')


def run(root,plan,pulse):
    root.mkdir(parents=True,exist_ok=False)
    functions,source_proof=exact_functions(plan)
    write(root/'EXACT_FUNCTIONS.json',source_proof)
    pilot=plan['task']=='profile-pilot'; count=1 if pilot else configuration()['worlds']
    points=[]; controls=[]
    for world in range(count):
        predictions_by_condition={}
        for condition in (('planted',) if pilot else CONDITIONS):
            records,labels,truth=construct(world,condition,functions,pilot=pilot)
            cell=root/f'w{world:02d}-{condition}'
            write(cell/'READER.json',records); write(cell/'LABELS.json',labels)
            write(cell/'EVALUATOR.json',truth)
            start=time.process_time()
            predictions=functions['predict_all'](records,labels,lambda x:pulse(phase='profile-fit',world=world,condition=condition,**x))
            validate_predictions(records,predictions)
            write(cell/'PREDICTIONS.json',predictions)
            write(cell/'fit-cost.json',dict(cpu_seconds=time.process_time()-start))
            scores={a:[] for a in ARMS}
            for r,label in zip(predictions,labels):
                for arm in ARMS:
                    p=np.asarray(r['arms'][arm]['probabilities'])
                    scores[arm].append(float(.5*np.square(p-np.eye(4)[label['action']]).sum()))
            points.append(dict(world=world,condition=condition,
                gain_baseline=float(np.mean(scores['baseline'])-np.mean(scores['own'])),
                gain_donor=float(np.mean(scores['donor'])-np.mean(scores['own'])),
                losses={a:float(np.mean(s)) for a,s in scores.items()}))
            predictions_by_condition[condition]=predictions
            if condition=='planted':
                original_records,original_labels=records,labels
        if not pilot:
            original=predictions_by_condition['planted']; prefix=configuration()['future_prefix']
            changed=deepcopy(original_labels); cutoff=original_records[prefix-1]['open_at']
            for i,r in enumerate(original_records):
                if r['resolved_at']>=cutoff: changed[i]['action']=(changed[i]['action']+1)%4
            # A full call is used below, retaining all future Choice rows. The
            # suffix is not scored as an invariance comparison because it may
            # legitimately train on the mutated historical labels.
            future=functions['predict_all'](original_records,changed,lambda x:pulse(phase='future-control',world=world,**x))
            permutation=[2,0,3,1]; inverse=np.argsort(permutation)
            permuted=deepcopy(original_records); perm_labels=deepcopy(original_labels)
            for row,label in zip(permuted,perm_labels):
                for arm in ARMS: row['features'][arm]=np.asarray(row['features'][arm])[permutation].tolist()
                label['action']=int(inverse[label['action']])
            reordered=functions['predict_all'](permuted,perm_labels,lambda x:pulse(phase='option-control',world=world,**x))
            delta_future=max(abs(np.asarray(a['arms'][arm]['probabilities'])-b['arms'][arm]['probabilities']).max() for a,b in zip(original[:prefix],future[:prefix]) for arm in ARMS)
            delta_option=max(abs(np.asarray(a['arms'][arm]['probabilities'])-np.asarray(b['arms'][arm]['probabilities'])[inverse]).max() for a,b in zip(original,reordered) for arm in ARMS)
            delta_constant=max(abs(np.asarray(r['arms']['own']['probabilities'])-r['arms']['donor']['probabilities']).max() for r in predictions_by_condition['constant'])
            write(root/f'w{world:02d}-future/PREDICTIONS.json',future)
            write(root/f'w{world:02d}-option/PREDICTIONS.json',reordered)
            controls.append(dict(world=world,future_max_difference=float(delta_future),option_max_difference=float(delta_option),constant_max_difference=float(delta_constant)))
        write(root/f'w{world:02d}-POINTS.json',[r for r in points if r['world']==world])
    write(root/'POINTS.json',points); write(root/'CONTROLS.json',controls)
    write(root/'SUMMARY.json',summarize(points,controls))

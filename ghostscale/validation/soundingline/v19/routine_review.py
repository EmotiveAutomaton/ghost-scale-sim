"""Independent full-state, policy, outcome and population review of revision.

Consumes completed evidence; no producer revision or scoring function is used.
Native trajectory validity uses the separate crossed-law reconstruction ruler.
"""
from collections import defaultdict
from itertools import product
import gzip
import json
import math
import numpy as np
from ..v18_3.io import read, write, file_digest, digest
from ..v18_3.world import rng
from .crossed_review import execute, population, near, GOALS, OPS

ARMS = ('keep', 'native', 'request-adopted', 'visible-task', 'full-state-oracle')
METRICS = ('success', 'net_success', 'operation_cost', 'revision_rate',
           'repaired_failure', 'destroyed_success')
AXES = ('changed_request', 'hidden_presentation', 'purpose', 'skill', 'routine')


def task(a, request):
    return float(a[0]*a[1] if request == 0 else a[2])


def parameters(lineage):
    random = rng('v19-local-world', lineage)
    return tuple(float(x) for x in (random.uniform(.8,1.2), random.uniform(.65,.85), random.uniform(.1,.3)))


def distribution(maker, a, b, request, hidden, arm, coefficients):
    purpose, skill, belief, routine = maker
    if arm == 'keep': return {(None, 'inspect'): 1.}
    if arm in ('native', 'request-adopted'):
        strength, rate, routine_weight = coefficients
        purpose = request if arm == 'request-adopted' else purpose
        weights = [1.5 if purpose == 0 else .6,
                   1.4 if (a[0]^belief) != a[1] else .5,
                   1.8 if purpose == 1 else .5]
        weights[2 if routine else 0] += routine_weight
        weights[2 if request else 0] += .2
        weights = [v**strength for v in weights]; total = math.fsum(weights)
        result = {}
        for i, goal in enumerate(GOALS):
            op = ('accept-tool' if skill else 'edit-claim', 'repair-evidence', 'replace-presentation')[i]
            result[goal, op] = weights[i]/total*rate
            result[goal, 'undo' if i == 1 else 'inspect'] = weights[i]/total*(1-rate)
        return result
    if arm not in ('visible-task', 'full-state-oracle'): raise ValueError('unknown arm')
    masked = hidden and arm == 'visible-task'
    assignments = product(range(2), repeat=2) if masked else [(a[2], b[2])]
    scores = defaultdict(list)
    for current, previous in assignments:
        x, y = (a[0],a[1],current), (b[0],b[1],previous)
        for op in OPS:
            if op == 'accept-tool' and not skill: continue
            z = execute(x,y,op,skill,belief,'original')
            scores[op].append(task(z,request) - (.0 if op == 'inspect' else .05))
    scores = {op:math.fsum(v)/len(v) for op,v in scores.items()}
    best = max(scores.values()); chosen = [op for op,v in scores.items() if abs(best-v) <= 1e-12]
    return {(None,op):1/len(chosen) for op in chosen}


def outcome(maker, a, b, request, arm, op):
    z = execute(a,b,op,maker[1],maker[2],'original')
    before, after = task(a,request), task(z,request)
    fee = 0. if op == 'inspect' else .05
    return dict(after=list(z), success=after,
        net_success=after-fee-(0. if arm == 'keep' else .01),
        operation_cost=fee, revision_rate=float(op != 'inspect'),
        repaired_failure=(1-before)*after, destroyed_success=before*(1-after))


def observation(a,b,old,request,hidden):
    return dict(artifact=[a[0],a[1],None if hidden else a[2]],
                undo_artifact=[b[0],b[1],None if hidden else b[2]],
                original_request=old,current_request=request,
                observed_units=[0,1] if hidden else [0,1,2])


def row_check(row, maker, a, b, request, hidden, coefficients):
    arm = row['arm']; expected = distribution(maker,a,b,request,hidden,arm,coefficients)
    outcomes = {(r['goal'],r['operation']):r for r in row['outcomes']}
    if len(outcomes) != len(row['outcomes']) or set(expected) != set(outcomes):
        raise ValueError('revision outcome support mismatch')
    values = {k:[] for k in METRICS}; error = 0.
    for key,p in expected.items():
        actual = outcomes[key]; checked = outcome(maker,a,b,request,arm,key[1])
        error = max(error,near(actual['probability'],p,1e-12))
        if actual['after'] != checked['after']: raise ValueError('revision artifact mismatch')
        for metric in METRICS:
            error = max(error,near(actual[metric],checked[metric],1e-12))
            values[metric].append(p*checked[metric])
    result = {k:math.fsum(v) for k,v in values.items()}
    error = max(error,near([row[k] for k in METRICS],[result[k] for k in METRICS],1e-12))
    return result,error


def controls():
    m=(0,1,0,0); a=(0,0,0); b=(1,1,1); c=(1.,.75,.2)
    keep=outcome(m,a,b,1,'keep','inspect')
    visible=distribution(m,a,b,0,False,'visible-task',c)
    native=distribution(m,a,b,1,False,'native',c)
    adopted=distribution(m,a,b,1,False,'request-adopted',c)
    illegal=False; corrupt=False
    try: outcome((0,0,0,0),a,b,0,'native','accept-tool')
    except ValueError: illegal=True
    try: near([0.,1.],[1.,0.])
    except ValueError: corrupt=True
    return {'live:adoption_changes_goals':native != adopted,
        'positive:undo':outcome(m,a,b,0,'native','undo')['after']==list(b),
        'positive:legal_tool':illegal, 'positive:corruption_rejected':corrupt,
        'positive:task_repair':visible=={(None,'accept-tool'):.5,(None,'undo'):.5},
        'positive:net_task_cost':abs(outcome(m,a,b,0,'visible-task','undo')['net_success']-.94)<1e-14,
        'placebo:keep_identity':keep['success']==keep['net_success']==keep['revision_rate']==0.,
        'placebo:native_hidden':native==distribution(m,a,b,1,True,'native',c),
        'placebo:hidden_bits':distribution(m,a,b,1,True,'visible-task',c)==distribution(m,(0,0,1),(1,1,0),1,True,'visible-task',c),
        'positive:intact_oracle':visible==distribution(m,a,b,0,False,'full-state-oracle',c)}


def zipped(p): return json.loads(gzip.decompress(p.read_bytes()))


def regroup(cells,cfg):
    lineages=sorted({r['lineage'] for r in cells})
    index={(r['lineage'],*(r[k] for k in AXES),r['arm']):r for r in cells}
    if len(index)!=len(cells): raise ValueError('duplicate summary stratum')
    samples=np.random.default_rng(cfg['bootstrap_seed']).integers(len(lineages),size=(cfg['bootstrap_resamples'],len(lineages)))
    strata=list(product((False,True),(False,True),range(2),range(2),range(2)))+[(ch,h,None,None,None) for ch,h in product((False,True),repeat=2)]
    contrasts=[]; pooled=[]
    for stratum in strata:
        by_arm={}
        for arm in ARMS:
            by_lineage=[]
            for l in lineages:
                rows=[r for key,r in index.items() if key[0]==l and key[-1]==arm and all(s is None or v==s for v,s in zip(key[1:-1],stratum))]
                mass=math.fsum(r['population_mass'] for r in rows)
                if not rows or mass<=0: raise ValueError('missing matched population')
                by_lineage.append([math.fsum(r['population_mass']*r[k] for r in rows)/mass for k in METRICS])
            by_arm[arm]=np.asarray(by_lineage)
            pooled.append(dict(zip(AXES,stratum),arm=arm,**{k:float(by_arm[arm][:,i].mean()) for i,k in enumerate(METRICS)}))
        for arm,base in (('native','keep'),('request-adopted','native'),('visible-task','native'),('full-state-oracle','visible-task')):
            delta=by_arm[arm]-by_arm[base]; bs=delta[samples].mean(axis=1)
            for i,metric in enumerate(METRICS):
                contrasts.append(dict(zip(AXES,stratum),arm=arm,baseline=base,metric=metric,
                    mean=float(delta[:,i].mean()),low=float(np.quantile(bs[:,i],.025)),high=float(np.quantile(bs[:,i],.975)),lineage_values=delta[:,i].tolist()))
    return dict(cells=cells,means=pooled,contrasts=contrasts,lineages=lineages,
        uncertainty='paired coefficient lineages; exact native populations; no fitted or human population uncertainty')


def run(root,plan,pulse):
    checks=controls();write(root/'CONTROLS.json',checks)
    if not all(checks.values()):raise ValueError('independent revision controls failed')
    cfg=plan['design'];original=root/'inputs/original'
    for n,h in cfg['input_files'].items():
        if file_digest(root/'inputs'/n)!=h:raise ValueError('review input differs')
    if file_digest(original/'PLAN.json')!=cfg['target_plan_sha256']:raise ValueError('target plan differs')
    design=read(original/'PLAN.json')['design'];summary=read(original/'SUMMARY.json')
    if design['arms']!=list(ARMS) or design['operation_cost']!=.05 or design['inspection_cost']!=.01:raise ValueError('policy/cost admission differs')
    all_cells=[];public={};error=0.;paths=states_count=rows_count=0
    for lineage in design['lineages']:
        pulse(phase='independent-revision',lineage=lineage)
        records=zipped(original/'inputs'/f'lineage-{lineage}_points.json.gz')
        _,_,e=population(records,lineage,'original');error=max(error,e);paths+=len(records)
        grouped=defaultdict(list)
        for i,r in enumerate(records):
            grouped[tuple(r['maker']),tuple(r['final']),tuple(r['steps'][-1]['before']),r['requested_purpose']].append(i)
        expected={}
        for i,(key,indices) in enumerate(sorted(grouped.items())):
            maker,a,b,old=key;mass=math.fsum(records[j]['probability'] for j in indices)
            expected[i]=dict(state_index=i,maker=list(maker),artifact=list(a),undo_artifact=list(b),original_request=old,mass=mass,source_indices=indices)
        actual=read(original/'evaluator'/f'{lineage}-states.json')
        if len(actual)!=len(expected) or {r['state_index'] for r in actual}!=set(expected):raise ValueError('state denominator differs')
        for r in actual:
            other=expected[r['state_index']]
            if {k:v for k,v in r.items() if k!='mass'}!={k:v for k,v in other.items() if k!='mass'}:raise ValueError('state/source assignment differs')
            error=max(error,near(r['mass'],other['mass'],1e-12))
        rows=zipped(original/'raw'/f'{lineage}-revision_points.json.gz');seen=set();acc=defaultdict(list);values_by_state=defaultdict(dict)
        coefficients=parameters(lineage)
        for row in rows:
            sid,ch,hidden,arm=(row[k] for k in ('state_index','changed_request','hidden_presentation','arm'))
            key=(sid,ch,hidden,arm)
            if key in seen or ch not in (False,True) or hidden not in (False,True) or arm not in ARMS:raise ValueError('duplicate/invalid row')
            seen.add(key);s=expected[sid];maker=tuple(s['maker']);a=tuple(s['artifact']);b=tuple(s['undo_artifact']);old=s['original_request'];request=old^int(ch)
            error=max(error,near(row['population_mass'],s['mass'],1e-12))
            values,e=row_check(row,maker,a,b,request,hidden,coefficients);error=max(error,e)
            visible=observation(a,b,old,request,hidden);ident=digest(visible)
            if row['packet_sha256']!=ident:raise ValueError('public evidence hash differs')
            public[ident]=dict(inputs=visible,input_sha256=ident,passage_anchors=['unit-0','unit-1','unit-2'])
            acc[ch,hidden,maker[0],maker[1],maker[3],arm].append((s['mass'],values))
            values_by_state[sid,ch,hidden][arm]=values
        if len(rows)!=len(expected)*20 or len(seen)!=len(rows):raise ValueError('incomplete revision comparison')
        for (sid,ch,hidden),vv in values_by_state.items():
            if set(vv)!=set(ARMS):raise ValueError('incomplete arms')
            if vv['full-state-oracle']['net_success']+1e-12<max(vv[k]['net_success'] for k in ARMS if k!='keep'):raise ValueError('oracle dominance fails')
            if not hidden:near(list(vv['full-state-oracle'].values()),list(vv['visible-task'].values()),1e-12)
        for key,rr in sorted(acc.items()):
            mass=math.fsum(p for p,v in rr)
            all_cells.append(dict(lineage=lineage,**dict(zip(AXES,key[:5])),arm=key[5],population_mass=mass,states=len(rr),
                **{k:math.fsum(p*v[k] for p,v in rr)/mass for k in METRICS}))
        states_count+=len(expected);rows_count+=len(rows)
    def key(r):return (r['lineage'],*(r[k] for k in AXES),r['arm'])
    expected={key(r):r for r in all_cells};actual={key(r):r for r in summary['cells']}
    if len(actual)!=len(summary['cells']) or set(expected)!=set(actual):raise ValueError('summary denominator differs')
    for k,r in expected.items():
        if r['states']!=actual[k]['states']:raise ValueError('summary states differ')
        error=max(error,near([r[v] for v in (*METRICS,'population_mass')],[actual[k][v] for v in (*METRICS,'population_mass')],1e-12))
    packet=read(original/'PUBLIC_PACKET.json')
    if packet['schema']!='v19.revision.reader.1' or packet['cases']!=[public[k] for k in sorted(public)]:raise ValueError('reader projection differs')
    for k,v in dict(paths=paths,states=states_count,rows=rows_count,public_packets=len(public),fits=0).items():
        if summary[k]!=v:raise ValueError('overall denominator differs')
    checks.update(positive_complete_path_reconstruction=True,positive_complete_policy_reconstruction=True,
        positive_complete_public_projection=True,positive_complete_oracle_dominance=True)
    write(root/'INDEPENDENT_REGROUP.json',regroup(all_cells,cfg))
    return dict(controls=checks,paths=paths,states=states_count,rows=rows_count,cells=len(all_cells),public_packets=len(public),max_error=error,
        scope='independent exact population, policy, cost and observation review; no learned competence, historical uniqueness or human endorsement')

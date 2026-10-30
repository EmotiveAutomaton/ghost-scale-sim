"""Fixed-size, fresh-seed N16 robustness extension; no confirmation spending."""
from collections import defaultdict
from pathlib import Path
import math
import statistics

from ..v23 import ledger
from ..v23.common import canonical, digest, read, write
from ..v23.family import rows
from ..v23.pipeline import Stream
from ..v24 import n16, pilot, production

IDENTITY = 'v24-n16-robustness-20261030'
EXCLUDED = IDENTITY + '-excluded'
BLOCKS = 64
CASES = 3840
BUDGETS = (2, 4, 6)
POLICIES = ('P2', 'P3', 'P4', 'P5')
CONDITIONS = ('correct', 'omit-truth', 'omit-other', 'high-reliability')
SOURCES = ('N4d', 'N9', 'N11', 'N15')


def validate(plan):
    root = Path(plan['accounting_root'])
    acceptance = read(root/'ACCEPTANCE.json')
    if digest(root/'ACCEPTANCE.json') != plan['acceptance_sha256']:
        raise ValueError('extension acceptance changed')
    if acceptance['identity'] != IDENTITY or acceptance['blocks'] != BLOCKS:
        raise ValueError('fixed extension design changed')
    if plan['science_identity'] != (EXCLUDED if plan['kind'] == 'excluded' else IDENTITY):
        raise ValueError('extension seed identity mismatch')
    if (plan['family'] != 'N16R' or plan['kind'] not in ('excluded', 'robustness')
            or plan['epsilon'] != .1 or plan['cases_per_cell'] != 5
            or type(plan['block_index']) is not int or not 0 <= plan['block_index'] < BLOCKS
            or plan.get('confirmation') is not False or plan.get('production')):
        raise ValueError('extension design mismatch')
    if plan['variant_ids'] != [v['id'] for v in n16.model.variants()]:
        raise ValueError('incomplete variant roster')
    if digest(root/'PROTOCOL.json') != plan['protocol_sha256']:
        raise ValueError('frozen protocol changed')
    if digest(root/'PRIOR_TERMINAL.json') != acceptance['prior_terminal_sha256']:
        raise ValueError('prior confirmation binding changed')


def contrasts(evidence, mechanism, independent=False):
    """Equal weight within each chooser-error class; scoring retains full truth."""
    table = {(e['scenario'], e['policy']): e['mass'] for e in evidence['posteriors']}
    scenarios = sorted({s for s, _ in table})
    classified = defaultdict(list)
    for scenario in scenarios:
        group = ('omit-truth' if scenario == 'omit-'+str(mechanism) else 'omit-other') if scenario.startswith('omit-') else scenario
        if group not in CONDITIONS:
            raise ValueError('unexpected chooser scenario')
        classified[group].append(scenario)
    if set(classified) != set(CONDITIONS):
        raise ValueError('incomplete chooser condition roster')
    output = {}
    for condition in CONDITIONS:
        for policy in POLICIES:
            for budget in BUDGETS:
                values = []
                q = table['correct', 'P1'][budget-1][mechanism]
                for scenario in classified[condition]:
                    p = table[scenario, policy][budget-1][mechanism]
                    # Normalization can produce 1 + one floating-point step.
                    # Preserve every value; do not clip scores or probabilities.
                    for mass in (table['correct','P1'][budget-1],table[scenario,policy][budget-1]):
                        if any(not math.isfinite(x) or x<0 or x>1+1e-12 for x in mass) or not math.isclose(math.fsum(mass),1.,abs_tol=1e-12,rel_tol=0):
                            raise ValueError('invalid normalized posterior')
                    if not (0 < q <= 1+1e-12 and 0 <= p <= 1+1e-12):
                        raise ValueError('true-label scoring support')
                    # Two algebraically independent forms; no ratio clipping.
                    value = (math.log(.9*p+.1*q)-math.log(q)) if independent else math.log(.9*(p/q)+.1)
                    if not math.isfinite(value) or value < math.log(.1)-1e-11:
                        raise ValueError('mixture lower bound')
                    values.append(value)
                output[f'{condition}|{policy}|{budget}'] = math.fsum(values)/len(values)
    return output


def summarize(records, plan):
    # The native balance check rejects missing, duplicate and unequal-weight cells.
    primary = n16.aggregate([dict(identity=[0, *r['identity'][1:]], mixture_gain=r['mixture_gain']) for r in records], 5)[0]
    by_source = defaultdict(lambda: defaultdict(list))
    for row in records:
        if row['source'] not in SOURCES:
            raise ValueError('source family mismatch')
        for key, value in row['contrasts'].items():
            by_source[row['source']][key].append(value)
    cells = []
    for source in SOURCES:
        if len(by_source[source]) != 48 or any(len(v) != 960 for v in by_source[source].values()):
            raise ValueError('unbalanced source/contrast coverage')
        for key, values in sorted(by_source[source].items()):
            condition, policy, budget = key.split('|')
            cells.append(dict(source=source, condition=condition, policy=policy, probes=int(budget), cases=len(values), mean=math.fsum(values)/len(values)))
    for condition in CONDITIONS:
        for policy in POLICIES:
            for budget in BUDGETS:
                subset = [c for c in cells if c['source'] != 'pooled' and (c['condition'], c['policy'], c['probes']) == (condition, policy, budget)]
                cells.append(dict(source='pooled', condition=condition, policy=policy, probes=budget, cases=CASES, mean=math.fsum(c['mean'] for c in subset)/4))
    paired = next(c['mean'] for c in cells if (c['source'], c['condition'], c['policy'], c['probes']) == ('pooled','correct','P4',2))
    if not math.isclose(paired, primary, abs_tol=1e-11, rel_tol=0):
        raise ValueError('primary and robust aggregation disagree')
    return dict(state='COMPLETE_BALANCED_BLOCK', scientific_identity=plan['science_identity'], block_index=plan['block_index'], cases=len(records), excluded=plan['kind']=='excluded', primary_mean=primary, cells=cells, confirmation=False)


def execute(root, plan, pulse):
    validate(plan)
    root=Path(root); root.mkdir(exist_ok=False)
    schema=production.dependency_check(plan)
    write(root/'CONTROLS.json',dict(passed=True,pre_data_controls_sha256=plan['controls_sha256']))
    streams={k:Stream(root/(k+'_points.jsonl.gz')) for k in ('public','truth','evidence','ledger')}
    scores=[]; block=plan['block_index']
    try:
        for law in range(16):
            for variant in n16.model.variants():
                for case in range(5):
                    identity=[block,law,variant['id'],case]
                    with production.repeated_distribution_cache(), production.unused_descriptive_metrics(True):
                        public,truth,ev,score=n16.case(variant,plan['science_identity'],block,law,case,.1)
                    bundle=pilot.emit(schema,[plan['science_identity'],*identity],public)
                    for kind,payload in [('public',dict(public=public)),('truth',dict(truth=truth)),('evidence',dict(evidence=ev,score=score)),('ledger',dict(ledger=bundle))]:
                        streams[kind].write(dict(identity=identity,**payload))
                    scores.append(dict(identity=identity,source=variant['source'],mixture_gain=score['mixture_gain'],contrasts=contrasts(ev,truth['mechanism'])))
                pulse(setting=0,block=block,law=law,cases=len(scores))
    finally:
        for stream in streams.values(): stream.close()
    write(root/'SUMMARY.json',summarize(scores,plan))
    write(root/'RAW_INDEX.json',{p.name:dict(sha256=digest(p),bytes=p.stat().st_size) for p in sorted(root.glob('*_points.jsonl.gz'))})


def reconstruct(root, plan, pulse):
    validate(plan); root=Path(root); schema=production.dependency_check(plan)
    index=read(root/'RAW_INDEX.json')
    if set(index) != {k+'_points.jsonl.gz' for k in ('public','truth','evidence','ledger')}:
        raise ValueError('incomplete raw roster')
    for name,bound in index.items():
        if digest(root/name)!=bound['sha256'] or (root/name).stat().st_size!=bound['bytes']:
            raise ValueError('raw binding changed')
    scores=[]; variants={v['id']:v for v in n16.model.variants()}
    for public,truth,ev,transport in zip(*(rows(root/(k+'_points.jsonl.gz')) for k in ('public','truth','evidence','ledger')),strict=True):
        identity=public['identity']; block,law,vid,case=identity; variant=variants[vid]
        mechanism=(16*block+law+case)%5 if variant['source']=='N9' else variant['original']['settings']['mechanism']
        if block!=plan['block_index'] or any(r['identity']!=identity for r in (truth,ev,transport)) or truth['truth']!={'law':law,'mechanism':mechanism}:
            raise ValueError('raw identity/truth mismatch')
        ledger.validate(transport['ledger'],schema)
        if canonical(transport['ledger'])!=canonical(pilot.emit(schema,[plan['science_identity'],*identity],public['public'])):
            raise ValueError('canonical ledger mismatch')
        score=production.reconstruct_case(variant,public['public'],truth['truth'],ev['evidence'],.1)
        if any(not math.isclose(score[k],ev['score'][k],abs_tol=1e-11,rel_tol=0) for k in score):
            raise ValueError('independent primary mismatch')
        scores.append(dict(identity=identity,source=variant['source'],mixture_gain=score['mixture_gain'],contrasts=contrasts(ev['evidence'],mechanism,True)))
        if len(scores)%48==0: pulse(setting=0,block=block,cases=len(scores))
    calculated=summarize(scores,plan); saved=read(root/'SUMMARY.json')
    for a,b in zip(calculated['cells'],saved['cells'],strict=True):
        if {k:v for k,v in a.items() if k!='mean'}!={k:v for k,v in b.items() if k!='mean'} or not math.isclose(a['mean'],b['mean'],abs_tol=1e-11,rel_tol=0):
            raise ValueError('independent robustness cell mismatch')
    if not math.isclose(calculated['primary_mean'],saved['primary_mean'],abs_tol=1e-11,rel_tol=0):
        raise ValueError('independent block primary mismatch')
    if {k:v for k,v in calculated.items() if k not in ('primary_mean','cells')}!={k:v for k,v in saved.items() if k not in ('primary_mean','cells')}:
        raise ValueError('summary metadata mismatch')
    return dict(state='INDEPENDENT_RECONSTRUCTION_PASSED',cases=CASES,block_index=plan['block_index'],cells=len(calculated['cells']),summary_sha256=digest(root/'SUMMARY.json'))


def aggregate_blocks(summaries):
    from scipy.stats import t
    n=len(summaries)
    if n<2 or [s['block_index'] for s in summaries]!=list(range(n)) or n>BLOCKS:
        raise ValueError('nonconsecutive fixed-size block prefix')
    if any(s['excluded'] or s['cases']!=CASES or s['scientific_identity']!=IDENTITY for s in summaries):
        raise ValueError('excluded or foreign block in analysis')
    expected={(source,condition,policy,budget) for source in (*SOURCES,'pooled') for condition in CONDITIONS for policy in POLICIES for budget in BUDGETS}
    for summary in summaries:
        keys=[(c['source'],c['condition'],c['policy'],c['probes']) for c in summary['cells']]
        if len(keys)!=len(expected) or set(keys)!=expected:raise ValueError('incomplete robustness cell roster')
    cells=[]
    for i,first in enumerate(summaries[0]['cells']):
        labels={k:v for k,v in first.items() if k not in ('mean','cases')}
        values=[]
        for summary in summaries:
            cell=summary['cells'][i]
            if {k:v for k,v in cell.items() if k not in ('mean','cases')}!=labels:
                raise ValueError('cell roster mismatch')
            values.append(cell['mean'])
        mean=math.fsum(values)/n; se=statistics.stdev(values)/math.sqrt(n);half=float(t.ppf(.975,n-1))*se
        cells.append(dict(**labels,blocks=n,mean=mean,standard_error=se,pointwise_95_t_interval=[mean-half,mean+half],negative_blocks=sum(v<0 for v in values)))
    return dict(state='COMPLETE_FIXED_SIZE_DESCRIPTIVE' if n==BLOCKS else 'INTERIM_DESCRIPTIVE_ONLY',blocks=n,target_blocks=BLOCKS,cases=n*CASES,cells=cells,confirmation=False,limitations=['Fresh-seed robustness under the same declared constructed laws; no human or engine admission.','Chooser omission is not scoring-model omission; the scoring posterior retains the true mechanism.','Intervals are pointwise descriptive t intervals over complete independent blocks, not simultaneous guarantees or a new confirmation test.','The original 175-block confirmation and its alpha/e-process remain unchanged.'])


def controls():
    native=production.controls()
    def fixture(p,q):
        return dict(posteriors=[dict(scenario=s,policy='P'+str(k),mass=[[q if s=='correct' and k==1 else p,1-(q if s=='correct' and k==1 else p)]]*6) for s in ('correct','omit-0','omit-1','high-reliability') for k in range(1,6)])
    zero=contrasts(fixture(.4,.4),0)
    positive=contrasts(fixture(.8,.2),0)
    negative=contrasts(fixture(.02,.8),0)
    assert len(zero)==48 and all(abs(v)<1e-12 for v in zero.values())
    assert all(abs(v-math.log(3.7))<1e-12 for v in positive.values())
    assert all(v<0 and v>=math.log(.1) for v in negative.values())
    independent=contrasts(fixture(.8,.2),0,True)
    assert all(abs(independent[k]-positive[k])<1e-12 for k in positive)
    return dict(passed=True,native=native,equal_reader_zero=True,positive_fixture=True,negative_fixture=True,correct_model_comparator=True,independent_arithmetic=True,confirmation_spending=False)

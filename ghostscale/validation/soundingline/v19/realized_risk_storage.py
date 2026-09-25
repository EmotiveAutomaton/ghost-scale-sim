"""Expected source coverage subject to a guarantee for every realized mask."""
import gzip
from fractions import Fraction as F
from .randomized_source_storage import randomize, fixture
from .robust_source_mass import from_parent
from .source_mass_frontier import PRIORS, POLICIES
from ..v18_3.io import read, write, canonical, file_digest

FLOORS = ((0,1),(1,2),(1,1))


def constrain(selection, multiplier):
    multiplier = F(multiplier)
    if multiplier not in map(lambda x:F(*x), FLOORS):
        raise ValueError('floor multiplier')
    unrestricted = randomize(selection)
    baseline = F(*unrestricted['deterministic_minimum_mass'])
    floor = multiplier*baseline
    survivors = [r for r in selection['candidates'] if min(F(*x) for x in r['masses']) >= floor]
    if not survivors:
        raise ValueError('deterministic baseline missing')
    result = randomize(dict(selection, candidates=survivors))
    rational = lambda x:[x.numerator,x.denominator]
    assert F(*result['worst_realized_mass']) >= floor
    return dict(floor_multiplier=rational(multiplier), realized_mass_floor=rational(floor),
        surviving_masks=sorted(r['mask'] for r in survivors), surviving_count=len(survivors),
        excluded_masks=sorted(r['mask'] for r in selection['candidates'] if r not in survivors),
        lottery=result, unrestricted_lottery=unrestricted['selected_lottery'],
        unrestricted_minimum_expected_mass=unrestricted['minimum_expected_mass'],
        expected_mass_cost=rational(F(*unrestricted['minimum_expected_mass'])-F(*result['minimum_expected_mass'])))


def controls():
    s=fixture([1,2,3],[1,1,1],1,{'a':[1],'b':[2],'c':[3]},
        [[F(7,10),F(1,10),F(1,5)],[F(1,10),F(7,10),F(1,5)]])
    a,b,c=[constrain(s,F(*x)) for x in FLOORS]
    empty=constrain(fixture([1,2],[1,1],0,{'none':[]},[[F(1,2)]*2]),F(1))
    return {'live:expected_gain_survives_half_floor': b['lottery']['expected_gain']==[1,5],
        'positive:full_floor_removes_known_gain': c['lottery']['minimum_expected_mass']==[1,5] and c['expected_mass_cost']==[1,5],
        'positive:zero_floor_identity': a['lottery']==randomize(s),
        'placebo:empty_no_gain': empty['lottery']['expected_gain']==[0,1]}


def run(root,plan,pulse):
    cfg=plan['design']
    if cfg['source_priors']!=list(PRIORS) or cfg['policies']!=list(POLICIES) or cfg['floor_multipliers']!=[list(x) for x in FLOORS]:
        raise ValueError('design')
    checks=controls();write(root/'CONTROLS.json',checks)
    if not all(checks.values()):raise ValueError('controls')
    for name,h in cfg['input_files'].items():
        if file_digest(root/'inputs'/name)!=h:raise ValueError('input binding')
    population=read(root/'inputs/POPULATION.json');structures=read(root/'inputs/STRUCTURES.json')
    parent=read(root/'inputs/PARENT_SELECTIONS.json');groups={}
    for s in parent:groups.setdefault((tuple(s['times']),tuple(s['costs']),s['capacity_bytes']),[]).append(s)
    selections=[];lookup={}
    for key,records in groups.items():
        s=from_parent(records);s['risk_levels']=[constrain(s,F(*x)) for x in FLOORS];s['id']=len(selections)
        selections.append(s);lookup[key]=s
    rows=[];paired={}
    for i,r in enumerate(population):
        if i%256==0:pulse(phase='realized-risk-storage',row=i)
        st=structures[r['structure']];times=tuple(r['times']);costs=tuple(st['source_costs'][t-1] for t in times)
        if len(times)!=r['report_sources'] or any(t>st['checkpoint'] for t in times):raise ValueError('roster')
        pair=tuple(r[k] for k in ('lineage','structure','draw','initial_maker','kind','switched','duplicates'))
        if r['evidence']=='aware':paired[pair]=times
        elif paired[pair]!=times:raise ValueError('source pairing')
        for budget,threshold in [('half',st['checkpoint']//2),('quarter',3*st['checkpoint']//4)]:
            recent=[j for j,t in enumerate(times) if t>threshold];count=len(recent);order=sorted(range(len(times)),key=times.__getitem__)
            spaced=[order[(2*j+1)*len(times)//(2*count)] for j in range(count)]
            capacity=min(sum(costs[j] for j in recent),sum(costs[j] for j in spaced));s=lookup[(times,costs,capacity)]
            for level in s['risk_levels']:
                a=level['lottery']
                row=dict(**{k:v for k,v in r.items() if k!='times'},budget=budget,selection_id=s['id'],
                    floor_multiplier=level['floor_multiplier'],surviving_masks=level['surviving_masks'],
                    surviving_count=level['surviving_count'],excluded_masks=level['excluded_masks'],
                    fixed_bytes=st['weighted_overhead'],total_budget_bytes=st['weighted_overhead']+capacity,
                    expected_total_used_bytes=st['weighted_overhead']+float(F(*a['expected_used_bytes'])),
                    maximum_total_realized_bytes=st['weighted_overhead']+a['maximum_realized_bytes'],
                    expected_masses=[float(F(*x)) for x in a['expected_masses']],selected_lottery=a['selected_lottery'],
                    optimal_basic_count=a['optimal_basic_count'],feasible_basic_count=a['feasible_basic_count'],unique_candidates=len(s['candidates']))
                for key in ('expected_retained_sources','minimum_expected_mass','deterministic_minimum_mass','expected_gain','worst_realized_mass'):
                    row[key]=float(F(*a[key]))
                for key in ('realized_mass_floor','unrestricted_minimum_expected_mass','expected_mass_cost'):
                    row[key]=float(F(*level[key]))
                rows.append(row)
    (root/'raw').mkdir(exist_ok=True)
    (root/'raw/realized_risk_points.json.gz').write_bytes(gzip.compress(canonical(rows),mtime=0))
    write(root/'SELECTIONS.json',selections)
    write(root/'EVIDENCE_ROLES.json',dict(reader='no new reader inputs',evaluator='source rosters,structural costs,three supplied priors and verified finite mask library',
        scope='finite-library expected coverage subject to realized mass floors;prior fixed before draw;each realized mask separately byte feasible'))
    return dict(controls=checks,population_rows=len(population),rows=len(rows),unique_selections=len(selections),
        risk_problems=sum(len(s['risk_levels']) for s in selections),candidate_count=sum(len(s['candidates']) for s in selections),numerical_acceptance=False,
        scope='exact finite-library risk constrained lotteries;no sampling,forecast accuracy,learned provenance or historical correspondence')

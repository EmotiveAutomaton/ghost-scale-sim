"""Exact basic lotteries minimizing maximum expected source-prior regret."""
import gzip
from fractions import Fraction as F
from itertools import combinations
from .randomized_source_storage import solve, randomize, fixture
from .robust_source_mass import from_parent
from .source_mass_frontier import PRIORS, POLICIES
from ..v18_3.io import read, write, canonical, file_digest


def minimize(selection):
    coverage = randomize(selection)
    rows = sorted(selection['candidates'],key=lambda r:r['mask'])
    masses = [[F(*x) for x in row['masses']] for row in rows]
    optima = [F(*x) for x in selection['best_prior_mass']]
    n,p = len(rows),len(optima)
    if p != len(masses[0]) or any(max(row[j] for row in masses)!=optima[j] for j in range(p)):
        raise ValueError('parent optimum')
    regrets = [[optima[j]-row[j] for j in range(p)] for row in masses]
    for r,rr in zip(rows,regrets):
        if rr != [F(*x) for x in r['prior_regrets']] or max(rr)!=F(*r['maximum_regret']):
            raise ValueError('parent regret')
    vertices = {}
    for k in range(1,min(n,p)+1):
        for support in combinations(range(n),k):
            for active in combinations(range(p),k):
                matrix = [[1]*k+[0]]+[[masses[i][j] for i in support]+[-1] for j in active]
                solution = solve(matrix,[1]+[optima[j] for j in active])
                if solution is None: continue
                weights,z = solution[:-1],solution[-1]
                if any(x<0 for x in weights):continue
                expected = [sum(weights[h]*masses[i][j] for h,i in enumerate(support)) for j in range(p)]
                if any(expected[j]-optima[j]<z for j in range(p)):continue
                full = tuple(next((weights[h] for h,i in enumerate(support) if i==a),F()) for a in range(n))
                assert sum(full)==1 and min(expected[j]-optima[j] for j in range(p))==z
                vertices[full]=(z,expected)
    if not vertices:raise ValueError('no feasible basic solution')
    best=max(a[0] for a in vertices.values());ties=sorted(w for w,(z,_) in vertices.items() if z==best)
    weights=ties[-1];expected=vertices[weights][1];expected_regrets=[a-b for a,b in zip(optima,expected)]
    rational=lambda x:[x.numerator,x.denominator]
    encode=lambda w:[dict(mask=rows[i]['mask'],weight=rational(x)) for i,x in enumerate(w) if x]
    coverage_regret=max(optima[j]-F(*coverage['expected_masses'][j]) for j in range(p))
    deterministic=min(map(max,regrets))
    return dict(optimal_basic_lotteries=[encode(w) for w in ties],selected_lottery=encode(weights),
        optimal_basic_count=len(ties),feasible_basic_count=len(vertices),expected_masses=list(map(rational,expected)),
        expected_prior_regrets=list(map(rational,expected_regrets)),maximum_expected_regret=rational(-best),
        coverage_lottery_regret=rational(coverage_regret),regret_improvement=rational(coverage_regret+best),
        deterministic_minimax_regret=rational(deterministic),gain_over_deterministic=rational(deterministic+best),
        minimum_expected_mass=rational(min(expected)),minimum_mass_cost=rational(F(*coverage['minimum_expected_mass'])-min(expected)),
        worst_realized_regret=rational(max(max(regrets[i]) for i,w in enumerate(weights) if w)),
        expected_used_bytes=rational(sum(w*rows[i]['used_bytes'] for i,w in enumerate(weights))),
        maximum_realized_bytes=max(rows[i]['used_bytes'] for i,w in enumerate(weights) if w),
        expected_retained_sources=rational(sum(w*len(rows[i]['selected_times']) for i,w in enumerate(weights))),
        coverage_lottery=coverage['selected_lottery'])


def controls():
    priors=[[F(2,5),F(3,5),F(0)],[F(2,5),F(3,10),F(3,10)]]
    a=minimize(fixture([1,2,3],[1,1,1],1,{'a':[1],'b':[2],'c':[3]},priors))
    empty=minimize(fixture([1,2],[1,1],0,{'empty':[]},[[F(1,2)]*2]))
    full=minimize(fixture([1,2],[1,1],2,{'all':[1,2]},[[F(1,2)]*2]))
    return {'live:strict_lottery_regret_gain':a['maximum_expected_regret']==[1,15] and a['gain_over_deterministic']==[1,30],
        'positive:coverage_disagreement':a['minimum_mass_cost']==[1,15] and a['regret_improvement']==[2,15],
        'placebo:empty_no_opportunity':empty['maximum_expected_regret']==[0,1],
        'positive:full_zero_regret':full['maximum_expected_regret']==[0,1]}


def run(root,plan,pulse):
    cfg=plan['design']
    if cfg['source_priors']!=list(PRIORS) or cfg['policies']!=list(POLICIES):raise ValueError('design')
    checks=controls();write(root/'CONTROLS.json',checks)
    if not all(checks.values()):raise ValueError('controls')
    for name,h in cfg['input_files'].items():
        if file_digest(root/'inputs'/name)!=h:raise ValueError('input binding')
    population=read(root/'inputs/POPULATION.json');structures=read(root/'inputs/STRUCTURES.json')
    parent=read(root/'inputs/PARENT_SELECTIONS.json');groups={}
    for s in parent:groups.setdefault((tuple(s['times']),tuple(s['costs']),s['capacity_bytes']),[]).append(s)
    selections=[];lookup={}
    for key,records in groups.items():
        s=from_parent(records);s['lottery']=minimize(s);s['id']=len(selections)
        selections.append(s);lookup[key]=s
    rows=[];paired={}
    for i,r in enumerate(population):
        if i%256==0:pulse(phase='randomized-source-regret',row=i)
        st=structures[r['structure']];times=tuple(r['times']);costs=tuple(st['source_costs'][t-1] for t in times)
        if len(times)!=r['report_sources'] or any(t>st['checkpoint'] for t in times):raise ValueError('roster')
        pair=tuple(r[k] for k in ('lineage','structure','draw','initial_maker','kind','switched','duplicates'))
        if r['evidence']=='aware':paired[pair]=times
        elif paired[pair]!=times:raise ValueError('source pairing')
        for budget,threshold in [('half',st['checkpoint']//2),('quarter',3*st['checkpoint']//4)]:
            recent=[j for j,t in enumerate(times) if t>threshold];count=len(recent);order=sorted(range(len(times)),key=times.__getitem__)
            spaced=[order[(2*j+1)*len(times)//(2*count)] for j in range(count)]
            capacity=min(sum(costs[j] for j in recent),sum(costs[j] for j in spaced));s=lookup[(times,costs,capacity)];a=s['lottery']
            row=dict(**{k:v for k,v in r.items() if k!='times'},budget=budget,selection_id=s['id'],
                fixed_bytes=st['weighted_overhead'],total_budget_bytes=st['weighted_overhead']+capacity,
                expected_total_used_bytes=st['weighted_overhead']+float(F(*a['expected_used_bytes'])),
                maximum_total_realized_bytes=st['weighted_overhead']+a['maximum_realized_bytes'],
                selected_lottery=a['selected_lottery'],coverage_lottery=a['coverage_lottery'],
                optimal_basic_count=a['optimal_basic_count'],feasible_basic_count=a['feasible_basic_count'],unique_candidates=len(s['candidates']))
            for key in ('expected_retained_sources','maximum_expected_regret','coverage_lottery_regret','regret_improvement',
                'deterministic_minimax_regret','gain_over_deterministic','minimum_expected_mass','minimum_mass_cost','worst_realized_regret'):
                row[key]=float(F(*a[key]))
            for key in ('expected_masses','expected_prior_regrets'):row[key]=[float(F(*x)) for x in a[key]]
            rows.append(row)
    (root/'raw').mkdir(exist_ok=True)
    (root/'raw/randomized_regret_points.json.gz').write_bytes(gzip.compress(canonical(rows),mtime=0))
    write(root/'SELECTIONS.json',selections)
    write(root/'EVIDENCE_ROLES.json',dict(reader='no new reader inputs',evaluator='source rosters,structural costs,three supplied source priors and verified finite mask library',
        scope='finite-library minimum maximum expected opportunity loss;prior fixed before draw;realized regret separately retained'))
    return dict(controls=checks,population_rows=len(population),rows=len(rows),unique_selections=len(selections),
        candidate_count=sum(len(s['candidates']) for s in selections),numerical_acceptance=False,
        scope='exact lotteries with prior fixed before draw;no sampling,forecast accuracy,learned provenance or historical correspondence')

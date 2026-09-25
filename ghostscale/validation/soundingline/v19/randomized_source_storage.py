"""Exact basic lotteries over individually byte-feasible source masks."""
import gzip
from fractions import Fraction
from itertools import combinations
from .robust_source_mass import from_parent, choose
from .source_mass_frontier import PRIORS, POLICIES
from ..v18_3.io import read, write, canonical, file_digest


def solve(matrix, rhs):
    """Gauss-Jordan over fractions; singular active sets are not basic vertices."""
    n = len(rhs)
    a = [[Fraction(x) for x in row]+[Fraction(y)] for row,y in zip(matrix,rhs)]
    if len(a)!=n or any(len(row)!=n+1 for row in a): raise ValueError('matrix')
    for col in range(n):
        pivot = next((i for i in range(col,n) if a[i][col]),None)
        if pivot is None: return None
        a[col],a[pivot] = a[pivot],a[col]
        divisor = a[col][col]; a[col] = [x/divisor for x in a[col]]
        for i in range(n):
            if i == col: continue
            multiplier = a[i][col]
            a[i] = [x-multiplier*y for x,y in zip(a[i],a[col])]
    return [row[-1] for row in a]


def randomize(selection):
    rows = sorted(selection['candidates'],key=lambda r:r['mask'])
    if not rows or len({r['mask'] for r in rows})!=len(rows): raise ValueError('library')
    times,costs,capacity = (selection[k] for k in ('times','costs','capacity_bytes'))
    if len(times)!=len(costs) or len(set(times))!=len(times): raise ValueError('items')
    if type(capacity) is not int or capacity<0 or any(type(x) is not int or x<0 for x in costs): raise ValueError('costs')
    charges=dict(zip(times,costs)); ranks=dict(zip(sorted(times),range(len(times))))
    for row in rows:
        chosen=row['selected_times']
        if len(set(chosen))!=len(chosen) or not set(chosen)<=set(times): raise ValueError('source identities')
        if sum(2**ranks[t] for t in chosen)!=row['mask']: raise ValueError('mask')
        used=sum(charges[t] for t in chosen)
        if used!=row['used_bytes'] or used>capacity: raise ValueError('realized byte feasibility')
    masses=[[Fraction(*x) for x in row['masses']] for row in rows]
    p=len(masses[0]); n=len(rows)
    if not 1<=p<=3 or any(len(row)!=p or any(not 0<=x<=1 for x in row) for row in masses): raise ValueError('prior masses')
    vertices={}
    for k in range(1,min(n,p)+1):
        for support in combinations(range(n),k):
            for active in combinations(range(p),k):
                matrix=[[1]*k+[0]]+[[masses[i][j] for i in support]+[-1] for j in active]
                solution=solve(matrix,[1]+[0]*k)
                if solution is None: continue
                w,z=solution[:-1],solution[-1]
                if any(x<0 for x in w): continue
                expected=[sum((w[h]*masses[i][j] for h,i in enumerate(support)),Fraction()) for j in range(p)]
                if any(x<z for x in expected): continue
                assert sum(w)==1 and min(expected)==z
                full=tuple(next((w[h] for h,i in enumerate(support) if i==a),Fraction()) for a in range(n))
                vertices[full]=(z,expected)
    if not vertices: raise ValueError('no feasible basic solution')
    best=max(x[0] for x in vertices.values()); optimal=sorted(w for w,(z,_) in vertices.items() if z==best)
    selected=optimal[-1]; expected=vertices[selected][1]
    rational=lambda x:[x.numerator,x.denominator]
    encode=lambda w:[dict(mask=rows[i]['mask'],weight=rational(v)) for i,v in enumerate(w) if v]
    deterministic=max(map(min,masses))
    if Fraction(*selection['minimum_mass'])!=deterministic: raise ValueError('deterministic baseline')
    return dict(optimal_basic_lotteries=[encode(w) for w in optimal],selected_lottery=encode(selected),
        optimal_basic_count=len(optimal),feasible_basic_count=len(vertices),
        expected_masses=list(map(rational,expected)),minimum_expected_mass=rational(best),
        deterministic_minimum_mass=rational(deterministic),expected_gain=rational(best-deterministic),
        worst_realized_mass=rational(min(min(masses[i]) for i,w in enumerate(selected) if w)),
        expected_used_bytes=rational(sum((w*rows[i]['used_bytes'] for i,w in enumerate(selected)),Fraction())),
        maximum_realized_bytes=max(rows[i]['used_bytes'] for i,w in enumerate(selected) if w),
        expected_retained_sources=rational(sum((w*len(rows[i]['selected_times']) for i,w in enumerate(selected)),Fraction())))


def fixture(times,costs,capacity,candidates,priors):
    return dict(times=times,costs=costs,capacity_bytes=capacity,**choose(times,costs,capacity,candidates,priors))


def controls():
    a=randomize(fixture([1,2],[1,1],1,{'one':[1],'two':[2]},[[Fraction(1),Fraction(0)],[Fraction(0),Fraction(1)]]))
    empty=randomize(fixture([1,2],[1,1],0,{'none':[]},[[Fraction(1,2)]*2]))
    full=randomize(fixture([1,2],[1,1],2,{'all':[1,2]},[[Fraction(1,2)]*2]))
    return {'live:diversification_gain':a['minimum_expected_mass']==[1,2] and a['expected_gain']==[1,2],
        'placebo:no_realized_guarantee':a['worst_realized_mass']==[0,1],
        'positive:empty_zero':empty['minimum_expected_mass']==[0,1],
        'positive:full_one':full['minimum_expected_mass']==[1,1]}


def run(root,plan,pulse):
    cfg=plan['design']
    if cfg['source_priors']!=list(PRIORS) or cfg['policies']!=list(POLICIES): raise ValueError('design')
    checks=controls();write(root/'CONTROLS.json',checks)
    if not all(checks.values()): raise ValueError('controls')
    for name,h in cfg['input_files'].items():
        if file_digest(root/'inputs'/name)!=h: raise ValueError('input binding')
    population=read(root/'inputs/POPULATION.json'); structures=read(root/'inputs/STRUCTURES.json')
    parent=read(root/'inputs/PARENT_SELECTIONS.json'); groups={}
    for s in parent: groups.setdefault((tuple(s['times']),tuple(s['costs']),s['capacity_bytes']),[]).append(s)
    selections=[];lookup={}
    for key,records in groups.items():
        s=from_parent(records);s['lottery']=randomize(s);s['id']=len(selections)
        selections.append(s);lookup[key]=s
    rows=[];paired={}
    for i,r in enumerate(population):
        if i%256==0:pulse(phase='randomized-source-storage',row=i)
        st=structures[r['structure']];times=tuple(r['times']);costs=tuple(st['source_costs'][t-1] for t in times)
        if len(times)!=r['report_sources'] or any(t>st['checkpoint'] for t in times):raise ValueError('roster')
        pair=(r['lineage'],r['structure'],r['draw'],r['initial_maker'],r['kind'],r['switched'],r['duplicates'])
        if r['evidence']=='aware':paired[pair]=times
        elif paired[pair]!=times:raise ValueError('source pairing')
        for budget,threshold in [('half',st['checkpoint']//2),('quarter',3*st['checkpoint']//4)]:
            recent=[j for j,t in enumerate(times) if t>threshold];count=len(recent);order=sorted(range(len(times)),key=times.__getitem__)
            spaced=[order[(2*j+1)*len(times)//(2*count)] for j in range(count)]
            capacity=min(sum(costs[j] for j in recent),sum(costs[j] for j in spaced));s=lookup[(times,costs,capacity)];a=s['lottery']
            rows.append(dict(**{k:v for k,v in r.items() if k!='times'},budget=budget,selection_id=s['id'],
                fixed_bytes=st['weighted_overhead'],total_budget_bytes=st['weighted_overhead']+capacity,
                expected_total_used_bytes=st['weighted_overhead']+float(Fraction(*a['expected_used_bytes'])),
                maximum_total_realized_bytes=st['weighted_overhead']+a['maximum_realized_bytes'],
                expected_retained_sources=float(Fraction(*a['expected_retained_sources'])),
                minimum_expected_mass=float(Fraction(*a['minimum_expected_mass'])),
                expected_masses=[float(Fraction(*x)) for x in a['expected_masses']],
                deterministic_minimum_mass=float(Fraction(*a['deterministic_minimum_mass'])),
                expected_gain=float(Fraction(*a['expected_gain'])),worst_realized_mass=float(Fraction(*a['worst_realized_mass'])),
                selected_lottery=a['selected_lottery'],optimal_basic_count=a['optimal_basic_count'],
                feasible_basic_count=a['feasible_basic_count'],unique_candidates=len(s['candidates'])))
    (root/'raw').mkdir(exist_ok=True)
    (root/'raw/randomized_storage_points.json.gz').write_bytes(gzip.compress(canonical(rows),mtime=0))
    write(root/'SELECTIONS.json',selections)
    write(root/'EVIDENCE_ROLES.json',dict(reader='no new reader inputs',evaluator='source rosters,structural costs,three supplied source priors and verified finite mask library',
        scope='finite-library maximum minimum expected mass over individually feasible masks; no pointwise improvement guarantee or adaptive-prior claim'))
    return dict(controls=checks,population_rows=len(population),rows=len(rows),unique_selections=len(selections),
        candidate_count=sum(len(s['candidates']) for s in selections),numerical_acceptance=False,
        scope='exact lotteries with prior fixed before the draw; no sampling,forecast accuracy,learned provenance or historical correspondence')

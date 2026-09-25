from itertools import product
from fractions import Fraction
import pytest
from ghostscale.validation.soundingline.v19 import source_mass_frontier as M


def exhaustive(times,costs,capacity,prior):
    weights=[Fraction(1) if prior=='uniform' else Fraction(t) if prior=='time-proportional' else Fraction(1,t) for t in times]
    ranks={t:i for i,t in enumerate(sorted(times))}
    choices=[]
    for selected in product((False,True),repeat=len(times)):
        if sum(k*s for k,s in zip(costs,selected))<=capacity:
            choices.append((sum(w*s for w,s in zip(weights,selected)),sum((1<<ranks[t])*s for t,s in zip(times,selected)),selected))
    _,_,chosen=max(choices)
    return sorted((t for t,s in zip(times,chosen) if s),reverse=True)


@pytest.mark.parametrize('prior',M.PRIORS)
@pytest.mark.parametrize('costs',[(0,2,3,5),(1,1,1,1),(7,2,1,4),(8,3,5,2)])
def test_full_small_frontier_against_exhaustive(prior,costs):
    for capacity in range(sum(costs)+2):
        got=M.allocate((1,3,4,7),costs,capacity,prior)
        assert got['optimal']['selected_times']==exhaustive((1,3,4,7),costs,capacity,prior)
        for row in got.values():assert row['used_bytes']<=capacity and row['gap_to_optimum']>=0


@pytest.mark.parametrize('prior',M.PRIORS)
def test_permutation_and_free_items(prior):
    a=M.allocate((1,3,7),(3,0,4),4,prior);b=M.allocate((7,1,3),(4,3,0),4,prior)
    assert a==b
    assert all(3 in r['selected_times'] for r in a.values())


def test_greedy_skips_and_is_not_optimal():
    r=M.allocate((1,2,3),(2,2,3),4,'uniform')
    assert r['recent']['selected_times']==[3] and r['optimal']['selected_times']==[2,1]
    r=M.allocate((1,2,3),(1,2,20),3,'time-proportional')
    assert r['probability']['selected_times']==[2,1]


def test_empty_full_and_ties():
    assert M.allocate((),(),0,'uniform')['optimal']['retained_mass']==0
    assert all(M.controls().values())
    assert M.allocate((1,2,3),(1,1,1),2,'uniform')['optimal']['selected_times']==[3,2]


@pytest.mark.parametrize('costs,cap',[((-1,2),2),((1.,2),2),((1,2),-1),((1,2),2.)])
def test_corrupted_cost_fails(costs,cap):
    with pytest.raises(ValueError):M.allocate((1,2),costs,cap,'uniform')


def test_known_structural_storage():
    spec=dict(hypotheses=[['none',0,s] for s in range(16)],membership=list(range(16)),signatures=[[s] for s in range(16)],checkpoint=3)
    costs,overhead=M.structural_costs(spec)
    assert costs==[140]*3 and overhead==8*(16+13+32)


def test_handler_full_pair_and_export_roles(tmp_path):
    from ghostscale.validation.soundingline.v18_3.io import write,file_digest,read
    (tmp_path/'inputs').mkdir()
    base=dict(lineage=190000,structure='4-4',draw=1,initial_maker=0,kind='purpose',switched=False,duplicates=False,report_sources=4,times=[1,2,3,4])
    write(tmp_path/'inputs/POPULATION.json',[dict(base,evidence=e) for e in ('aware','omitted')])
    write(tmp_path/'inputs/STRUCTURES.json',{'4-4':dict(checkpoint=4,source_costs=[2,3,4,5],weighted_overhead=400)})
    cfg=dict(source_priors=list(M.PRIORS),policies=list(M.POLICIES),input_files={p.name:file_digest(p) for p in (tmp_path/'inputs').iterdir()})
    got=M.run(tmp_path,dict(design=cfg),lambda **kw:None)
    assert got['rows']==48 and not got['numerical_acceptance']
    assert read(tmp_path/'EVIDENCE_ROLES.json')['reader']=='no new reader inputs'
    population=read(tmp_path/'inputs/POPULATION.json');population[1]['times']=[1,2,3,5]
    write(tmp_path/'inputs/POPULATION.json',population,immutable=False)
    with pytest.raises(ValueError,match='input binding'):M.run(tmp_path,dict(design=cfg),lambda **kw:None)

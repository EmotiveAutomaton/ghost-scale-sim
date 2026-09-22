"""Known-answer, exhaustive pair coverage and corruption controls."""
from itertools import combinations
import copy
import gzip
import json
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import marginal_transition as M
from ghostscale.validation.soundingline.v18_3.io import write, file_digest


def test_controls_and_binary64_alias_distinction():
    assert all(M.controls().values())
    law=np.full((16,4,8),1/8)
    law[1,0,0]+=1e-14;law[1,0,1]-=1e-14
    row=next(r for r in M.distances(law) if (r['left'],r['right'],r['context'])==(0,1,0))
    assert not row['identical_binary64'] and not row['above_1e_12'] and row['total_variation']>0
    law[0,0,0]=-1
    with pytest.raises(ValueError,match='endpoint law'):M.distances(law)


def test_all_pairs_match_direct_enumeration_and_corruption_fails():
    hs=[('none',0,0),('purpose',8,0),('purpose',9,0),('skill',8,0),('purpose',8,8),('none',0,8)]
    for checkpoint in (8,9,10):
        for future in range(checkpoint,12):
            r=M.partitions(hs,checkpoint,future);M.check_partition(hs,r)
            actual={tuple(sorted((i,j))) for b in r['blocks'] for i in r['groups'][b['left']]['members'] for j in r['groups'][b['right']]['members']}
            expected={(i,j) for i,j in combinations(range(len(hs)),2) if M.direct_state(hs[i],checkpoint)==M.direct_state(hs[j],checkpoint) and M.direct_state(hs[i],future)!=M.direct_state(hs[j],future)}
            assert actual==expected and len(actual)==r['pairs']
    r=M.partitions(hs,8,10);bad=copy.deepcopy(r);bad['groups'][0]['members'].pop()
    with pytest.raises(ValueError,match='membership'):M.check_partition(hs,bad)
    bad=copy.deepcopy(r);bad['blocks'].pop()
    with pytest.raises(ValueError,match='coverage'):M.check_partition(hs,bad)
    with pytest.raises(ValueError):M.partitions(hs,10,9)


def test_known_forecast_witness_and_same_time_identity():
    law=np.zeros((16,4,8));law[:,:,0]=1;law[8,:,0]=0;law[8,:,1]=1
    r=next(r for r in M.distances(law) if (r['left'],r['right'],r['context'])==(0,8,0))
    assert r['total_variation']==r['max_absolute_difference']==1 and r['above_1e_12']
    hs=[('none',0,0),('purpose',8,0)]
    assert M.partitions(hs,8,8)['pairs']==0 and M.partitions(hs,8,9)['pairs']==1


def test_complete_handler_and_input_binding(tmp_path):
    law=np.full((16,4,8),1/8);p=tmp_path/'inputs/190969-law.json';write(p,law.tolist())
    plan={'design':dict(lineages=[190969],lengths=[32,128],checkpoints=[32],input_files={p.name:file_digest(p)})}
    summary=M.run(tmp_path,plan,lambda **kw:None)
    assert all(summary['controls'].values()) and summary['forecast_cells']==392
    assert [r['hypotheses'] for r in summary['counts']]==[560,3632]
    cells=json.loads(gzip.decompress((tmp_path/'raw/forecast_summary_points.json.gz').read_bytes()))
    assert all(r['pairs']==r['exact_binary64_forecast_aliases'] and r['max_total_variation']==0 for r in cells)
    p.write_text('[]')
    with pytest.raises(ValueError,match='input differs'):M.run(tmp_path,plan,lambda **kw:None)

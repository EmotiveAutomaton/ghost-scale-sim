from itertools import product
import pytest
from ghostscale.validation.soundingline.v18_2 import direct_control as d,model as m,assembly_maker as a
from ghostscale.validation.soundingline.v18_2.verify import replay


def test_direct_public_law_rival_executes_all_declared_goals():
    for target in range(16):
        if target.bit_count()<=3:assert replay(d.board(m.world('test',0),target))==target
    with pytest.raises(ValueError):d.board(m.world('test',0),15)
    for split in ('train','dev','test'):
        for index in range(8):
            w=a.world(split,index)
            for target in product((0,1),repeat=3):
                program=d.assembly(w,target)
                assert a.replay(w,program)==sum(value<<i for i,value in enumerate(target))
                assert len(program)<=7

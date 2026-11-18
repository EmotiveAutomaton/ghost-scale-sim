"""The recipe trigger applies to the declared heavy-floor diary branch too."""
import pytest
from ghostscale.validation.soundingline.v31.common import read,write,digest
from runners.prepare_v31_followons import prepare


@pytest.mark.parametrize('family',['D_CORE','D_HEAVY'])
def test_recipe_trigger_covers_each_diary_family(tmp_path,family):
    c=tmp_path/'campaign';job=c/'jobs/test'
    write(job/'COMPLETE.json',dict(state='VERIFIED'))
    setting=dict(people=2272,days=10,prompts=6)
    write(job/'PLAN.json',dict(family=family,setting=setting))
    summary=job/'output/study-original/SUMMARY.json'
    write(summary,dict(rows=[dict(measure='random_slope',spread=.5),dict(measure='split',spread=1.)]))
    write(job/'WORKER_COMPLETE.json',dict(summary_sha256=digest(summary)))
    prepare(tmp_path,c)
    rows=read(c/'TRIGGER_STATUS.json')['rows']
    assert len(rows)==1 and rows[0]['eligible'] is True
    assert rows[0]['spread_ratio']==.5 and rows[0]['kind']=='D_optional_recipe'

import pytest
from runners.run_v31 import resource_contract
from ghostscale.validation.soundingline.v31.common import write,digest


def contract(tmp_path):
    original=dict(cpu_limits=dict(D=180000,reserve=72000,M=126000),total_cpu_seconds=576000,drain='unchanged')
    write(tmp_path/'ACCEPTANCE.json',original)
    first=dict(approved=True,owner_authorization='original transfer',acceptance_sha256=digest(tmp_path/'ACCEPTANCE.json'),
               cpu_limits=original['cpu_limits'],total_cpu_seconds=576000)
    write(tmp_path/'RESOURCE_REALLOCATION.json',first)
    later=dict(approved=True,owner_authorization='two-hour transfer',acceptance_sha256=digest(tmp_path/'ACCEPTANCE.json'),
               parent_supplement_sha256=digest(tmp_path/'RESOURCE_REALLOCATION.json'),
               cpu_limits=dict(D=172800,reserve=79200,M=126000),total_cpu_seconds=576000)
    return later


def test_exact_conserved_transfer_preserves_clock(tmp_path):
    later=contract(tmp_path);write(tmp_path/'RESOURCE_REALLOCATION_20261118.json',later)
    result=resource_contract(tmp_path)
    assert result['cpu_limits']['reserve']==79200 and result['drain']=='unchanged'


@pytest.mark.parametrize('change',[dict(parent_supplement_sha256='wrong'),dict(approved=False),dict(cpu_limits=dict(D=172800,reserve=80000,M=126000))])
def test_unbound_or_larger_transfer_refused(tmp_path,change):
    later=contract(tmp_path);write(tmp_path/'RESOURCE_REALLOCATION_20261118.json',dict(later,**change))
    with pytest.raises(ValueError):resource_contract(tmp_path)

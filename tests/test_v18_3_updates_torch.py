import pytest
torch=pytest.importorskip('torch',reason='optional shared CPU PyTorch runtime required')
from ghostscale.validation.soundingline.v18_3 import torch_worker as T
from runners.benchmark_v18_3_updates import cached_update


def test_incremental_update_matches_full_recurrent_state():
    torch.manual_seed(9183)
    for kind in ('flat','split'):
        model=T.Reader(kind,20,4,24);h=torch.randn(8,32,20)
        with torch.no_grad():
            prior=model.encode(h,torch.full((8,),6))
            actual=cached_update(model,prior,h[:,6:7])
            expected=model.encode(h,torch.full((8,),7))
        assert torch.allclose(actual,expected,atol=1e-6,rtol=0)

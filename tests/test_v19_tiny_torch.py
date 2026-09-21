import io
import numpy as np
import pytest
torch=pytest.importorskip('torch')
from ghostscale.validation.soundingline.v19.tiny_worker import Reader,encode,objective,structural_controls,learning_control

@pytest.mark.parametrize('kind',['transformer','recurrent'])
def test_sequence_known_answer_placebo_and_causal_fixtures(kind):
    c=structural_controls(kind);assert all(c[k] for k in ('causal_prefix','no_op_decode','duplicate_passthrough','normalized'))
    l=learning_control(kind);assert l['learning_passed'] and l['uniform_null']

def test_frozen_parameter_counts_are_close():
    n=[sum(p.numel() for p in Reader(k).parameters()) for k in ('transformer','recurrent')]
    assert abs(n[0]-n[1])/max(n)<.05

@pytest.mark.parametrize('kind',['transformer','recurrent'])
def test_checkpoint_optimizer_rng_and_next_update_exact(kind):
    torch.manual_seed(196011);model=Reader(kind);opt=torch.optim.AdamW(model.parameters(),lr=.001)
    tokens=torch.tensor([[32,0,1],[32,2,3]]);pos=torch.tensor([[1,2],[1,2]])
    target=torch.ones(2,2,4,8)/8;mask=torch.ones((2,2),dtype=torch.bool)
    def step(m,o):
        order=torch.randperm(2);o.zero_grad();loss=objective(m(tokens[order],pos[order])[0],target[order],mask[order]);loss.backward();o.step()
    step(model,opt);buffer=io.BytesIO();torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),rng=torch.get_rng_state()),buffer)
    step(model,opt);expected={k:v.clone() for k,v in model.state_dict().items()}
    buffer.seek(0);ck=torch.load(buffer,weights_only=True);restored=Reader(kind);o=torch.optim.AdamW(restored.parameters(),lr=.001)
    restored.load_state_dict(ck['model']);o.load_state_dict(ck['optimizer']);torch.set_rng_state(ck['rng']);step(restored,o)
    assert all(torch.equal(v,restored.state_dict()[k]) for k,v in expected.items())

"""Reconstruct V20's separate observed-inspection retention witness."""
import numpy as np
from ghostscale.validation.soundingline.v18_3.io import read
from ghostscale.validation.soundingline.v20 import world as W

def verify(root):
    design=read(root/'PLAN.json')['design'];summary=read(root/'SUMMARY.json')
    saved={(r['lineage'],r['context'],r['route'],r['method']):r for r in summary['rows']}
    records=[]
    for lineage in design['lineages']:
        rows=W.enumerate_world(lineage);kernel=np.zeros((32,8,8))
        endpoint=(rows[:,11]*4+rows[:,12]*2+rows[:,13]).astype(int)
        np.add.at(kernel,(rows[:,0].astype(int),rows[:,1].astype(int),endpoint),rows[:,-1])
        kernel/=kernel.sum(2,keepdims=True)
        for context in range(8):
            for route in range(W.CONTEXTS[context][2]+1):
                mask=(rows[:,1]==context)&(rows[:,2]==route)&(rows[:,4]==1)&(rows[:,6]==0)
                states=[];banks=[]
                for inspection in (0,1):
                    subset=rows[mask&(rows[:,5]==inspection)]
                    posterior=np.bincount(subset[:,0].astype(int),weights=subset[:,-1],minlength=32)
                    posterior/=posterior.sum();states.append(posterior)
                    banks.append((posterior@kernel.reshape(32,64)).reshape(8,8))
                state_gap=float(abs(states[0]-states[1]).max());bank_gap=float(abs(banks[0]-banks[1]).max())
                indistinguishable=W.law(lineage)['inspect']==5
                if (state_gap<1e-13)!=indistinguishable or bank_gap>=1e-13:
                    raise ValueError('observed-inspection witness differs')
                for method in ('raw-history','maker-state','prediction-bank','state-plus-ledger'):
                    expected=np.log(2) if method=='prediction-bank' or (method=='maker-state' and indistinguishable) else 0
                    if saved[lineage,context,route,method]['recorded_inspection_log_loss']!=expected:
                        raise ValueError('inspection diagnostic score differs')
                records.append(dict(lineage=lineage,context=context,route=route,maker_posterior_max_gap=state_gap,future_bank_max_gap=bank_gap,maker_state_cannot_distinguish=indistinguishable))
    return dict(passed=True,records=records,maximum_future_bank_gap=max(r['future_bank_max_gap'] for r in records),indistinguishable_lineages=sorted({r['lineage'] for r in records if r['maker_state_cannot_distinguish']}),scope='Distinct observed-inspection witness; original order-witness byte costs do not measure inspection-specific storage. Hidden inspection remains unknown.')

"""Independent row-weight reference, without aggregated channel tables."""
import numpy as np


def reference(rows,rate,false,mechanism,reader):
    out=np.zeros((8,3,128))
    artifact=rows[:,11:14].astype(int)@np.array([4,2,1]);inspected=rows[:,5].astype(int)
    for key in range(8):
        selected=rows[artifact==key];bits=inspected[artifact==key];prior=selected[:,-1]
        missing=np.full(len(selected),rate) if mechanism=='independent' else np.array([min(1,1.5*rate) if b else .5*rate for b in bits])
        for record in (-1,0,1):
            weight=prior.copy()
            if reader in ('naive-negative','full-record-oracle'):weight*=bits==max(0,record)
            elif reader!='endpoint':
                if record==-1:
                    if reader in ('observation-aware','matched-direct'):weight*=missing
                else:
                    weight*=np.array([1-false if b==record else false for b in bits])
                    if reader in ('observation-aware','matched-direct'):weight*=1-missing
            if not weight.sum():weight=prior.copy()
            out[key,record+1]=np.bincount(selected[:,10].astype(int),weights=weight,minlength=128)/weight.sum()
    return out

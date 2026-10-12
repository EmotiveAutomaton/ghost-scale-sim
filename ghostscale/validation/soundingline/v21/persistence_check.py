"""Separate probability-domain persistence reference and table reconstruction."""
import numpy as np
from . import world as W


def check_tables(rows,table):
    from .checker import key
    current={}
    for row in rows:
        k=np.asarray(key(row,W.CONTEXTS[int(row[1])],'context',False),dtype=np.int8).tobytes()
        if k not in current:current[k]=np.zeros((32,128))
        current[k][int(row[0]),int(row[10])]+=float(row[-1])
    for k,mass in current.items():assert np.allclose(table['joint'][table['current'][k]],mass,atol=1e-14,rtol=1e-12)
    for maker in range(32):
        selected=rows[rows[:,0]==maker];history=W.project(selected,'complete',True)
        for context in range(8):
            mask=selected[:,1]==context;total=selected[mask,-1].sum();counts={}
            for r,weight in zip(history[mask],selected[mask,-1]):counts[r.tobytes()]=counts.get(r.tobytes(),0)+float(weight)
            for key,mass in counts.items():assert abs(table['emission'][maker,table['history'][key]]-mass/total)<1e-12
    assert abs(table['joint'].sum()-1)<1e-12


def reference(table,x):
    rates=(1.,.9,.5,0.);by_rate={str(r):[] for r in rates};inferred=[];base=[]
    for row in x:
        current=table['joint'][table['current'][row[:13].tobytes()]];masses=[]
        for rate in rates:
            weight=np.ones(32)
            for episode in row[13:].reshape(-1,13):
                i=table['history'][episode.tobytes()]
                weight*=rate*table['emission'][:,i]+(1-rate)*table['background'][i]
            mass=weight@current;masses.append(mass)
            fallback=current.sum(0)
            by_rate[str(rate)].append(mass/mass.sum() if mass.sum()>0 else fallback/fallback.sum())
        mean=np.mean(masses,axis=0);inferred.append(mean/mean.sum());b=current.sum(0);base.append(b/b.sum())
    return dict(by_rate={k:np.array(v) for k,v in by_rate.items()},inferred=np.array(inferred),no_history=np.array(base))

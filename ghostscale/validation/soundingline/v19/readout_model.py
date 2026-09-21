"""Fixed architecture probability readouts; all model choices live in the plan."""
import io,zipfile
import numpy as np

WIDTH=512
HIDDEN=128
RIDGE=.01

def padded(x):
    x=np.asarray(x,float)
    if x.ndim!=2 or x.shape[1]>WIDTH:raise ValueError('undeclared feature width')
    return np.pad(x,((0,0),(0,WIDTH-x.shape[1])))

def basis(x,seed):
    x=padded(x);r=np.random.default_rng(seed)
    weights=r.normal(size=(WIDTH,HIDDEN))/np.sqrt(WIDTH)
    bias=r.normal(scale=.2,size=HIDDEN)
    return np.column_stack((np.ones(len(x)),np.tanh(x@weights+bias))),weights,bias

def fit(x,y):
    penalty=np.eye(x.shape[1])*RIDGE;penalty[0,0]=0.
    return np.linalg.solve(x.T@x+penalty,x.T@y)

def probabilities(raw,sizes):
    parts=[];invalid=[]
    for start,size in zip(np.cumsum([0]+sizes[:-1]),sizes):
        p=raw[:,start:start+size]
        invalid.append((p.min(1)<0)|(p.max(1)>1)|(abs(p.sum(1)-1)>1e-6))
        p=np.maximum(p,0)+1e-6;p/=p.sum(1,keepdims=True);parts.append(p)
    return np.concatenate(parts,axis=1),np.stack(invalid,axis=1)

def scores(truth,p,sizes):
    out=[];brier=[]
    for start,size in zip(np.cumsum([0]+sizes[:-1]),sizes):
        y=truth[:,start:start+size];q=p[:,start:start+size]
        out.append(-np.sum(y*np.log(np.maximum(q,1e-300)),axis=1))
        brier.append(np.sum((y-q)**2,axis=1))
    return np.stack(out,1),np.stack(brier,1)

def save_arrays(path,**arrays):
    path.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(path,'x',zipfile.ZIP_DEFLATED) as z:
        for name,array in sorted(arrays.items()):
            b=io.BytesIO();np.lib.format.write_array(b,np.asarray(array),allow_pickle=False)
            info=zipfile.ZipInfo(name+'.npy',date_time=(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(info,b.getvalue())

def controls():
    x=np.column_stack((np.ones(12),np.tile([0.,1.],6)))
    y=np.column_stack((1-x[:,1],x[:,1]))
    p,_=probabilities(x@fit(x,y),[2])
    null=np.tile([.5,.5],(12,1));q,_=probabilities(x@fit(x,null),[2])
    return dict(live_known_separation=bool(np.max(abs(y-p))<.01),
        placebo_constant_target=bool(np.max(abs(q-.5))<1e-10),
        positive_probability_repair=bool(np.allclose(probabilities(np.array([[-2.,3.]]),[2])[0].sum(1),1.)))

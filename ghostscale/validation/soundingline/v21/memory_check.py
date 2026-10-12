"""Independent byte parser and count-likelihood B05 reference."""
import hashlib,math
import numpy as np

def maker_reference(records,emission):
    counts=[sum(2*r[1]+r[2]==v for r in records) for v in range(4)]
    lp=[sum(counts[v]*math.log(float(emission[s,v])) for v in range(4)) for s in range(8)]
    p=[math.exp(v-max(lp)) for v in lp];total=math.fsum(p)
    return np.array([v/total for v in p])

def reference(blob,emission,queries,case_id):
    assert blob[:4]==b'GSM1' and blob[5]==0
    tag=blob[4];count=int.from_bytes(blob[6:8],'little');cursor=8;belief=None;prediction=None
    if tag in (1,4):belief=np.frombuffer(blob[cursor:cursor+64],dtype='<f8').tolist();cursor+=64
    if tag==2:prediction=np.frombuffer(blob[cursor:cursor+32],dtype='<f8').tolist();cursor+=32
    rows=[]
    for _ in range(count):
        link=None
        if tag==0:link=blob[cursor:cursor+16];cursor+=16
        index=int.from_bytes(blob[cursor:cursor+4],'little');x,y,order,inspection=blob[cursor+4:cursor+8];cursor+=8
        assert all(v in (0,1) for v in (x,y,order,inspection))
        if link is not None:assert link==hashlib.sha256((case_id+':'+str(index)).encode()).digest()[:16]
        rows.append((index,x,y,order,inspection))
    assert cursor==len(blob) and len({r[0] for r in rows})==len(rows)
    if prediction is None:
        if belief is None:
            belief=maker_reference(rows,emission)
        prediction=[math.fsum(belief[s]*float(emission[s,v]) for s in range(8)) for v in range(4)]
    historical=[]
    for query in queries:
        match=[r for r in rows if r[0]==query];p=[.25]*4
        if match:p=[float(label==2*match[0][3]+match[0][4]) for label in range(4)]
        historical.append(p)
    return np.array(prediction),np.array(historical),rows

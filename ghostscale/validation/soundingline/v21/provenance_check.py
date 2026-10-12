"""Independent source-first dynamic program over used-root sets."""
import itertools
import numpy as np


def reference(reports,ids,roots,mode,rate):
    totals=[0.,0.]
    for bits in itertools.product((0,1),repeat=roots):
        paths={0:1.}
        for report,label in zip(reports,ids):
            updated={}
            for mask,mass in paths.items():
                for root in range(roots):
                    if bits[root]!=report:continue
                    if mode=='hidden':weight=rate if label<0 else (1-rate)*int(label==root)
                    else:weight=(1-rate)*int(label==root)+rate/roots
                    new=mask|(1<<root);updated[new]=updated.get(new,0.)+mass*weight
            paths=updated
        likelihood=paths.get((1<<roots)-1,0.)
        for goal in (0,1):
            prior=1.
            for bit in bits:prior*=.8 if bit==goal else .2
            totals[goal]+=likelihood*prior
    assert sum(totals)>0
    return np.array(totals)/sum(totals)

"""Q conditional event-absence diagnostic, using L's statistic not its wrong law."""
import numpy as np
from .q_rank import generate,controls as base_controls
from .d_diary import summarize
from .common import write

def infer(raw,s,reference=False,pulse=lambda **kw:None):
    out={k:[] for k in ('own_exact_no_trigger','rank_exact_no_trigger','neighbour_drop_removed','baseline_show_without_drop')}
    for w,path in enumerate(raw['records']):
        show=path[:,1]>0;own=path[:,0]<0;rank=path[:,0]-s['weight']*path[:,2]<0;count=int(show.sum())
        if reference:
            own_no=sum(bool(y and not a) for y,a in zip(show,own));rank_no=sum(bool(y and not a) for y,a in zip(show,rank));removed=sum(bool(y and not a and b) for y,a,b in zip(show,own,rank))
        else:own_no=np.sum(show&~own);rank_no=np.sum(show&~rank);removed=np.sum(show&~own&rank)
        out['own_exact_no_trigger'].append(own_no/count if count else np.nan);out['rank_exact_no_trigger'].append(rank_no/count if count else np.nan);out['neighbour_drop_removed'].append(removed/count if count else np.nan)
        out['baseline_show_without_drop'].append(float(np.mean(show[~rank])) if np.any(~rank) else np.nan);pulse(world=w+1,phase='infer')
    return {k:np.array(v) for k,v in out.items()}
def pilot_exclusion(s,m):return None
def controls(dest,pulse=lambda **kw:None):
    checks=base_controls(dest/'Q')['checks'];path=np.zeros((4,11));path[:,1]=1;path[:,0]=[0,-1,1,0];path[:,2]=[1,0,0,0]
    out=infer(dict(records=path[None]),dict(weight=1.))
    checks['exact_event_count']=out['own_exact_no_trigger'][0]==.75 and out['rank_exact_no_trigger'][0]==.5 and out['neighbour_drop_removed'][0]==.25
    write(dest/'CONTROLS.json',checks);return dict(passed=all(checks.values()),checks=checks)

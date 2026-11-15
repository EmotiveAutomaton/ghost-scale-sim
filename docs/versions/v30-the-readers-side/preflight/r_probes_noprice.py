"""Family R pre-flight, added after the independent check: the two engine questions the miniature models (DEVALUE = Satiate,
AUDIENCE = Switch) without the PRICE question. Remove is not modelled here. Exact enumeration over the four answer patterns."""
import numpy as np, json, itertools
from r_probes import keep, PROBES, C
def asked(q,b,names):
    out={}
    for truth in C:
        tot=0.0; res=0.0; vs_habit=0.0
        for ans in itertools.product([0,1],repeat=len(names)):
            pa={c:1.0 for c in C}
            for n,a in zip(names,ans):
                p,nn,w=PROBES[n]
                for c in C:
                    pk=q*keep(c,p,nn,w)+(1-q)*b; pa[c]*=pk if a else 1-pk
            lo=np.log(pa[truth])-max(np.log(pa[c]) for c in C if c!=truth)
            tot+=pa[truth]*lo; res+=pa[truth]*(lo>=3)
            if truth=='RULE': vs_habit+=pa[truth]*(np.log(pa['RULE'])-np.log(pa['HABIT']))
        out[truth]={'expected_log_odds_vs_best_rival':round(float(tot),2),'share_settled':round(float(res),2)}
        if truth=='RULE': out[truth]['expected_log_odds_vs_HABIT']=round(float(vs_habit),2)
    return out
R={}
for q,b in ((1.0,0.5),(0.9,0.5),(0.75,0.5)):
    R[f'q={q},b={b}']=asked(q,b,['DEVALUE','AUDIENCE']); print('DEVALUE and AUDIENCE only, q=%.2f:'%q,json.dumps(R[f'q={q},b={b}']))
o=json.load(open('r_probes_out.json')); o['asked_without_price']=R; json.dump(o,open('r_probes_out.json','w'),indent=1)

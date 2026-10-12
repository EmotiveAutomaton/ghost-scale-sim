"""Independent scalar probability-domain filter for B04's finite path library."""

def reference(evidence,preference,skill,context,reader):
    states=[(p,s,c) for p in (0,1) for s in (0,1) for c in (0,1)]
    emissions=[]
    for p,s,c in states:
        row=[]
        for obs in range(4):
            x,y=divmod(int(obs),2);mass=0.
            for intention in (0,1):
                a=preference[p] if intention==x else 1-preference[p]
                b=skill[s] if intention==y else 1-skill[s]
                mass+=(context[c] if x else 1-context[c])*a*b
            row.append(float(mass))
        emissions.append(row)
    if reader in ('static','all-fast','all-slow'):
        rate={'static':0.,'all-fast':1/8,'all-slow':1/64}[reader];belief=[1/8]*8
        for t,obs in enumerate(evidence):
            if t:belief=[(1-rate)*v+rate/8 for v in belief]
            belief=[v*emissions[i][int(obs)] for i,v in enumerate(belief)];total=sum(belief);belief=[v/total for v in belief]
        return [(1-rate)*v+rate/8 for v in belief]
    before=[1/8]*8;after=[0.]*8
    for t in range(len(evidence)+1):
        if t:
            hazard=(.75/95)/(.25+.75*(96-t)/95) if t<96 else 0.
            addition=[0.]*8
            for i,si in enumerate(states):
                allowed=[j for j,sj in enumerate(states) if (sum(a!=b for a,b in zip(si,sj))==1 if reader=='role-mixture' else i!=j)]
                for j in allowed:addition[j]+=before[i]*hazard/len(allowed)
            after=[a+b for a,b in zip(after,addition)];before=[b*(1-hazard) for b in before]
        if t==len(evidence):break
        obs=int(evidence[t]);before=[v*emissions[i][obs] for i,v in enumerate(before)];after=[v*emissions[i][obs] for i,v in enumerate(after)]
        total=sum(before)+sum(after);before=[v/total for v in before];after=[v/total for v in after]
    return [a+b for a,b in zip(before,after)]

"""H1: exact hidden piece/episode segmentation, with supported mismatch."""
import math
import numpy as np
from .common import rng,posterior,scores,record,close
from ..v25.common import supported_posterior

def _world(setting,identity):
    r=rng('library',*identity);nt=setting.get('tops',6);ns=max(8,nt);npieces=setting.get('pieces',12)
    # Shared two-step cores ensure no individual step identifies a top goal.
    # Distinguish pieces by extra suffixes; all strings retain a common core.
    maxlen=setting.get('max_piece_length',4)
    strings=[tuple([i%10,(i+1)%10]+r.integers(0,10,int(r.integers(0,maxlen-1))).tolist()) for i in range(npieces)]
    sub=r.dirichlet(np.full(npieces,2.),ns);top=np.zeros((nt,ns));overlap=setting['overlap']
    for g in range(nt):
        if overlap=='none':top[g,g]=1
        else:
            shared=2 if overlap=='partial' else 4
            # Slightly different weights on shared support; private support is weak.
            top[g,:shared]=r.dirichlet(np.full(shared,8.))*.9
            top[g,shared+(g%(ns-shared))]+=.1
    # Check the no-near-exclusive-step invariant before retaining the world.
    counts=np.zeros((npieces,10))
    for i,s in enumerate(strings):
        for x in s:counts[i,x]+=1
    expected=top@sub@counts
    share=np.divide(expected,expected.sum(0),out=np.zeros_like(expected),where=expected.sum(0)>0)
    if share.max()>.5:raise ValueError('library violates no-exclusive-step check')
    if setting.get('mixed_tops'):
        # All unordered two-goal equal mixtures, treated as their own finite hypotheses.
        top=np.array([(top[i]+top[j])/2 for i in range(nt) for j in range(i+1,nt)])
    return strings,sub,top

def world(setting,identity):
    for attempt in range(1000):
        try:return _world(setting,[*identity,'library-attempt',attempt])
        except ValueError as exc:
            if str(exc)!='library violates no-exclusive-step check':raise
    raise ValueError('no-exclusive-step redraw limit exhausted')

def document(r,strings,sub,top,g,episodes):
    chunks=[]
    for _ in range(episodes):
        s=int(r.choice(len(sub),p=top[g]));pieces=r.choice(len(strings),int(r.integers(1,4)),p=sub[s])
        chunks.append(tuple(x for k in pieces for x in strings[k]))
    return tuple(x for c in chunks for x in c),chunks

def transitions(text,strings,weights,unknown=.05):
    n=len(text);limit=max(4,max(map(len,strings),default=4));out=np.zeros((len(weights),n+1,limit+1))
    for start in range(n):
        for i,s in enumerate(strings):
            if tuple(text[start:start+len(s)])==tuple(s):out[:,start,len(s)]+=(1-unknown)*weights[:,i]
        for k in (2,3,4):
            if start+k<=n:out[:,start,k]+=unknown/(3*10**k)
    return out

def episode_table(t):
    ns,n1,l=t.shape;n=n1-1;span=3*(l-1);ep=np.zeros((ns,n+1,span+1))
    for start in range(n):
        paths=np.zeros((ns,span+1));paths[:,0]=1
        for _ in range(3):
            nxt=np.zeros_like(paths)
            for used in range(span+1):
                if start+used>n:break
                for k in range(2,l):
                    if used+k<=span and start+used+k<=n:nxt[:,used+k]+=paths[:,used]*t[:,start+used,k]
            ep[:,start]+=nxt/3;paths=nxt
    return ep

def likelihood(text,strings,sub,top,episodes,independent=False,flat_pieces=False,novice=False):
    t=transitions(text,strings,sub,unknown=1. if novice else .05)
    if flat_pieces:
        t=transitions(text,strings,top@sub);ep=episode_table(t);mix=ep
    else:
        ep=episode_table(t);mix=np.einsum('gs,snl->gnl',top,ep)
    n=len(text);ng=len(top);cur=np.zeros((ng,n+1));cur[:,0]=1
    for _ in range(episodes):
        nxt=np.zeros_like(cur)
        for start in range(n):
            stop=min(mix.shape[-1],n-start+1)
            if independent:
                for g in range(ng):
                    for k in range(2,stop):nxt[g,start+k]+=float(cur[g,start])*float(mix[g,start,k])
            else:nxt[:,start+2:start+stop]+=cur[:,start,None]*mix[:,start,2:stop]
        cur=nxt
    return cur[:,n]

def flat_train(strings,sub,top,identity,episodes):
    r=rng('flat-training-excluded',*identity);first=np.ones((len(top),10));bi=np.ones((len(top),10,10))
    for g in range(len(top)):
        previous=next((j for j in range(g) if np.array_equal(top[j],top[g])),None)
        if previous is not None:first[g]=first[previous];bi[g]=bi[previous];continue
        goals=r.choice(len(sub),size=(10000,episodes),p=top[g]);counts=r.integers(1,4,(10000,episodes))
        pieces=np.zeros((10000,episodes,3),int)
        for s in range(len(sub)):
            mask=goals==s;pieces[mask]=r.choice(len(strings),size=(sum(mask.ravel()),3),p=sub[s])
        mask=np.arange(3)[None,None,:]<counts[:,:,None]
        frequencies=np.bincount(pieces[mask],minlength=len(strings))
        np.add.at(first[g],[strings[i][0] for i in pieces[:,0,0]],1)
        for i,s in enumerate(strings):
            for a,b in zip(s[:-1],s[1:]):bi[g,a,b]+=frequencies[i]
        for row,m in zip(pieces.reshape(10000,-1),mask.reshape(10000,-1)):
            seq=row[m]
            np.add.at(bi[g],([strings[i][-1] for i in seq[:-1]],[strings[i][0] for i in seq[1:]]),1)
    return first/first.sum(1,keepdims=True),bi/bi.sum(2,keepdims=True)

def block(setting,identity,independent=False):
    strings,sub,top=world(setting,identity);ep=setting.get('episodes',6);nt=len(top)
    first,bi=flat_train(strings,sub,top,identity,ep)
    r=rng('observations',*identity);mismatch=rng('mismatch',*identity)
    libraries={}
    for fraction in (1.,.75,.5):
        own=list(strings)
        for i in range(round(len(strings)*fraction),len(strings)):
            own[i]=tuple((x+1+int(mismatch.integers(9)))%10 for x in strings[i])
        libraries['expert-'+str(int(100*fraction))]=own
    levels=setting.get('prefixes',[1,2,4,8,16]);out=[];raw=[];alias=[]
    for maker in range(nt):
        truth=maker;ll={k:np.zeros(nt) for k in [*libraries,'novice','flat','pieces','boundaries']};history=[]
        for d in range(max(levels)):
            text,chunks=document(r,strings,sub,top,truth,ep);history.append(dict(steps=text,episodes=chunks))
            for reader,own in libraries.items():ll[reader]+=np.log(likelihood(text,own,sub,top,ep,independent))
            ll['novice']+=0 # exact equal likelihood cancels from posterior
            ll['flat']+=np.log(first[:,text[0]])+np.log(bi[:,text[:-1],text[1:]]).sum(1)
            if not setting.get('omit_pieces'):ll['pieces']+=np.log(likelihood(text,strings,sub,top,ep,independent,flat_pieces=True))
            ll['boundaries']+=sum((np.log(likelihood(c,strings,sub,top,1,independent)) for c in chunks),np.zeros(nt))
            if d+1 not in levels:continue
            posts={k:supported_posterior(v,independent=independent) for k,v in ll.items()}
            for name,p in posts.items():
                if name=='pieces' and setting.get('omit_pieces'):continue
                metric=scores(p,truth) if p is not None else dict(truth_probability=None,log_gain=None,half_brier=None,accuracy=None)
                out.append(record(dict(metric,unavailable=float(p is None)),maker=maker,reader=name,history=d+1))
            def contrast(a,b):
                return math.log(posts[a][truth]/posts[b][truth]) if posts[a] is not None and posts[b] is not None and posts[a][truth]>0 and posts[b][truth]>0 else None
            out.append(record(dict(primary=contrast('expert-75','flat'),mismatch_minus_flat=contrast('expert-50','flat'),mismatch_gain=contrast('expert-50','novice'),boundary_gain=contrast('boundaries','expert-100')),maker=maker,reader='contrasts',history=d+1))
            alias.append(dict(maker=maker,history=d+1,exact_posterior=posts['expert-100'].tolist() if posts['expert-100'] is not None else None))
        raw.append(dict(truth=truth,documents=history))
    return dict(public=dict(strings=strings,subgoals=sub.tolist(),tops=top.tolist(),flat_first=first.tolist(),flat_bigrams=bi.tolist(),libraries=libraries),truth_and_observations=raw,posteriors=alias,readouts=out)

def brute(text,strings,sub,top,episodes):
    # Independent exhaustive segmentation on tiny fixtures only.
    def piece_options(pos,s):
        out=[]
        for k,v in enumerate(strings):
            if tuple(text[pos:pos+len(v)])==tuple(v):out.append((pos+len(v),.95*sub[s,k]))
        for length in (2,3,4):
            if pos+length<=len(text):out.append((pos+length,.05/(3*10**length)))
        return out
    def episode(pos,s,count):
        if count==0:return [(pos,1.)]
        return [(end,p*q) for nxt,p in piece_options(pos,s) for end,q in episode(nxt,s,count-1)]
    def docs(pos,g,left):
        if left==0:return float(pos==len(text))
        return math.fsum(top[g,s]*p*docs(end,g,left-1)/3 for s in range(len(sub)) for count in (1,2,3) for end,p in episode(pos,s,count))
    return np.array([docs(0,g,episodes) for g in range(len(top))])

def controls():
    strings=[(0,1),(1,0)];sub=np.array([[.8,.2],[.3,.7]]);top=np.array([[.9,.1],[.2,.8]])
    for text,episodes in [((0,1,1,0),1),((0,1,0,1),2)]:
        p=likelihood(text,strings,sub,top,episodes);close(p.tolist(),brute(text,strings,sub,top,episodes).tolist(),tol=1e-12)
        close(p.tolist(),likelihood(text,strings,sub,top,episodes,True).tolist(),tol=1e-12)
    t=np.full((2,2),.5);p=likelihood((0,1,1,0),strings,sub,t,1);assert abs(p[0]-p[1])<1e-12
    first,bi=flat_train(strings,sub,t,['equal-top-fixture'],1);assert np.array_equal(first[0],first[1]) and np.array_equal(bi[0],bi[1])
    assert np.all(likelihood((9,9),strings,sub,top,1)>0)
    return dict(passed=True,dp_brute_two_documents=True,identical_top_prior=True,expert100_exact_same_implementation=True,unknown_support=True)

def design_check(settings):
    rows=[]
    for setting in settings:
        below=0;count=0
        for i in range(16):
            identity=['check-only-H1',setting,i];strings,sub,top=world(setting,identity);r=rng(*identity)
            truth=int(r.integers(len(top)));text,_=document(r,strings,sub,top,truth,setting.get('episodes',6))
            p=posterior(np.log(likelihood(text,strings,sub,top,setting.get('episodes',6))))
            below+=p[truth]<.8;count+=1
        rows.append(dict(setting=setting,uncertain_share=below/count,passed=setting['overlap']=='none' or below/count>=.25))
    return dict(passed=all(x['passed'] for x in rows),seed_domain='check-only-H1',checks=rows)

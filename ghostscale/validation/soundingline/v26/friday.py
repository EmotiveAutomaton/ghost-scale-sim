"""Prospective November 4 exploratory cards and supplement-aware admission.

Prior science is immutable. These cards are new descriptive maps, not confirmation.
"""
import itertools
import math
from datetime import datetime, timedelta
import numpy as np
from . import contract, registry
from .common import rng, posterior, close


def crossing_metrics(path):
    """One-nat crossing, recrossing and finite-horizon persistence; no censor imputation."""
    x=[float(v) for v in path]
    if not x or not all(math.isfinite(v) for v in x):raise ValueError('finite nonempty evidence path required')
    hits=[i+1 for i,v in enumerate(x) if v>=1.]
    last_below=max((i+1 for i,v in enumerate(x) if v<1.),default=0)
    return dict(first_crossing=hits[0] if hits else None,
                sustained_crossing=last_below+1 if last_below<len(x) else None,
                ever_crossed=float(bool(hits)),final_above=float(x[-1]>=1.),
                recrossed_below=float(bool(hits) and any(v<1 for v in x[hits[0]:])),
                fraction_above=sum(v>=1 for v in x)/len(x),final_log_odds=x[-1])


def cards():
    result=[]
    def add(f, settings, blocks, question, case):
        i=1+sum(c['family']==f for c in result)
        result.append(dict(id=f'{f}-X{i:02d}',family=f,settings=settings,blocks=blocks,
            question=question,case=case,precision=f'{blocks} independent worlds per cell; maximum binomial Monte Carlo standard error {0.5/math.sqrt(blocks):.4f}; log-score uncertainty estimated across worlds.',
            interpretation='Exploratory constructed-method map; miniature, architecture untested; no confirmation or human inference.',
            routes=list(registry.ROUTES[f]),seed_domain=f'v26-friday-20261104-{f.lower()}-x{i:02d}',
            probe_rule='One excluded world at every declared setting; complete original, replay and scalar reconstruction. Per-pass cost uses the largest observed whole-world time with 50% margin, plus overhead.',
            controls='Inherited exact-law known answers plus Friday-specific alias, persistence, source and setting checks. Controls never impose a desired science effect.',
            construction='Makers are nested within worlds. Reader comparisons share observations within a setting. Different setting cells use independent worlds; no paired causal claim. All unavailable outputs remain explicit.'))
    for tops,pieces in itertools.product((4,6,10),(8,12,20)):
        add('H1',[dict(tops=tops,pieces=pieces,episodes=ep,overlap=o,prefixes=[1,2,4,8,16]) for ep,o in itertools.product((3,6,12),('partial','heavy'))],64,
            f'How does library overlap interact with episode length at {tops} top goals and {pieces} pieces?',
            'At least 25% of excluded worlds remain uncertain about the true top goal after one document, at every setting.')
    for eta,share in itertools.product((.0025,.005,.02),(.5,.75,1.)):
        horizon=2048 if eta<.02 else 1024
        environments=['stationary','forced','trade']+(['shift'] if share<1 else [])
        add('E1',[dict(eta=eta,theta=.6,specialists=share,environment=e,max_history=horizon,
                       histories=[64,128,256,512,1024]+([2048] if horizon==2048 else []),truth_types=['A','B'],evidence_path=True) for e in environments],64,
            f'At learning rate {eta} and specialist share {share}, do one-nat crossings persist, and how often do fixed-value makers cross falsely?',
            'Every menu condition has differing fixed/circulating policies after an informative outcome in at least 25% of deterministic grid fixtures. Shift is tested only where it changes menus; threshold theta is not a B-truth intervention.')
    for d in (.3,.6):
        add('H2',[dict(d=d,q=q,units=n,hard=h) for q,n,h in itertools.product((.2,.5),(16,32,64),(1,3))],64,
            f'With deliberation probability {d}, how do observation dose and difficulty burden change the value and misreading of pauses?',
            'At least 25% of excluded worlds remain uncertain under text-only reading at each setting; both deliberation and difficulty have positive probability.')
    for rate in (.3,.6,.9):
        add('H3',[dict(r=rate,partial=p,discovery_fraction=f,units=n) for p,f,n in itertools.product((0.,.5,.75),(.25,.75),(12,48))],64,
            f'With revision rate {rate}, how do incomplete revision and discovery timing change initial versus settled-goal recovery?',
            'Changed-goal cases have positive probability .5; at least 25% of excluded changed-goal documents remain uncertain under the final-text reader at each setting.')
    for ends,missing in itertools.product((3,4),(False,True)):
        add('E2',[dict(kappa=k,overlap=o,ends=ends,missing_piece=missing) for k,o in itertools.product((.5,2.,8.,16.),(3,4,6))],128,
            f'With {ends} ends and missing-piece proxy {missing}, where do own, population and mixture priors cross as prior concentration and library mismatch vary?',
            'Every finite maker weight has candidates in all four declared distance bins; bin assignment is balanced by construction.')
    for overlap,fear in itertools.product(('partial','heavy'),(False,True)):
        add('E3',[dict(overlap=overlap,fear=fear,rate=rate,innate_share=mix,truth='cached') for rate,mix in itertools.product((.05,.1,.3,.5),(0.,.25,.75))],64,
            f'Under {overlap} library overlap and fear channel {fear}, when do learning rate and innate mixing change amplifier-reader harm?',
            'At least 25% of excluded eligible observations give amplifier and cached signature probabilities differing by .2, at every setting.')
    for strength in (.25,.5,.75,1.):
        add('U1',[dict(s=strength,tau=1.,claims=n,training_lengths=list(range(1,65))) for n in (16,32,64,128)],128,
            f'At source strength {strength}, how does new-claim dose change the source-history learning curve and its uncertainty?',
            'Enumerated revealed pools distinguish the generating source from unbiased selection by more than .1 with probability at least .25 under each hypothesis. Strengths .25/.75 are deliberately outside the reader grid, not exact-model consistency tests.')
    for structure in ('constant','varying'):
        add('R1',[dict(beta=b,structure=structure,switch_gap=g) for b,g in itertools.product((1,3),(0.,.125,.25,.5,1.))],64,
            f'Under {structure} passive effects, how does actual intervention strength alter split-alias recovery?',
            'Passive split aliases hold exactly at every setting; positive-gap switch interventions separate every split pair, and zero gap is an exact negative control. Remove cannot split the retained pair; constant-effect satiation remains aliased.')
    return result


def design_check(card):
    """Excluded grid checks inspect laws/short fixtures, never cost or science draws."""
    from . import hierarchy, values, pauses, revision, self_model, cached, opacity, alias_repair
    f=card['family'];rows=[]
    for index,s in enumerate(card['settings']):
        shares=[];detail={}
        if f=='H1':
            check=hierarchy.design_check([s]);shares=[check['checks'][0]['uncertain_share']]
        elif f in ('H2','H3'):
            uncertain=0
            for b in range(16):
                ident=['friday-check-only',card['id'],index,b];r=rng(*ident);w=pauses.weights(ident)
                if f=='H2':
                    laws=np.array([pauses.law(h,g,s['d'],s['q'],tuple(range(s['hard']))).sum((1,2)) for g in w for h in pauses.HABITS]);ll=np.zeros(18)
                    for t in range(s['units']):
                        obs,_=pauses.observe(r,pauses.HABITS[b%3],w[b%6],s['d'],s['q'],tuple(range(s['hard'])));ll+=np.log(laws[:,obs[0]])
                    p=posterior(ll).reshape(6,3).sum(1);uncertain+=p[b%6]<.8
                else:
                    for attempt in range(1000):
                        obs=revision.draw(r,w,0,0,s['units'],s['r'],s['partial'],s['discovery_fraction'])
                        if obs['settled']!=0:break
                    else:raise ValueError('excluded changed-goal fixture exhausted')
                    hs=revision.hypotheses(s['units'],s['discovery_fraction']);prior=np.array([(.5 if a==g else .1)/108 for a,g,p,rt,h in hs])
                    p=posterior(revision.document_likelihood(w,obs,s['units'],'final',partial=s['partial'],fraction=s['discovery_fraction']),prior)
                    ps=np.bincount([h[1] for h in hs],weights=p,minlength=6);uncertain+=ps[obs['settled']]<.8
            shares=[uncertain/16]
        elif f=='E1':
            differences=0;changed=0
            for b in range(32):
                ident=['friday-check-only',card['id'],index,b];r=rng(*ident);menu,_,_,_=values.menu_at(r,129,s|dict(environment_dimension=0))
                hs=[('A',0,2,np.array([.25,.25,.5])),('B',s['eta'],2,np.array([.25,.25,.5]))];state=values.states(hs)
                values.update(hs,state,np.array([1.,0.,0.]),1)
                differences+=np.max(abs(values.policy(menu,hs,state)[0]-values.policy(menu,hs,state)[1]))>1e-8
                before=values.menu_at(rng(*ident),129,s|dict(environment='stationary',environment_dimension=0))[0]
                changed+=not np.array_equal(menu,before)
            shares=[differences/32];detail['changed_menu_share']=changed/32
            if s['environment']=='shift':shares.append(changed/32)
        elif f=='E2':
            w=values.grid(s['ends']);ok=[]
            for x in w:
                tv=np.abs(w-x).sum(1)/2
                ok.append(all(np.any(abs(tv-d)<1e-12) for d in (0,.25,.5)) and np.any(tv>=.75))
            shares=[sum(ok)/len(ok)];detail['all_bins_available']=all(ok)
        elif f=='E3':
            different=eligible=0;r=rng('friday-check-only',card['id'],index)
            for b in range(32):
                e=cached.library(s['overlap'],r);w=cached.simplex(4)[int(r.integers(15))];a=np.zeros(4)
                for t in range(128):
                    si=t%4;law=cached.softmax(3*w@e[list(cached.TRAIN[si])].T);piece=cached.TRAIN[si][int(r.choice(len(law),p=law))]
                    features=r.integers(0,2,4);ids=np.flatnonzero(features);blocked=r.random()<.02+.16*features[0]
                    available=r.random()<.02+.16*features[1] if s['fear'] else blocked
                    appraisal=(1-s['innate_share'])*(a[ids].mean() if len(ids) else 0)+s['innate_share']*features[0]
                    if available:eligible+=1;different+=abs(cached.fire(w@e[piece])-cached.fire(appraisal))>.2
                    a[ids]+=s['rate']*(int(blocked)-a[ids])
            shares=[different/eligible]
        elif f=='U1':
            for p in (.3,.7):
                shares.append(sum(math.comb(8,k)*p**k*(1-p)**(8-k) for k in range(9) if abs(opacity.conditional(k,s['s'])-opacity.conditional(k,0))>.1))
            detail['reader_misspecified']=s['s'] not in (0,.5,1.)
        elif f=='R1':
            hs=alias_repair.hypotheses(s['beta']);pairs=alias_repair.split_pairs(s['beta'],hs)
            passive=True;split=[]
            for b in range(4):
                e=alias_repair.library(rng('friday-check-only',card['id'],index,b),s['structure'],s['switch_gap'])
                for menu in alias_repair.TRAIN:
                    law=alias_repair.laws(e,menu,hs);passive &= all(np.max(abs(law[i]-law[j]))<1e-12 for i,j in pairs)
                laws=[alias_repair.laws(e,menu,hs) for menu in alias_repair.SWITCH]
                split.extend(any(np.max(abs(law[i]-law[j]))>1e-12 for law in laws) for i,j in pairs)
            expected=s['switch_gap']>0;shares=[float(passive),sum(v==expected for v in split)/len(split)];detail['switch_separation_share']=sum(split)/len(split)
        else:raise ValueError(f)
        rows.append(dict(setting=index,shares=shares,passed=all(v>=.25 for v in shares),**detail))
    return dict(passed=all(r['passed'] for r in rows),card=card['id'],seed_domain='friday-check-only; hierarchy uses check-only-H1',floor=.25,checks=rows,case=card['case'])


def admission(plan,accepted,supplement,current,charges,running=0):
    if accepted!=contract.acceptance(accepted['opened'],accepted['specification_sha256']):raise ValueError('unknown original acceptance')
    if current.tzinfo is None:raise ValueError('aware clock required')
    pools=supplement['pools_seconds']
    if sum(pools.values())!=110*3600 or supplement['total_cpu_seconds']!=110*3600:raise ValueError('unknown supplement total')
    if any(not math.isfinite(v) or v<0 for v in charges.values()):raise ValueError('invalid charges')
    if any(charges.get(k,0)>v for k,v in pools.items()):raise ValueError('existing pool overrun')
    if not datetime.fromisoformat(accepted['science_open'])<=current<min(datetime.fromisoformat(supplement['review_target']),datetime.fromisoformat(accepted['new_starts_close'])):raise ValueError('outside new admission window')
    if running>=2:raise ValueError('two shared scientific slots')
    if plan['numerical_threads']!=1 or plan['priority']!='below-normal':raise ValueError('resource mismatch')
    if plan['kind'] not in ('probe','study','healing'):raise ValueError('Friday exploratory packet mismatch')
    expected_pool='followup_healing' if plan['kind']=='healing' else 'studies'
    if plan['pool']!=expected_pool:raise ValueError('Friday exploratory pool mismatch')
    if plan['kind']=='healing' and plan.get('healing_number') not in (1,2):raise ValueError('healing limit')
    cap=plan['cpu_cap_seconds'];family=plan['family'];passes=plan['pass_caps']
    if not math.isfinite(cap) or cap<=0:raise ValueError('invalid cap')
    if set(passes)!={'original','replay','review'} or any(not math.isfinite(x) or x<=0 for x in passes.values()) or sum(passes.values())>cap:raise ValueError('pass reservation')
    if plan['kind']=='probe' and cap>1800:raise ValueError('probe ceiling')
    if charges.get(expected_pool,0)+cap>pools[expected_pool] or sum(charges.get(k,0) for k in pools)+cap>supplement['total_cpu_seconds']:raise ValueError('pool ceiling')
    if charges.get('family:'+family,0)+cap>supplement['family_seconds'][family]:raise ValueError('family ceiling')
    projection=plan['projected_wall_seconds']
    if not math.isfinite(projection) or projection<=0:raise ValueError('invalid wall projection')
    if current+timedelta(seconds=projection)>=datetime.fromisoformat(supplement['review_target']):raise ValueError('insufficient Friday review margin')
    if not plan.get('design_check_sha256') or not plan.get('supplement_sha256'):raise ValueError('missing prospective bindings')
    return True

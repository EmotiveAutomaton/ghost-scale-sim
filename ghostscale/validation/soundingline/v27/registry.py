"""Declared grids, provisional choices and all prospective expansion routes."""
import itertools
from . import contract

MODULES=dict(K1='timing',K2='motives',K3='setpoints',K4='stress',K5='recheck',K6='recursive',K7='laughter')
QUESTIONS=dict(K1='Do decision times distinguish a locked rule from a strong preference after learning personal speed?',
 K2='Can a reader separate several motives for one act, including feedback from acting?',
 K3='Can restoring efforts locate hidden set points and distinguish moving from dulling or gain-up?',
 K4='Which records distinguish predicted harm from uncertainty about outcomes or actions?',
 K5='How do re-reading, fading tags, warnings, group priors and source libraries change capture?',
 K6='Does iterative interpretation recover information or reinforce its initial mistake?',
 K7='Do completion markers reveal hidden structure when surprise or social state can also cause them?')
ROUTES=dict(
 K1=['Sounding Line: timing robustness can nominate later suggestion-latency work; no human timing test is admitted here.','ToMpathy: compare timing benefit with false identity calls before adding an identity mark.'],
 K2=['Sounding Line: no human test this week; retain the constructed explaining-away requirement.','ToMpathy: compare exact explaining-away, uncertainty-tracking and associative readers before implementing a network.'],
 K3=['Sounding Line: CH4 and SP need restoring actions and reserves; report whether probes are necessary to distinguish moving and dulling.','ToMpathy: report the assumptions required to identify a set point and the indistinguishability of a restore-only band from its lower edge.'],
 K4=['Sounding Line: CH2/CH3 and IN2/IN3 need channels that separate stakes, policy uncertainty and outcome uncertainty.','ToMpathy: separate FEAR from stress only when their observation laws are validated.'],
 K5=['Sounding Line: no human test this week.','ToMpathy: RR/RT scale comparisons do not license a retroactive tag; working position C-1107-17 remains untested by this miniature.'],
 K6=['Sounding Line: AT3 and GO2 are the human analogues of recursive reading.','ToMpathy: compare joint inference, sequential inference and guarded iteration across panels.'],
 K7=['Sounding Line: LA1 compares resolution with surprise; LA2 compares shared background with co-laughter.','ToMpathy: quantify social base-rate dilution and misleading markers before adding a structural cue.'])
EARLIER=dict(K1=['G-V26-DAY-G4-1','G-V26-DAY-G2-1','C-1107-9'],
 K2=['C-1107-10'],K3=['G-V26-DAY-G1-1','G-V26-DAY-G5-1','C-1107-3','C-1107-6'],
 K4=['C-1103-1','C-1107-2','C-1107-8'],K5=['G-V26-DAY-G3-1','G-V26-DAY-G3-T09-1','C-1107-17'],
 K6=['G-V26-H2-1','C-1107-14'],K7=['G-V26-H1-1','C-1107-15'])


def primary_settings(f):
    if f=='K1':return [dict(truth=t,records=256,makers=8,ordinary=64,jitter=0.,ladder=[1,2,4,8],shared_speed=False,n_share=.5) for t in ('A90','A100','L','Ls','Ah','N')]
    if f=='K2':return [dict(records=64,makers=8,targets=4,phi=0.,tournament=q,payment=False,rho=.4) for q in (.5,.8)]
    if f=='K3':
        return [dict(records=512,makers=4,dynamic=d,beta=b,width=w,needs=2,consumption=3.,probe=pr,lapse=.02,stock_noise=0.)
            for d,b,w,pr in [(d,0.,0,pr) for d in ('P0','P+','P-','S','G') for pr in (0.,1/64,1/16)]
              +[('P0',1.,0,1/64),('P0',0.,4,1/64)]]
    if f=='K4':return [dict(records=128,makers=8,fear=fear,stress=stress,magnitude=1.,trained_share=.5,mix=mix,latency_noise=1.)
        for mix in ('designed','observational') for fear in range(2) for stress in range(3)]
    if f=='K5':return [dict(records=64,worlds_per_block=4,sources=32,theta=.7,delta=0.,switch=16,rho=.05,
        strengths=[.5],x=x,tau=tau,zeta=zeta,context_switch=switch,warning=w,bias=.1)
        for x,tau,zeta,switch,w in [(x,tau,zeta,sw,.8) for x in (.25,.5,.75) for tau in (.5,.9) for zeta,sw in [(0.,False),(.02,False),(0.,True)]]
        +[(.5,.5,0.,False,w) for w in (.6,.95)]]
    if f=='K6':return [dict(records=64,makers=8,goals=4,d=.3,q=.5,hard=2,information=1.,rounds=20,correlated=False)]
    if f=='K7':return [dict(structures=32,makers=8,law=law,qc=.8,q0=.2,fake=fake,listeners=4,overlap=1.,episodes=3)
        for law in ('M1','M2','M3') for fake in (0.,.1)]
    raise ValueError(f)


def variants(f):
    """Full declared Cartesian grids, unique deviations; out-of-case last."""
    axes=dict(
        K1=dict(jitter=[0.,.1,.3],ordinary=[0,16,64],ladder=[[1,2,4,8],[1,2,4,8,16]],type_mix=['all','no-Ls-Ah','double-N']),
        K2=dict(phi=[0.,.1,.3],tournament=[.2,.5,.8],targets=[2,4,8],payment=[False,True]),
        K3=dict(beta=[0.,.5],width=[0,2],needs=[2,3],consumption=[3.,6.],lapse=[.02,.05]),
        K4=dict(magnitude=[1.,2.],trained_share=[.25,.5,.75],latency_noise=[1.,2.],channel=['all','arousal']),
        K5=dict(delta=[0.,.05,.1],switch=[16,32],theta=[.7,.5],bias=[.1,.2],strengths=[[.5],[.25,.5,.75,1.]]),
        K6=dict(hard=[2,4],goals=[4,8],information=[1.,.5],rounds=[20,5]),
        K7=dict(q0=[.05,.2,.5],qc=[.4,.8],listeners=[1,4,8],overlap=[.5,1.]))[f]
    out=[]
    for values in itertools.product(*axes.values()):out.append((dict(zip(axes,values)),False))
    extra=dict(K1=[dict(shared_speed=True)],K2=[dict(tournament=0.),dict(tournament=1.)],
        K3=[dict(stock_noise=1.)],K4=[dict(mix='observational')],K5=[dict(strengths=[.5,1.])],
        K6=[dict(correlated=True)],K7=[dict(law='M3',q0=0.)])
    out.extend((x,True) for x in extra[f]);return out


def settings_with(f,changes):
    settings=primary_settings(f)
    if f=='K1' and changes.get('type_mix')=='no-Ls-Ah':settings=[s for s in settings if s['truth'] not in ('Ls','Ah')]
    if f=='K1' and changes.get('type_mix')=='double-N':
        settings=[s|dict(makers=2*s['makers']) if s['truth']=='N' else s for s in settings]
    return [s|changes for s in settings]


def cards():
    out=[]
    for f in contract.ORDER:
        common=dict(family=f,question=QUESTIONS[f],routes=ROUTES[f],earlier_ids=EARLIER[f],blocks=16,
            interpretation='Constructed method; miniature, architecture untested. Pointwise descriptive intervals; no human or engine validation.')
        out.append(common|dict(id=f,lane='study',settings=primary_settings(f),declared_out_of_case=False,order=contract.ORDER.index(f)))
        seen={str(primary_settings(f))}
        for changes,outside in variants(f):
            settings=settings_with(f,changes)
            if str(settings) in seen:continue
            seen.add(str(settings));n=sum(x['family']==f and x['lane']=='texture' for x in out)+1
            out.append(common|dict(id=f+'-T'+str(n).zfill(3),lane='texture',settings=settings,
                declared_out_of_case=outside,changes=changes,order=100+1000*contract.ORDER.index(f)+n))
    return out


def expansion_menu(f):
    """Three prospectively distinct axes; generation two requires parent decision."""
    return dict(K1=[('stress',dict(jitter=.3)),('dose',dict(records=512)),('mechanism',dict(ordinary=0))],
      K2=[('stress',dict(phi=.3)),('dose',dict(records=128)),('mechanism',dict(rho=0.))],
      K3=[('stress',dict(consumption=4.)),('dose',dict(records=1024)),('mechanism',dict(probe=1/16))],
      K4=[('stress',dict(latency_noise=4.)),('dose',dict(records=256)),('mechanism',dict(mix='mixed'))],
      K5=[('stress',dict(theta=.5)),('dose',dict(records=128)),('mechanism',dict(strengths=[.5,1.]))],
      K6=[('stress',dict(q=.5,d=.6)),('dose',dict(records=128)),('mechanism',dict(information=.5))],
      K7=[('stress',dict(q0=.5)),('dose',dict(structures=64)),('mechanism',dict(fake=.1))])[f]

"""Independent scalar G3 class and weight likelihood reconstruction."""
import math
import numpy as np
from numba import njit


@njit(cache=True)
def reconstruct(stocks,needs,actions,truth,weights):
    worlds,steps=stocks.shape
    last=np.full((worlds,7),-1);final=np.empty((worlds,7));sep=np.empty((worlds,7))
    exposure_world=np.zeros((worlds,3),np.bool_);exposure_step=np.zeros((worlds,3))
    before=np.zeros((worlds,3));weight_sep=np.empty(worlds);post=np.empty(worlds)
    reserves=(2,4,10,15,20);pairs=((2.,2.),(3.,1.),(1.,3.))
    for w in range(worlds):
        logs=np.zeros((7,3));weight_last=-1;moments=np.zeros((steps,3),np.bool_)
        for t in range(steps):
            stock=stocks[w,t];a=needs[w,t,0];b=needs[w,t,1];chosen=actions[w,t]
            for c in range(7):
                cost=(3. if stock<=reserves[c] else 0.) if c<5 else (3. if c==5 else 5.)
                for j in range(3):
                    pa=math.exp(pairs[j][0]*a-cost) if stock>0 and a>0 else 0.
                    pb=math.exp(pairs[j][1]*b-cost) if stock>0 and b>0 else 0.
                    number=1. if chosen==0 else (pa if chosen==1 else pb)
                    logs[c,j]+=math.log(number/(1+pa+pb)) if number else -math.inf
            evidence=np.empty(7)
            for c in range(7):
                high=max(logs[c,0],logs[c,1],logs[c,2]);total=0.
                for j in range(3):total+=math.exp(logs[c,j]-high)
                evidence[c]=high+math.log(total/3)
            for c in range(7):
                final[w,c]=evidence[truth]-evidence[c]
                if final[w,c]<1:last[w,c]=t
            ev=np.empty(3)
            for j in range(3):
                high=-math.inf
                for c in range(7):high=max(high,logs[c,j])
                total=0.
                for c in range(7):total+=math.exp(logs[c,j]-high)
                ev[j]=high+math.log(total/7)
            if ev[1]-ev[2]<1:weight_last=t
            for j,reserve in enumerate((10,15,20)):
                moments[t,j]=stock>reserve and (a>0 or b>0)
                if moments[t,j]:exposure_world[w,j]=True;exposure_step[w,j]+=1./steps
        for c in range(7):sep[w,c]=math.inf if final[w,c]<1 else last[w,c]+1
        for j in range(3):
            for t in range(last[w,j+2]+1):
                if moments[t,j]:before[w,j]+=1
        weight_sep[w]=math.inf if ev[1]-ev[2]<1 else weight_last+1
        high=max(ev[0],ev[1],ev[2]);den=0.
        for j in range(3):den+=math.exp(ev[j]-high)
        post[w]=math.exp(ev[weights[w]]-high)/den
    return sep,last,final,exposure_world,exposure_step,before,weight_sep,post


def infer(raw,setting):
    values=reconstruct(raw['stocks'],raw['needs'],raw['actions'],int(setting['truth']),raw['weights'])
    return dict(zip(('separation','last_below','final_odds','exposure_world','exposure_step',
        'exposure_before','weight_separation','weight_posterior'),values))

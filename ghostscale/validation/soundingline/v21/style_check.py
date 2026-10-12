"""E03 scalar count fit and archive selection independent of producer helpers."""
import numpy as np


def fit_reference(target,noise,reliability):
    result=np.zeros((2,2,2,2,2))
    for m in range(2):
      for s in range(2):
       for decoder in range(2):
        counts=[[.5,.5],[.5,.5]]
        for y,u in zip(target,noise):
            x=int(y)
            if m:x=1-x
            if u>=reliability:x=1-x
            if decoder==0 and s:x=1-x
            counts[x][int(y)]+=1
        for x in range(2):
            total=sum(counts[x])
            for y in range(2):result[m,s,decoder,x,y]=counts[x][y]/total
    return result


def predict_reference(table,mechanism,style,token,locality,decoder):
    selected_m=mechanism if locality in ('both','mechanism') else 1-mechanism
    selected_s=style if locality in ('both','style') else 1-style
    feature=1-token if decoder=='decoded' and style else token
    return table[selected_m,selected_s,1 if decoder=='decoded' else 0,feature].copy()

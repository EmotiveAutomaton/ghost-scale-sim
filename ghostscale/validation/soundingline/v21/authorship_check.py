"""Independent elementary topology/role enumeration, no production catalog calls."""
import numpy as np


def reference(roles,conflict,prior,evidence,tier,reader):
    masses={};endpoint=int(evidence[0]);count=0 if tier=='none' else roles-1 if tier=='partial' else roles
    for kind in range(3):
        goals=[];owners=[]
        for j in range(roles):
            goals.append(endpoint if kind==0 or not conflict or (roles-1-j)%2==0 else 1-endpoint)
            owners.append(0 if kind!=2 or j==0 else 1)
        allowed=True
        if reader!='endpoint':
            for j in range(count):allowed=allowed and goals[j]==evidence[2+j]
            for j in range(1,count):allowed=allowed and int(owners[j]==owners[j-1])==evidence[2+roles+j-1]
        if reader=='single-person' and kind==2:allowed=False
        label=kind*8
        for j in range(roles):label+=goals[j]*2**(2-j)
        masses[label]=float(prior[kind]/2) if allowed else 0.
    if not sum(masses.values()):
        for label in masses:masses[label]=float(prior[label//8]/2) if label//8<2 else 0.
    q=np.zeros(128);total=sum(masses.values())
    for label,mass in masses.items():q[label]=mass/total
    return q

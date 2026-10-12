"""Independent scalar execution and posterior for E01 plan selection."""
import math

def execute(proposals,choice,cosmetic_edits):
    content=[]
    for i in range(len(proposals[choice])):content.append(int(proposals[choice][i]))
    style=0
    for _ in range(cosmetic_edits):style=1-style
    return content,style

def posterior(proposals,artifact,prior,record):
    masses=[]
    for choice in (0,1):
        possible=all(int(proposals[choice][i])==int(artifact[i]) for i in range(len(artifact)))
        masses.append((prior if choice else 1-prior) if possible and (record is None or choice==record) else 0.)
    total=math.fsum(masses)
    if total==0:raise ValueError('inconsistent selection evidence')
    return [x/total for x in masses]

def effects(proposals,choice,edits):
    actual,_=execute(proposals,choice,edits);alternative,_=execute(proposals,1-choice,edits)
    return [sum(a!=b for a,b in zip(actual,alternative))]+[0]*edits

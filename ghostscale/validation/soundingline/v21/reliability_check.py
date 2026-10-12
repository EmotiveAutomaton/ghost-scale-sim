"""Independent scalar probability-domain noisy-record update."""
import math
import numpy as np


def reference(prior,reports,reliability,hard=False):
    answer=[]
    for row,reported in zip(prior,reports):
        weights=[]
        for target,mass in enumerate(row):
            match=((target//4)%2)==int(reported)
            weight=float(match) if hard else reliability if match else 1-reliability
            weights.append(float(mass)*weight)
        denominator=math.fsum(weights)
        answer.append([p/denominator for p in weights] if denominator else list(row))
    return np.array(answer)

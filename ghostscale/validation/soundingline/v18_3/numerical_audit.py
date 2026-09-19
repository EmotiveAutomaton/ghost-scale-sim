"""Sensitivity of finite compression to numerically tied old-task optima."""
import numpy as np
from .compression import codebooks,code_losses


def tied_codebooks(ph,old,new,cardinality,tolerance=1e-10,query_entropy=0.):
    codes,masks,sizes=codebooks();oldloss,_=code_losses(masks,ph,old);newloss,_=code_losses(masks,ph,new)
    candidates=np.flatnonzero(sizes==cardinality);minimum=float(oldloss[candidates].min())
    tied=candidates[oldloss[candidates]<=minimum+tolerance]
    return dict(cardinality=cardinality,candidates=len(candidates),old_minimum=minimum,numerically_optimal_codes=len(tied),tolerance=tolerance,
        best_new_loss_among_ties=float(newloss[tied].min()-query_entropy),worst_new_loss_among_ties=float(newloss[tied].max()-query_entropy),
        scope='post-hoc sensitivity range; future loss does not select or replace the original old-task code')

from tasks.config import cfg
from math import log

from pdb import set_trace as pause

def get_adaptative_lambda(inner_iter):
 #### TODO LAMBDA
    lambda_ign = 0

    if cfg.ADAPTATIVE_SUP.TYPE == 'log':
        
        lb = cfg.ADAPTATIVE_SUP.LB
        lambda_gt  = (log(inner_iter+lb)-log(lb))/(log(cfg.SOLVER.MAX_ITER+lb)-log(lb))/2

        if cfg.ADAPTATIVE_SUP.DO_TRICK:
            if cfg.ADAPTATIVE_SUP.ADAPTATIVE_IGN:
                lambda_ign = 0.51 - lambda_gt
            else:
                lambda_ign = cfg.TRAIN.BG_THRESH

    return lambda_gt, lambda_ign 
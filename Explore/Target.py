import numpy as np
import math

def same_target_check(target_loc : tuple, target_list : tuple, same_target_threshold_horiz,
                      same_target_threshold_verti):
    
    if len(target_list) != 0:
        target_loc = np.array(target_loc).reshape(1,-1)
        target_list = np.concatenate(target_list,axis = 0)        
        dist = np.linalg.norm(target_loc[:,0:2] - target_list[:,0:2], axis=-1)
        idx = np.where(dist < same_target_threshold_horiz)[0]
        if idx.shape[0] == 0:
            return True
        else:
            sub_targets = target_list[idx]
            dist = abs(target_loc[:,-1] - sub_targets[:,-1])
            idx = np.where(dist < same_target_threshold_verti)[0]
            if idx.shape[0] == 0:
                return True
        return False
    else:
        return True
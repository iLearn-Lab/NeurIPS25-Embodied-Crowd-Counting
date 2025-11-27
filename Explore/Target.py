import numpy as np
import math

def target_distance_check(target_loc, location_list : list, dist_threshold):
    '''
        return if the target is in range of one of the locations of location_list
    '''
    if len(location_list) != 0:
        location = np.array(location_list)
        target_loc = np.array(target_loc).reshape(1,-1)               
        residual = target_loc[:,0:2] - location[:,0:2]
        dist = np.linalg.norm(residual, axis=-1)
        idx = np.where(dist < dist_threshold)
        if len(idx[0]) != 0:
            return True
        else:
            return False
    else:
        return False

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

def scanned_area_check(target_loc : tuple, location_list : list, pose_list : list, camera : int, fov, distance_threshold):

    if len(pose_list) != 0:
        location = np.array(location_list)
        pose = np.array(pose_list)
        if camera == 1:
            pose[:,2] = pose[:,2] - 90
        elif camera == 2:
            pose[:,2] = pose[:,2] + 90
        elif camera == 4:
            pose[:,2] = pose[:,2] + 180  
        target_loc = np.array(target_loc).reshape(1,-1)               
        residual = target_loc[:,0:2] - location[:,0:2]
        dist = np.linalg.norm(residual, axis=-1)
        new_pos = np.arctan2(residual[:,1], residual[:,0]) * 180 / math.pi
        bias1 = abs(pose[:,2] - new_pos)
        bias2 = 360 - bias1
        bias = np.minimum(bias1, bias2)   
        idx1 = dist < distance_threshold
        idx2 = bias <= fov/2
        idx = idx1 & idx2
        idx = np.where(idx == True)[0]
        if idx.shape[0] == 0: 
            return False
        else:
            return True
    else:
        return False      
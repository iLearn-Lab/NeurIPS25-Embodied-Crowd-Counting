import warnings
import os
import numpy as np  
np.random.seed(42)
import random  
random.seed(42)
import shutil

from Simulator.Simulator import Simulator
from Drone.Control import drone_controller
from Point_cloud.Point_cloud import point_cloud
from utils.saver import image_saver_plt
from Explore.Explore import explore
from Explore.DroneLift import DroneLift

from Point_cloud.Map_element import (
    UNKNOWN,
    OBSTACLE,
    EXPLORED,
    FRONTIER)

warnings.filterwarnings("ignore")

def init_explore(config):
    
    path = config['Record_root'] + config['now'] + '/explore_config/'
    if not os.path.exists(path):  
        os.makedirs(path)
    shutil.copy2('Configs/FBEConfig.yml', path + 'Config.yml')    

    SM = Simulator(config['Simulator'])
    change, route = SM.get_route()
    ground_truth = SM.get_ground_truth(change)
    SM.save_simulator(change,route,ground_truth,config['Record_root'],config['now'] + '/')    

    DC = drone_controller(config['Drone'])
    DC.to_cv_mode()
    
    PT = point_cloud()
    
    EX = explore(config)
    
    path_saver = image_saver_plt(config['now'],config['Record_root'],'path')     
    
    camera = [0,1,4,2]
    records = dict()
    records['visual'] = dict()
    for cam in camera:
        records['visual'][cam] = dict()
        records['visual'][cam]['bgr'] = []
        records['visual'][cam]['depth'] = []
    records['location'] = []
    records['pose'] = []
    records['path'] = []

    DL = DroneLift(config)
    location, pose = DC.get_world_location_pose()

    DL.to_target_height(DC, location, pose, records, down=True)
    return DC, PT, EX, records, path_saver, camera

def run_fbe(config):

    DC, PT, EX, records, path_saver, camera = init_explore(config)
    FBE(EX, DC,PT,camera,records,config,path_saver)
    EX.calculate_all_path(records)
    DC.save_records(config, records)
    
def FBE(EX: explore,
        DC: drone_controller,
        PT: point_cloud,
        camera,records,config,path_saver):

    global_2D_map = dict()
    global_2D_map[OBSTACLE] = np.array([])
    global_2D_map[EXPLORED] = np.array([]) 

    global_X = []
    global_Y = []
    global_Z = []
    global_bgr = []

    from utils.saver import image_saver_plt
    global_map_saver = image_saver_plt(config['now'],config['Record_root'],'global_map')

    while(1):
        X,Y,Z,bgr,location,pose = EX.get_current_point_cloud(DC,PT,camera,records,global_X,global_Y,global_Z,global_bgr)
            
        map, change = EX.get_map_at_current_height(X,Y,Z,PT,location)
        explore_range = config['Explore']['explore_range']
        explored_map = EX.get_explored_map(map, location, change, explore_range)
        EX.update_global_map(explored_map, location, change, global_2D_map)
        global_explored_map, change = EX.get_global_explored_map(location,global_2D_map)
        frontiers, frontiers_map = EX.get_frontiers(global_explored_map, change)
        global_map_image = EX.draw_global_map_and_frontier(frontiers,global_2D_map)
            
        global_map_saver.save(global_map_image) 
        if len(frontiers) == 0:
            break       
        mini_frontier_size = config['Explore']['frontier_divide_size']

        EX.divide_frontiers_to_bins(frontiers, location, mini_frontier_size)
        path, navi_point = EX.find_surround_frontier(location, frontiers, frontiers_map, change)
        path_image = EX.draw_path(global_map_image,path,location,navi_point)
        path_saver.save(path_image)
        EX.move_along_path(path, navi_point, DC, records,capture_rate = 2, if_capture = False, sleep = 0)
        EX.save_path(path,records)
        
    return global_2D_map, global_X, global_Y, global_Z, global_bgr 
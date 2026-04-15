from Drone.Control import drone_controller
from Explore.DensityGuided import DensityGuided
from Perception.GeneralizedLoss import gereralizedloss
from Point_cloud.Point_cloud import point_cloud
from Explore.Explore import explore
from utils.saver import image_saver_plt
import warnings
import os
import numpy as np  
np.random.seed(42)
import random  
random.seed(42)

warnings.filterwarnings("ignore")

def init_explore(config):
    import shutil
    path = config['Record_root'] + config['now'] + '/explore_config/'
    if not os.path.exists(path):  
        os.makedirs(path)
    shutil.copy2('Configs/FBEWithDGConfig.yml', path + 'Config.yml')    

    from Simulator.Simulator import Simulator
    SM = Simulator(config['Simulator'])
    change, route = SM.get_route()
    ground_truth = SM.get_ground_truth(change)
    SM.save_simulator(change,route,ground_truth,config['Record_root'],config['now'] + '/')

    DC = drone_controller(config['Drone'])

    DC.to_cv_mode()
    PT = point_cloud()
    path_saver = image_saver_plt(config['now'],config['Record_root'],'path')     
    EX = explore(config)
    GL = gereralizedloss()
    DG = DensityGuided()

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

    from Explore.DroneLift import DroneLift
    DL = DroneLift(config)
    location, pose = DC.get_world_location_pose()
    DL.to_target_height(DC, location, pose, records, down=True)
    return GL, DG, DC, PT, EX, records, path_saver, camera

def run_fbe_with_density_guide(config):

    GL, DG, DC, PT, EX, records, path_saver, camera = init_explore(config)

    from Methods.FBE import FBE
    global_2D_map, global_X, global_Y, global_Z, global_bgr = \
        FBE(EX, DC,PT,camera,records,config,path_saver)
    
    records = DG.run_fbe_method(config, global_X, global_Y, global_Z, global_bgr, records, DC, GL, EX, camera)
    
    EX.calculate_all_path(records)
    DC.save_records(config, records)
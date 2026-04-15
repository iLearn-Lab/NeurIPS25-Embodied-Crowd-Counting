import warnings
import os
import shutil
import airsim
import numpy as np  
np.random.seed(42)
import random  
random.seed(42)

from Simulator.Simulator import Simulator
from Drone.Control import drone_controller
from Point_cloud.Point_cloud import point_cloud
from utils.saver import image_saver_plt
from Explore.OurExplore import modify_explore
from Perception.GeneralizedLoss import gereralizedloss
from Others.ValueMap.ValueMap import ValueMap
from Explore.DroneLift import DroneLift
from Others.IntuitionMap.gpt4o_integration import GPT
from Explore.DensityGuided import DensityGuided
from Others.IntuitionMap.qwen_integration import Qwen

warnings.filterwarnings("ignore")

def init_explore(config):
    
    SM = Simulator(config['Simulator'])
    change, route = SM.get_route()
    ground_truth = SM.get_ground_truth(change)
    SM.save_simulator(change,route,ground_truth,config['Record_root'],config['now'] + '/')    

    path = config['Record_root'] + config['now'] + '/explore_config/'
    if not os.path.exists(path):  
        os.makedirs(path)
    config_path = 'Configs/OurMethodConfig.yml'
    shutil.copy2(config_path, path + 'Config.yml')    

    DC = drone_controller(config['Drone'])
    DC.to_cv_mode()

    PT = point_cloud()
    path_saver = image_saver_plt(config['now'],config['Record_root'],'path')     

    VMP = ValueMap(config)
    DL = DroneLift(config)
    if config['OurExplore']['LLM_name'] == 'GPT4o':
        LLM = GPT(config)
    else:
        LLM = Qwen(config)
    EX = modify_explore(config, VMP, DL, LLM)
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
    records['low_location'] = []
    records['pose'] = []
    records['low_pose'] = []
    records['path'] = []

    location, pose = DC.get_world_location_pose()

    DL.to_target_height(DC, location, pose, records, down=True)    

    return GL, DG, DC, PT, EX, camera, records, path_saver

def run_our_method(config):
    GL, DG, DC, PT, EX, camera, records, path_saver = init_explore(config)

    global_X, global_Y, global_Z, global_bgr, up_global_x, up_global_y, up_global_z = EX.main_loop(DC,PT,camera,records,config,path_saver)
    records['location'] = records['low_location']
    records['pose'] = records['low_pose']
    
    records = DG.run_our_method(config, global_X, global_Y, global_Z, global_bgr, up_global_x, up_global_y, up_global_z, records, DC, GL, EX, camera)

    EX.calculate_all_path(records)
    DC.save_records(config, records)

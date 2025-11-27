import warnings
import os
import numpy as np  
np.random.seed(42)
import random  
random.seed(42)
import airsim

warnings.filterwarnings("ignore")

def init_explore(config):
    from Simulator.Simulator import Simulator

    SM = Simulator(config['Simulator'])
    change, route = SM.get_route()
    ground_truth = SM.get_ground_truth(change)
    SM.save_simulator(change,route,ground_truth,config['Record_root'],config['now'] + '/')    

    import shutil
    path = config['Record_root'] + config['now'] + '/explore_config/'
    if not os.path.exists(path):  
        os.makedirs(path)
    config_path = 'Methods/OurMethodConfig.yml'
    shutil.copy2(config_path, path + 'Config.yml')    

    from Drone.Control import drone_controller
    DC = drone_controller(config['Drone'])
    '''设置相机参数'''
    DC.to_cv_mode()
    location, pose = DC.get_world_location_pose()
    DC.client.simSetVehiclePose(                    
    airsim.Pose(
        airsim.Vector3r(location[0], location[1], location[2]),
        airsim.to_quaternion(0, 0, 0)
        ), 
        True)

    from Point_cloud.Point_cloud import point_cloud
    PT = point_cloud()
    from utils.saver import image_saver_plt
    path_saver = image_saver_plt(config['now'],config['Record_root'],'path')     

    from Explore.OurExplore import modify_explore
    from Perception.GeneralizedLoss import gereralizedloss
    from Others.ValueMap.ValueMap import ValueMap
    from Others.DroneLift.DroneLift import DroneLift
    from Others.IntuitionMap.gpt4o_integration import LLMJudge
    from Others.DensityGuided.DensityGuided import DensityGuided
    from Others.IntuitionMap.qwen_integration import Qwen

    '''使用到的组件'''
    VMP = ValueMap(config)
    DL = DroneLift(config)
    if config['LLM_name'] == 'GPT4o':
        LLM = LLMJudge(config)
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

    from Others.DroneLift.DroneLift import DroneLift
    DL = DroneLift(config)
    location, pose = DC.get_world_location_pose()
    # 升到目标高度
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

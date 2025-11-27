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
    shutil.copy2('Methods/FBEConfig.yml', path + 'Config.yml')    

    from Simulator.Simulator import Simulator
    SM = Simulator(config['Simulator'])
    change, route = SM.get_route()
    ground_truth = SM.get_ground_truth(change)
    SM.save_simulator(change,route,ground_truth,config['Record_root'],config['now'] + '/')    

    from Drone.Control import drone_controller
    DC = drone_controller(config['Drone'])
    '''设置相机参数'''
    DC.to_cv_mode()
    from Point_cloud.Point_cloud import point_cloud
    PT = point_cloud()
    from utils.saver import image_saver_plt
    path_saver = image_saver_plt(config['now'],config['Record_root'],'path')     
    from Explore.Explore import explore
    EX = explore(config['Explore'])

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

    from Others.DroneLift.DroneLift import DroneLift
    DL = DroneLift(config)
    location, pose = DC.get_world_location_pose()
    # 升到目标高度
    DL.to_target_height(DC, location, pose, records, down=True)
    return DC, PT, EX, records, path_saver, camera

def run_fbe(config):
    # 采集人群图像
    DC, PT, EX, records, path_saver, camera = init_explore(config)

    EX.FBE(DC,PT,camera,records,config,path_saver)

    EX.calculate_all_path(records)
    DC.save_records(config, records)
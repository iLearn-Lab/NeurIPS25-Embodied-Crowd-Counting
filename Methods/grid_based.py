import os
import numpy as np

def init_explore(config):
    import shutil
    path = config['Record_root'] + config['now'] + '/explore_config/'
    if not os.path.exists(path):  
        os.makedirs(path)
    shutil.copy2('Config.yml', path + 'Config.yml')    

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

def run_grid_based(config):

    DC, PT, EX, records, path_saver, camera = init_explore(config)

    # first construct global density map using greedy
    FBE_heights = config['Explore']['FBE_heights']
    voxel_size = config['Explore']['voxel_size']
    start_location, start_pose = DC.get_world_location_pose()
    navi_points = ()

    global_2D_map, global_X, global_Y, global_Z, global_bgr = EX.FBE(DC,PT,camera,records,config,path_saver) 
    global_point_cloud = EX.get_global_point_cloud(global_X, global_Y, global_Z)  
    
    for height in FBE_heights: 
        explored = EX.get_explored_at_height(height, global_point_cloud, start_location)
        explored = EX.mask_explored_at_height(global_2D_map, explored)
        navi_point = EX.get_global_2D_map_navi_point(explored, height, voxel_size)
        navi_points += (navi_point, )        
    navi_points = np.concatenate(navi_points,axis=0)

    from Explore.path_3D import path_planning_3d
    path_3d = path_planning_3d(global_point_cloud)
    voxelized_cloud = path_3d.voxelized_cloud

    navi_points = EX.cancel_occupied_navi_points(navi_points, voxelized_cloud)

    import open3d
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(voxelized_cloud)
    pcd2 = open3d.geometry.PointCloud()
    pcd2.points = open3d.utility.Vector3dVector(navi_points) 
    pcd2.colors = open3d.utility.Vector3dVector(np.zeros_like(navi_points))
    open3d.visualization.draw_geometries([pcd,pcd2])    
    
    from utils.saver import numpy_saver
    grid_navipoint_save = numpy_saver(config['now'],config['Record_root'],'grid_navipoint')
    grid_navipoint_save.save(navi_points, 'grid_navi_point.npy')

    navi_points = EX.array_to_dict(navi_points)

    for navi_point in navi_points:

        import airsim
        DC.client.simSetVehiclePose(
            airsim.Pose(airsim.Vector3r(navi_point[0], navi_point[1], navi_point[2]),
            airsim.to_quaternion(0, 0, 0)), True)
        
        DC.record_data(records, camera)


    # while len(navi_points) != 0:
        # navi_point = EX.reach_targets_greedy(navi_points, DC)
        # path = path_3d.path_planning_3d(navi_point, DC)
        # if path.shape[0] != 0:
            # EX.move_along_path_only_capture_when_reach(path,navi_point,camera,DC,records, sleep = 0)
        #     EX.save_path(path,records)
    # EX.calculate_all_path(records)
    DC.save_records(config, records)
import numpy as np
from Perception.GroundingDINO import GroundingDINO_detector
from utils.saver import image_saver, image_saver_plt, numpy_saver
from Point_cloud.Point_cloud import point_cloud
from Drone.Control import drone_controller    
from utils.video import load_video, load_numpy, get_img_path
from utils.flight import load_flight_data
from Explore.Target import target_distance_check, same_target_check
import math
import cv2
from tqdm import tqdm
import torch
import os
import pickle
import matplotlib.pyplot as plt
import json 
from Perception.GeneralizedLoss import gereralizedloss
from Explore.Explore import explore
from Count.Count import if_new_target, save_results, save_checks
from Explore.Explore import explore
from Point_cloud.Map_element import OBSTACLE, EXPLORED
from collections import defaultdict
import glob
import cv2


def load_results(record_path):
    ground_truth = np.load(record_path + 'simulator/ground_truth.npy')
    return ground_truth

def draw_gd_and_result(ground_truth, targets):

    plt.figure()
    plt.scatter(ground_truth[:, 1], ground_truth[:, 0], 
                c='black',s=1) 
    if targets.shape[0] !=0:
        plt.scatter(targets[:, 1], targets[:, 0], 
                c='red',s=0.2)
    plt.title('Target: %d Groundtruth: %d ' % (targets.shape[0], ground_truth.shape[0]))
    return plt.gca()

def get_tmp_path(path, st, ed, search_start):
    i = search_start
    ret = []
    st = np.array(st, dtype=np.int32)
    ed = np.array(ed, dtype=np.int32)
    assert tuple(path[i].tolist()) == tuple(st.tolist())
    i += 1
    while True:
        if tuple(path[i].tolist()) != tuple(ed.tolist()):
            ret.append(path[i])
        else: break
        i += 1
    i += 1
    return ret, i

path_count = 0

def visualize_path(config, path, DC:drone_controller):
    global path_count
    camera = [0]
    records = dict()
    records['visual'] = dict()
    for cam in camera:
        records['visual'][cam] = dict()
        records['visual'][cam]['bgr'] = []
        records['visual'][cam]['depth'] = []
    records['location'] = []
    records['pose'] = []
    records['path'] = []
    loc, _ = DC.get_world_location_pose()
    DC.move_and_capture_along_route_cv_mode(loc, path, camera,records,capture_rate=2)
    now = config['Count']['record_path'].split('/')[0]
    from utils.saver import image_saver_plt, image_saver
    image_saver = image_saver(now,config['Record_root'],f'vis_path/{path_count}')
    for i in range(len(records['visual'][0]['bgr'])):
        img = records['visual'][0]['bgr'][i]
        image_saver.save(img)
    path_count += 1

def count_GD(config):
    from Drone.Control import drone_controller
    DC = drone_controller(config['Drone'])
    DC.to_cv_mode()

    ground_truth = load_results(config['Record_root'] + config['Count']['record_path'])

    record_path = config['Record_root'] + config['Count']['record_path']
    cameras = config['Count']['camera']
    flight_data = load_flight_data(record_path + 'flight_data/0.json')
    location = flight_data['location']
    pose = flight_data['pose']
    distance_threshold = config['Count']['distance_threshold']
    same_target_threshold_horiz = config['Count']['same_target_threshold_horiz']
    same_target_threshold_verti = config['Count']['same_target_threshold_verti']

    data_path = 'Record/demo/path_points/'
    path_real_point = np.load(data_path + 'path.npy')
    path_point = path_real_point.astype(np.int32)
    search_start = 0

    GD = GroundingDINO_detector(config['GroundingDINO'])    
    DC = drone_controller(config['Drone'])
    PT = point_cloud() 
    EX = explore(config['Explore'])
    global_2D_map = dict()
    global_2D_map[OBSTACLE] = np.array([])
    global_2D_map[EXPLORED] = np.array([]) 
    global_threshold = (49, -184, 189, -3)
    # SAM2 = SAM2_detector(config['SAM2'])
    
    now = config['Count']['record_path'].split('/')[0]
    max_area = config['Count']['max_area']

    global_map_saver = image_saver_plt(now,config['Record_root'],'vid_path_figs')
    ground_truth_map_saver = image_saver_plt(now,config['Record_root'],'vid_ground_truth_figs')


    targets = ()
    targets_label = []
    all_checks = dict()

    for camera in cameras:
        all_checks[camera] = []

    bgrs = np.array([])
    depths = np.array([])
    detection = defaultdict(lambda: [])
    bgr_list = []
    depth_list = []

    for camera in cameras:

        bgr_list.append(load_video(record_path + 'bgr/0/camera_' + str(camera), if_RGB=False))
        depth_list.append(load_numpy(record_path + 'depth/0/camera_' + str(camera)))

        detection['boxes'].append(torch.load(record_path + 'detection/GD/camera_' + str(camera) + '/boxes.pth'))
        detection['logits'].append(torch.load(record_path + 'detection/GD/camera_' + str(camera) + '/logits.pth'))
        detection['phrases'].append(torch.load(record_path + 'detection/GD/camera_' + str(camera) + '/phrases.pth'))

        intrinsic_matrix = DC.get_intrinsic_matrix(camera)

    
    for j in tqdm(range(len(detection['boxes'][0])), desc="Counting: "):
        
        EX.history_location = []
        if j > 40: 
            global_2D_map[EXPLORED] = np.array([])
        target_num = 0
        

        X = dict(); Y = dict(); Z = dict()

        for i in range(len(cameras)):        
            bgr = bgr_list[i][j]
            depth = depth_list[i][j]
            loc = location[j]
            pos = pose[j]

            intrinsic_matrix = DC.get_intrinsic_matrix(cameras[i])
            pose_matrix = DC.get_pose_matrix(pos,cameras[i])                
            x,y,z = PT.get_point_clouds_from_depth(intrinsic_matrix, 
                                                pose_matrix,
                                                depth,
                                                camera = cameras[i])            
            x,y,z = PT.get_global_point_cloud(loc, x,y,z)
            
            X.update(x); Y.update(y); Z.update(z)


        map, change = PT.get_map_at_current_height(X,Y,Z,loc,threshold_draw=global_threshold)
        explore_range = config['Explore']['explore_range']
        explored_map = EX.get_explored_map(map, loc, change, explore_range)
        EX.update_global_map(explored_map, loc, change, global_2D_map)
        global_explored_map, change = EX.get_global_explored_map(loc,global_2D_map)
        global_map_image = EX.draw_global_map_and_frontier([],global_2D_map)


        if j > 40:
            # 大于40帧时，开始计数，否则只绘制地图 

            for i in range(len(cameras)):
                boxes = detection['boxes'][i][j]
                logits = detection['logits'][i][j]
                phrases = detection['phrases'][i][j]
                # gd_saver.save(detection['annotated_frame'])

                phrased_boxes = (GD.phrase_GD_boxes(boxes, depth))

                if len(phrased_boxes) != 0:
                    for k in range(len(phrased_boxes)): 
                        area = phrased_boxes[k]['area']
                        if area < max_area:      
                            distance, target_loc = GD.get_target_loc(phrased_boxes[k],loc,X,Y,Z,str(cameras[i]))
                            if if_new_target(distance, target_loc, distance_threshold,same_target_threshold_horiz,same_target_threshold_verti,targets): 

                                targets += (np.array(target_loc).reshape(1,-1),)

                                target_num += 1
            

        if j != 0:
            EX.history_location = [location[j-1], loc]
            tmp_path, nxt = get_tmp_path(path_point, location[j-1], loc, search_start)            
            visualize_path(config, np.array(path_real_point[search_start:nxt+1]), DC)
            search_start = nxt
        else:
            visualize_path(config, np.array([path_real_point[0]]), DC)
        
        global_map_image = EX.draw_global_map_and_frontier([],global_2D_map)
        if j != 0: 
            tmp_path = np.array(tmp_path)
            if tmp_path.shape[0] != 0: global_map_image.plot(tmp_path[:,1], tmp_path[:,0], c='green')
            history_location = np.array(EX.history_location)
            global_map_image.scatter(history_location[0,1], history_location[0,0], c='red', s=5)
            global_map_image.scatter(history_location[1,1], history_location[1,0], c='yellow', s=5)

        
        if j > 40:
            for target in targets:
                global_map_image.scatter(target[:, 1], target[:, 0], c = 'b', alpha=0.5, s=2)

        global_map_image.set_xlim(-3, 189)
        global_map_image.set_ylim(-184, 49)
        global_map_image.axis('off')
        for spine in global_map_image.spines.values():
            spine.set_visible(False)
        global_map_saver.save(global_map_image)
        
        # 基准图
        if j > 40: ground_truth_map_image = draw_gd_and_result(ground_truth, np.concatenate(targets,axis = 0))
        else: ground_truth_map_image = draw_gd_and_result(ground_truth, np.array([]))

        ground_truth_map_image.set_xlim(-3, 189)
        ground_truth_map_image.set_ylim(-184, 49)
        ground_truth_map_image.axis('off')
        for spine in ground_truth_map_image.spines.values():
            spine.set_visible(False)
        ground_truth_map_saver.save(ground_truth_map_image)

            
    targets = np.concatenate(targets,axis = 0)
    # generate_vid('Record/demo/vid_figs2/')
    # save_checks(all_checks,now,config['Record_root'])
    # save_results(targets, targets_label, record_path, 'GD')            
    return targets


def generate_vid(path):
# 获取所有PNG文件并按名称排序（确保顺序正确）
    all_files = glob.glob(os.path.join(path, "*"))  # 获取目录下所有文件
    img_files = sorted([f for f in all_files if f.lower().endswith(".png")])
    img_files = sorted(img_files, key=lambda x: int(''.join(filter(str.isdigit, x))))

    print(f"找到的 PNG 文件：{img_files}")
    # 读取第一张图片以获取尺寸信息
    img = cv2.imread(img_files[0])
    height, width, channels = img.shape

    # 设置视频参数
    output_filename = "output.mp4"
    fps = 30  # 帧率（Frames Per Second）
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 编码器（根据系统调整，如 'XVID' 对应 AVI）
    video = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

    # 计算每张图片需要写入的帧数（0.5秒 * 帧率）
    frame_count = int(fps * 0.8)

    # 遍历所有图片并写入视频
    for i, file in enumerate(img_files):
        img = cv2.imread(file)
        if i <= 40: img = cv2.putText(img, 'FBE Part', (200, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 10, cv2.LINE_AA)
        else: img = cv2.putText(img, 'Density Guide Part', (200, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 10, cv2.LINE_AA)
        # cv2.imshow('img', cv2.resize(img, (500, 500)))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        for _ in range(frame_count):
            video.write(img)

    # 释放资源
    video.release()
    print(f"视频已生成：{output_filename}")

def generate_vid2(vid_path_figs, vid_ground_truth_figs, vis_path):
    
    all_files = glob.glob(os.path.join(vid_path_figs, "*"))  # 获取目录下所有文件
    img_files = sorted([f for f in all_files if f.lower().endswith(".png")])
    path_img_files = sorted(img_files, key=lambda x: int(''.join(filter(str.isdigit, x))))

    all_files = glob.glob(os.path.join(vid_ground_truth_figs, "*"))  # 获取目录下所有文件
    img_files = sorted([f for f in all_files if f.lower().endswith(".png")])
    ground_img_files = sorted(img_files, key=lambda x: int(''.join(filter(str.isdigit, x))))

    output_filename = "output2.mp4"
    fps = 60 # 帧率（Frames Per Second）
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 编码器（根据系统调整，如 'XVID' 对应 AVI）
    video = cv2.VideoWriter(output_filename, fourcc, fps, (1200, 1200))

    for i in range(166):
        all_files = glob.glob(os.path.join(vis_path + str(i) + '/', "*"))  # 获取目录下所有文件
        img_files = sorted([f for f in all_files if f.lower().endswith(".png")])
        vis_img_files = sorted(img_files, key=lambda x: int(''.join(filter(str.isdigit, x))))
        if i<= 40: frame_count = int(fps * 0.2)
        else: frame_count = int(fps * 0.4)
        img2 = cv2.resize(cv2.imread(path_img_files[i]), (550,550))
        img3 = cv2.resize(cv2.imread(ground_img_files[i]), (550,550))
        for j in range(len(vis_img_files)):
            total_img = np.zeros((1200, 1200, 3), dtype=np.uint8).fill(128)
            img1 = cv2.resize(cv2.imread(vis_img_files[j]), (550,550)) 
            total_img[33:33+img1.shape[0], 33:33+img1.shape[1]] = img1
            total_img[33:33+img2.shape[0], 33*2+img1.shape[1]:33*2+img1.shape[1]+img2.shape[1]] = img2
            total_img[33*2+img1.shape[0]:33*2+img1.shape[0]+img3.shape[0], 33*2+img1.shape[1]:33*2+img1.shape[1]+img2.shape[1]] = img3
            for _ in range(frame_count):
                video.write(total_img)
    video.release()
    print(f"视频已生成：{output_filename}")



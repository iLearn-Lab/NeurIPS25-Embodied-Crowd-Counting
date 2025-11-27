import numpy as np
from Perception.GroundingDINO import GroundingDINO_detector
from utils.saver import image_saver, image_saver_plt, numpy_saver
from Point_cloud.Point_cloud import point_cloud
from Drone.Control import drone_controller    
from utils.video import load_video, load_numpy, get_img_path, load_partial_numpy, load_partial_video
from utils.flight import load_flight_data
from Explore.Target import target_distance_check, same_target_check

from tqdm import tqdm
import torch
import os
import pickle
import matplotlib.pyplot as plt
import json 
from Perception.GeneralizedLoss import gereralizedloss
from Explore.Explore import explore
import shutil

def load_results(config, method_name):

    record_path = config['Record_root'] + config['Count']['record_path']
    change = np.load(record_path + 'simulator/change.npy')
    ground_truth = np.load(record_path + 'simulator/ground_truth.npy')
    targets = np.load(record_path + 'result/' + method_name + '/count.npy')
    with open(record_path + '/result/' + method_name + '/label.json', 'r') as f:
        labels = json.load(f)
    return targets, labels, change, ground_truth

def target_detection_GD(config):

    record_path = config['Record_root'] + config['Count']['record_path']
    cameras = config['Count']['camera']
    prompt = config['Count']['detection_prompt']
    GD = GroundingDINO_detector(config['GroundingDINO'])
    for camera in cameras:
        camera = str(camera)
        bgr_list = load_video(record_path + 'bgr/0/camera_' + camera ,if_RGB=False)          
        detection, imgs = GD.inference_batch_images(bgr_list,16,prompt)
        path = record_path + 'detection/GD/camera_' + camera + '/'
        if not os.path.exists(path):
            os.makedirs(path)
        for key in detection.keys():
            save_path = path + key + '.pth'        
            torch.save(detection[key], save_path)
        now = config['Count']['record_path'].split('/')[0]
        saver = image_saver(now, config['Record_root'], 'detection/GD/camera_' + camera + '/')
        saver.save_list(imgs)
        
def partial_target_detection_GD(config, end_idx):

    record_path = config['Count']['record_path']
    cameras = config['Count']['camera']
    prompt = config['Count']['detection_prompt']
    GD = GroundingDINO_detector(config['GroundingDINO'])
    for camera in cameras:
        camera = str(camera)
        bgr_list = load_partial_video(record_path + 'bgr/0/camera_' + camera , end_idx,if_RGB=False)          
        detection, imgs = GD.inference_batch_images(bgr_list,16,prompt)
        path = record_path + 'detection/GD/camera_' + camera + '/'
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            shutil.rmtree(path)   
            os.makedirs(path)
        for key in detection.keys():
            save_path = path + key + '.pth'
            torch.save(detection[key], save_path)
        now = ''
        saver = numpy_saver(now, record_path, 'detection/GD/camera_' + camera + '/')
        saver.save_list(imgs)

def target_detection_GL(config):

    record_path = config['Record_root'] + config['Count']['record_path']
    cameras = config['Count']['camera']
    GL = gereralizedloss()
    for camera in cameras:
        camera = str(camera)
        bgr_list = load_video(record_path + 'bgr/0/camera_' + camera ,if_RGB=False)          
        density_maps = GL.inference_images(bgr_list)
        path = record_path + 'detection/GL/camera_' + camera + '/'
        if not os.path.exists(path):
            os.makedirs(path)

        now = config['Count']['record_path'].split('/')[0]
        saver = numpy_saver(now, config['Record_root'], 'detection/GL/camera_' + camera + '/')
        saver.save_list(density_maps)

def target_detection_PseCo(config):

    record_path = config['Record_root'] + config['Count']['record_path']
    cameras = config['Count']['camera']
    PSECO = PseCo()
    for camera in cameras:
        camera = str(camera)
        img_path_list = get_img_path(record_path + 'bgr/0/camera_' + camera)  
        detection = PSECO.inference(img_path_list)
        path = record_path + 'detection/PseCo/camera_' + camera + '/'
        if not os.path.exists(path):
            os.makedirs(path)

        now = config['Count']['record_path'].split('/')[0]
        saver = numpy_saver(now, config['Record_root'],'detection/PseCo/camera_' + camera)
        saver.save_list(detection)

def target_detection_MP(config):

    record_path = config['Record_root'] + config['Count']['record_path']
    cameras = config['Count']['camera']
    MP = MPcount(config)
    for camera in cameras:
        camera = str(camera)
        img_path = record_path + 'bgr/0/camera_' + camera
        density_maps, axs = MP.inference(img_path, if_draw = False)     
        path = record_path + 'detection/MPcount/camera_' + camera + '/'
        if not os.path.exists(path):
            os.makedirs(path)

        now = config['Count']['record_path'].split('/')[0]
        saver = numpy_saver(now, config['Record_root'], 'detection/MPcount/camera_' + camera + '/')       
        saver.save_list(density_maps)
        saver = image_saver_plt(now, config['Record_root'], 'detection/MPcount/imgs/camera_' + camera + '/')    
        saver.save_list(axs)    

def calculate_point_cloud(config):

    record_path = config['Record_root'] + config['Count']['record_path']
    depth_list = load_numpy(record_path + 'depth_numpy/0/camera_0')
    flight_data = load_flight_data(record_path + 'flight_data/0.json')
    location = flight_data['location']
    pose = flight_data['pose']

    DC = drone_controller(config['Drone'])
    PT = point_cloud()
    camera = 0
    intrinsic_matrix = DC.get_intrinsic_matrix(camera)

    Xs = dict()
    Ys = dict()
    Zs = dict()

    for i in tqdm(range(len(location)), desc="Calculating point cloud: "):

        loc = location[i]
        pos = pose[i]
        depth = depth_list[i]

        pose_matrix = DC.get_pose_matrix(pos, camera)
        X, Y, Z = PT.get_point_clouds_from_depth( 
                                intrinsic_matrix,
                                pose_matrix,
                                depth,
                                camera)
        X, Y, Z = PT.get_global_point_cloud(loc, X , Y, Z) 

        Xs[str(i)] = X
        Ys[str(i)] = Y
        Zs[str(i)] = Z
    
    path = record_path + 'point_cloud/0/camera_0/'
    if not os.path.exists(path):
        os.makedirs(path)    
    with open(path + 'X.pkl', 'wb') as f:  
        pickle.dump(Xs, f)
    with open(path + 'Y.pkl', 'wb') as f:  
        pickle.dump(Ys, f)  
    with open(path + 'Z.pkl', 'wb') as f:  
        pickle.dump(Zs, f)          

def count_GD(config):

    record_path = config['Record_root'] + config['Count']['record_path']
    cameras = config['Count']['camera']
    flight_data = load_flight_data(record_path + 'flight_data/0.json')
    location = flight_data['location']
    pose = flight_data['pose']
    distance_threshold = config['Count']['distance_threshold']
    same_target_threshold_horiz = config['Count']['same_target_threshold_horiz']
    same_target_threshold_verti = config['Count']['same_target_threshold_verti']

    GD = GroundingDINO_detector(config['GroundingDINO'])    
    DC = drone_controller(config['Drone'])
    PT = point_cloud() 
    # SAM2 = SAM2_detector(config['SAM2'])
    
    now = config['Count']['record_path'].split('/')[0]
    max_area = config['Count']['max_area']

    targets = ()
    targets_label = []
    all_checks = dict()

    for camera in cameras:
        all_checks[camera] = []
               
    for camera in cameras:
        bgr_list = load_video(record_path + 'bgr/0/camera_' + str(camera), if_RGB=False)
        depth_list = load_numpy(record_path + 'depth/0/camera_' + str(camera))
        detection = dict()
        detection['boxes'] = torch.load(record_path + 'detection/GD/camera_' + str(camera) + '/boxes.pth')
        detection['logits'] = torch.load(record_path + 'detection/GD/camera_' + str(camera) + '/logits.pth')
        detection['phrases'] = torch.load(record_path + 'detection/GD/camera_' + str(camera) + '/phrases.pth')

        intrinsic_matrix = DC.get_intrinsic_matrix(camera)
    
        for i in tqdm(range(len(detection['boxes'])), desc="Counting: "):

            checks = []
            target_num = 0
            
            bgr = bgr_list[i]
            depth = depth_list[i]
            loc = location[i]
            pos = pose[i]

            # get point cloud
            pose_matrix = DC.get_pose_matrix(pos, camera)
            X, Y, Z = PT.get_point_clouds_from_depth( 
                                    intrinsic_matrix,
                                    pose_matrix,
                                    depth,
                                    camera)
            X, Y, Z = PT.get_global_point_cloud(loc, X , Y, Z)

            boxes = detection['boxes'][i]
            logits = detection['logits'][i]
            phrases = detection['phrases'][i]
            # gd_saver.save(detection['annotated_frame'])

            phrased_boxes = GD.phrase_GD_boxes(boxes, depth)

            # use SAM2 
            # if len(phrased_boxes) != 0:
            #     _,_,bboxes = SAM2.get_points_labels_bboxes(phrased_boxes)
            #     masks = SAM2.inference_image_memory(bgr,bboxes=bboxes)
            #     for j in range(masks.shape[0]):
            #         if np.max(masks[j]) > 0:
            #             check = GD.draw_single_box_on_image(bgr,boxes[j],logits[j],phrases[j])                       
            #             distance, target_loc = get_target_loc_SAM2(masks[j],loc,PT,X,Y,Z,camera)
            #             if if_new_target(distance, target_loc, location, pose, i, fov, distance_threshold):
            #                 check = GD.draw_single_box_on_image(bgr,boxes[j],logits[j],phrases[j])
            #                 counts.append(check)
            #     count_saver.save_list(counts)

            # use gd center
            if len(phrased_boxes) != 0:
                for j in range(len(phrased_boxes)): 
                    area = phrased_boxes[j]['area']
                    if area < max_area:      
                        distance, target_loc = GD.get_target_loc(phrased_boxes[j],loc,X,Y,Z,str(camera))
                        if if_new_target(distance, target_loc, distance_threshold,same_target_threshold_horiz,same_target_threshold_verti,targets): 
                            check = GD.draw_single_box_on_image(bgr,boxes[j],logits[j],phrases[j])
                            checks.append(check)
                            targets += (np.array(target_loc).reshape(1,-1),)
                            targets_label.append('camera_' + str(camera) + '-' + str(i) + '-' + str(target_num))
                            target_num += 1
            all_checks[camera].append(checks)
            
    targets = np.concatenate(targets,axis = 0)
    # save_checks(all_checks,now,config['Record_root'])
    save_results(targets, targets_label, record_path, 'GD')            
    return targets

def partial_count_GD(config, end_idx):

    record_path = config['Count']['record_path']
    cameras = config['Count']['camera']
    flight_data = load_flight_data(record_path + 'flight_data/0.json')
    location = flight_data['location']
    pose = flight_data['pose']
    distance_threshold = config['Count']['distance_threshold']
    same_target_threshold_horiz = config['Count']['same_target_threshold_horiz']
    same_target_threshold_verti = config['Count']['same_target_threshold_verti']

    GD = GroundingDINO_detector(config['GroundingDINO'])    
    DC = drone_controller(config['Drone'])
    PT = point_cloud() 
    # SAM2 = SAM2_detector(config['SAM2'])
    
    now = config['Count']['record_path'].split('/')[0]
    max_area = config['Count']['max_area']

    targets = ()
    targets_label = []
    all_checks = dict()

    for camera in cameras:
        all_checks[camera] = []
               
    for camera in cameras:
        bgr_list = load_partial_video(record_path + 'bgr/0/camera_' + str(camera), end_idx, if_RGB=False)
        depth_list = load_partial_numpy(record_path + 'depth/0/camera_' + str(camera), end_idx)
        detection = dict()
        detection['boxes'] = torch.load(record_path + 'detection/GD/camera_' + str(camera) + '/boxes.pth')
        detection['logits'] = torch.load(record_path + 'detection/GD/camera_' + str(camera) + '/logits.pth')
        detection['phrases'] = torch.load(record_path + 'detection/GD/camera_' + str(camera) + '/phrases.pth')

        intrinsic_matrix = DC.get_intrinsic_matrix(camera)
    
        for i in tqdm(range(len(detection['boxes'])), desc="Counting: "):

            checks = []
            target_num = 0
            
            bgr = bgr_list[i]
            depth = depth_list[i]
            loc = location[i]
            pos = pose[i]

            # get point cloud
            pose_matrix = DC.get_pose_matrix(pos, camera)
            X, Y, Z = PT.get_point_clouds_from_depth( 
                                    intrinsic_matrix,
                                    pose_matrix,
                                    depth,
                                    camera)
            X, Y, Z = PT.get_global_point_cloud(loc, X , Y, Z)

            boxes = detection['boxes'][i]
            logits = detection['logits'][i]
            phrases = detection['phrases'][i]
            # gd_saver.save(detection['annotated_frame'])

            phrased_boxes = GD.phrase_GD_boxes(boxes, depth)

            # use SAM2 
            # if len(phrased_boxes) != 0:
            #     _,_,bboxes = SAM2.get_points_labels_bboxes(phrased_boxes)
            #     masks = SAM2.inference_image_memory(bgr,bboxes=bboxes)
            #     for j in range(masks.shape[0]):
            #         if np.max(masks[j]) > 0:
            #             check = GD.draw_single_box_on_image(bgr,boxes[j],logits[j],phrases[j])                       
            #             distance, target_loc = get_target_loc_SAM2(masks[j],loc,PT,X,Y,Z,camera)
            #             if if_new_target(distance, target_loc, location, pose, i, fov, distance_threshold):
            #                 check = GD.draw_single_box_on_image(bgr,boxes[j],logits[j],phrases[j])
            #                 counts.append(check)
            #     count_saver.save_list(counts)

            # use gd center
            if len(phrased_boxes) != 0:
                for j in range(len(phrased_boxes)): 
                    area = phrased_boxes[j]['area']
                    if area < max_area:      
                        distance, target_loc = GD.get_target_loc(phrased_boxes[j],loc,X,Y,Z,str(camera))
                        if if_new_target(distance, target_loc, distance_threshold,same_target_threshold_horiz,same_target_threshold_verti,targets): 
                            check = GD.draw_single_box_on_image(bgr,boxes[j],logits[j],phrases[j])
                            checks.append(check)
                            targets += (np.array(target_loc).reshape(1,-1),)
                            targets_label.append('camera_' + str(camera) + '-' + str(i) + '-' + str(target_num))
                            target_num += 1
            all_checks[camera].append(checks)
            
    targets = np.concatenate(targets,axis = 0)
    # save_checks(all_checks,now,config['Record_root'])
    save_results(targets, targets_label, record_path, 'GD')            
    return targets

def count_GL(config):

    record_path = config['Record_root'] + config['Count']['record_path']
    cameras = config['Count']['camera']
    flight_data = load_flight_data(record_path + 'flight_data/0.json')
    location = flight_data['location']
    pose = flight_data['pose']
    distance_threshold = config['Count']['distance_threshold']
    density_threshold = config['Count']['density_threshold']
    same_target_threshold_horiz = config['Count']['same_target_threshold_horiz']
    same_target_threshold_verti = config['Count']['same_target_threshold_verti']
    
    DC = drone_controller(config['Drone'])
    PT = point_cloud() 
    # SAM2 = SAM2_detector(config['SAM2'])
    
    now = config['Count']['record_path'].split('/')[0]
    max_area = config['Count']['max_area']

    targets = ()
    targets_label = []
    all_checks = dict()

    for camera in cameras:
        all_checks[camera] = []
               
    for camera in cameras:
        bgr_list = load_video(record_path + 'bgr/0/camera_' + str(camera), if_RGB=False)
        depth_list = load_numpy(record_path + 'depth/0/camera_' + str(camera))
        density_map_list = load_numpy(record_path + 'detection/GL/camera_' + str(camera) + '/0/')

        intrinsic_matrix = DC.get_intrinsic_matrix(camera)
    
        for i in tqdm(range(len(density_map_list)), desc="Counting: "):

            checks = []
            target_num = 0
            
            bgr = bgr_list[i]
            depth = depth_list[i]
            loc = location[i]
            pos = pose[i]

            # get point cloud
            pose_matrix = DC.get_pose_matrix(pos, camera)
            X, Y, Z = PT.get_point_clouds_from_depth( 
                                    intrinsic_matrix,
                                    pose_matrix,
                                    depth,
                                    camera)
            X, Y, Z = PT.get_global_point_cloud(loc, X , Y, Z)

            density_map = density_map_list[i]
            idx = np.where(density_map >= density_threshold)
            if idx[0].shape[0] == 0:
                continue
            idx_0 = (idx[0] / density_map.shape[0] * X[str(camera)].shape[0]).astype(int)
            idx_1 = (idx[1] / density_map.shape[1] * X[str(camera)].shape[1]).astype(int) 

            x = X[str(camera)][idx_0.reshape(1,-1),idx_1.reshape(1,-1)].reshape(-1,1)
            y = Y[str(camera)][idx_0.reshape(1,-1),idx_1.reshape(1,-1)].reshape(-1,1)
            z = Z[str(camera)][idx_0.reshape(1,-1),idx_1.reshape(1,-1)].reshape(-1,1)
            idx1 = ~np.isnan(x)
            idx2 = ~np.isnan(y)
            idx3 = ~np.isnan(z)
            idx = idx1 & idx2 & idx3

            x = x[idx].reshape(-1,1)
            y = y[idx].reshape(-1,1)  
            z = z[idx].reshape(-1,1)          
            target_locs = np.concatenate((x,y,z),axis = 1)
            for target_loc in target_locs:
                target_dist = np.linalg.norm(np.array(loc) - np.array(target_loc))
                if if_new_target(target_dist, target_loc, distance_threshold, same_target_threshold_horiz, same_target_threshold_verti, targets):
                    targets += (np.array(target_loc).reshape(1,-1),)
                    targets_label.append('camera_' + str(camera) + '-' + str(i) + '-' + str(target_num))
                    target_num += 1
                    
    targets = np.concatenate(targets,axis = 0)
    save_results(targets, targets_label, record_path, 'GL')            
    return targets

def count_MP(config):

    record_path = config['Record_root'] + config['Count']['record_path']
    cameras = config['Count']['camera']
    flight_data = load_flight_data(record_path + 'flight_data/0.json')
    location = flight_data['location']
    pose = flight_data['pose']
    distance_threshold = config['Count']['distance_threshold']
    same_target_threshold_horiz = config['Count']['same_target_threshold_horiz']
    same_target_threshold_verti = config['Count']['same_target_threshold_verti']
    
    DC = drone_controller(config['Drone'])
    PT = point_cloud() 
    # SAM2 = SAM2_detector(config['SAM2'])
    
    now = config['Count']['record_path'].split('/')[0]
    max_area = config['Count']['max_area']

    targets = ()
    targets_label = []
    all_checks = dict()

    for camera in cameras:
        all_checks[camera] = []
               
    for camera in cameras:
        bgr_list = load_video(record_path + 'bgr/0/camera_' + str(camera), if_RGB=False)
        depth_list = load_numpy(record_path + 'depth/0/camera_' + str(camera))
        density_map_list = load_numpy(record_path + 'detection/MPcount/camera_' + str(camera) + '/0/')

        intrinsic_matrix = DC.get_intrinsic_matrix(camera)
    
        for i in tqdm(range(len(density_map_list)), desc="Counting: "):

            checks = []
            target_num = 0
            
            bgr = bgr_list[i]
            depth = depth_list[i]
            loc = location[i]
            pos = pose[i]

            # get point cloud
            pose_matrix = DC.get_pose_matrix(pos, camera)
            X, Y, Z = PT.get_point_clouds_from_depth( 
                                    intrinsic_matrix,
                                    pose_matrix,
                                    depth,
                                    camera)
            X, Y, Z = PT.get_global_point_cloud(loc, X , Y, Z)

            density_map = density_map_list[i]
            idx = np.where(density_map == 1)
            if idx[0].shape[0] == 0:
                continue
            idx_0 = (idx[0] / density_map.shape[0] * X[str(camera)].shape[0]).astype(int)
            idx_1 = (idx[1] / density_map.shape[1] * X[str(camera)].shape[1]).astype(int) 

            x = X[str(camera)][idx_0.reshape(1,-1),idx_1.reshape(1,-1)].reshape(-1,1)
            y = Y[str(camera)][idx_0.reshape(1,-1),idx_1.reshape(1,-1)].reshape(-1,1)
            z = Z[str(camera)][idx_0.reshape(1,-1),idx_1.reshape(1,-1)].reshape(-1,1)
            idx1 = ~np.isnan(x)
            idx2 = ~np.isnan(y)
            idx3 = ~np.isnan(z)
            idx = idx1 & idx2 & idx3

            x = x[idx].reshape(-1,1)
            y = y[idx].reshape(-1,1)  
            z = z[idx].reshape(-1,1)          
            target_locs = np.concatenate((x,y,z),axis = 1)
            for target_loc in target_locs:
                target_dist = np.linalg.norm(np.array(loc) - np.array(target_loc))
                if if_new_target(target_dist, target_loc, distance_threshold, same_target_threshold_horiz, same_target_threshold_verti, targets):
                    targets += (np.array(target_loc).reshape(1,-1),)
                    targets_label.append('camera_' + str(camera) + '-' + str(i) + '-' + str(target_num))
                    target_num += 1
                    
    targets = np.concatenate(targets,axis = 0)
    save_results(targets, targets_label, record_path, 'MPcount')            
    return targets

def save_checks(all_checks : dict, now, record_root):
    for key in all_checks.keys():
        item = all_checks[key]
        count_saver = image_saver(now ,record_root,'count/camera_'+str(key))
        for checks in item:
            count_saver.save_list(checks)

def save_results(targets, targets_label, record_path, method_name):

    path = record_path + 'result/' + method_name + '/'
    if not os.path.exists(path):
        os.makedirs(path)
    np.save(path + 'count.npy', targets)
    with open(path + 'label.json', "w") as file:  
        json.dump(targets_label, file)    

def draw_gd_and_result(ground_truth, targets, config):

    record_path = config['Record_root'] + config['Count']['record_path']
    path = record_path + 'visualization/'
    if not os.path.exists(path):
        os.makedirs(path)
    plt.figure()
    plt.scatter(ground_truth[:, 1], ground_truth[:, 0], 
                c='black',s=1) 
    plt.scatter(targets[:, 1], targets[:, 0], 
                c='red',s=0.2)
    MAPE = abs(ground_truth.shape[0]-targets.shape[0]) / ground_truth.shape[0]
    plt.title('Target: %d Groundtruth: %d MAPE: %.5f' % (targets.shape[0], ground_truth.shape[0], MAPE))
    return plt.gca()
    
def draw_targets(ground_truth, targets, targets_label):

    fig, ax = plt.subplots()
    # scatter = ax.scatter(ground_truth[:, 1], ground_truth[:, 0], picker=True, s = 1, c = 'black')
    scatter = ax.scatter(targets[:, 1], targets[:, 0], picker=True, s = 0.5, c = 'red')
    labels = targets_label

    def onpick(event):  
        ind = event.ind  
        points = scatter.get_offsets()[ind]  
        label = labels[ind[0]] if len(ind) > 0 else None  
        plt.gcf().canvas.manager.set_window_title(f'Point: {label}\nX: {points[0][0]:.2f}, Y: {points[0][1]:.2f}')  
        print(f'You clicked on point: {label}\nX: {points[0][0]:.2f}, Y: {points[0][1]:.2f}')

    fig.canvas.mpl_connect('pick_event', onpick)   
    plt.show()
    
def get_target_loc_SAM2(mask, current_loc, PT : point_cloud, X, Y, Z, camera):

    x,y,z = PT.get_point_cloud_from_mask(X[camera],Y[camera],Z[camera],mask)
    point_cloud = PT.convert_to_open3d(x,y,z)
    current_loc_np = np.array(current_loc).reshape(1,-1)
    distances = np.linalg.norm(point_cloud[:,0:2] - current_loc_np[:,0:2], axis=1)
    idx = np.where(distances == min(distances))[0]
    distance = distances[idx]
    target_loc = tuple(np.squeeze(point_cloud[idx,:]))
    return distance, target_loc

def if_new_target(target_dist, target_loc, distance_threshold, same_target_threshold_horiz, same_target_threshold_verti, targets):   

    in_distance = target_dist < distance_threshold
    same_target = same_target_check(target_loc, targets, same_target_threshold_horiz = same_target_threshold_horiz, same_target_threshold_verti=same_target_threshold_verti)
    # if same_target:
    if in_distance and same_target:
        return True
    return False

def qualification(ground_truth, targets):

    for i in range(ground_truth.shape[0]):

        gt = ground_truth[i,:]
        residual = targets - gt
        dist = np.linalg.norm(residual, axis=1)

def select_point_cloud_height(point_cloud, height_low, height_high, labels = None):

    condition1 = point_cloud[:,-1] >= height_low
    condition2 = point_cloud[:,-1] <= height_high
    idx = np.where(np.logical_and(condition1, condition2))[0]
    if labels != None:    
        labels_temp = np.array(labels)
        return point_cloud[idx,:], labels_temp[idx]
    else:
        return point_cloud[idx,:]
    
def match_target_with_gd(ground_truth, targets, cancel_count_range):

    new_targets = ()
    for target in targets:
        dist = np.linalg.norm(target[0:2] - ground_truth[:,0:2], axis = -1)
        idx = np.where(dist < cancel_count_range)[0]
        if idx.shape[0] != 0:
            new_targets += (target.reshape(1,-1), )
    new_targets = np.concatenate(new_targets,axis=0)
    return new_targets

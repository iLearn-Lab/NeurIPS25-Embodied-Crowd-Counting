import airsim
import numpy as np
import math
import cv2
import time
import threading
import json
from tqdm import tqdm
from Point_cloud.Point_cloud import point_cloud
from Point_cloud.Map_element import (
    UNKNOWN,
    OBSTACLE,
    EXPLORED,
    FRONTIER)
import os

class drone_controller:

    def __init__(self, config):

        self.config = config
        self.client = airsim.MultirotorClient(ip=config['ip'])
        self.image_client = airsim.MultirotorClient(ip=config['ip'])
        self.location_client = airsim.MultirotorClient(ip=config['ip'])

        self.vx = 0
        self.vy = 0
        self.vz = 0
        self.yaw_rate = 0

        self.init_local_location_pose((0,0,0),(0,0,0))
        self.prepare_intrinsic_matrixs()
    
    def to_cv_mode(self):
        
        '''front'''
        camera_pose = airsim.Pose(airsim.Vector3r(0, 0, 0), 
                                  airsim.to_quaternion(0, 0, math.radians(0))) #radians
        self.client.simSetCameraPose("0", camera_pose)
        '''left'''
        camera_pose = airsim.Pose(airsim.Vector3r(0, 0, 0), 
                                  airsim.to_quaternion(0, 0, math.radians(-90))) #radians
        self.client.simSetCameraPose("1", camera_pose)
        '''fix backward camera'''
        camera_pose = airsim.Pose(airsim.Vector3r(0, 0, 0), 
                                  airsim.to_quaternion(0, 0, math.radians(-180))) #radians
        self.client.simSetCameraPose("4", camera_pose)
        camera_pose = airsim.Pose(airsim.Vector3r(0, 0, 0), 
                                  airsim.to_quaternion(0, 0, math.radians(90))) #radians
        self.client.simSetCameraPose("2", camera_pose)                
        
    def get_world_location_pose(self):
        '''
            system_loc : x_val -> x
                         y_val -> y
                         z_val -> z
        '''
        state = self.location_client.simGetGroundTruthKinematics()
        system_location = state.position
        (pitch, roll, yaw) = airsim.to_eularian_angles(state.orientation)        
        location = (system_location.x_val,
                    system_location.y_val,
                    system_location.z_val)
        pose = (pitch * 180 / math.pi,
                roll * 180 / math.pi,
                yaw * 180 / math.pi)
        return location, pose

    def init_local_location_pose(self, 
                                 init_location : tuple, 
                                 init_pose : tuple):

        world_location, world_pose = self.get_world_location_pose()
        self.location_bias = tuple(a - b for a, b in zip(
            init_location, 
            world_location)
        )
        self.pose_bias = tuple(a - b for a, b in zip(
            init_pose, 
            world_pose)
        )
        
    def get_single_bgr_image(self, camera=0):

        responses = self.image_client.simGetImages(
            [airsim.ImageRequest(camera, airsim.ImageType.Scene, pixels_as_float=False, compress=False)]
        )

        img_1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
        img_bgr = img_1d.reshape(responses[0].height, responses[0].width, 3)
        return img_bgr
    
    def get_single_segment_image(self, camera=0):

        responses = self.image_client.simGetImages(
            [airsim.ImageRequest(camera, airsim.ImageType.Segmentation, pixels_as_float=False, compress=False)]
            )

        img_1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
        img_rgb = img_1d.reshape(responses[0].height, responses[0].width, 3)
        img_pixel = img_1d.reshape(-1,3)
        
        types = np.unique(img_pixel, axis=0)
        idx = []

        for item in types:
            idx.append(
                np.where(
                    (img_rgb[:,:,0] == item[0]) & 
                    (img_rgb[:,:,1] == item[1]) & 
                    (img_rgb[:,:,2] == item[2])
                )
            )
        
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        return types, idx, img_bgr

    def get_single_depth_image(self, camera=0):

        responses = self.image_client.simGetImages([
            airsim.ImageRequest(camera, airsim.ImageType.DepthPlanar, pixels_as_float=True, compress=False)])
        
        img_depth_planar = np.array(responses[0].image_data_float).reshape(responses[0].height, responses[0].width)

        # threshold
        img_depth_planar[img_depth_planar > self.config['depth_threshold']] = math.inf
        return img_depth_planar
    
    def get_bgr_and_depth(self, camera : list):

        bgr_requests = []
        depth_requests = []
        for cam in camera:
            bgr_requests.append(
                airsim.ImageRequest(cam,
                                    airsim.ImageType.Scene,
                                    pixels_as_float=False,
                                    compress=False)
                                    )
            depth_requests.append(
                airsim.ImageRequest(cam,
                                    airsim.ImageType.DepthPlanar,
                                    pixels_as_float=True,
                                    compress=False)
                                    )            
        bgr_responses = self.image_client.simGetImages(bgr_requests)
        depth_responses = self.image_client.simGetImages(depth_requests)
        img_bgrs = []
        img_depth_planars = []
        for i in range(len(bgr_responses)):
            img_1d = np.frombuffer(bgr_responses[i].image_data_uint8, dtype=np.uint8)
            img_bgr = img_1d.reshape(bgr_responses[i].height, bgr_responses[i].width, 3)
            img_bgrs.append(img_bgr)
            img_depth_planar = np.array(depth_responses[i].image_data_float).reshape(depth_responses[i].height, depth_responses[i].width)
            img_depth_planar[img_depth_planar > self.config['depth_threshold']] = math.inf
            img_depth_planars.append(img_depth_planar)

        return img_bgrs, img_depth_planars
 
    def get_images(self , camera=0):
        '''
            camera = 
            0: front
            1: left
            2: right
            3: down
            4: back
        '''
        scene_bgr = self.get_single_bgr_image(camera=camera)
        depth = self.get_single_depth_image(camera=camera)
        segment = self.get_single_segment_image(camera=camera)

        data = {
            'scene_bgr' : scene_bgr,
            'depth' : depth,
            'segment' : segment
        }
        return data
    
    def get_fov(self, camera):
        fov = self.client.simGetCameraInfo(camera, external=False).fov
        return fov    
        
    def prepare_intrinsic_matrixs(self):
        self.intrinsic_matrixs = dict()
        for i in range(0,5):
            intrinsic_matrix = np.zeros([3, 3])
            fov = self.get_fov(i)
            request = [airsim.ImageRequest(i, airsim.ImageType.DepthPlanar, False, False)]
            responses = self.client.simGetImages(request)
            img_width = responses[0].width
            img_height = responses[0].height
            intrinsic_matrix[0, 0] = img_width / 2 / math.tan(math.radians(fov / 2))
            intrinsic_matrix[1, 1] = img_width / 2 / math.tan(math.radians(fov / 2))
            intrinsic_matrix[0, 2] = img_width / 2
            intrinsic_matrix[1, 2] = img_height / 2
            intrinsic_matrix[2, 2] = 1
            self.intrinsic_matrixs[i] = intrinsic_matrix

    def get_intrinsic_matrix(self, camera):

        return self.intrinsic_matrixs[camera]
        
    def get_pose_matrix(self, pose : tuple, camera : int):

        pitch = pose[0]
        roll = pose[1]
        yaw = pose[2]

        if camera == 1:
            yaw = yaw - 90
        elif camera == 2:
            yaw = yaw + 90
        elif camera == 4:
            yaw = yaw + 180
        elif camera == 3:
            roll = roll - 90       

        pose_matrix1 = np.zeros([3, 3])
        pose_matrix1[0,0] = 1
        pose_matrix1[1,1] = math.cos(math.radians(-roll))
        pose_matrix1[1,2] = -math.sin(math.radians(-roll))
        pose_matrix1[2,1] = math.sin(math.radians(-roll))
        pose_matrix1[2,2] = math.cos(math.radians(-roll))

        pose_matrix2 = np.zeros([3, 3])
        pose_matrix2[0,0] = math.cos(math.radians(-pitch))
        pose_matrix2[0,2] = math.sin(math.radians(-pitch))
        pose_matrix2[1,1] = 1
        pose_matrix2[2,0] = -math.sin(math.radians(-pitch))
        pose_matrix2[2,2] = math.cos(math.radians(-pitch))

        pose_matrix3 = np.zeros([3, 3])
        pose_matrix3[0,0] = math.cos(math.radians(-yaw))
        pose_matrix3[0,1] = -math.sin(math.radians(-yaw))
        pose_matrix3[1,0] = math.sin(math.radians(-yaw))
        pose_matrix3[1,1] = math.cos(math.radians(-yaw))
        pose_matrix3[2,2] = 1

        return pose_matrix3 @ pose_matrix1 @ pose_matrix2
        
    def dump_flight_data(self,location,pose):

        flight_data = {
            'location': location,
            'pose': pose
        }
        
        json_data = json.dumps(flight_data)

        return json_data
    
    def record_data(self, results, camera : list):

        img_bgrs, depths = self.get_bgr_and_depth(camera)
        for i in range(len(camera)):
            img_bgr = img_bgrs[i]
            depth = depths[i]
            results['visual'][camera[i]]['bgr'].append(img_bgr)
            results['visual'][camera[i]]['depth'].append(depth)
        location, pose = self.get_world_location_pose()
        results['location'].append(location)
        results['pose'].append(pose)
        return img_bgrs, depths
    
    def save_records(self, config, records):

        from utils.saver import image_saver, json_saver, numpy_saver
        bgr_saver = image_saver(config['now'], config['Record_root'], 'bgr')
        for key in records['visual'].keys():
            bgr_saver.save_list(records['visual'][key]['bgr'],'camera_' + str(key), if_next=False)
        bgr_saver.next()

        depth_saver = numpy_saver(config['now'], config['Record_root'], 'depth')
        for key in records['visual'].keys():
            depth_saver.save_list(records['visual'][key]['depth'],'camera_' + str(key), if_next=False)
        depth_saver.next()

        flight_data = self.dump_flight_data(records['location'], records['pose'])
        JS = json_saver(config['now'], config['Record_root'], 'flight_data')
        JS.save(flight_data)

        cost = records['cost']
        path = config['Record_root'] + config['now'] + '/cost/'
        if not os.path.exists(path):
            os.makedirs(path)        
        with open(path + 'cost.txt', 'w') as file:  
            file.write(str(cost))  

        all_path = ()
        navi_points = ()
        for item in records['path']:
            all_path += (item, )
            navi_points += (item[-1].reshape(1,-1),)
        all_path = np.concatenate(all_path,axis=0)
        navi_points = np.concatenate(navi_points,axis=0)
        path = config['Record_root'] + config['now'] + '/path_points/'
        if not os.path.exists(path):
            os.makedirs(path)        
        np.save(path + 'path.npy', all_path) 
        np.save(path + 'navi_points.npy', navi_points)

    def _ensure_3d_target(self, target_point, current_height):

        target_point = np.array(target_point).squeeze()
        if target_point.shape[-1] != 3:
            target_point = np.array([target_point[0], target_point[1], current_height])
        return target_point

    def _shortest_yaw_delta(self, current_yaw, target_yaw):

        delta = target_yaw - current_yaw
        alternative = delta - 360 if delta > 0 else delta + 360
        return delta if abs(delta) < abs(alternative) else alternative

    def _set_vehicle_pose(self, location, yaw):

        self.client.simSetVehiclePose(
            airsim.Pose(
                airsim.Vector3r(location[0], location[1], location[2]),
                airsim.to_quaternion(0, 0, math.radians(yaw))
            ),
            True
        )
    
    def get_to_location_and_capture_vector(self, navi_point, camera, records, capture_rate = 0.5, if_capture = True, sleep = 0, final_yaw = None):

        location, pose = self.get_world_location_pose()
        location = np.array(location)
        yaw = pose[2]
        navi_point = self._ensure_3d_target(navi_point, location[2])
        residual = navi_point - location
        
        # Rotate toward the target first.
        new_pose = math.atan2(residual[1], residual[0]) * 180 / math.pi
        move_degree = self._shortest_yaw_delta(yaw, new_pose)
        yaw_rate = self.config['yaw_rate']
        yaw_unit = yaw_rate * capture_rate
        direction = 1 if move_degree > 0 else -1
        yaw_unit = yaw_unit * direction
        rotated = 0
        current_yaw = yaw
        while rotated < abs(move_degree):
            if rotated + abs(yaw_unit) > abs(move_degree):
                step = (abs(move_degree) - rotated) * direction
            else:
                step = yaw_unit
            rotated += abs(yaw_unit)
            current_yaw += step
            self._set_vehicle_pose(location, current_yaw)
            if if_capture:
                self.record_data(records, camera)
            time.sleep(sleep)
         
        # Move in a straight line toward the target.
        distance = np.linalg.norm(residual)
        direction_vector = np.array([residual[0], residual[1], residual[2]]) / distance 
        speed = self.config['speed'] 
        move_unit = direction_vector * speed * capture_rate
        moved_dist = 0
        while moved_dist < distance:          
            if moved_dist + speed * capture_rate > distance:
                move = navi_point - location
            else:
                move = move_unit
            moved_dist += speed * capture_rate
            location[0] += move[0]
            location[1] += move[1]
            location[2] += move[2]
            self._set_vehicle_pose(location, new_pose)
            if if_capture:
                self.record_data(records, camera)        
            time.sleep(sleep)

    def get_to_location_only_reach_capture_vector(self, cord, camera, records, capture_rate = 0.5, if_capture = True, sleep = 0):

        location, pose = self.get_world_location_pose()
        location = np.array(location)
        yaw = pose[2]
        cord = self._ensure_3d_target(cord, location[2])
        residual = cord - location
        
        # Rotate toward the destination before moving.
        new_pose = math.atan2(residual[1], residual[0]) * 180 / math.pi
        move_degree = self._shortest_yaw_delta(yaw, new_pose)
        yaw_rate = self.config['yaw_rate']
        yaw_unit = yaw_rate * capture_rate
        direction = 1 if move_degree > 0 else -1
        yaw_unit = yaw_unit * direction
        rotated = 0
        current_yaw = yaw
        while rotated < abs(move_degree):
            if rotated + abs(yaw_unit) > abs(move_degree):
                step = (abs(move_degree) - rotated) * direction
            else:
                step = yaw_unit
            rotated += abs(yaw_unit)
            current_yaw += step
            self._set_vehicle_pose(location, current_yaw)
            time.sleep(sleep)
         
        # Move until the destination is reached.
        distance = np.linalg.norm(residual)
        direction_vector = np.array([residual[0], residual[1], residual[2]]) / distance 
        speed = self.config['speed'] 
        move_unit = direction_vector * speed * capture_rate
        moved_dist = 0
        while moved_dist < distance:          
            if moved_dist + speed * capture_rate > distance:
                move = cord - location
            else:
                move = move_unit
            moved_dist += speed * capture_rate
            location[0] += move[0]
            location[1] += move[1]
            location[2] += move[2]
            self._set_vehicle_pose(location, new_pose)
            time.sleep(sleep)
        if if_capture:
            self.record_data(records, camera)  
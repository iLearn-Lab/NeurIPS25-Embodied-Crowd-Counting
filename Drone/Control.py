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
        
    def takeoff(self):
        # get control
        self.client.enableApiControl(True)   
        # unlock
        self.client.armDisarm(True)
        # takeoff          
        self.client.takeoffAsync().join() 
        points = [airsim.Vector3r(5, 0, -3),
          airsim.Vector3r(5, 8, -3),
          airsim.Vector3r(8, 12, -3),
          airsim.Vector3r(4, 9, -3)]
        self.client.moveToZAsync(self.config['takeoff_height'], self.config['takeoff_speed']).join()
        # self.client.moveToPositionAsync(0, 0, -9, self.config['takeoff_speed']).join()  
        # self.client.moveOnPathAsync(points, self.config['takeoff_speed']).join()
        # self.get_to_location(location = (0,0,self.config['takeoff_height']))

        # init location
        # self.init_location(self.location[-1])

        time.sleep(2)

    def land(self):
        self.client.landAsync().join()
        self.client.armDisarm(False)
        self.client.enableApiControl(False)

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

        # 通过对DepthPlanar的处理，手动生成DepthVis深度可视图
        # 1. 获取DepthPlanar图片
        responses = self.image_client.simGetImages([
            airsim.ImageRequest(camera, airsim.ImageType.DepthPlanar, pixels_as_float=True, compress=False)])
        
        img_depth_planar = np.array(responses[0].image_data_float).reshape(responses[0].height, responses[0].width)

        # 2. 距离100米以上的像素设为白色（此距离阈值可以根据自己的需求来更改）
        # img_depth_planar = img_depth_planar/threshold
        # img_depth_vis[img_depth_vis > 1] = 1.
        # # 3. 转换为整形
        # img_depth_vis = (img_depth_vis*255).astype(np.uint8)
        # 4. 保存为文件

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
    
    def speed_change_vector(self, current_speed, changed_speed, 
                     acceleration, location : tuple, interval = 0.1):
                
        if current_speed == changed_speed:
            return
        
        # calculate target vector
        distance = np.linalg.norm([location[0], location[1], location[2]])
        direction_vector = np.array([location[0], location[1], location[2]]) / distance

        #linear
        duration = abs(changed_speed - current_speed) / acceleration       
        times = np.linspace(0,duration,int(duration/interval+1))

        if changed_speed < current_speed:
            acceleration = acceleration * -1
        for i in range(len(times)):
            speed = times[i] * acceleration + current_speed
            speeds = direction_vector * speed
            self.vx = speeds[0]
            self.vy = speeds[1]
            self.vz = speeds[2]
            self.client.moveByVelocityBodyFrameAsync(vx = self.vx, 
                                                     vy = self.vy, 
                                                     vz = self.vz, duration = interval).join()
            
        self.client.moveByVelocityBodyFrameAsync(vx = self.vx, 
                                                 vy = self.vy, 
                                                 vz = self.vz, duration = math.inf)
        
        return abs(current_speed * times[-1] + \
                   1 / 2 * acceleration * times[-1]**2)

    def start_move_vector(self, location : tuple):

        current_speed = np.linalg.norm([self.vx, self.vy, self.vz])    
        self.speed_change_vector(current_speed = current_speed, 
                                 changed_speed = self.config['speed'],
                                 acceleration = self.config['acceleration'],
                                 location = location)
    
    def stop_move_vector(self, location : tuple):

        current_speed = np.linalg.norm([self.vx, self.vy, self.vz])    
        self.speed_change_vector(current_speed = current_speed, 
                                 changed_speed = 0,
                                 acceleration = self.config['acceleration'],
                                 location = location)
        self.client.hoverAsync().join()
        
    def get_to_location(self, location : tuple):

        '''
            location: (x,y,z)
            x > 0 : move forward
            y > 0 : move right
            z > 0 : move down
            use local cord
        '''
        # all distance
        distance = np.linalg.norm([location[0], location[1], location[2]])
        # acceleration distance
        acc_duration = abs(self.config['speed']) / self.config['acceleration']
        acc_distance = self.config['acceleration'] * acc_duration**2
        # remain distance
        distance = distance - acc_distance
        # start
        self.start_move_vector(location)   
        # maintain speed duration
        duration = distance / self.config['speed']
        # if need to move
        if duration > 0:
            self.client.moveByVelocityBodyFrameAsync(
                vx=self.vx, 
                vy=self.vy, 
                vz=self.vz, 
                duration = duration
            ).join()      
        # stop
        self.stop_move_vector(location)
        time.sleep(3)

    def yaw_rate_change(self, current_yaw_rate, changed_yaw_rate,
                        acceleration, interval = 0.1):
        
        if current_yaw_rate == changed_yaw_rate:
            return
        
        #linear
        duration = abs(changed_yaw_rate - current_yaw_rate) / acceleration       
        times = np.linspace(0,duration,int(duration/interval+1))

        if changed_yaw_rate < current_yaw_rate:
            acceleration = acceleration * -1
        for i in range(len(times)):
            yaw_rate = times[i] * acceleration + current_yaw_rate
            self.yaw_rate = yaw_rate
            self.client.rotateByYawRateAsync(
                yaw_rate = yaw_rate, 
                duration = interval
            ).join()

        self.client.rotateByYawRateAsync(
                yaw_rate = yaw_rate, 
                duration = math.inf
        )
        
        return abs(current_yaw_rate * times[-1] + \
                   1 / 2 * acceleration * times[-1]**2)
    
    def stop_rotate(self):

        current_yaw_rate = self.yaw_rate    
        self.yaw_rate_change(current_yaw_rate = current_yaw_rate, 
                             changed_yaw_rate = 0,
                             acceleration = self.config['yaw_acceleration'])
        self.client.hoverAsync().join()
        
    def rotate(self, yaw, yaw_rate = None):

        current_yaw_rate = self.yaw_rate

        if yaw_rate == None:
            yaw_rate = self.config['yaw_rate']
        if yaw < 0:
            yaw_rate = yaw_rate * -1

        traveled = self.yaw_rate_change(current_yaw_rate=current_yaw_rate,
                                        changed_yaw_rate=yaw_rate,
                                        acceleration = self.config['yaw_acceleration']
                                        )
        
        remain = abs(yaw) - 2 * traveled
        duration = remain / abs(yaw_rate)

        if duration > 0:

            self.client.rotateByYawRateAsync(
                yaw_rate = self.yaw_rate, 
                duration = duration
            ).join() 

        self.stop_rotate()
        time.sleep(2)

    def rotate_no_acc(self, yaw):
        
        yaw_rate = self.config['yaw_rate']
        if yaw<0:
            yaw_rate = yaw_rate * -1
        self.client.rotateByYawRateAsync(
                yaw_rate = yaw_rate, 
                duration = abs(yaw/yaw_rate)
        ).join()
        time.sleep(2)

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
        
    def capture_video(self, results : dict, camera : list, interval = 0.5):
       
        def _take_picture(stop_event, results):

            time.sleep(5)
            print('Capture start')
            while not stop_event.is_set():
                for cam in camera:
                    img_bgr, depth = self.get_bgr_and_depth(cam)
                    results['visual'][cam]['bgr'].append(img_bgr)
                    results['visual'][cam]['depth'].append(depth)
                location, pose = self.get_world_location_pose()
                time.sleep(interval)
                results['location'].append(location)
                results['pose'].append(pose)
            print('Capture stop')
                
        stop_event = threading.Event()
        results['visual'] = dict()
        for cam in camera:
            results['visual'][cam] = dict()
            results['visual'][cam]['bgr'] = []
            results['visual'][cam]['depth'] = []
        results['location'] = []
        results['pose'] = []
        take_picture = threading.Thread(target=_take_picture, 
                                        args=(stop_event, results))
        take_picture.start()
        return stop_event, take_picture

    def stop_capture_video(self,stop_event,take_picture):
        stop_event.set()
        take_picture.join()

    def dump_flight_data(self,location,pose):

        flight_data = {
            'location': location,
            'pose': pose
        }
        
        json_data = json.dumps(flight_data)

        return json_data
    
    def move_along_route(self, change, route):

        for cord in tqdm(route, desc="Moving: "):

            location, pose = self.get_world_location_pose()
            yaw = pose[2]
            location = np.array(location)
            location += change
            residual = cord - location
            new_pose = math.atan2(residual[1], residual[0]) * 180 / math.pi
            move_degree1 = new_pose-yaw 
            if move_degree1 > 0:
                move_degree2 = move_degree1 - 360
            else:
                move_degree2 = move_degree1 + 360
            move_degree = move_degree1 if abs(move_degree1) < abs(move_degree2) else move_degree2
            self.rotate_no_acc(move_degree)
            self.get_to_location((np.linalg.norm([residual[0], residual[1]]),0,0))

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

    def get_to_location_and_capture(self, cord, camera, records, X, Y, Z, PT : point_cloud, capture_rate = 0.5, if_capture = True, sleep = 0):

        # vertical move
        def vertical_move(residual, location, pose, speed):            
            distance = abs(residual[2])
            dir = 1 if residual[2] > 0 else -1
            move_unit = dir * speed * capture_rate
            if abs(move_unit) > distance:
                move = dir * distance
            else:
                move = move_unit
            residual[2] -= move            
            self.client.simSetVehiclePose(
                airsim.Pose(
                    airsim.Vector3r(location[0], location[1], location[2] + move),
                    airsim.to_quaternion(0, 0, math.radians(pose))
                    ), 
                    True)
            location[2] += move
            if if_capture:
                self.record_data(records, camera)
            time.sleep(sleep) 
            return residual, location        

        location, pose = self.get_world_location_pose()
        location = np.array(location)
        yaw = pose[2]
        cord = np.array(cord).squeeze()
        if cord.shape[-1] != location.shape[-1]:
            cord = np.array([cord[0],cord[1],location[2]])
        residual = cord - location
        
        # rotate
        new_pose = math.atan2(residual[1], residual[0]) * 180 / math.pi
        move_degree1 = new_pose-yaw 
        if move_degree1 > 0:
            move_degree2 = move_degree1 - 360
        else:
            move_degree2 = move_degree1 + 360
        move_degree = move_degree1 if abs(move_degree1) < abs(move_degree2) \
        else move_degree2            
        yaw_rate = self.config['yaw_rate']
        yaw_unit = yaw_rate * capture_rate
        dir = 1 if move_degree > 0 else -1
        yaw_unit = yaw_unit * dir
        moved = 0
        start = yaw
        while moved < abs(move_degree):
            if moved + abs(yaw_unit) > abs(move_degree):
                move = (abs(move_degree) - moved) * dir
            else:
                move = yaw_unit
            moved += abs(yaw_unit)
            start += move
            self.client.simSetVehiclePose(                    
                airsim.Pose(
                    airsim.Vector3r(location[0], location[1], location[2]),
                    airsim.to_quaternion(0, 0, math.radians(start))
                    ), 
                    True)
            if if_capture:
                self.record_data(records, camera)
            time.sleep(sleep)
         
        # horizen move                     
        distance = np.linalg.norm([residual[0], residual[1]])
        direction_vector = np.array([residual[0], residual[1]]) / distance 
        speed = self.config['speed'] 
        move_unit = direction_vector * speed * capture_rate
        moved_dist = 0
        start_x = location[0]
        start_y = location[1]
        while moved_dist < distance:          
            if moved_dist + speed * capture_rate > distance:
                move = np.array([cord[0] - start_x, cord[1] - start_y])
            else:
                move = move_unit
            moved_dist += speed * capture_rate
            start_x += move[0]
            start_y += move[1]
            if abs(residual[2]) > 0:
                while(1):
                    map, change = PT.get_map_at_current_height(X,Y,Z,location)
                    if map[int(start_x - change[0]), 
                        int(start_y - change[1])] == OBSTACLE:
                        residual,location = vertical_move(residual,location,new_pose,speed)
                    else:
                        break
            self.client.simSetVehiclePose(
                airsim.Pose(
                    airsim.Vector3r(start_x, start_y, location[2]),
                    airsim.to_quaternion(0, 0, math.radians(new_pose))
                    ),
                    True)
            location[0] = start_x
            location[1] = start_y
            if if_capture:
                self.record_data(records, camera)        
            time.sleep(sleep)

        while abs(residual[2]) > 0:
            residual,location = vertical_move(residual,location,new_pose,speed)

    def move_and_capture_along_route_cv_mode(self, change, route, camera, records, capture_rate = 0.5):

        for cord in tqdm(route, desc="Moving: "):
            cord = cord
            # cord[2] = - cord[2]
            location, pose = self.get_world_location_pose()                       
            self.get_to_location_and_capture_vector(cord,camera,records,capture_rate)
            path = np.concatenate((np.array(location).reshape(1,-1), np.array(cord).reshape(1,-1)),axis=0)
            records['path'].append(path)
    
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
    
    def save_partial_record(self, root, records):
        from utils.saver import image_saver, json_saver, numpy_saver
        bgr_saver = image_saver('', root, 'bgr')
        for key in records['visual'].keys():
            bgr_saver.save_list(records['visual'][key]['bgr'],'camera_' + str(key), if_next=False)
        bgr_saver.next()

        depth_saver = numpy_saver('', root, 'depth')
        for key in records['visual'].keys():
            depth_saver.save_list(records['visual'][key]['depth'],'camera_' + str(key), if_next=False)
        depth_saver.next()

        flight_data = self.dump_flight_data(records['location'], records['pose'])
        JS = json_saver('', root, 'flight_data')
        JS.save(flight_data)

    def get_to_location_and_capture_vector(self, navi_point, camera, records, capture_rate = 0.5, if_capture = True, sleep = 0, final_yaw = None):

        location, pose = self.get_world_location_pose()
        location = np.array(location)
        yaw = pose[2]
        navi_point = np.array(navi_point).squeeze()
        if navi_point.shape[-1] != location.shape[-1]:
            navi_point = np.array([navi_point[0],navi_point[1],location[2]])
        residual = navi_point - location
        
        # rotate
        new_pose = math.atan2(residual[1], residual[0]) * 180 / math.pi
        move_degree1 = new_pose-yaw 
        if move_degree1 > 0:
            move_degree2 = move_degree1 - 360
        else:
            move_degree2 = move_degree1 + 360
        move_degree = move_degree1 if abs(move_degree1) < abs(move_degree2) \
        else move_degree2            
        yaw_rate = self.config['yaw_rate']
        yaw_unit = yaw_rate * capture_rate
        dir = 1 if move_degree > 0 else -1
        yaw_unit = yaw_unit * dir
        moved = 0
        start = yaw
        while moved < abs(move_degree):
            if moved + abs(yaw_unit) > abs(move_degree):
                move = (abs(move_degree) - moved) * dir
            else:
                move = yaw_unit
            moved += abs(yaw_unit)
            start += move
            self.client.simSetVehiclePose(                    
                airsim.Pose(
                    airsim.Vector3r(location[0], location[1], location[2]),
                    airsim.to_quaternion(0, 0, math.radians(start))
                    ), 
                    True)
            if if_capture:
                self.record_data(records, camera)
            time.sleep(sleep)
         
        # horizen move                     
        distance = np.linalg.norm([residual[0], residual[1], residual[2]])
        direction_vector = np.array([residual[0], residual[1], residual[2]]) / distance 
        speed = self.config['speed'] 
        move_unit = direction_vector * speed * capture_rate
        moved_dist = 0
        while moved_dist < distance:          
            if moved_dist + speed * capture_rate > distance:
                move = np.array([navi_point[0] - location[0], 
                                 navi_point[1] - location[1], 
                                 navi_point[2] - location[2]])
            else:
                move = move_unit
            moved_dist += speed * capture_rate
            location[0] += move[0]
            location[1] += move[1]
            location[2] += move[2]
            self.client.simSetVehiclePose(
                airsim.Pose(
                    airsim.Vector3r(location[0], location[1], location[2]),
                    airsim.to_quaternion(0, 0, math.radians(new_pose))
                    ),
                    True)
            if if_capture:
                self.record_data(records, camera)        
            time.sleep(sleep)

    def get_to_location_only_reach_capture_vector(self, cord, camera, records, capture_rate = 0.5, if_capture = True, sleep = 0):

        location, pose = self.get_world_location_pose()
        location = np.array(location)
        yaw = pose[2]
        cord = np.array(cord).squeeze()
        if cord.shape[-1] != location.shape[-1]:
            cord = np.array([cord[0],cord[1],location[2]])
        residual = cord - location
        
        # rotate
        new_pose = math.atan2(residual[1], residual[0]) * 180 / math.pi
        move_degree1 = new_pose-yaw 
        if move_degree1 > 0:
            move_degree2 = move_degree1 - 360
        else:
            move_degree2 = move_degree1 + 360
        move_degree = move_degree1 if abs(move_degree1) < abs(move_degree2) \
        else move_degree2            
        yaw_rate = self.config['yaw_rate']
        yaw_unit = yaw_rate * capture_rate
        dir = 1 if move_degree > 0 else -1
        yaw_unit = yaw_unit * dir
        moved = 0
        start = yaw
        while moved < abs(move_degree):
            if moved + abs(yaw_unit) > abs(move_degree):
                move = (abs(move_degree) - moved) * dir
            else:
                move = yaw_unit
            moved += abs(yaw_unit)
            start += move
            self.client.simSetVehiclePose(                    
                airsim.Pose(
                    airsim.Vector3r(location[0], location[1], location[2]),
                    airsim.to_quaternion(0, 0, math.radians(start))
                    ), 
                    True)
            time.sleep(sleep)
         
        # horizen move                     
        distance = np.linalg.norm([residual[0], residual[1], residual[2]])
        direction_vector = np.array([residual[0], residual[1], residual[2]]) / distance 
        speed = self.config['speed'] 
        move_unit = direction_vector * speed * capture_rate
        moved_dist = 0
        while moved_dist < distance:          
            if moved_dist + speed * capture_rate > distance:
                move = np.array([cord[0] - location[0], 
                                 cord[1] - location[1], 
                                 cord[2] - location[2]])
            else:
                move = move_unit
            moved_dist += speed * capture_rate
            location[0] += move[0]
            location[1] += move[1]
            location[2] += move[2]
            self.client.simSetVehiclePose(
                airsim.Pose(
                    airsim.Vector3r(location[0], location[1], location[2]),
                    airsim.to_quaternion(0, 0, math.radians(new_pose))
                    ),
                    True)        
            time.sleep(sleep)
        if if_capture:
            self.record_data(records, camera)  


    def turn_to_cord_and_capture(self, cord, camera, records, capture_rate = 0.5, if_capture = True, sleep = 0):

        location, pose = self.get_world_location_pose()
        location = np.array(location)
        yaw = pose[2]
        cord = np.array(cord).squeeze()
        if cord.shape[-1] != location.shape[-1]:
            cord = np.array([cord[0],cord[1],location[2]])
        residual = cord - location
        
        # rotate
        new_pose = math.atan2(residual[1], residual[0]) * 180 / math.pi
        move_degree1 = new_pose-yaw 
        if move_degree1 > 0:
            move_degree2 = move_degree1 - 360
        else:
            move_degree2 = move_degree1 + 360
        move_degree = move_degree1 if abs(move_degree1) < abs(move_degree2) \
        else move_degree2            
        yaw_rate = self.config['yaw_rate']
        yaw_unit = yaw_rate * capture_rate
        dir = 1 if move_degree > 0 else -1
        yaw_unit = yaw_unit * dir
        moved = 0
        start = yaw
        while moved < abs(move_degree):
            if moved + abs(yaw_unit) > abs(move_degree):
                move = (abs(move_degree) - moved) * dir
            else:
                move = yaw_unit
            moved += abs(yaw_unit)
            start += move
            self.client.simSetVehiclePose(                    
                airsim.Pose(
                    airsim.Vector3r(location[0], location[1], location[2]),
                    airsim.to_quaternion(0, 0, math.radians(start))
                    ), 
                    True)            
            time.sleep(sleep)
        if if_capture:
            self.record_data(records, camera)

    









        

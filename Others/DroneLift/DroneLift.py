import numpy as np
from Drone.Control import drone_controller
import airsim
import math
import cv2
from Point_cloud.Point_cloud import point_cloud
from Point_cloud.Map_element import UNKNOWN, OBSTACLE, EXPLORED
import queue
from utils.saver import image_saver, image_saver_plt
from collections import defaultdict
import copy

class DroneLift:
    def __init__(self, config):
        self.height = config['DroneLift']['HBE_height']
        self.bottom_height = config['DroneLift']['LBE_height']
        self.config = config
        self.saver = image_saver(config['now'],config['Record_root'], f'DroneLift/camera_up')
        self.down_saver = image_saver(config['now'],config['Record_root'], f'DroneLift/camera_down')
        self.points_up = np.array([], dtype = np.int32).reshape(0, 3)
        self.points_down = np.array([], dtype = np.int32).reshape(0, 3)
        
        '''地面高度'''
        self.ground_height = None
    
    def squeeze_point_cloud(self, X: dict, Y: dict, Z: dict) -> np.ndarray:
        x = np.concatenate([X[key] for key in X.keys()], axis=0)
        y = np.concatenate([Y[key] for key in Y.keys()], axis=0)
        z = np.concatenate([Z[key] for key in Z.keys()], axis=0)

        valid_mask = ~np.isnan(x) & ~np.isnan(y) & ~np.isnan(z)
        x_valid = x[valid_mask].astype(np.int32)
        y_valid = y[valid_mask].astype(np.int32)
        z_valid = z[valid_mask].astype(np.int32)
        
        points = np.stack((x_valid, y_valid, z_valid), axis=1)
        return np.unique(points, axis=0)
    
    def bold_point_cloud(self, points):
        ret_points = copy.deepcopy(points)
        add_items = ()
        for i in range(-6,7):
            for j in range(-6,7):
                item = ret_points + np.array([[i,j,0]])
                add_items += (item, )
        add_items = np.concatenate(add_items, axis = 0)
        ret_points = np.unique(np.concatenate((ret_points, add_items), axis = 0), axis=0)
        return ret_points
    
    def meger_points_down(self, points):
        '''合并下方的点云'''
        points_down = points[np.where(-points[:,2] <= self.bottom_height + 1)]
        self.points_down = np.concatenate([points_down, self.points_down], axis = 0)

    def vert_move(self, current_2D_map, points, location, down=False):

        # import open3d
        # pcd = open3d.geometry.PointCloud()
        # pcd.points = open3d.utility.Vector3dVector(points)
        # pcd.paint_uniform_color([0, 0, 1.0])
        # open3d.visualization.draw_geometries([pcd])       
        
        def find_nearest_point(point, point_cloud):
            distance = np.linalg.norm(np.array(point) - np.array(point_cloud), axis = 1)
            idx = np.where(distance == min(distance))[0][0]
            return point_cloud[idx]        

        if down == False:
            current_height = self.bottom_height
            target_height = self.height       
        else:
            current_height = self.height
            target_height = self.bottom_height

        min_height = min(current_height, target_height)
        max_height = max(current_height, target_height)
        idx = -points[:,2] >= min_height
        vert_points = points[idx]

        location = location[0:2]
        vert_point_cloud_2D = vert_points[:,0:2]

        current_explored = current_2D_map[EXPLORED].copy()

        while True:
            vert_point = find_nearest_point(location, current_explored)
            idx = (vert_point == vert_point_cloud_2D).all(axis=-1)
            heights = vert_points[idx][:,-1]
            # test if the vertical route is blocked
            idx = (-heights > min_height) & (-heights < max_height)
            if idx.shape[0] == 0 or not np.all(idx):
                break
            idx = ~(vert_point == current_explored).all(axis=-1)
            current_explored = current_explored[idx]
        
        return vert_point

    def search_point(self, points, location, global_2D_map, down=False, low_obstacles=[]):
        # if down: 
        #     points = self.bold_point_cloud(points)

        import open3d
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(points)
        pcd.paint_uniform_color([0, 0, 1.0])

        x = points[:, 0]; y = points[:, 1]; z = -points[:, 2]; 
        if down: min_z = z.min()
        
        st = np.array(location, dtype = np.int32)[:2].tolist()
        dx, dy = [0, 0, 1, -1], [1, -1, 0, 0]
        all_explore = global_2D_map[EXPLORED].tolist()
        q = queue.Queue()
        q.put(st)
        vis = defaultdict(lambda: False)
        point = None
        while q.empty() == False:
            cur = q.get()

            vis[tuple(cur)] = True
            for inc_x, inc_y in zip(dx, dy):
                nxt = cur + np.array([inc_x, inc_y], dtype = np.int32)
                nxt = nxt.tolist()
                if nxt in all_explore and vis[tuple(nxt)] == False:
                    q.put(nxt)
                    vis[tuple(nxt)] = True

            if cur in low_obstacles: continue

            if down: bnd = 4; limit = (bnd * 2 + 1) ** 2
            else: bnd = 1; limit = 9
            idx1 = np.where((x <= cur[0] + bnd) & (x >= cur[0] - bnd))
            idx2 = np.where((y <= cur[1] + bnd) & (y >= cur[1] - bnd))
            # 去掉重复的x，y点对
            idx2 = np.intersect1d(idx1, idx2)

            if down:
                tmp_z = z[idx2]
                idx3 = np.where((tmp_z >= self.bottom_height + 1))
                if idx3[0].shape[0] == 0:
                    v_vis = set()
                    stay = []
                    for i in range(idx2.shape[0]):
                        if (x[idx2][i], y[idx2][i]) not in v_vis:
                            stay.append(i)
                            v_vis.add((x[idx2][i], y[idx2][i]))
                    idx2 = (idx2[stay],)
                    if idx2[0].shape[0] >= limit:
                        point = cur
                        break
            else:
                if idx2.shape[0] == 0:
                    point = cur
                    break
                tmp_z = z[idx2]
                idx3 = np.where((tmp_z <= self.height + 1))
                if idx3[0].shape[0] == 0:
                    point = cur
                    break
        assert point is not None

        return point
    
    def clear_points_up(self):
        self.points_up = np.array([], dtype = np.int32).reshape(0, 3)
    
    def turn_to_up(self, DC:drone_controller):
        '''将摄像头旋转至上方''' 
        camera_pose = airsim.Pose(airsim.Vector3r(0, 0, 0), 
                                  airsim.to_quaternion(math.radians(-270), 0, 0)) #radians
        DC.client.simSetCameraPose("3", camera_pose)
    
    def turn_to_down(self, DC:drone_controller):
        '''将摄像头旋转至下方'''
        camera_pose = airsim.Pose(airsim.Vector3r(0, 0, 0), 
                                  airsim.to_quaternion(math.radians(-90), 0, 0)) #radians
        DC.client.simSetCameraPose("3", camera_pose)
    
    def get_record(self, DC:drone_controller, PT:point_cloud, points:np.ndarray):
        bgr = DC.get_single_bgr_image(3)
        self.saver.save(bgr)
        depth = DC.get_single_depth_image(3)
        location, pose = DC.get_world_location_pose()
        intrinsic_matrix = DC.get_intrinsic_matrix(3)
        pose_matrix = self.get_pose_matrix(pose)
        X_up, Y_up, Z_up = PT.get_point_clouds_from_depth(intrinsic_matrix, 
                                            pose_matrix,
                                            depth,
                                            3)
        '''上方点云已经构建完了'''
        X_up, Y_up, Z_up = PT.get_global_point_cloud(location, X_up, Y_up, Z_up)
        points_up = self.squeeze_point_cloud(X_up, Y_up, Z_up)
        # idx = np.where((points[:,2] < location[2] - 1) )
        # points_up = np.concatenate((points_up, points[idx]), axis = 0)
        self.points_up = np.concatenate([points_up, self.points_up], axis = 0)
    
    def get_record_down(self, DC:drone_controller, PT:point_cloud, points:np.ndarray):
        bgr = DC.get_single_bgr_image(3)
        self.down_saver.save(bgr)
        depth = DC.get_single_depth_image(3)
        location, pose = DC.get_world_location_pose()
        intrinsic_matrix = DC.get_intrinsic_matrix(3)
        '''注意姿态'''
        pose_matrix = self.get_pose_matrix(pose, down=True)
        X_down, Y_down, Z_down = PT.get_point_clouds_from_depth(intrinsic_matrix, 
                                            pose_matrix,
                                            depth,
                                            3)
        '''下方点云已经构建完了'''
        X_down, Y_down, Z_down = PT.get_global_point_cloud(location, X_down, Y_down, Z_down)
        points_down = self.squeeze_point_cloud(X_down, Y_down, Z_down)
        
        '''只需要当前高度以下的点云'''
        # need_idx = np.where((-points_down[:,2] <= self.bottom_height + 1) )
        # points_down = points_down[need_idx]
        
        # idx = np.where((-points[:,2] <= self.bottom_height + 1))
        # points_down = np.concatenate((points_down, points[idx]), axis = 0)
        self.points_down = np.concatenate([points_down, self.points_down], axis = 0)
        return bgr

    def Fall(self, global_2D_map, DC:drone_controller, low_obstacles, records, view_saver):
        location, pose = DC.get_world_location_pose()
        landing_point = self.search_point(self.points_down, location, global_2D_map, down=True, low_obstacles=low_obstacles)

        bgr = DC.get_single_bgr_image(3)
        view_saver.save(bgr)        
        
        return landing_point
    
    def Lift(self, current_global_2D_map, DC:drone_controller, records, view_saver):
        location, pose = DC.get_world_location_pose()

        takeoff_point = self.search_point(self.points_up, location, current_global_2D_map)

        bgr = DC.get_single_bgr_image(3)
        view_saver.save(bgr)   

        return takeoff_point     
            
    def to_target_height(self, DC:drone_controller, location, pose, records, down=False):
        location = list(location)
        if down: residual = -self.bottom_height - location[2]  
        else: residual = -self.height - location[2]         
        while abs(residual) > 0:
            distance = abs(residual)
            dir = 1 if residual > 0 else -1
            speed = self.config['DroneLift']['speed']
            move_unit = dir * speed 
            if abs(move_unit) > distance:
                move = dir * distance
            else:
                move = move_unit
            residual -= move   
            # DC.get_to_location_and_capture_vector(np.array((location[0], location[1], location[2] + move)), [0], records, if_capture=False, final_yaw = pose[2])

            DC.client.simSetVehiclePose(
                airsim.Pose(
                    airsim.Vector3r(location[0], location[1], location[2] + move),
                    airsim.to_quaternion(math.radians(pose[0]), math.radians(pose[1]), math.radians(pose[2]))
                    ), 
                    True)
            location[2] += move

    
    def get_pose_matrix(self, pose : tuple, down=False):

        pitch = pose[0]
        roll = pose[1]
        yaw = pose[2]
        
        '''相机往上'''
        if down: roll = roll - 90
        else: roll = roll + 90     

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
    
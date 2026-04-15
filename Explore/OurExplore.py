from Explore.Frontier import frontier
from Point_cloud.Map_element import (
    UNKNOWN,
    OBSTACLE,
    EXPLORED,
    FRONTIER)
import numpy as np
import math
import matplotlib.pyplot as plt 
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
from matplotlib.axes import Axes
from pathfinding.core.diagonal_movement import DiagonalMovement
from Drone.Control import drone_controller
from sklearn.cluster import KMeans
from Point_cloud.Point_cloud import point_cloud
import matplotlib.pyplot as plt  
from utils.saver import image_saver, image_saver_plt
from numpy.linalg import norm
from Others.ValueMap.ValueMap import ValueMap
from Explore.DroneLift import DroneLift
import csv
from Explore.Explore import explore as base_explore

class modify_explore(base_explore):

    def __init__(self, config, VMP, DL, LLM):

        self.config = config
        self.frontier = frontier(config['OurExplore'])
        self.VMP: ValueMap = VMP
        self.DL: DroneLift = DL
        self.up_height = self.DL.height
        self.LLM = LLM
        # self.AG: Agent = AG
        self.lowBEF_count = 0

        self.get_navigation_bound()

    def get_navigation_bound(self):

        path = self.config['Simulator']['route_path']
        route = ()
        with open(path, encoding='utf-8')as f:
            reader = csv.reader(f)
            for row in reader:
                cord = [float(x) / 100 for x in row]
                cord = np.array([cord])
                route += (cord, )
        route = np.concatenate(route, axis = 0)
        change = route[0,:]        
        
        self.boundary = dict()
        navigation_path = self.config['Simulator']['navigation_path']
        with open(navigation_path, 'r') as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if i == 0:
                    self.boundary['x_min'] = float(row[0]) / 100 - change[0]
                    self.boundary['x_max'] = float(row[1]) / 100 - change[0]
                    self.boundary['y_min'] = float(row[2]) / 100 - change[1]
                    self.boundary['y_max'] = float(row[3]) / 100 - change[1]        

    def get_frontiers(self, global_explored_map, change, frontier_size):
        self.frontier.config['frontier_size'] = frontier_size
        return super().get_frontiers(global_explored_map, change)
    
    def merge_points(self, point1, point2):
        return super().merge_points(point1, point2)
    
    def update_global_map(self, map, location, change, global_2D_map, history_location):

        loc = (location[0], location[1])
        history_location.append(loc)

        idx = np.where(map == OBSTACLE)
        x = idx[0]; y = idx[1]
        x = x + change[0]
        y = y + change[1]
        temp = np.concatenate((x.reshape(-1,1),y.reshape(-1,1)), axis = 1)
        if global_2D_map[OBSTACLE].shape[0] == 0:
            global_2D_map[OBSTACLE] = temp
        else:
            global_2D_map[OBSTACLE] = self.merge_points(global_2D_map[OBSTACLE], temp)

        idx = np.where(map == EXPLORED)
        x = idx[0]; y = idx[1]
        x = x + change[0]
        y = y + change[1]
        temp = np.concatenate((x.reshape(-1,1),y.reshape(-1,1)), axis = 1)
        if global_2D_map[EXPLORED].shape[0] == 0:
            global_2D_map[EXPLORED] = temp
        else:
            global_2D_map[EXPLORED] = self.merge_points(global_2D_map[EXPLORED], temp)

    def draw_current_map(self, map, location, change):

        idx = np.where(map == OBSTACLE)
        x = idx[0]; y = idx[1]
        x = x + change[0]
        y = y + change[1]
        plt.figure() 
        plt.scatter(y, x, 
                    c='black',s=1)
        
        idx = np.where(map == EXPLORED)
        x = idx[0]; y = idx[1]
        x = x + change[0]
        y = y + change[1] 
        plt.scatter(y, x, 
                    c='gray',s=1)

        # frontiers_squeeze = ()
        # labels = ()
        # for i in range(len(frontiers)):
        #     frontiers_squeeze += (frontiers[i]['frontier'],)
        #     labels += (i * np.ones(frontiers[i]['frontier'].shape[0]),)
        # frontiers_squeeze = np.concatenate(frontiers_squeeze,axis=0)
        # labels = np.concatenate(labels)
        # plt.scatter(frontiers_squeeze[:, 1], frontiers_squeeze[:, 0], 
        # c=labels, cmap=plt.cm.gist_rainbow, s=1)

        plt.scatter(location[1], location[0], 
            c='red',s=3) 
        return plt.gca()

    def mask_explored_at_height(self, global_2D_map, explored_at_height):
        
        explored = global_2D_map[EXPLORED]
        obstacle = global_2D_map[OBSTACLE]

        # get global cant go
        min_point = np.min(np.concatenate((explored,obstacle)), axis=0).astype(int)  
        max_point = np.max(np.concatenate((explored,obstacle)), axis=0).astype(int)
        Len = (max_point - min_point + 1).astype(int) 
        map = np.zeros((Len[0], Len[1]), dtype=np.int8)
        map_point = explored.astype(int) - np.repeat(min_point.reshape(1,-1), explored.shape[0], axis=0)
        map[
            map_point[:,0].reshape(1,-1),
            map_point[:,1].reshape(1,-1),
        ] = 1  
        cant_go = np.array(np.where(map == 0)).T + min_point      

        # mask explored at height
        min_point = np.min(np.concatenate((explored_at_height,cant_go)), axis=0).astype(int)  
        max_point = np.max(np.concatenate((explored_at_height,cant_go)), axis=0).astype(int)
        Len = (max_point - min_point + 1).astype(int)
        mask = np.ones((Len[0], Len[1]), dtype=np.int8)
        mask_point = cant_go.astype(int) - np.repeat(min_point.reshape(1,-1), cant_go.shape[0], axis=0)
        mask[
            mask_point[:,0].reshape(1,-1),
            mask_point[:,1].reshape(1,-1),
        ] = 0
        map = np.zeros((Len[0], Len[1]), dtype=np.int8)
        map_point = explored_at_height.astype(int) - np.repeat(min_point.reshape(1,-1), explored_at_height.shape[0], axis=0)
        map[
            map_point[:,0].reshape(1,-1),
            map_point[:,1].reshape(1,-1),
        ] = 1
        map = map * mask
        new_explored_at_height = np.array(np.where(map == 1)).T + min_point

        plt.figure() 
        plt.scatter(new_explored_at_height[:, 1], new_explored_at_height[:, 0], c='gray',s=1)   
        # plt.savefig('test.png')     
        return new_explored_at_height
                       
    def draw_global_map_and_frontier(self, frontiers,global_2D_map,history_location):
        
        obstacle = global_2D_map[OBSTACLE]
        explored = global_2D_map[EXPLORED]
        plt.figure() 
        if explored.shape[0] != 0:
            plt.scatter(explored[:, 1], explored[:, 0], 
                        c='gray',s=1)
        if obstacle.shape[0] != 0:
            plt.scatter(obstacle[:, 1], obstacle[:, 0], 
                        c='black',s=1)                
        history_location = np.array(history_location)
        if history_location.shape[0] != 0:
            if history_location.shape[0] > 1:
                plt.scatter(history_location[:-1, 1], history_location[:-1, 0], 
                            c='red', s=3)
            plt.scatter(history_location[-1, 1], history_location[-1, 0], 
                        c='blue', marker='*', s=18)
        
        if len(frontiers) != 0:
            frontiers_squeeze = ()
            labels = ()
            for i in range(len(frontiers)):
                frontiers_squeeze += (frontiers[i]['frontier'],)
                labels += (i * np.ones(frontiers[i]['frontier'].shape[0]),)
            frontiers_squeeze = np.concatenate(frontiers_squeeze,axis=0)
            labels = np.concatenate(labels)
            plt.scatter(frontiers_squeeze[:, 1], frontiers_squeeze[:, 0], 
            c=labels, cmap=plt.cm.gist_rainbow, s=1)
        return plt.gca()
    
    def get_explored_map(self, map, location, change, explore_range):
        return super().get_explored_map(map, location, change, explore_range)
    
    def squeeze_frontiers(self, frontiers):

        new_frontiers = ()
        for item in frontiers:
            frontier = item['frontier']
            new_frontiers += (frontier, )
        return np.concatenate(new_frontiers, axis = 0)
       
    def divide_points_to_directions(self, points, current_loc, bin_num):
        
        bins = [() for _ in range(bin_num)]
        bins_info = dict()
        points = np.array(points)
        current_loc = np.array(current_loc).reshape(1,-1)  
        if points.shape[0] != 0:          
            residual = points[:,0:2] - current_loc[:,0:2]
            pose = (np.arctan2(residual[:,1], residual[:,0]) * 180 / math.pi + 180) % 360
            bin_index = (pose / (360 / bin_num)).astype(int)
            for i in range(bin_index.shape[0]):
                bins[bin_index[i]] += (points[i,:].reshape(1,-1),)
            size = []
            for i in range(len(bins)):
                if len(bins[i]) != 0:
                    bins[i] = np.concatenate(bins[i], axis=0)
                    size.append(bins[i].shape[0])
                else:
                    bins[i] = np.array([])
                    size.append(0)
            bins_info['size'] = size
        return bins, bins_info
    
    def divide_frontiers_to_bins(self, frontiers, current_loc, mini_frontier_size):
        return super().divide_frontiers_to_bins(frontiers, current_loc, mini_frontier_size)
                
    def point_to_move(self, current_loc : tuple, navi_point: tuple):
        return super().point_to_move(current_loc, navi_point)

    def draw_frontiers(self, global_map_image : Axes, frontiers):
        
        for item in frontiers:
            centers = item['centers']
            global_map_image.scatter(centers[:,1], centers[:,0], c='green',s=3)
        return global_map_image
        
    def calculate_route(self, navi_point, location, map, change):
        return super().calculate_route(navi_point, location, map, change)
    
    def draw_path(self, global_map_image : Axes, path, location, navi_point):
        return super().draw_path(global_map_image, path, location, navi_point)
    
    def move_along_path(self, path: np.ndarray, navi_point, DC : drone_controller, records,capture_rate = 1, if_capture = True, sleep = 0):

        for i in range(1, path.shape[0]-1):
            cord = path[i,:]
            DC.get_to_location_and_capture_vector(cord, [0,1,2], records, capture_rate=capture_rate, if_capture = if_capture, sleep = sleep)    
        DC.get_to_location_and_capture_vector(navi_point, [0,1,2], records, capture_rate=capture_rate, if_capture = if_capture, sleep = sleep)
    
    def move_along_path_only_capture_when_reach(self, path, navi_point, camera, DC : drone_controller, records,capture_rate = 1, sleep = 0):

        for i in range(1, path.shape[0]-1):
            cord = path[i,:]
            DC.get_to_location_and_capture_vector(cord, camera, records, capture_rate=capture_rate, if_capture = False, sleep = sleep)    
        DC.get_to_location_only_reach_capture_vector(navi_point, camera, records, capture_rate=capture_rate, if_capture = True, sleep = sleep)

    def find_surround_frontier(self, location, frontiers, map, change):
        
        min_dist = math.inf
        new_navi_point = None
        for item in frontiers:
            centers = item['centers']
            for center in centers:
                residual = np.array([location[0],location[1]]).reshape(1,-1) - \
                    np.array([center[0],center[1]]).reshape(1,-1)
                dist = np.linalg.norm(residual, axis=1)
                # path = self.calculate_route(mean, location, map, change)
                # dist = len(path)
                if dist < min_dist:
                    min_dist = dist
                    new_navi_point = center
        path = self.calculate_route(new_navi_point, location, map, change)
        return path, new_navi_point

    def get_global_explored_map(self, location, global_2D_map):
        return super().get_global_explored_map(location, global_2D_map)
    
    def draw_targets(self, targets):
        return super().draw_targets(targets)
    
    def get_current_point_cloud(self, DC : drone_controller, PT : point_cloud, camera: list, records, up=False):
        
        if up == False: 
            bgr ,depth = DC.record_data(records,camera)
            location = records['location'][-1]
            pose = records['pose'][-1]
        else: 
            bgr ,depth = DC.get_bgr_and_depth(camera)
            location, pose = DC.get_world_location_pose()
            records['location'].append(location)
            records['pose'].append(pose)

        X = dict(); Y = dict(); Z = dict()
        location, pose = DC.get_world_location_pose()
        
        '''debug camera direction'''
        
        for i in range(len(camera)):                         
            intrinsic_matrix = DC.get_intrinsic_matrix(camera[i])
            pose_matrix = DC.get_pose_matrix(pose,camera[i])                
            x,y,z = PT.get_point_clouds_from_depth(intrinsic_matrix, 
                                                pose_matrix,
                                                depth[i],
                                                camera = camera[i])            
            x,y,z = PT.get_global_point_cloud(location, x,y,z)
            
            '''generate point cloud'''
        
            X.update(x); Y.update(y); Z.update(z)
        
        '''visualize point cloud'''
        
        return X,Y,Z,bgr,location,pose

    def save_explore_bgr(self, config,global_bgr):
        return super().save_explore_bgr(config, global_bgr)
    
    def calculate_all_path(self, records):
        return super().calculate_all_path(records)

    def save_path(self, path, record):
        return super().save_path(path, record)

    def array_to_dict(self,array):
        return super().array_to_dict(array)

    def get_explored_at_height(self, height, global_point_cloud, start_location):
        return super().get_explored_at_height(height, global_point_cloud, start_location)
    
    def _get_map_at_current_height(self, x: dict, y: dict, z: dict, location: tuple, PT: point_cloud):
        x, y, z = PT.squeeze_point_cloud(x, y, z)
        x_all, y_all, z_all = x.flatten(), y.flatten(), z.flatten()
        
        valid_idx = ~np.isnan(x_all) & ~np.isnan(y_all) & ~np.isnan(z_all)
        x_all, y_all, z_all = x_all[valid_idx], y_all[valid_idx], z_all[valid_idx]
        threshold = (
            int(self.boundary['x_max']),
            int(self.boundary['x_min']),
            int(self.boundary['y_max']),
            int(self.boundary['y_min'])
        )
        height_idx = (z_all <= location[2] + 0.1) & (z_all >= location[2] - 0.1)
        x, y = x_all[height_idx].astype(np.int32), y_all[height_idx].astype(np.int32)
        points = np.unique(np.vstack((x_all, y_all, z_all)).T.astype(np.int32), axis=0)
        map, change = PT.create_2D_map(x, y, threshold)
        return map.astype(int), change, points

    def get_map_at_current_low_height(self, x: dict, y: dict, z: dict, 
                                  location: tuple, explore_range: int, PT: point_cloud):
        return self._get_map_at_current_height(x, y, z, location, PT)
    
    def get_map_at_current_upper_height(self, x: dict, y: dict, z: dict, 
                                  location: tuple, PT: point_cloud):
        return self._get_map_at_current_height(x, y, z, location, PT)
    
    def check_meger_known_points(self, points1: list, points2: np.ndarray):
        points1_set = set(map(tuple, points1))
        points2_set = set(map(tuple, points2))
        for point in points1_set:
            for i in range(-1,2):
                for j in range(-1,2):
                    neighbour = (point[0] + i, point[1] + j)
                    if neighbour in points2_set:
                        points2_set.add(point)
        return points2

    def main_loop(self,DC:drone_controller,PT:point_cloud,camera,records,config,path_saver):

        self.global_lower_2D_map = dict()
        self.global_lower_2D_map[OBSTACLE] = np.array([])
        self.global_lower_2D_map[EXPLORED] = np.array([])
        self.low_history_location = []
        self.low_global_map_saver = image_saver_plt(config['now'],config['Record_root'],f'low/low_global_map')
        self.lift_path_saver = image_saver_plt(config['now'],config['Record_root'],f'low/lift_path')
        self.lift_up_view = image_saver(config['now'],config['Record_root'],'low/lift_up_view')
        
        self.global_higher_2D_map = dict()
        self.global_higher_2D_map[OBSTACLE] = np.array([])
        self.global_higher_2D_map[EXPLORED] = np.array([])
        self.high_history_location = []
        self.high_global_map_saver = image_saver_plt(config['now'],config['Record_root'],f'high/high_global_map')
        self.fall_path_saver = image_saver_plt(config['now'],config['Record_root'],f'high/fall_path')
        self.high_path_saver = image_saver_plt(config['now'],config['Record_root'],f'high/high_path')
        self.fall_down_view = image_saver(config['now'],config['Record_root'],'high/fall_down_view')
       
        self.ret_global_x, self.ret_global_y, self.ret_global_z, self.ret_global_bgr = [], [], [], []
        self.ret_up_global_x, self.ret_up_global_y, self.ret_up_global_z = [], [], []

        location, pose = DC.get_world_location_pose()
        self.DL.to_target_height(DC, location, pose, records)
        self.first_lift = True

        while True:

            self.DL.turn_to_down(DC)
            up_frontiers_map, up_change, if_lower_explore, if_no_frontier = self.upFBE(DC,PT,camera,records,config,path_saver)
            
            landing_point = self.DL.Fall(self.global_higher_2D_map, DC, self.global_lower_2D_map[OBSTACLE].tolist(), records, self.fall_down_view)
            location, pose = DC.get_world_location_pose()
            path = self.calculate_route(landing_point, location, up_frontiers_map, up_change)
            img = self.draw_global_map_and_frontier([], self.global_higher_2D_map, self.high_history_location)
            path_img = self.draw_path(img, path, location, landing_point)
            self.fall_path_saver.save(path_img)
            self.move_along_path(path, np.array(landing_point), DC, records,capture_rate = 1, if_capture = False, sleep = 0)

            down_bgr = DC.get_single_bgr_image(3)
            self.fall_down_view.save(down_bgr)

            location, pose = DC.get_world_location_pose()
            height = location[-1]
            height_vector = np.ones((path.shape[0],1)) * height
            path = np.concatenate((path, height_vector),axis = -1)

            self.DL.to_target_height(DC, location, pose, records, down=True)
            self.save_path(path, records)

            if if_lower_explore:

                self.landing_location, self.landing_pose = DC.get_world_location_pose()
                lower_explored = self.global_lower_2D_map[EXPLORED]
                lower_obstacle = self.global_lower_2D_map[OBSTACLE]
                
                self.current_global_2D_map = dict()
                self.current_global_2D_map[EXPLORED] = np.array([])
                self.current_global_2D_map[OBSTACLE] = lower_obstacle
                self.current_known_global_2D_map = []
                self.current_low_history_location = []
                
                for loc in lower_explored:
                    self.current_known_global_2D_map.append(loc)                
                
                self.DL.turn_to_up(DC)
                self.DL.clear_points_up()

                frontiers_map, change = self.lowFBE(DC,PT,camera,records,config,path_saver)
                self.lowBEF_count += 1

                self.VMP.clear_current_target_map()               
                
                img = self.draw_global_map_and_frontier([], self.global_lower_2D_map, self.low_history_location)
                if config['OurExplore']['VMP_active']:
                    global_target = np.array(list(self.VMP.global_target_2D_map))
                    if global_target.shape[0] != 0:
                        img.scatter(global_target[:,1], global_target[:,0], c='blue', s=1)                
                self.low_global_map_saver.save(img)

            if not if_no_frontier:

                takeoff_point = self.DL.Lift(self.global_lower_2D_map, DC, records, self.lift_up_view)
                location, pose = DC.get_world_location_pose()
                path = self.calculate_route(takeoff_point, location, frontiers_map, change)
                img = self.draw_global_map_and_frontier([], self.global_lower_2D_map, self.low_history_location)
                path_img = self.draw_path(img, path, location, takeoff_point)
                self.lift_path_saver.save(path_img)
                self.move_along_path(path, np.array(takeoff_point), DC, records,capture_rate = 1, if_capture = False, sleep = 0)

                up_bgr = DC.get_single_bgr_image(3)
                self.lift_up_view.save(up_bgr)

                location, pose = DC.get_world_location_pose()
                height = location[-1]
                height_vector = np.ones((path.shape[0],1)) * height
                path = np.concatenate((path, height_vector),axis = -1)

                self.DL.to_target_height(DC, location, pose, records)
                self.save_path(path, records)
                self.first_lift = False

            else:      
                return self.ret_global_x, self.ret_global_y, self.ret_global_z, self.ret_global_bgr, self.ret_up_global_x, self.ret_up_global_y, self.ret_up_global_z
        
    def upFBE(self,DC:drone_controller,PT:point_cloud,camera,records,config,path_saver):

        up_count = 0
        if_no_frontier = False
        if_lower_explore = False

        while True:

            camera = [0,1,2,3,4]
            X,Y,Z,bgr,location,pose = self.get_current_point_cloud(DC,PT,camera,records, up=True)
            
            # 0 1 4 2 front left back right

            self.ret_up_global_x.append(X); self.ret_up_global_y.append(Y); self.ret_up_global_z.append(Z)

            explore_range = config['OurExplore']['up_explore_range']
            map, change, points_3d = self.get_map_at_current_upper_height(
                X,Y,Z,location, PT)
            down_img = self.DL.get_record_down(DC, PT, points_3d)
            
            explored_map = self.get_explored_map(map, location, change, explore_range)
            self.update_global_map(explored_map, location, change, self.global_higher_2D_map, self.high_history_location)
            
            global_explored_map, change = self.get_global_explored_map(location, self.global_higher_2D_map)
            frontier_size = self.config['OurExplore']['up_frontier_size']
            frontiers, frontiers_map = self.get_frontiers(
                global_explored_map, change, frontier_size)

            frontier_stay = []
            for idx, frontier in enumerate(frontiers):
                if frontier['size'] < config['OurExplore']['up_frontier_size']: continue
                else: frontier_stay.append(idx)
            frontiers = [frontiers[i] for i in frontier_stay]

            global_map_image = self.draw_global_map_and_frontier(frontiers,self.global_higher_2D_map, self.high_history_location)
            self.high_global_map_saver.save(global_map_image)
            
            if len(frontiers) == 0:
                if_no_frontier = True            

            if up_count or self.first_lift:
                # panoramic_img, panoramic_segment = self.LLM.get_panoramic_image_down(bgr)
                if_mark = []
                for i in range(len(bgr)):
                    if_mark.append(False)
                panoramic_image, marked_imgs = self.LLM.get_visual_prompt(bgr, if_mark, 1)
                top_down_img = bgr[3]
                llm_score_thresh = config['OurExplore']['llm_score_thresh']
                if_lower_explore = self.LLM.LLMChooseDown2(top_down_img, llm_score_thresh)

            if if_lower_explore:
                return frontiers_map, change, if_lower_explore, if_no_frontier
 
            else:
                if not if_no_frontier:

                    mini_frontier_size = config['OurExplore']['up_frontier_divide_size']
                    self.divide_frontiers_to_bins(frontiers, location, mini_frontier_size)                          
                    
                    path, navi_point = self.find_surround_frontier(location, frontiers, frontiers_map, change)
                        
                    path_image = self.draw_path(global_map_image,path,location,navi_point)
                    self.high_path_saver.save(path_image)
                    self.move_along_path(path, navi_point, DC, records,capture_rate = 1, if_capture = False, sleep = 0)
                    self.save_path(path, records)
                    up_count += 1
                else:
                    return frontiers_map, change, if_lower_explore, if_no_frontier
                
    def lowFBE(self,DC:drone_controller,PT:point_cloud,camera,records,config,path_saver):
        
        global_map_saver = image_saver_plt(config['now'],config['Record_root'],f'lowFBE/{self.lowBEF_count}/global_map')
        target_map_saver = image_saver_plt(config['now'],config['Record_root'],f'lowFBE/{self.lowBEF_count}/target_map')
        path_saver = image_saver_plt(config['now'],config['Record_root'],f'lowFBE/{self.lowBEF_count}/path')
        ori_global_map_saver = image_saver_plt(config['now'],config['Record_root'],f'lowFBE/{self.lowBEF_count}/ori_global_map')

        while True:
            X,Y,Z,bgr,location,pose = self.get_current_point_cloud(DC,PT,camera,records)

            explore_range = config['OurExplore']['low_explore_range']
            map, change, points_3d = self.get_map_at_current_low_height(X,Y,Z,location,explore_range, PT)

            self.ret_global_x.append(X); self.ret_global_y.append(Y); self.ret_global_z.append(Z); self.ret_global_bgr.append(bgr)
            records['low_location'].append(location); 
            records['low_pose'].append(pose)

            self.DL.get_record(DC, PT, points_3d)
            
            explored_map = self.get_explored_map(map, location, change, explore_range)
            self.update_global_map(explored_map, location, change, self.current_global_2D_map, self.current_low_history_location)
            self.global_lower_2D_map[EXPLORED] = self.merge_points(self.current_global_2D_map[EXPLORED], self.global_lower_2D_map[EXPLORED])
            self.global_lower_2D_map[OBSTACLE] = self.merge_points(self.current_global_2D_map[OBSTACLE], self.global_lower_2D_map[OBSTACLE])
            for loc in self.current_low_history_location:
                self.low_history_location.append(loc)          
            
            limit = config['OurExplore']['low_explore_range'] * config['OurExplore']['low_explore_limit']
            
            self.current_global_2D_map[EXPLORED] = self.check_meger_known_points(self.current_known_global_2D_map, self.current_global_2D_map[EXPLORED])
            
            global_explored_map, change = self.get_global_explored_map(location, self.global_lower_2D_map)
            frontier_size = config['OurExplore']['low_frontier_size']
            frontiers, frontiers_map = self.get_frontiers(
                global_explored_map, change, frontier_size)
            
            global_map_image = self.draw_global_map_and_frontier(frontiers,self.global_lower_2D_map, self.low_history_location)
            ori_global_map_saver.save(global_map_image) 

            # update vmp
            if config['OurExplore']['VMP_active']:
                current_target_loc = self.VMP.get_global_target_loc(bgr, X, Y, Z)
                current_target_loc_2D = np.array(np.array(current_target_loc)[:, :2]).astype(np.int32) if len(current_target_loc) != 0 else []            
                self.VMP.update_current_target_map(current_target_loc_2D)
                self.VMP.meger_current_to_global(self.current_global_2D_map[EXPLORED])

            frontier_stay = []
            for idx, frontier in enumerate(frontiers):
                points = frontier['frontier'].tolist()
                stay = []
                for i, point in enumerate(points):
                    dist = np.linalg.norm(np.array(point)[:2] - \
                                          np.array(self.landing_location)[:2])
                    if dist > limit:
                        continue
                    stay.append(i)
                if len(stay) < config['OurExplore']['low_frontier_size']: 
                    continue
                else:
                    frontier['frontier'] = frontier['frontier'][stay]
                    frontier['size'] = len(stay)
                    if_stay = True
                    if config['OurExplore']['VMP_active']:
                        target_num = self.VMP.check_frontier_around_target(
                            frontier['frontier'], self.config['OurExplore']['frontier_check_range'])
                        min_target_num = config['OurExplore']['minimum_target_num']
                        give_up_prob = config['OurExplore']['give_up_prob']
                        if target_num <= min_target_num or \
                           np.random.rand() < give_up_prob: if_stay = False
                    if if_stay:
                        frontier_stay.append(idx)
            frontiers = [frontiers[i] for i in frontier_stay]
            
            global_map_image = self.draw_global_map_and_frontier(
                frontiers,self.global_lower_2D_map, self.low_history_location)
            global_map_saver.save(global_map_image)
            
            if config['OurExplore']['VMP_active']:
                self.VMP.draw_current_map_and_frontier_and_target(
                    frontiers, self.global_lower_2D_map, self.low_history_location,map_saver=target_map_saver)
            
            if len(frontiers) == 0:
                return frontiers_map, change
            mini_frontier_size = config['OurExplore']['low_frontier_divide_size']
            self.divide_frontiers_to_bins(frontiers, location, mini_frontier_size)
            
            if len(frontiers) > 1 and config['OurExplore']['VMP_active']:
                navi_point, _ = self.VMP.select_navi_point_by_dist_and_LLM(frontiers, None)
                path = self.calculate_route(navi_point, np.array(location, dtype=np.int32), frontiers_map, change)
            else:
                path, navi_point = self.find_surround_frontier(np.array(location, dtype=np.int32), frontiers, frontiers_map, change)
                
            path_image = self.draw_path(global_map_image,path,location,navi_point)
            path_saver.save(path_image)
            self.move_along_path(path, navi_point, DC, records,capture_rate = 1, if_capture = False, sleep = 0)
            self.save_path(path, records)    
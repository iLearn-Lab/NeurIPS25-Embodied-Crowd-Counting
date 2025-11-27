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
from Others.DroneLift.DroneLift import DroneLift
from Others.IntuitionMap.gpt4o_integration import LLMJudge
import csv

'''还差一个和大模型结合的部分'''
class modify_explore():

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

    def get_frontiers(self, global_explored_map, change):
        
        frontiers, frontiers_map = self.frontier.get_frontiers(global_explored_map, change)
        return frontiers, frontiers_map
    
    def merge_points(self, point1, point2):

        unique_points = set(tuple(point) for point in point1)  
        unique_points.update(tuple(point) for point in point2)
        merged_point_cloud = list(unique_points)
        return np.array(merged_point_cloud)
    
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
        plt.scatter(history_location[:, 1], history_location[:, 0], 
                    c='red',s=3)
        
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

        explored_map = self.frontier.get_explored(map, location, change, explore_range)
        return explored_map
    
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
        
        bins = []
        current_loc = np.array(current_loc).reshape(1,-1) 
        for item in frontiers:
            frontier = item['frontier']
            frontier_size = frontier.shape[0]
            n_clusters = np.ceil(frontier_size / mini_frontier_size).astype(int)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)  
            kmeans.fit(frontier)
            centers = kmeans.cluster_centers_
            labels = kmeans.fit_predict(frontier)
            adjusted_centroids = np.zeros_like(centers)  
            for i, centroid in enumerate(centers):  
                # 找到距离聚类中心最近的点云中的点  
                distances = np.linalg.norm(frontier - centroid, axis=1)  
                closest_idx = np.argmin(distances)  
                adjusted_centroids[i] = frontier[closest_idx]
            item['centers'] = adjusted_centroids.astype(int)
            item['labels'] = labels
                
    def point_to_move(self, current_loc : tuple, navi_point: tuple):

        return tuple(a - b for a , b in zip(navi_point, current_loc))  

    def draw_frontiers(self, global_map_image : Axes, frontiers):
        
        for item in frontiers:
            centers = item['centers']
            global_map_image.scatter(centers[:,1], centers[:,0], c='green',s=3)
        return global_map_image
        
    def calculate_route(self, navi_point, location, map, change):

        navi_point = np.array(navi_point).reshape(1,-1)[:,0:2]
        location = np.array(location).reshape(1,-1)[:,0:2]
        navi_point_r = np.array([int(navi_point[:,0]),int(navi_point[:,1])])
        location_r = np.array([int(location[:,0]),int(location[:,1])])        
        route_map = np.ones_like(map)        
        idx = np.where(map == OBSTACLE)
        route_map[idx[0].reshape(1,-1),
                  idx[1].reshape(1,-1)] = 0 
        idx = np.where(map == UNKNOWN)
        route_map[idx[0].reshape(1,-1),
                  idx[1].reshape(1,-1)] = 0             
        navi_point_local = navi_point_r - np.array(change)
        location_local = location_r - np.array(change)
        grid = Grid(matrix = route_map) 
        start = grid.node(location_local[1], location_local[0])
        end = grid.node(navi_point_local[1], navi_point_local[0])
        finder = AStarFinder(diagonal_movement=DiagonalMovement.always)          
        path, _ = finder.find_path(start, end, grid)
        if len(path) != 0:
            path = [(p.x, p.y) for p in path]
            path = np.array(path)[:, ::-1]
            path += change
        return path
    
    def draw_path(self, global_map_image : Axes, path, location, navi_point):

        location = np.array(location).reshape(1,-1)
        navi_point = np.array(navi_point).reshape(1,-1)
        global_map_image.scatter(path[:,1], path[:,0], c='blue',s=3)
        global_map_image.scatter(location[:,1], location[:,0], c='red',s=3)
        global_map_image.scatter(navi_point[:,1], navi_point[:,0], c='red',s=3)
        return global_map_image
    
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

    def move_along_path_and_trun_to_center(self, path, navi_point, center, camera, DC : drone_controller, records,capture_rate = 1, sleep = 0):

        for i in range(1, path.shape[0]-1):
            cord = path[i,:]
            DC.get_to_location_and_capture_vector(cord, camera, records, capture_rate=capture_rate, if_capture = False, sleep = sleep)    
        DC.get_to_location_and_capture_vector(navi_point, camera, records, capture_rate=capture_rate, if_capture = False, sleep = sleep) 
        DC.turn_to_cord_and_capture(center, camera, records,capture_rate, sleep = sleep)       

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

        explored = global_2D_map[EXPLORED]
        obstacles = global_2D_map[OBSTACLE]
        x_all = np.concatenate((explored[:,0], obstacles[:,0]),axis = 0)
        y_all = np.concatenate((explored[:,1], obstacles[:,1]),axis = 0)
        threshold = (int(max(location[0], x_all.max())), 
                     int(min(location[0], x_all.min())), 
                     int(max(location[1], y_all.max())), 
                     int(min(location[1], y_all.min())))            
        x_max = threshold[0]
        x_min = threshold[1]
        y_max = threshold[2]
        y_min = threshold[3]
        map = np.zeros(
            (x_max - x_min + 1, y_max - y_min + 1)
        ).astype(int)  
        map[obstacles[:,0].reshape(1,-1) - x_min,
            obstacles[:,1].reshape(1,-1) - y_min] = OBSTACLE
        map[explored[:,0].reshape(1,-1) - x_min,
            explored[:,1].reshape(1,-1) - y_min] = EXPLORED                
        change = (x_min, y_min)
        return map, change  
    
    def draw_targets(self, targets):

        plt.figure()
        plt.scatter(targets[:, 1], targets[:, 0], c='red', s=1)
        return plt.gca()
    
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
        for i, item in enumerate(global_bgr):
            explore_saver = image_saver(config['now'],config['Record_root'],'explore/' + str(i))
            explore_saver.save_list(item)
    
    def calculate_all_path(self, records):
        
        all_dist = 0
        for item in records['path']:
            for i in range(item.shape[0]-1):
                start = item[i]
                end = item[i+1]
                dist = norm(start-end)
                all_dist += dist 
        records['cost'] = all_dist 

    def save_path(self, path, record):

        if path.shape[-1] == 3:
            record['path'].append(path)
        else:
            location = record['location'][-1]
            height = location[-1]
            height_vector = np.ones((path.shape[0],1)) * height
            path = np.concatenate((path, height_vector),axis = -1)
            record['path'].append(path)

    def array_to_dict(self,array):

        dict_array = dict()
        for item in array:
            dict_array[tuple(item.tolist())] = []
        return dict_array 

    def get_explored_at_height(self, height, global_point_cloud, start_location):
        
        from Point_cloud.Map_element import OBSTACLE, EXPLORED, UNKNOWN
        idx = np.where((global_point_cloud[:,-1] <= height + 0.1) & (global_point_cloud[:,-1] >= height - 0.1))
        map_point = global_point_cloud[idx][:,0:2]

        min_point = np.min(map_point, axis=0).astype(int)  
        temp = np.concatenate((min_point.reshape(1,-1),np.array(start_location[0:2]).reshape(1,-1)),axis=0)
        min_point = np.min(temp, axis=0).astype(int)
        max_point = np.max(map_point, axis=0).astype(int)
        temp = np.concatenate((max_point.reshape(1,-1),np.array(start_location[0:2]).reshape(1,-1)),axis=0)        
        max_point = np.max(temp, axis=0).astype(int)        
        map_point = map_point.astype(int) - np.repeat(min_point.reshape(1,-1), map_point.shape[0], axis=0)
        Len = (max_point - min_point + 1).astype(int)
        matrix = np.ones((Len[0], Len[1]), dtype=np.int8) * UNKNOWN
        matrix[
            map_point[:,0].reshape(1,-1),
            map_point[:,1].reshape(1,-1),
        ] = OBSTACLE

        for w in range(0,2):
            add_items = ()
            map_point = np.where(matrix == OBSTACLE)
            map_point = np.array(map_point).T
            for i in range(-1,2):
                for j in range(-1,2):
                        item = map_point + np.array([[i,j]])
                        add_items += (item, )
            add_items = np.concatenate(add_items, axis = 0)
            for i in range(add_items.shape[-1]):
                add_items = add_items[(add_items[:, i] >= 0) & (add_items[:, i] < matrix.shape[i])]
            map_point = np.concatenate((map_point, add_items), axis = 0)
            matrix[
                map_point[:,0].reshape(1,-1),
                map_point[:,1].reshape(1,-1),
            ] = OBSTACLE 

        # get explored area
        explored_map = np.copy(matrix)
        stack = []
        stack.append(np.array(start_location).astype(int)[0:2] - min_point)
        while len(stack) !=0 :
            point = stack.pop()
            explored_map[point[0],point[1]] = EXPLORED
            neighbours = ()
            for i in range(-1,2):
                for j in range(-1,2):
                        neighbour = point + np.array([i,j])
                        neighbours += (neighbour.reshape(1,-1), )
            neighbours = np.concatenate(neighbours, axis = 0)
            for i in range(neighbours.shape[-1]):
                neighbours = neighbours[(neighbours[:, i] >= 0) & (neighbours[:, i] < matrix.shape[i])]
            for neighbour in neighbours:
                if explored_map[neighbour[0],neighbour[1]] == UNKNOWN:
                    stack.append(neighbour)
        ax = self.draw_current_map(explored_map,start_location,min_point)
        ax.figure.savefig('test.png',dpi=300)
        print('ok')

        explored = np.where(explored_map == EXPLORED)
        explored = np.array(explored).T
        explored = explored + min_point
        return explored
    
    def get_map_at_current_low_height(self, x: dict, y: dict, z: dict, 
                                  location: tuple, explore_range: int, PT: point_cloud):
        x, y, z = PT.squeeze_point_cloud(x, y, z)
        x_all, y_all, z_all = x.flatten(), y.flatten(), z.flatten()
        
        valid_idx = ~np.isnan(x_all) & ~np.isnan(y_all) & ~np.isnan(z_all)
        x_all, y_all, z_all = x_all[valid_idx], y_all[valid_idx], z_all[valid_idx]

        # threshold = (
        #     int(max(location[0], x_all.max())), 
        #     int(min(location[0], x_all.min())), 
        #     int(max(location[1], y_all.max())), 
        #     int(min(location[1], y_all.min()))
        # )
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
    
    def get_map_at_current_upper_height(self, x: dict, y: dict, z: dict, 
                                  location: tuple, PT: point_cloud):
        x, y, z = PT.squeeze_point_cloud(x, y, z)
        x_all, y_all, z_all = x.flatten(), y.flatten(), z.flatten()
        
        valid_idx = ~np.isnan(x_all) & ~np.isnan(y_all) & ~np.isnan(z_all)
        x_all, y_all, z_all = x_all[valid_idx], y_all[valid_idx], z_all[valid_idx]
        # threshold = (
        #     int(max(location[0], x_all.max())), 
        #     int(min(location[0], x_all.min())), 
        #     int(max(location[1], y_all.max())), 
        #     int(min(location[1], y_all.min()))
        # )
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

    def lowFBE(self,DC:drone_controller,PT:point_cloud,camera,records,config,path_saver):
        
        global_map_saver = image_saver_plt(config['now'],config['Record_root'],f'lowFBE/{self.lowBEF_count}/global_map')
        target_map_saver = image_saver_plt(config['now'],config['Record_root'],f'lowFBE/{self.lowBEF_count}/target_map')
        path_saver = image_saver_plt(config['now'],config['Record_root'],f'lowFBE/{self.lowBEF_count}/path')
        ori_global_map_saver = image_saver_plt(config['now'],config['Record_root'],f'lowFBE/{self.lowBEF_count}/ori_global_map')

        while True:
            X,Y,Z,bgr,location,pose = self.get_current_point_cloud(DC,PT,camera,records)

            '''注意bgr对应的顺序为0，1，4，2，前左后右'''
            global_target_loc = self.VMP.get_global_target_loc(bgr, X, Y, Z)
            global_tagrget_loc_2D = np.array(np.array(global_target_loc)[:, :2]).astype(np.int32) if len(global_target_loc) != 0 else []

            explore_range = config['OurExplore']['low_explore_range']
            map, change, points_3d = self.get_map_at_current_low_height(X,Y,Z,location,explore_range, PT)

            '''记录低空的点云'''
            self.ret_global_x.append(X); self.ret_global_y.append(Y); self.ret_global_z.append(Z); self.ret_global_bgr.append(bgr)
            records['low_location'].append(location); 
            records['low_pose'].append(pose)

            self.DL.get_record(DC, PT, points_3d)
            
            explored_map = self.get_explored_map(map, location, change, explore_range)
            self.update_global_map(explored_map, location, change, self.current_global_2D_map, self.current_low_history_location)
            
            # '''裁剪explore_map'''
            # self.current_global_2D_map[EXPLORED] = np.unique(self.current_global_2D_map[EXPLORED], axis = 0)
            # self.current_global_2D_map[OBSTACLE] = np.unique(self.current_global_2D_map[OBSTACLE], axis = 0)
            # total = self.current_global_2D_map[EXPLORED].tolist()
            # # limit = config['OurExplore']['low_explore_range'] * 2
            limit = config['OurExplore']['low_explore_range'] * config['OurExplore']['low_explore_limit']
            
            # stay = []
            # for i, loc in enumerate(total):
            #     if abs(loc[0] - self.landing_location[0]) > limit or abs(loc[1] - self.landing_location[1]) > limit:
            #         continue
            #     stay.append(i)
            # self.current_global_2D_map[EXPLORED] = self.current_global_2D_map[EXPLORED][stay]

            '''检查是否需要合并已知点'''
            self.current_global_2D_map[EXPLORED] = self.check_meger_known_points(self.current_known_global_2D_map, self.current_global_2D_map[EXPLORED])
            
            '''计算探索率'''
            explored_rate = len(self.current_global_2D_map[EXPLORED]) / (explore_range * 2) ** 2
            global_explored_map, change = self.get_global_explored_map(location, self.current_global_2D_map)
            frontiers, frontiers_map = self.get_frontiers(global_explored_map, change)
            
            global_map_image = self.draw_global_map_and_frontier(frontiers,self.current_global_2D_map, self.current_low_history_location)
            ori_global_map_saver.save(global_map_image) 

            '''update target and draw'''
            self.VMP.update_current_target_map(global_tagrget_loc_2D)

            '''裁剪frontier并且check frontier周围的target数量'''
            frontier_stay = []
            for idx, frontier in enumerate(frontiers):
                points = frontier['frontier'].tolist()
                stay = []
                for i, point in enumerate(points):
                    dist = np.linalg.norm(np.array(point)[:2] - \
                                          np.array(self.landing_location)[:2])
                    # if abs(point[0] - self.landing_location[0]) > limit or abs(point[1] - self.landing_location[1]) > limit:
                    if dist > limit:
                        continue
                    stay.append(i)
                if len(stay) < config['OurExplore']['low_frontier_size']: 
                    continue
                else:
                    frontier['frontier'] = frontier['frontier'][stay]
                    frontier['size'] = len(stay)
                    '''检查会不会放弃这个frontier'''
                    target_num = self.VMP.check_frontier_around_target(frontier['frontier'], self.config['OurExplore']['frontier_check_range'])
                    if target_num <= explored_rate * self.config['OurExplore']['minimum_target_num'] and np.random.rand() < self.config['OurExplore']['give_up_prob']: 
                        continue
                    else: frontier_stay.append(idx)
                    # frontier_stay.append(idx)
            frontiers = [frontiers[i] for i in frontier_stay]
            
            global_map_image = self.draw_global_map_and_frontier(frontiers,self.current_global_2D_map, self.current_low_history_location)
            self.VMP.draw_current_map_and_frontier_and_target(frontiers, self.current_global_2D_map, self.current_low_history_location,map_saver=target_map_saver)
            
            global_map_saver.save(global_map_image) 
            if len(frontiers) == 0:
                return frontiers_map, change
            mini_frontier_size = config['OurExplore']['low_frontier_divide_size']
            '''分割较大的frontier'''
            self.divide_frontiers_to_bins(frontiers, location, mini_frontier_size)
            
            if len(frontiers) > 1:
                navi_point, _ = self.VMP.select_navi_point_by_dist_and_LLM(frontiers, None)
                path = self.calculate_route(navi_point, np.array(location, dtype=np.int32), frontiers_map, change)
            else:
                path, navi_point = self.find_surround_frontier(np.array(location, dtype=np.int32), frontiers, frontiers_map, change)
                
            path_image = self.draw_path(global_map_image,path,location,navi_point)
            path_saver.save(path_image)
            self.move_along_path(path, navi_point, DC, records,capture_rate = 1, if_capture = False, sleep = 0)
            self.save_path(path, records)

    def main_loop(self,DC:drone_controller,PT:point_cloud,camera,records,config,path_saver):
        '''低空全局saver'''
        self.global_lower_2D_map = dict()
        self.global_lower_2D_map[OBSTACLE] = np.array([])
        self.global_lower_2D_map[EXPLORED] = np.array([])
        self.low_history_location = np.array([])
        self.low_global_map_saver = image_saver_plt(config['now'],config['Record_root'],f'low/low_global_map')
        self.lift_path_saver = image_saver_plt(config['now'],config['Record_root'],f'low/lift_path')
        self.lift_up_view = image_saver(config['now'],config['Record_root'],'low/lift_up_view')
        
        '''高空全局saver'''
        self.global_higher_2D_map = dict()
        self.global_higher_2D_map[OBSTACLE] = np.array([])
        self.global_higher_2D_map[EXPLORED] = np.array([])
        self.high_history_location = []
        self.high_global_map_saver = image_saver_plt(config['now'],config['Record_root'],f'high/high_global_map')
        self.fall_path_saver = image_saver_plt(config['now'],config['Record_root'],f'high/fall_path')
        self.high_path_saver = image_saver_plt(config['now'],config['Record_root'],f'high/high_path')
        self.fall_down_view = image_saver(config['now'],config['Record_root'],'high/fall_down_view')

        '''人群识别saver'''
        self.target_record = np.array([])
        
        self.ret_global_x, self.ret_global_y, self.ret_global_z, self.ret_global_bgr = [], [], [], []
        self.ret_up_global_x, self.ret_up_global_y, self.ret_up_global_z = [], [], []

        '''首先在高空做FBE'''
        location, pose = DC.get_world_location_pose()
        self.DL.to_target_height(DC, location, pose, records)
        self.first_lift = True

        while True:

            self.DL.turn_to_down(DC)
            '''up_frontiers_map以及up_change用来移到可降落位置上方,flg表示是否高空探索没有frontier了'''
            up_frontiers_map, up_change, if_lower_explore, if_no_frontier = self.upFBE(DC,PT,camera,records,config,path_saver)
            
            '''降落'''
            landing_point = self.DL.Fall(self.global_higher_2D_map, DC, self.global_lower_2D_map[OBSTACLE].tolist(), records, self.fall_down_view)
            location, pose = DC.get_world_location_pose()
            path = self.calculate_route(landing_point, location, up_frontiers_map, up_change)
            img = self.draw_global_map_and_frontier([], self.global_higher_2D_map, self.high_history_location)
            path_img = self.draw_path(img, path, location, landing_point)
            self.fall_path_saver.save(path_img)
            self.move_along_path(path, np.array(landing_point), DC, records,capture_rate = 1, if_capture = False, sleep = 0)
            '''保存视角'''
            down_bgr = DC.get_single_bgr_image(3)
            self.fall_down_view.save(down_bgr)
            '''记录轨迹,在path最后一个维度加高度'''
            location, pose = DC.get_world_location_pose()
            height = location[-1]
            height_vector = np.ones((path.shape[0],1)) * height
            path = np.concatenate((path, height_vector),axis = -1)

            '''降落'''
            self.DL.to_target_height(DC, location, pose, records, down=True)
            self.save_path(path, records)

            if if_lower_explore:

                '''结束高空探索的位置'''
                self.landing_location, self.landing_pose = DC.get_world_location_pose()
                lower_explored = self.global_lower_2D_map[EXPLORED]
                lower_obstacle = self.global_lower_2D_map[OBSTACLE]
                
                '''低空探索范围是一个正方形'''
                self.current_global_2D_map = dict()
                self.current_global_2D_map[EXPLORED] = np.array([])
                self.current_global_2D_map[OBSTACLE] = lower_obstacle
                self.current_known_global_2D_map = []
                self.current_low_history_location = []
                # limit = self.config['OurExplore']['low_explore_range'] * 2
                limit = config['OurExplore']['low_explore_range'] * config['OurExplore']['low_explore_limit']
                self.DL.turn_to_up(DC)
                self.DL.clear_points_up()
                # self.VMP.clear_global_target_map()
                self.VMP.meger_current_to_global(self.global_lower_2D_map[EXPLORED])
                self.VMP.clear_current_target_map()
                '''
                筛选出本次低空探索范围内的过去已探索点，考虑全部的障碍点，不考虑前面已经detect到的目标点
                current_known_global_2D_map用于补全当前低空位置探索范围内的已探索点
                '''
                for loc in lower_explored:
                    # if abs(loc[0] - self.landing_location[0]) <= limit and abs(loc[1] - self.landing_location[1]) <= limit:
                    self.current_known_global_2D_map.append(loc)
                
                frontiers_map, change = self.lowFBE(DC,PT,camera,records,config,path_saver)
                self.lowBEF_count += 1

                '''更新全局地图'''
                self.global_lower_2D_map[EXPLORED] = self.merge_points(self.current_global_2D_map[EXPLORED], self.global_lower_2D_map[EXPLORED])
                self.global_lower_2D_map[OBSTACLE] = self.merge_points(self.current_global_2D_map[OBSTACLE], self.global_lower_2D_map[OBSTACLE])
                self.low_history_location = self.merge_points(np.array(self.current_low_history_location), self.low_history_location)
                self.target_record = self.merge_points(self.target_record, np.array(list(self.VMP.current_target_2D_map), dtype=np.int32))
                img = self.draw_global_map_and_frontier([], self.global_lower_2D_map, self.low_history_location)
                img.scatter(self.target_record[:,1], self.target_record[:,0], c='blue', s=1)
                self.low_global_map_saver.save(img)

            if not if_no_frontier:

                '''低空探索结束，升空'''
                takeoff_point = self.DL.Lift(self.global_lower_2D_map, DC, records, self.lift_up_view)
                location, pose = DC.get_world_location_pose()
                path = self.calculate_route(takeoff_point, location, frontiers_map, change)
                img = self.draw_global_map_and_frontier([], self.global_lower_2D_map, self.low_history_location)
                path_img = self.draw_path(img, path, location, takeoff_point)
                self.lift_path_saver.save(path_img)
                self.move_along_path(path, np.array(takeoff_point), DC, records,capture_rate = 1, if_capture = False, sleep = 0)
                '''保存视角'''
                up_bgr = DC.get_single_bgr_image(3)
                self.lift_up_view.save(up_bgr)
                '''记录轨迹'''
                location, pose = DC.get_world_location_pose()
                height = location[-1]
                height_vector = np.ones((path.shape[0],1)) * height
                path = np.concatenate((path, height_vector),axis = -1)

                '''升空'''
                self.DL.to_target_height(DC, location, pose, records)
                self.save_path(path, records)
                self.first_lift = False

            else:  
                '''高空探索结束'''      
                return self.ret_global_x, self.ret_global_y, self.ret_global_z, self.ret_global_bgr, self.ret_up_global_x, self.ret_up_global_y, self.ret_up_global_z
        
    def upFBE(self,DC:drone_controller,PT:point_cloud,camera,records,config,path_saver):

        up_count = 0
        if_no_frontier = False
        if_lower_explore = False

        while True:

            camera = [0,1,2,3,4]
            X,Y,Z,bgr,location,pose = self.get_current_point_cloud(DC,PT,camera,records, up=True)
            
            '''注意bgr对应的顺序为0，1，4，2，前左后右'''

            self.ret_up_global_x.append(X); self.ret_up_global_y.append(Y); self.ret_up_global_z.append(Z)

            explore_range = config['OurExplore']['up_explore_range']
            map, change, points_3d = self.get_map_at_current_upper_height(X,Y,Z,location, PT)
            down_img = self.DL.get_record_down(DC, PT, points_3d)
            
            explored_map = self.get_explored_map(map, location, change, explore_range)
            self.update_global_map(explored_map, location, change, self.global_higher_2D_map, self.high_history_location)
            
            global_explored_map, change = self.get_global_explored_map(location, self.global_higher_2D_map)
            frontiers, frontiers_map = self.get_frontiers(global_explored_map, change)

            '''高空再筛一遍'''
            frontier_stay = []
            for idx, frontier in enumerate(frontiers):
                if frontier['size'] < config['OurExplore']['up_frontier_size']: continue
                else: frontier_stay.append(idx)
            frontiers = [frontiers[i] for i in frontier_stay]

            global_map_image = self.draw_global_map_and_frontier(frontiers,self.global_higher_2D_map, self.high_history_location)
            self.high_global_map_saver.save(global_map_image) 

            # 若没有frontier，标记
            if len(frontiers) == 0:
                if_no_frontier = True            

            '''加入大模型的部分'''
            if up_count or self.first_lift:
                # panoramic_img, panoramic_segment = self.LLM.get_panoramic_image_down(bgr)
                if_mark = []
                for i in range(len(bgr)):
                    if_mark.append(False)
                panoramic_image, marked_imgs = self.LLM.get_visual_prompt(bgr, if_mark, 1)
                '''只传一张从上到下的图片'''
                top_down_img = bgr[3]
                llm_score_thresh = config['OurExplore']['llm_score_thresh']
                if_lower_explore = self.LLM.LLMChooseDown2(top_down_img, llm_score_thresh)

            if if_lower_explore:
                return frontiers_map, change, if_lower_explore, if_no_frontier
 
            # 大模型决定不下降，去往下一个frontier
            else:
                if not if_no_frontier:

                    mini_frontier_size = config['OurExplore']['up_frontier_divide_size']
                    '''分割较大的frontier'''
                    self.divide_frontiers_to_bins(frontiers, location, mini_frontier_size)                          
                    
                    path, navi_point = self.find_surround_frontier(location, frontiers, frontiers_map, change)
                        
                    path_image = self.draw_path(global_map_image,path,location,navi_point)
                    self.high_path_saver.save(path_image)
                    self.move_along_path(path, navi_point, DC, records,capture_rate = 1, if_capture = False, sleep = 0)
                    self.save_path(path, records)
                    up_count += 1
                else:
                    return frontiers_map, change, if_lower_explore, if_no_frontier
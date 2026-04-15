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
from numpy.linalg import norm
import csv
import airsim

class explore():

    def __init__(self, config):

        self.config = config
        self.frontier = frontier(config['Explore'])
        self.history_location = []

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
    
    def update_global_map(self, map, location, change, global_2D_map):

        loc = (location[0], location[1])
        self.history_location.append(loc)

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

    def draw_global_map_and_frontier(self, frontiers,global_2D_map):
        
        obstacle = global_2D_map[OBSTACLE]
        explored = global_2D_map[EXPLORED]
        plt.figure() 
        plt.scatter(explored[:, 1], explored[:, 0], 
                    c='gray',s=3)
        plt.scatter(obstacle[:, 1], obstacle[:, 0], 
                    c='black',s=3)                
        history_location = np.array(self.history_location)
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
                distances = np.linalg.norm(frontier - centroid, axis=1)  
                closest_idx = np.argmin(distances)  
                adjusted_centroids[i] = frontier[closest_idx]
            item['centers'] = adjusted_centroids.astype(int)
            item['labels'] = labels
                
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
    
    def move_along_path(self, path: np.ndarray, navi_point, DC : drone_controller, records,capture_rate = 1, if_capture = True, sleep = 0, by_what_mean="Pose"):
        if by_what_mean == "Pose":
            for i in range(1, path.shape[0]-1):
                cord = path[i,:]
                DC.get_to_location_and_capture_vector(cord, [0], records, capture_rate=capture_rate, if_capture = if_capture, sleep = sleep)    
            DC.get_to_location_and_capture_vector(navi_point, [0,1,2,4], records, capture_rate=capture_rate, if_capture = if_capture, sleep = sleep)
        elif by_what_mean == "Path":
            location, pose = DC.get_world_location_pose()
            path = path.tolist()
            cords = [airsim.Vector3r(path[i][0], path[i][1], int(location[2])) for i in range(1, len(path))]
            DC.client.moveOnPathAsync(cords, 1).join()
        else:
            raise Exception(f"no implementation for '{by_what_mean}'")
    
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

    def get_current_point_cloud(self, DC : drone_controller, PT : point_cloud, camera: list, records, global_X,global_Y,global_Z,global_bgr):
        
        bgr ,depth = DC.record_data(records,camera)
        location = records['location'][-1]
        pose = records['pose'][-1]
        X = dict(); Y = dict(); Z = dict()
        location, pose = DC.get_world_location_pose()
                
        for i in range(len(camera)):                         
            intrinsic_matrix = DC.get_intrinsic_matrix(camera[i])
            pose_matrix = DC.get_pose_matrix(pose,camera[i])                
            x,y,z = PT.get_point_clouds_from_depth(intrinsic_matrix, 
                                                pose_matrix,
                                                depth[i],
                                                camera = camera[i])            
            x,y,z = PT.get_global_point_cloud(location, x,y,z)
            
            X.update(x); Y.update(y); Z.update(z)
        global_X.append(X)
        global_Y.append(Y) 
        global_Z.append(Z)
        global_bgr.append(bgr)
        
        return X,Y,Z,bgr,location,pose
            
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

    def get_map_at_current_height(self, x : dict, y : dict, z : dict, PT: point_cloud,
                                  location : tuple, threshold_draw : tuple = None):

        x,y,z = PT.squeeze_point_cloud(x,y,z)
        x_all = x.reshape(1,-1); y_all = y.reshape(1,-1); z_all = z.reshape(1,-1)
        idx1 = ~np.isnan(x_all)
        idx2 = ~np.isnan(y_all)
        idx3 = ~np.isnan(z_all)
        idx = idx1 & idx2 & idx3
        x_all = x_all[idx]
        
        y_all = y_all[idx]
        z_all = z_all[idx]

        threshold = (
            int(self.boundary['x_max']),
            int(self.boundary['x_min']),
            int(self.boundary['y_max']),
            int(self.boundary['y_min'])
        )        

        # get the boundry that is in the same height of the drone
        idx = np.where((z_all <= location[2] + 0.1) & (z_all >= location[2] - 0.1))
        x = x_all[idx]
        y = y_all[idx]
        z = z_all[idx]
       
        # go to grid
        x = x.astype(np.int32)
        y = y.astype(np.int32)

        # get map
        if threshold_draw is None:
            map, change = PT.create_2D_map(x,y,threshold)
        else:
            map, change = PT.create_2D_map(x,y,threshold_draw)
        
        return map.astype(int), change
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
from Perception.GroundingDINO import GroundingDINO_detector
from Point_cloud.Point_cloud import point_cloud
from Perception.GeneralizedLoss import gereralizedloss
import cv2
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import open3d
import matplotlib.pyplot as plt  
import matplotlib.cm as cm 
from utils.saver import image_saver, image_saver_plt
from matplotlib.cm import get_cmap
from numpy.linalg import norm
import itertools 
from scipy.spatial import distance
import airsim
from Others.ValueMap.ValueMap import ValueMap
from Others.DroneLift.DroneLift import DroneLift
from Others.IntuitionMap.gpt4o_integration import LLMJudge
import csv

class explore():

    def __init__(self, config):

        self.config = config
        self.frontier = frontier(config)
        self.history_location = []

        self.get_navigation_bound()

    def get_navigation_bound(self):

        path = self.config['route_path']
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
        navigation_path = self.config['navigation_path']
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
            exit() 
    
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

    def get_target_detection(self, config, bgr : list, GD : GroundingDINO_detector, location,X,Y,Z,camera,GD_saver):

        prompt = config['Count']['detection_prompt']
        detection, imgs = GD.inference_batch_images(bgr,16,prompt)
        GD_saver.save_list(imgs)
        dists = []
        target_locs = []
        explore_range = config['Explore']['explore_range']
        for i in range(len(detection['boxes'])):
            boxes = detection['boxes'][i]
            phrased_boxes = GD.phrase_GD_boxes(boxes, X['0'])
            if len(phrased_boxes) != 0:
                for j in range(len(phrased_boxes)):
                    distance, target_loc = GD.get_target_loc(phrased_boxes[j],location,X,Y,Z,str(camera[i]))
                    if distance <= explore_range:
                        dists.append(distance)
                        target_locs.append(target_loc)
        return target_locs, dists

        # targets_bin, targets_bin_info = EX.divide_points_to_directions(target_locs, location, bin_num)

    def get_targets_from_density_maps(self, GL : gereralizedloss, density_threshold, cameras, config ,location_list,global_X,global_Y,global_Z,global_bgr,height_range = None):

        bgrs = []
        axs = []
        for i, item in enumerate(global_bgr):
            bgrs += item
        density_maps = GL.inference_images(bgrs)
        # density_imgs = GL.draw_density_maps(density_maps)
        # for i in range(0, len(density_imgs), len(cameras)):  
        #     img_batch = density_imgs[i:i + len(cameras)]        
        #     density_saver = image_saver_plt(config['now'],config['Record_root'],'density/'+str(int(i/len(cameras))))            
        #     density_saver.save_list(img_batch)       
        targets = ()
        see_target_location = ()
        for i, density_map in enumerate(density_maps):
            camera = str(cameras[int(i % len(cameras))])
            X = global_X[int(i / len(cameras))] 
            Y = global_Y[int(i / len(cameras))]
            Z = global_Z[int(i / len(cameras))]            
            # density_map = (density_map - density_map.min()) / \
            #     (density_map.max() - density_map.min())
            idx = np.where(density_map >= density_threshold)
            idx_0 = (idx[0] / density_map.shape[0] * X[camera].shape[0]).astype(int)
            idx_1 = (idx[1] / density_map.shape[1] * X[camera].shape[1]).astype(int) 

            # idx_0_show = (idx[0] / density_map.shape[0] * bgrs[i].shape[0]).astype(int)
            # idx_1_show = (idx[1] / density_map.shape[1] * bgrs[i].shape[1]).astype(int)
            # plt.figure()
            # plt.imshow(cv2.cvtColor(bgrs[i], cv2.COLOR_BGR2RGB))
            # plt.scatter(idx_1_show, idx_0_show, c = 'red', s=0.5)
            # axs.append(plt.gca())

            x = X[camera][idx_0.reshape(1,-1),idx_1.reshape(1,-1)].reshape(-1,1)
            y = Y[camera][idx_0.reshape(1,-1),idx_1.reshape(1,-1)].reshape(-1,1)
            z = Z[camera][idx_0.reshape(1,-1),idx_1.reshape(1,-1)].reshape(-1,1)
            idx1 = ~np.isnan(x)
            idx2 = ~np.isnan(y)
            idx3 = ~np.isnan(z)
            idx = idx1 & idx2 & idx3
            if height_range != None:
                idx4 = (z>=height_range[1]) & (z<=height_range[0])
                idx = idx & idx4
            x = x[idx].reshape(-1,1)
            y = y[idx].reshape(-1,1)  
            z = z[idx].reshape(-1,1)          
            target = np.concatenate((x,y,z),axis = 1)
            targets += (target,)
            location = np.array(location_list[int(i / len(cameras))]).reshape(1,-1)
            location = np.repeat(location,target.shape[0],axis=0)
            see_target_location += (location,)

        targets = np.concatenate(targets,axis = 0)
        # pcd = open3d.geometry.PointCloud()
        # pcd.points = open3d.utility.Vector3dVector(targets) 
        # open3d.visualization.draw_geometries([pcd])        
            
        return targets, np.concatenate(see_target_location,axis = 0),axs
    
    def draw_targets(self, targets):

        plt.figure()
        plt.scatter(targets[:, 1], targets[:, 0], c='red', s=1)
        return plt.gca()
    
    def voxelize_point_cloud(self, points, voxel_size):  
        """  
        将点云划分到具有不同长、宽、高的三维体素中。  
    
        参数：  
        points (numpy.ndarray): 形状为 (N, 3) 的点云数组，其中 N 是点的数量。  
        voxel_sizes (tuple): 一个包含三个浮点数的元组，分别表示体素的长、宽、高。  
    
        返回：  
        voxels (dict): 一个字典，键为体素索引元组，值为体素中包含的点列表。  
        voxel_grid_dims (tuple): 体素网格的维度（长、宽、高方向上的体素数量）。  
        min_point (numpy.ndarray): 点云中的最小点坐标。  
        """  
        # 确定点云的最小和最大坐标  
        min_point = np.min(points, axis=0)  
        max_point = np.max(points, axis=0)  
    
        # 计算体素网格的维度 
        for i, item in enumerate(voxel_size):
            if item == -1:
                voxel_size[i] = (max_point - min_point + 1)[i]
        voxel_size = tuple(voxel_size)
        voxel_grid_dims = np.ceil((max_point - min_point) / voxel_size).astype(int)  
    
        # 初始化体素字典  
        voxels = {} 
        voxels_idx = {} 
    
        # 遍历点云中的每个点  
        for i, point in enumerate(points):  
            # 计算体素索引  
            voxel_index = tuple(((point - min_point) / voxel_size).astype(int))

            # 将点添加到对应的体素中  
            if voxel_index not in voxels:  
                voxels[voxel_index] = [] 
                voxels_idx[voxel_index] = []
            voxels[voxel_index].append(point) 
            voxels_idx[voxel_index].append(i)
    
        return voxels, voxels_idx, voxel_size, min_point
    
    def get_current_point_cloud(self, DC : drone_controller, PT : point_cloud, camera: list, records, global_X,global_Y,global_Z,global_bgr):
        
        bgr ,depth = DC.record_data(records,camera)
        location = records['location'][-1]
        pose = records['pose'][-1]
        # DC.get_bgr_and_depth(camera)
        # img_saver.save_list(bgr)
        X = dict(); Y = dict(); Z = dict()
        location, pose = DC.get_world_location_pose()
        
        '''debug camera direction'''
        # points = []
        # cv2.namedWindow('debug_front',cv2.WINDOW_NORMAL)
        # cv2.namedWindow("debug_left",cv2.WINDOW_NORMAL)
        # cv2.namedWindow("debug_back",cv2.WINDOW_NORMAL)
        # cv2.namedWindow("debug_right",cv2.WINDOW_NORMAL)
        # cv2.imshow('debug_front',bgr[0])
        # cv2.imshow("debug_left",bgr[1])
        # cv2.imshow("debug_back",bgr[2])
        # cv2.imshow("debug_right",bgr[3])    
        # cv2.waitKey(0)
        
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
    
    def GaussianMixture(self, targets, GM_cluster_size):

        targets = np.array(targets)
        n_components = np.ceil(targets.shape[0] / GM_cluster_size).astype(int)
        gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)  
        # gmm.fit(targets[:,0:2])
        gmm.fit(targets) 
        # labels = gmm.predict(targets[:,0:2])
        labels = gmm.predict(targets)
        centers = gmm.means_

        # find nearest target as center
        new_centers = ()
        for i, label in enumerate(np.unique(labels)):
            center = centers[i].reshape(1,-1)
            idx = np.where(labels == label)[0]
            target = targets[idx]
            residual = target - center
            dist = norm(residual, axis = -1)
            idx = np.where(dist == min(dist))[0]
            new_center = target[idx]
            new_centers += (new_center.reshape(1,-1), )
        new_centers = np.concatenate(new_centers, axis=0)
        centers = new_centers

        cmap = get_cmap('jet')    
        colors = ()
        # 为每个点根据其标签着色  
        for label in labels:  
            # 获取当前标签对应的颜色（归一化到0-1之间）  
            color = np.array(cmap(label / np.max(labels)))[:3]
            colors += (color.reshape(1,-1), )
        colors = np.concatenate(colors, axis = 0)

        # pcd = open3d.geometry.PointCloud()
        # pcd.points = open3d.utility.Vector3dVector(targets)
        # pcd.colors = open3d.utility.Vector3dVector(colors)       
        # open3d.visualization.draw_geometries([pcd])    

        # 可视化聚类结果
        plt.figure()  
        plt.scatter(targets[:, 1], targets[:, 0], c=labels, s=40, cmap='viridis')   
        plt.scatter(centers[:, 1], centers[:, 0], c='red', s=200, alpha=0.75, marker='X')  
        plt.title('Gaussian Mixture Model Clustering')  
        plt.xlabel('Feature 1')  
        plt.ylabel('Feature 2') 
        ax = plt.gca() 
        # plt.savefig('test.png')        

        # # visit_targets = dict()
        # # for i in range(0,labels.max() + 1):
        # #     idx = np.where(labels == i)
        # #     center = centers[i]
        # #     current_targets = targets[idx]
        # #     height = np.mean(current_targets[:,2])
        # #     visit_targets[(center[0],center[1],height)] = []        
        return labels, centers, ax
    
    def GaussianMixture_voxel(self, targets, GM_cluster_size, voxels, voxels_idx, voxel_size, min_point, see_target_locations, global_point_cloud, voxelized_cloud):

        start_label = 0
        all_labels = ()
        all_centers = ()
        all_idx = ()
        for key in voxels.keys():

            target = np.array(voxels[key])
            if target.shape[0] < 2:
                continue
            n_components = np.ceil(target.shape[0] / GM_cluster_size).astype(int)
            gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
            # gmm.fit(targets[:,0:2])
            gmm.fit(target[:,0:2])
            # labels = gmm.predict(targets[:,0:2])
            labels = gmm.predict(target[:,0:2])
            labels += start_label
            start_label = max(labels) + 1
            centers = gmm.means_
            height = np.ones((centers.shape[0],1)) * ((key[-1] + 0.5) * voxel_size[-1] + min_point[-1])
            centers = np.concatenate((centers, height), axis=-1)
            idx = voxels_idx[key]
            all_labels += (labels,)
            all_centers += (centers,)
            all_idx += (idx,)

        all_labels = np.concatenate(all_labels)
        all_idx = np.concatenate(all_idx)
        all_centers = np.concatenate(all_centers,axis=0)
        labels = all_labels
        centers = all_centers
        targets = targets[all_idx]
        see_target_locations = see_target_locations[all_idx]

        # pcd2 = open3d.geometry.PointCloud()
        # pcd2.points = open3d.utility.Vector3dVector(targets)
        # pcd2.colors = open3d.utility.Vector3dVector(np.zeros_like(targets))  

        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(voxelized_cloud)

        pcd3 = open3d.geometry.PointCloud()
        pcd3.points = open3d.utility.Vector3dVector(see_target_locations)
        pcd3.colors = open3d.utility.Vector3dVector(np.zeros_like(see_target_locations))  

        # 定义起点和终点（向量）  
        start_points = targets
        end_points = see_target_locations                 
        points = np.vstack((start_points, end_points))  
        lines = []  
        for i in range(len(start_points)):  
            lines.append([i, i + len(start_points)])  # i是起点索引，i + len(start_points)是终点索引    
        # line_set2 = open3d.geometry.LineSet()  
        # line_set2.points = open3d.utility.Vector3dVector(points)    
        # line_set2.lines = open3d.utility.Vector2iVector(np.array(lines))      

        # open3d.visualization.draw_geometries([pcd,pcd3,line_set2])        

        # find nearest target as center
        new_centers = ()
        see_target_vectors = ()
        for i, label in enumerate(np.unique(labels)):
            center = centers[i].reshape(1,-1)
            idx = np.where(labels == label)[0]
            target = targets[idx]
            see_target_location = see_target_locations[idx]
            residual = target - center
            dist = norm(residual, axis = -1)
            idx = np.where(dist == min(dist))[0]
            new_center = target[idx]
            new_centers += (new_center.reshape(1,-1), )
            # get the center see_vector
            see_target_location = see_target_location[idx]
            vector = see_target_location - center
            n = norm(vector, axis=-1).reshape(-1,1)
            n = np.repeat(n, vector.shape[-1], axis=-1)
            vector = vector / n 
            see_target_vectors += (vector.reshape(1,-1),)
            
        new_centers = np.concatenate(new_centers, axis=0)
        see_target_vectors = np.concatenate(see_target_vectors,axis = 0)
        centers = new_centers

        # 定义起点和终点（向量）  
        start_points = centers 
        end_points = see_target_vectors * 5 + centers                 
        points = np.vstack((start_points, end_points))  
        lines = []  
        for i in range(len(start_points)):  
            lines.append([i, i + len(start_points)])  # i是起点索引，i + len(start_points)是终点索引    
        # line_set2 = open3d.geometry.LineSet()  
        # line_set2.points = open3d.utility.Vector3dVector(points)    
        # line_set2.lines = open3d.utility.Vector2iVector(np.array(lines))
        
        # pcd = open3d.geometry.PointCloud()
        # pcd.points = open3d.utility.Vector3dVector(end_points)
        # pcd.colors = open3d.utility.Vector3dVector(np.zeros_like(end_points))

        # pcd2 = open3d.geometry.PointCloud()
        # pcd2.points = open3d.utility.Vector3dVector(voxelized_cloud)

        # open3d.visualization.draw_geometries([pcd,pcd2,line_set2])        

        # 可视化聚类结果
        plt.figure()
        plt.scatter(targets[:, 1], targets[:, 0], c=labels, s=40, cmap='viridis')   
        plt.scatter(centers[:, 1], centers[:, 0], c='red', s=200, alpha=0.75, marker='X')  
        plt.title('Gaussian Mixture Model Clustering')  
        plt.xlabel('Feature 1')  
        plt.ylabel('Feature 2') 
        ax = plt.gca()                
        return labels, centers, targets, see_target_vectors, ax  

    def GaussianMixture_voxel2(self, targets, GM_cluster_size, voxels, voxels_idx, voxel_size, min_point, see_target_locations, global_point_cloud, voxelized_cloud):

        start_label = 0
        all_labels = ()
        all_centers = ()
        all_idx = ()
        for key in voxels.keys():

            target = np.array(voxels[key])
            if target.shape[0] < 2:
                continue
            n_components = np.ceil(target.shape[0] / GM_cluster_size).astype(int)
            gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
            # gmm.fit(targets[:,0:2])
            gmm.fit(target[:,0:2])
            # labels = gmm.predict(targets[:,0:2])
            labels = gmm.predict(target[:,0:2])
            labels += start_label
            start_label = max(labels) + 1
            centers = gmm.means_
            height = np.ones((centers.shape[0],1)) * ((key[-1] + 0.5) * voxel_size[-1] + min_point[-1])
            centers = np.concatenate((centers, height), axis=-1)
            idx = voxels_idx[key]
            all_labels += (labels,)
            all_centers += (centers,)
            all_idx += (idx,)

        all_labels = np.concatenate(all_labels)
        all_idx = np.concatenate(all_idx)
        all_centers = np.concatenate(all_centers,axis=0)
        labels = all_labels
        centers = all_centers
        targets = targets[all_idx]
        see_target_locations = see_target_locations[all_idx]
                
        return labels, centers, targets      
            
    def reach_targets_greedy(self,targets, DC:drone_controller):
        location, pose = DC.get_world_location_pose()
        location = np.array(location).reshape(1,-1)
        target_locs = np.array(list(targets.keys()))
        residual = target_locs - location
        dist = np.linalg.norm(residual, axis=1)
        idx = np.where(dist == dist.min())[0][0]
        key = tuple(target_locs[idx].squeeze())
        targets.pop(key) 
        return target_locs[idx]       

    def get_global_point_cloud(self, global_X, global_Y, global_Z):

        temp_X = ()
        temp_Y = ()
        temp_Z = ()
        for i, item_x in enumerate(global_X):
            item_y = global_Y[i]
            item_z = global_Z[i]
            temp_x = ()
            temp_y = ()
            temp_z = ()
            for key in item_x.keys():
                temp_x += (item_x[key].reshape(-1,1),)
                temp_y += (item_y[key].reshape(-1,1),)
                temp_z += (item_z[key].reshape(-1,1),)
            temp_x = np.concatenate(temp_x,axis = 0)
            temp_y = np.concatenate(temp_y,axis = 0)
            temp_z = np.concatenate(temp_z,axis = 0)
            temp_X += (temp_x,)
            temp_Y += (temp_y,)
            temp_Z += (temp_z,)
        temp_X = np.concatenate(temp_X,axis = 0)
        temp_Y = np.concatenate(temp_Y,axis = 0)
        temp_Z = np.concatenate(temp_Z,axis = 0)

        idx1 = ~np.isnan(temp_X)
        idx2 = ~np.isnan(temp_Y)
        idx3 = ~np.isnan(temp_Z)
        idx = idx1 & idx2 & idx3
        temp_X = temp_X[idx].reshape(-1,1)
        temp_Y = temp_Y[idx].reshape(-1,1)
        temp_Z = temp_Z[idx].reshape(-1,1)
        point_cloud = np.concatenate((temp_X,temp_Y,temp_Z),axis = -1)

        # pcd = open3d.geometry.PointCloud()
        # pcd.points = open3d.utility.Vector3dVector(point_cloud)        
        # open3d.visualization.draw_geometries([pcd])
        return point_cloud

    # def get_map_from_global_point_cloud(self):

    def get_norm_vectors(self, targets, alignment_direction = np.array([0, 0, -1])):

        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(targets)
        # pcd = pcd.voxel_down_sample(2)
        pcd.colors = open3d.utility.Vector3dVector(np.zeros_like(targets))
        pcd.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=50, max_nn=100))
        pcd.orient_normals_to_align_with_direction(alignment_direction) 
        # open3d.visualization.draw_geometries([pcd], point_show_normal=True)

        normal_vectors = np.asarray(pcd.normals)
        dot_product = normal_vectors @ alignment_direction.T
        angle_rad = np.arccos(dot_product)
        idx = angle_rad < np.pi / 2
        normal_vectors = normal_vectors[idx]  
        targets = targets[idx]    
        return  normal_vectors, targets

    def save_explore_bgr(self, config,global_bgr):
        for i, item in enumerate(global_bgr):
            explore_saver = image_saver(config['now'],config['Record_root'],'explore/' + str(i))
            explore_saver.save_list(item)

    def draw_gd_and_target_3d(self, gd, targets):
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(targets)
        colors = np.repeat(np.array([1,0,0]).reshape(1,-1),targets.shape[0],axis = 0)
        pcd.colors = open3d.utility.Vector3dVector(colors)
        pcd2 = open3d.geometry.PointCloud()
        pcd2.points = open3d.utility.Vector3dVector(gd)
        pcd2.colors = open3d.utility.Vector3dVector(np.zeros_like(gd))
        open3d.visualization.draw_geometries([pcd,pcd2])

    def generate_vertical_vector(self, normal_vector, theta):  
        # 确保法向量是单位向量 
        n = norm(normal_vector, axis=-1).reshape(-1,1)
        n = np.repeat(n,normal_vector.shape[1],axis=-1) 
        normal_vector = normal_vector / n  
        
        # # 选择一个不与法向量共线的向量  
        # if abs(normal_vector[2]) < 0.9:  # 避免与 z 轴共线  
        #     a = np.array([1, 0, 0])  
        # else:  
        #     a = np.array([0, 1, 0])
        # 
        a = np.array([0, 1, 0]) 
        
        # 计算垂直于法向量的向量  
        u = np.dot(normal_vector, a.reshape(-1,1))
        u = np.repeat(u,normal_vector.shape[1],axis=-1) * normal_vector
        u = a - u
        n = norm(u, axis=-1).reshape(-1,1)
        n = np.repeat(n,u.shape[1],axis=-1)
        u = u / n  # 单位化  
        
        # 构造旋转矩阵
        results = ()
        for i, vector in enumerate(normal_vector):   
            axis = vector  
            theta_rad = np.radians(theta)  
            cos_theta = np.cos(theta_rad)  
            sin_theta = np.sin(theta_rad)  
            one_minus_cos_theta = 1 - cos_theta          
            rotation_matrix = np.array([  
                [cos_theta + axis[0]**2 * (1 - cos_theta),  
                axis[0] * axis[1] * one_minus_cos_theta - axis[2] * sin_theta,  
                axis[0] * axis[2] * one_minus_cos_theta + axis[1] * sin_theta],  
                [axis[1] * axis[0] * one_minus_cos_theta + axis[2] * sin_theta,  
                cos_theta + axis[1]**2 * (1 - cos_theta),  
                axis[1] * axis[2] * one_minus_cos_theta - axis[0] * sin_theta],  
                [axis[2] * axis[0] * one_minus_cos_theta - axis[1] * sin_theta,  
                axis[2] * axis[1] * one_minus_cos_theta + axis[0] * sin_theta,  
                cos_theta + axis[2]**2 * (1 - cos_theta)]  
            ])          
            # 计算旋转后的向量 
            U = u[i]
            v = np.dot(rotation_matrix, U)          
            # 确保 v 是单位向量  
            v = v / norm(v)
            results += (v.reshape(1,-1),)
        results = np.concatenate(results,axis=0)          
        return results    

    def get_potential_navi_vectors(self, targets, centers, norm_vectors_mean, interval, degree, global_point_cloud, see_target_vectors):
        '''
           degree : angle between navi_point and the plane of the norm_points
        '''

        thetas = np.arange(0, 360, interval)
        start_points = ()
        navi_vectors = ()
        references = ()
        idxs = ()
        show_navi_vectors = ()
        new_see_target_vectors = ()

        reference = np.copy(norm_vectors_mean)
        reference[:,-1] = 0
        n = norm(reference, axis=-1).reshape(-1,1)
        n = np.repeat(n,reference.shape[1],axis=-1)
        reference = reference / n

        for theta in thetas: 
            # first get target plane
            vertical_vector = self.generate_vertical_vector(norm_vectors_mean, theta)
       
            # only vertical_vector close to the ground stay
            # cos_theta = np.sum(vertical_vector * reference, axis = -1)  
            # angle_rad = np.arccos(cos_theta)
            # idx = np.where(angle_rad  < np.pi / 2)[0]
            idx = np.arange(0,vertical_vector.shape[0])
            idxs += (idx.reshape(1,-1),)

            # calculate navi_vector
            length = math.tan(degree / 180 * math.pi)
            vector = norm_vectors_mean[idx] * length
            # navi_vector is degree between the plane
            navi_vector = vector + vertical_vector[idx]
            # normalize navi_vector
            n = norm(navi_vector, axis=-1).reshape(-1,1)
            n = np.repeat(n,navi_vector.shape[1],axis=-1)
            navi_vector = navi_vector / n
            start_points += (centers[idx], )
            navi_vectors += (navi_vector, )
            show_navi_vectors += (5 * navi_vector + centers[idx], )
            references += (reference[idx], )
            new_see_target_vectors += (see_target_vectors,)

        start_points = np.concatenate(start_points,axis=0)
        navi_vectors = np.concatenate(navi_vectors,axis=0)
        show_navi_vectors = np.concatenate(show_navi_vectors,axis=0)
        idxs = np.concatenate(idxs,axis=-1)
        references = np.concatenate(references,axis=0)
        new_see_target_vectors = np.concatenate(new_see_target_vectors,axis=0)

        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(show_navi_vectors)
        pcd.colors = open3d.utility.Vector3dVector(np.zeros_like(show_navi_vectors))
        pcd2 = open3d.geometry.PointCloud()
        pcd2.points = open3d.utility.Vector3dVector(start_points) 
        pcd2.colors = open3d.utility.Vector3dVector(np.zeros_like(start_points)) 
        pcd3 = open3d.geometry.PointCloud()
        pcd3.points = open3d.utility.Vector3dVector(global_point_cloud)                      
                
        points = np.vstack((start_points, show_navi_vectors))  
        lines = []  
        for i in range(len(start_points)):  
            lines.append([i, i + len(start_points)])  # i是起点索引，i + len(start_points)是终点索引          
        # 将数据转换为Open3D的LineSet对象  
        # line_set = open3d.geometry.LineSet()  
        # line_set.points = open3d.utility.Vector3dVector(points)  
        # # 注意：LineSet的lines需要是(2, N)形状的数组，其中N是线段的数量  
        # # 每一行表示一个线段，包含两个点的索引  
        # line_set.lines = open3d.utility.Vector2iVector(np.array(lines))
        # open3d.visualization.draw_geometries([pcd,pcd2,pcd3,line_set])
       
        return start_points, navi_vectors, idxs, references, new_see_target_vectors

    def get_normal_vectors_mean(self, centers, targets, norm_vectors, labels, global_point_cloud):

        mean_vectors = ()
        # see_target_vectors = ()
        for label in np.unique(labels):
            idx = np.where(labels == label)[0]
            norm_vector = norm_vectors[idx]
            mean_vector = np.mean(norm_vector, axis = 0)
            mean_vector = mean_vector / norm(mean_vector)
            mean_vectors += (mean_vector.reshape(1,-1), )
            # see_target_location = see_target_locations[idx]
            # center = centers[label]
            # vector = see_target_location - center
            # n = norm(vector, axis=-1).reshape(-1,1)
            # n = np.repeat(n, vector.shape[-1], axis=-1)
            # vector = vector / n
            # cos_theta = np.sum(vector * np.repeat(mean_vector.reshape(1,-1),vector.shape[0],axis=0), axis = -1)  
            # angle_rad = np.arccos(cos_theta)
            # sorted_indices = np.argsort(angle_rad)
            # see_target_vectors += (vector[sorted_indices[0]].reshape(1,-1),)
            
        mean_vectors = np.concatenate(mean_vectors,axis = 0) 
        # see_target_vectors = np.concatenate(see_target_vectors,axis = 0)

        return mean_vectors
    
    def move_center_to_surface(self, centers, labels, targets, norm_vectors, global_point_cloud):

        new_centers = np.copy(centers)
        for i, label in enumerate(np.unique(labels)):
            idx = np.where(labels == label)[0]            
            center = centers[i]
            norm_vector = norm_vectors[i]
            target = targets[idx]
            vector = target - np.repeat(center.reshape(1,-1), target.shape[0], axis = 0)
            vector_n = norm(vector, axis=-1)
            dot_product = np.sum(vector * norm_vector, axis = -1)
            cos_theta = dot_product / vector_n 
            angle_rad = np.arccos(cos_theta)
            idx = np.where(angle_rad  < np.pi / 2)[0]
            dot_product = dot_product[idx]
            dist = abs(dot_product)
            if dist.shape[0] !=0 :
                new_centers[i] += dist.max() * norm_vector 

        # 定义起点和终点（向量）  
        start_points = new_centers  
        end_points = new_centers + norm_vectors * 5 

        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(end_points)
        pcd.colors = open3d.utility.Vector3dVector(np.zeros_like(end_points))
        pcd2 = open3d.geometry.PointCloud()
        pcd2.points = open3d.utility.Vector3dVector(start_points) 
        pcd2.colors = open3d.utility.Vector3dVector(np.zeros_like(start_points)) 
        pcd3 = open3d.geometry.PointCloud()
        pcd3.points = open3d.utility.Vector3dVector(global_point_cloud)                      
                 
        points = np.vstack((start_points, end_points))  
        lines = []  
        for i in range(len(start_points)):  
            lines.append([i, i + len(start_points)])  # i是起点索引，i + len(start_points)是终点索引  
        
        # # 将数据转换为Open3D的LineSet对象  
        # line_set = open3d.geometry.LineSet()  
        # line_set.points = open3d.utility.Vector3dVector(points)  
        # # 注意：LineSet的lines需要是(2, N)形状的数组，其中N是线段的数量  
        # # 每一行表示一个线段，包含两个点的索引  
        # line_set.lines = open3d.utility.Vector2iVector(np.array(lines))
        # open3d.visualization.draw_geometries([pcd,pcd2,pcd3,line_set])
        return new_centers  

    def get_navi_points(self, targets, centers_navi, idxs, navi_vectors, references, global_point_cloud, distance_threshold):

        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(global_point_cloud)
        downsampled = pcd.voxel_down_sample(1)
        global_point_cloud = np.asarray(downsampled.points)        

        nearest_points = ()
        for i, navi_vector in enumerate(navi_vectors):
            center = centers_navi[i]
            vector = global_point_cloud - center
            # 计算投影长度（点积）  
            projected_lengths = np.sum(np.repeat(navi_vector.reshape(1,-1),vector.shape[0],axis = 0) * vector, axis = -1)
            idx = np.where(projected_lengths > 0)[0]
            projected_lengths = projected_lengths[idx]       
            # 计算投影点的坐标  
            projected_points = center + projected_lengths[:, np.newaxis] * navi_vector 
            # 计算实际点到投影点的欧几里得距离  
            distances = np.linalg.norm(global_point_cloud[idx] - projected_points, axis=1)        
            # 找到最小距离的点
            dist_idx = np.argmin(distances)
            nearest_point = projected_points[dist_idx].reshape(1,-1)
            nearest_points += (nearest_point, )
        nearest_points = np.concatenate(nearest_points, axis = 0)

        # indice = np.where(idxs == 10)[1]
        # # 定义起点和终点（向量）  
        # start_points = centers_navi[indice]  
        # end_points = nearest_points[indice]
        # points = np.vstack((start_points, end_points))  
        # lines = []  
        # for i in range(len(start_points)):  
        #     lines.append([i, i + len(start_points)])
        
        # # 将数据转换为Open3D的LineSet对象  
        # line_set = open3d.geometry.LineSet()  
        # line_set.points = open3d.utility.Vector3dVector(points)  
        # # 注意：LineSet的lines需要是(2, N)形状的数组，其中N是线段的数量  
        # # 每一行表示一个线段，包含两个点的索引  
        # line_set.lines = open3d.utility.Vector2iVector(np.array(lines))

        # start_points = centers_navi[indice]  
        # end_points = centers_navi[indice] + navi_vectors[indice] * 5
        # points = np.vstack((start_points, end_points))  
        # lines = []  
        # for i in range(len(start_points)):  
        #     lines.append([i, i + len(start_points)])
        
        # # 将数据转换为Open3D的LineSet对象  
        # line_set2 = open3d.geometry.LineSet()  
        # line_set2.points = open3d.utility.Vector3dVector(points)  
        # # 注意：LineSet的lines需要是(2, N)形状的数组，其中N是线段的数量  
        # # 每一行表示一个线段，包含两个点的索引  
        # line_set2.lines = open3d.utility.Vector2iVector(np.array(lines))
        # colors = open3d.utility.Vector3dVector(np.array([[0, 1, 0]]))  # 红色  
        # line_set2.colors = open3d.utility.Vector3dVector(np.tile(colors, (len(lines), 1)))         

        # pcd3 = open3d.geometry.PointCloud()
        # pcd3.points = open3d.utility.Vector3dVector(global_point_cloud) 
        # open3d.visualization.draw_geometries([pcd3,line_set,line_set2])
        
        selected_navi_vectors = ()
        centers = ()
        for idx in np.unique(idxs):
            indice = np.where(idxs == idx)[1]
            nearest_point = nearest_points[indice]
            navi_vector = navi_vectors[indice]
            center_navi = centers_navi[indice]
            reference = references[indice]

            residual = nearest_point - center_navi
            distance = norm(residual, axis = -1)            

            cos_theta = np.sum(navi_vector * reference, axis = -1)  
            angle_rad = np.arccos(cos_theta)
            sorted_indices = np.argsort(angle_rad)

            for indice in sorted_indices:
                if distance[indice] > distance_threshold:
                    selected_navi_vectors += ((center_navi[indice] + distance_threshold * navi_vector[indice]).reshape(1,-1),)
                    centers += (center_navi[indice].reshape(1,-1), )
                    break
        selected_navi_vectors = np.concatenate(selected_navi_vectors,axis=0)
        centers = np.concatenate(centers,axis=0)

        # 定义起点和终点（向量）  
        start_points = selected_navi_vectors  
        end_points = centers
        points = np.vstack((start_points, end_points))  
        lines = []  
        for i in range(len(start_points)):  
            lines.append([i, i + len(start_points)])
        
        # 将数据转换为Open3D的LineSet对象  
        line_set = open3d.geometry.LineSet()  
        line_set.points = open3d.utility.Vector3dVector(points)  
        # 注意：LineSet的lines需要是(2, N)形状的数组，其中N是线段的数量  
        # 每一行表示一个线段，包含两个点的索引  
        line_set.lines = open3d.utility.Vector2iVector(np.array(lines))

        pcd2 = open3d.geometry.PointCloud()
        pcd2.points = open3d.utility.Vector3dVector(selected_navi_vectors)
        pcd2.colors = open3d.utility.Vector3dVector(np.zeros_like(selected_navi_vectors))        

        pcd3 = open3d.geometry.PointCloud()
        pcd3.points = open3d.utility.Vector3dVector(global_point_cloud)
        colors = np.repeat(np.array([1,0,0]).reshape(1,-1),global_point_cloud.shape[0],axis = 0)       
        pcd3.colors = open3d.utility.Vector3dVector(colors)
        open3d.visualization.draw_geometries([pcd2, pcd3,line_set])        

        navi_points = dict()
        for item in selected_navi_vectors:
            navi_points[tuple(item.tolist())] = []
        return navi_points
    
    def get_navi_points_2(self, global_point_cloud, centers_navi, idxs, navi_vectors,centers, references, navi_point_range,norm_vectors_mean):
            
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(global_point_cloud)
        downsampled = pcd.voxel_down_sample(1)  
        downsampled_point = np.asarray(downsampled.points)
        alignment_direction = np.array([0,0,1])
        global_norm_vectors_up, downsampled_point  = self.get_norm_vectors(downsampled_point,alignment_direction)
        alignment_direction = np.array([0,0,-1])
        global_norm_vectors_down, downsampled_point  = self.get_norm_vectors(downsampled_point,alignment_direction)        

        # downsampled_point = global_point_cloud
        # global_norm_vectors = self.get_norm_vectors(downsampled_point)

        # global_norm_vectors, downsampled_point = self.get_norm_vectors(downsampled_point)

        selected_navi_vectors = ()
        alignment_directions = ()
        alignment_centers = ()
        norm_vs = ()
        selected_navi_points = ()
        show_centers = ()        
        for idx in np.unique(idxs):
            indice = np.where(idxs == idx)[1]
            navi_vector = navi_vectors[indice]
            center = centers_navi[indice][0]
            reference = references[idx]
            norm_vector_mean = norm_vectors_mean[idx]

            idx_up = np.where((downsampled_point[:,0] >= center[0] - 10) & \
                           (downsampled_point[:,0] <= center[0] + 10) & \
                           (downsampled_point[:,1] >= center[1] - 10) & \
                           (downsampled_point[:,1] <= center[1] + 10) & \
                           (downsampled_point[:,2] >= center[2] - 5) & \
                           (downsampled_point[:,2] <= center[2]))
            local_norm_vectors_up = global_norm_vectors_up[idx_up]

            idx_down = np.where((downsampled_point[:,0] >= center[0] - 10) & \
                           (downsampled_point[:,0] <= center[0] + 10) & \
                           (downsampled_point[:,1] >= center[1] - 10) & \
                           (downsampled_point[:,1] <= center[1] + 10) & \
                           (downsampled_point[:,2] <= center[2] + 2) & \
                           (downsampled_point[:,2] >= center[2]))
            local_norm_vectors_down = global_norm_vectors_down[idx_down]  
        
            

            # pcd = open3d.geometry.PointCloud()
            # pcd.points = open3d.utility.Vector3dVector(local_point_cloud) 
            # pcd.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=50, max_nn=100))

            # # align local norm vector using target dir
            # align_center = center + 1 * norm_vector_mean
            # alignment_centers += (align_center.reshape(1,-1),)
            # alignment_direction = align_center - local_point_cloud
            # n = norm(alignment_direction, axis=-1).reshape(-1,1)
            # n = np.repeat(n,alignment_direction.shape[1],axis=-1)
            # alignment_direction = alignment_direction / n            
            # alignment_direction = np.mean(alignment_direction, axis=0)
            # alignment_direction[-1] = 0

            # # align local norm vector using target norm
            # # alignment_direction = reference            

            # alignment_directions += (alignment_direction.reshape(1,-1),)
            # norm_vs += (norm_vector_mean.reshape(1,-1),)
            
            # pcd.orient_normals_to_align_with_direction(alignment_direction)             
            # local_norm_vectors = np.asarray(pcd.normals)

            # use global estimated norm
            local_point_cloud = np.concatenate((downsampled_point[idx_up],downsampled_point[idx_down]),axis = 0)
            local_norm_vectors = np.concatenate((local_norm_vectors_up,local_norm_vectors_down),axis = 0)

            start_points = local_point_cloud
            end_points = local_point_cloud + local_norm_vectors * 10
            points = np.vstack((start_points, end_points))  
            lines = []  
            for i in range(len(start_points)):  
                lines.append([i, i + len(start_points)]) 
            line_set2 = open3d.geometry.LineSet()  
            line_set2.points = open3d.utility.Vector3dVector(points)    
            line_set2.lines = open3d.utility.Vector2iVector(np.array(lines))

            pcd3 = open3d.geometry.PointCloud()
            pcd3.points = open3d.utility.Vector3dVector(start_points)        


            # colors = np.repeat(np.array([1,0,0]).reshape(1,-1),centers_navi.shape[0],axis = 0)         
            # pcd2.colors = open3d.utility.Vector3dVector(np.zeros_like(colors))         

            # open3d.visualization.draw_geometries([pcd3, line_set2])


        
            mean_vector = np.mean(local_norm_vectors, axis = 0)            
            cos_theta = np.sum(navi_vector * mean_vector, axis = -1)  
            angle_rad = np.arccos(cos_theta)
            sorted_indices = np.argsort(angle_rad) 
            selected_navi_vector = navi_vector[sorted_indices[0]]
            selected_navi_vector = selected_navi_vector / norm(selected_navi_vector)
            selected_navi_point = center + navi_point_range * selected_navi_vector
            selected_navi_points += (selected_navi_point.reshape(1,-1), )
            selected_navi_vectors += (selected_navi_vector.reshape(1,-1), )
            show_centers += (center.reshape(1,-1), )

        # selected_navi_points = np.concatenate(selected_navi_points, axis=0)
        # show_centers = np.concatenate(show_centers, axis=0)
        # selected_navi_vectors = np.concatenate(selected_navi_vectors, axis=0)
        # alignment_directions = np.concatenate(alignment_directions, axis=0)
        # alignment_centers = np.concatenate(alignment_centers, axis=0)
        # norm_vs = np.concatenate(norm_vs, axis=0)

        # for i, navi_vector in enumerate(selected_navi_vectors):
        #     center = centers[i]
        #     safe_center = center + 2 * norm_vectors_mean[i]
        #     vector = downsampled_point - safe_center
        #     # 计算投影长度（点积）  
        #     projected_lengths = np.sum(np.repeat(navi_vector.reshape(1,-1),vector.shape[0],axis = 0) * vector, axis = -1)
        #     idx = np.where(projected_lengths > 0)[0]
        #     projected_lengths = projected_lengths[idx]       
        #     # 计算投影点的坐标  
        #     projected_points = safe_center + projected_lengths[:, np.newaxis] * navi_vector 
        #     # 计算实际点到投影点的欧几里得距离  
        #     distances = np.linalg.norm(global_point_cloud[idx] - projected_points, axis=1)        
        #     # 找到最小距离的点
        #     dist_idx = np.argmin(distances)
        #     nearest_point = projected_points[dist_idx]
        #     dist = norm(nearest_point - safe_center)
        #     if dist > navi_point_range:
        #         selected_navi_point = center + navi_point_range * navi_vector
        #         selected_navi_points += (selected_navi_point.reshape(1,-1), )
        #         show_centers += (center.reshape(1,-1), )

        selected_navi_points = np.concatenate(selected_navi_points, axis=0)
        show_centers = np.concatenate(show_centers, axis=0)        
             
        start_points = show_centers
        end_points = selected_navi_points
        points = np.vstack((start_points, end_points))  
        lines = []  
        for i in range(len(start_points)):  
            lines.append([i, i + len(start_points)]) 
        line_set = open3d.geometry.LineSet()  
        line_set.points = open3d.utility.Vector3dVector(points)    
        line_set.lines = open3d.utility.Vector2iVector(np.array(lines))

        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(start_points) 
        pcd.colors = open3d.utility.Vector3dVector(np.zeros_like(start_points))

        pcd2 = open3d.geometry.PointCloud()
        pcd2.points = open3d.utility.Vector3dVector(downsampled_point)        

        # start_points = alignment_centers
        # end_points = alignment_centers + alignment_directions * 20
        # points = np.vstack((start_points, end_points))  
        # lines = []  
        # for i in range(len(start_points)):  
        #     lines.append([i, i + len(start_points)]) 
        # line_set2 = open3d.geometry.LineSet()  
        # line_set2.points = open3d.utility.Vector3dVector(points)    
        # line_set2.lines = open3d.utility.Vector2iVector(np.array(lines))

        # pcd3 = open3d.geometry.PointCloud()
        # pcd3.points = open3d.utility.Vector3dVector(start_points) 
        # pcd3.colors = open3d.utility.Vector3dVector(np.zeros_like(start_points))        


        # # colors = np.repeat(np.array([1,0,0]).reshape(1,-1),centers_navi.shape[0],axis = 0)         
        # # pcd2.colors = open3d.utility.Vector3dVector(np.zeros_like(colors))         

        open3d.visualization.draw_geometries([pcd,pcd2, line_set])

        navi_points = dict()
        for item in selected_navi_points:
            navi_points[tuple(item.tolist())] = []
        return navi_points      


    def get_navi_points_3(self, navi_vectors, see_target_vectors, centers_navi, idxs, navi_point_range, centers, global_point_cloud):

        selected_navi_points = ()
        show_centers = ()
        for idx in np.unique(idxs):
            indice = np.where(idxs == idx)[1]
            navi_vector = navi_vectors[indice]
            see_target_vector = see_target_vectors[indice]
            center = centers_navi[indice]

            see_target_vector[:,-1] = 0
            n = norm(see_target_vector,axis = -1).reshape(-1,1)
            n = np.repeat(n,see_target_vector.shape[-1],axis=-1)
            see_target_vector = see_target_vector / n
            cos_theta = np.sum(navi_vector * see_target_vector, axis = -1)  
            angle_rad = np.arccos(cos_theta)
            sorted_indices = np.argsort(angle_rad) 
            selected_navi_vector = navi_vector[sorted_indices[0]]
            selected_navi_point = center[0] + navi_point_range * selected_navi_vector
            item = selected_navi_point.astype(int)
            dist = norm(global_point_cloud - item, axis = -1)
            idx = np.where(dist <= 1)
            if idx[0].shape[0] == 0:
                selected_navi_points += (selected_navi_point.reshape(1,-1),)
                show_centers += (center[0].reshape(1,-1), )
        selected_navi_points = np.concatenate(selected_navi_points, axis=0)
        show_centers = np.concatenate(show_centers, axis=0)

        start_points = show_centers
        end_points = selected_navi_points
        points = np.vstack((start_points, end_points))  
        # lines = []  
        # for i in range(len(start_points)):  
        #     lines.append([i, i + len(start_points)]) 
        # line_set = open3d.geometry.LineSet()  
        # line_set.points = open3d.utility.Vector3dVector(points)    
        # line_set.lines = open3d.utility.Vector2iVector(np.array(lines))

        # pcd = open3d.geometry.PointCloud()
        # pcd.points = open3d.utility.Vector3dVector(start_points) 
        # pcd.colors = open3d.utility.Vector3dVector(np.zeros_like(start_points))
        # pcd2 = open3d.geometry.PointCloud()
        # pcd2.points = open3d.utility.Vector3dVector(end_points) 
        # pcd2.colors = open3d.utility.Vector3dVector(np.zeros_like(end_points))        
        # pcd3 = open3d.geometry.PointCloud()
        # pcd3.points = open3d.utility.Vector3dVector(global_point_cloud)        

        # open3d.visualization.draw_geometries([pcd2,pcd3, line_set])
        return selected_navi_points, show_centers
    
    def get_navi_points_4(self, navi_vectors, see_target_vectors, centers_navi, idxs, navi_point_range, centers, global_point_cloud):

        selected_navi_points = ()
        show_centers = ()
        for idx in np.unique(idxs):
            indice = np.where(idxs == idx)[1]
            navi_vector = navi_vectors[indice]
            see_target_vector = see_target_vectors[indice]
            center = centers_navi[indice]
            import random
            selected_navi_vector = navi_vector[random.randint(0, navi_vector.shape[0]-1)]
            selected_navi_point = center[0] + navi_point_range * selected_navi_vector
            selected_navi_points += (selected_navi_point.reshape(1,-1),)
            show_centers += (center[0].reshape(1,-1), )
        selected_navi_points = np.concatenate(selected_navi_points, axis=0)
        show_centers = np.concatenate(show_centers, axis=0)

        start_points = show_centers
        end_points = selected_navi_points
        points = np.vstack((start_points, end_points))  
        lines = []  
        for i in range(len(start_points)):  
            lines.append([i, i + len(start_points)]) 
        line_set = open3d.geometry.LineSet()  
        line_set.points = open3d.utility.Vector3dVector(points)    
        line_set.lines = open3d.utility.Vector2iVector(np.array(lines))

        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(start_points) 
        pcd.colors = open3d.utility.Vector3dVector(np.zeros_like(start_points))
        pcd2 = open3d.geometry.PointCloud()
        pcd2.points = open3d.utility.Vector3dVector(end_points) 
        pcd2.colors = open3d.utility.Vector3dVector(np.zeros_like(end_points))        
        pcd3 = open3d.geometry.PointCloud()
        pcd3.points = open3d.utility.Vector3dVector(global_point_cloud)        

        open3d.visualization.draw_geometries([pcd2,pcd3, line_set])
        return selected_navi_points, show_centers    

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

        # threshold = (int(max(location[0], x_all.max())), 
        #              int(min(location[0], x_all.min())), 
        #              int(max(location[1], y_all.max())), 
        #              int(min(location[1], y_all.min())))

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

        # middle = int(x.shape[0] / 2)
        # x_middle = x[middle,:]
        # y_middle = y[middle,:]
        # x = x_middle
        # y = y_middle

        #cut nan
        # idx1 = ~np.isnan(x)
        # idx2 = ~np.isnan(y)

        # idx = idx1 & idx2
        # x = x[idx]
        # y = y[idx]        

        # go to grid
        x = x.astype(np.int32)
        y = y.astype(np.int32)

        # get map
        if threshold_draw is None:
            map, change = PT.create_2D_map(x,y,threshold)
        else:
            map, change = PT.create_2D_map(x,y,threshold_draw)
        # draw_location = tuple(a - b for a, b in zip(location, change))
        # image = self.draw_point(map, draw_location)
        
        return map.astype(int), change

    def FBE(self,DC:drone_controller,PT:point_cloud,camera,records,config,path_saver):

        global_2D_map = dict()
        global_2D_map[OBSTACLE] = np.array([])
        global_2D_map[EXPLORED] = np.array([]) 

        global_X = []
        global_Y = []
        global_Z = []
        global_bgr = []

        from utils.saver import image_saver_plt
        global_map_saver = image_saver_plt(config['now'],config['Record_root'],'global_map')

        while(1):
            X,Y,Z,bgr,location,pose = self.get_current_point_cloud(DC,PT,camera,records,global_X,global_Y,global_Z,global_bgr)
                
            map, change = self.get_map_at_current_height(X,Y,Z,PT,location)
            explore_range = config['Explore']['explore_range']
            explored_map = self.get_explored_map(map, location, change, explore_range)
            self.update_global_map(explored_map, location, change, global_2D_map)
            global_explored_map, change = self.get_global_explored_map(location,global_2D_map)
            frontiers, frontiers_map = self.get_frontiers(global_explored_map, change)
            global_map_image = self.draw_global_map_and_frontier(frontiers,global_2D_map)
                
            global_map_saver.save(global_map_image) 
            if len(frontiers) == 0:
                break       
            mini_frontier_size = config['Explore']['frontier_divide_size']
            '''分割较大的frontier，筛选小的frontier'''
            self.divide_frontiers_to_bins(frontiers, location, mini_frontier_size)
            path, navi_point = self.find_surround_frontier(location, frontiers, frontiers_map, change)
            path_image = self.draw_path(global_map_image,path,location,navi_point)
            path_saver.save(path_image)
            self.move_along_path(path, navi_point, DC, records,capture_rate = 2, if_capture = False, sleep = 0)
            self.save_path(path,records)

            # self.calculate_all_path(records)
            # cost = records['cost']
            # if cost > config['Explore']['max_path']:
            #     break
            
        return global_2D_map, global_X, global_Y, global_Z, global_bgr 
    
    def get_global_2D_map_navi_point(self, explored, height, voxel_size):

        heights = np.ones((explored.shape[0],1)) * height
        explored = np.concatenate((explored, heights), axis = -1)
        voxels, voxels_idx, voxel_size, min_point = self.voxelize_point_cloud(explored, voxel_size)
        min_point = np.array(min_point)
        voxel_size = np.array(voxel_size)
        keys = np.array(list(voxels.keys()))
        keys[:,0] = keys[:,0] + 0.5
        keys[:,1] = keys[:,1] + 0.5
        voxel_locs = keys * voxel_size + min_point 
        navi_points = ()
        for voxel_loc in voxel_locs:
            dist =  np.linalg.norm(explored - voxel_loc,axis = -1)
            idx = np.where(dist==min(dist))[0][0]
            navi_points += (explored[idx].reshape(1,-1),)   
        return np.concatenate(navi_points,axis=0)
    
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
    
    def cancel_occupied_navi_points(self,navi_points,global_point_cloud):
            
        new_navi_points = ()
        for navi_point in navi_points:
            dist = norm(global_point_cloud - navi_point, axis = -1)
            idx = np.where(dist <= 1)
            if idx[0].shape[0] == 0:
                new_navi_points += (navi_point.reshape(1,-1),)
        new_navi_points = np.concatenate(new_navi_points, axis=0)
        return new_navi_points    
    
    def find_nearest_navigable(self, target, global_2D_map : dict):

        target = np.array(target).astype(int)
        target = target[0:2]
        explored = global_2D_map[EXPLORED]
        dist = np.linalg.norm(explored - target, axis = -1).astype(int)
        idx = np.where(dist == min(dist))[0][0]
        navigable = explored[idx].squeeze()
        return navigable
    
    def draw_path_3d(self,data_path,voxelized_cloud):

        # draw path
        from matplotlib.colors import LinearSegmentedColormap

        path = np.load(data_path + 'path.npy')
        navi_points = np.load(data_path + 'navi_points.npy')
        lines = []
        for i in range(path.shape[0]-1):  
            lines.append([i, i + 1])
        line_set = open3d.geometry.LineSet()  
        line_set.points = open3d.utility.Vector3dVector(path)    
        line_set.lines = open3d.utility.Vector2iVector(np.array(lines))

        custom_cmap = LinearSegmentedColormap.from_list('custom_red_to_blue', ['red', 'blue'])
        # cmap = get_cmap('jet')    
        colors = ()
        # 为每个点根据其标签着色  
        for i in range(len(lines)):  
            # 获取当前标签对应的颜色（归一化到0-1之间）  
            color = np.array(custom_cmap(i / len(lines)))[:3]
            colors += (color.reshape(1,-1), )
        colors = np.concatenate(colors, axis = 0)
        # colors = np.array([0,1,0])
        # line_set.colors = open3d.utility.Vector3dVector(np.tile(colors, (len(lines), 1)))
        line_set.colors = open3d.utility.Vector3dVector(colors)

        idx = np.where(voxelized_cloud[:,-1] >= voxelized_cloud[:,-1].min() + 15)
        voxelized_cloud = voxelized_cloud[idx]
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(voxelized_cloud)
        pcd.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=50, max_nn=100))
        pcd.paint_uniform_color([0.5, 0.5, 0.5]) 

        start = path[0,:].reshape(1,-1)
        end = path[-1,:].reshape(1,-1)

        points = start
        pcd2 = open3d.geometry.PointCloud()
        pcd2.points = open3d.utility.Vector3dVector(points)
        colors = np.tile(np.array([1, 0, 0]), (points.shape[0], 1))
        pcd2.colors = open3d.utility.Vector3dVector(colors)

        points = end
        pcd3 = open3d.geometry.PointCloud()
        pcd3.points = open3d.utility.Vector3dVector(points)
        colors = np.tile(np.array([0, 0, 1]), (points.shape[0], 1))
        pcd3.colors = open3d.utility.Vector3dVector(colors)

        # dist = np.linalg.norm(navi_points-start,axis=-1)
        # idx1 = dist > min(dist)
        # dist = np.linalg.norm(navi_points-end,axis=-1)
        # idx2= dist > min(dist)
        # idx = idx1 & idx2
        # navi_points = navi_points[idx]
        # navi_points = navi_points[0:-1]
        pcd4 = open3d.geometry.PointCloud()
        pcd4.points = open3d.utility.Vector3dVector(navi_points)
        colors = np.array([0,1,0])
        colors = np.tile(colors, (navi_points.shape[0], 1))
        pcd4.colors = open3d.utility.Vector3dVector(colors)     

        open3d.visualization.draw_geometries([pcd2, pcd3, pcd4, pcd,line_set])

    def instructnav(self, DC : drone_controller, PT : point_cloud, VMP : ValueMap, camera,records, config, path_saver: image_saver_plt):

        global_2D_map = dict()
        global_2D_map[OBSTACLE] = np.array([])
        global_2D_map[EXPLORED] = np.array([]) 

        global_X = []
        global_Y = []
        global_Z = []
        global_bgr = []

        from utils.saver import image_saver_plt
        global_map_saver = image_saver_plt(config['now'],config['Record_root'],'global_map')

        while(1):
            X,Y,Z,bgr,location,pose = self.get_current_point_cloud(DC,PT,camera,records,global_X,global_Y,global_Z,global_bgr)

            '''注意bgr对应的顺序为0，1，4，2，前左后右'''
            global_tagrget_loc = VMP.get_global_target_loc(bgr, X, Y, Z)
            global_tagrget_loc_2D = np.array(np.array(global_tagrget_loc)[:, :2]).astype(np.int32) if len(global_tagrget_loc) != 0 else np.array([])
                
            map, change = PT.get_map_at_current_height(X,Y,Z,location)
            explore_range = config['Explore']['explore_range']
            explored_map = self.get_explored_map(map, location, change, explore_range)
            self.update_global_map(explored_map, location, change, global_2D_map)
            global_explored_map, change = self.get_global_explored_map(location,global_2D_map)
            # if self.DL_flg:
            #     x_up, y_up, z_up = self.DL.Lift(global_2D_map, DC, PT)
            frontiers, frontiers_map = self.get_frontiers(global_explored_map, change)
            global_map_image = self.draw_global_map_and_frontier(frontiers,global_2D_map)
            
            '''update target and draw'''
            # if self.VMP_flg:
            # VMP.update_current_target_map(global_tagrget_loc_2D)
            VMP.update_global_target_map(global_tagrget_loc_2D)
            VMP.draw_global_map_and_frontier_and_target(frontiers, global_2D_map, self.history_location)
            # VMP.meger_current_to_global(global_2D_map[EXPLORED])
                
            global_map_saver.save(global_map_image) 
            if len(frontiers) == 0:
                break       
            mini_frontier_size = config['Explore']['frontier_divide_size']
            '''分割较大的frontier，筛选小的frontier'''
            self.divide_frontiers_to_bins(frontiers, location, mini_frontier_size)
            # frontiers_image = self.draw_frontiers(global_map_image, frontiers)
            # frontiers_mean_saver.save(frontiers_image)
            # if self.VMP_flg:
            navi_point = VMP.select_navi_point_by_dist(frontiers)
            if navi_point is None: 
                path, navi_point = self.find_surround_frontier(location, frontiers, frontiers_map, change)
            else:
                path = self.calculate_route(navi_point, location, frontiers_map, change)
            # else:
            #     path, navi_point = self.find_surround_frontier(location, frontiers, frontiers_map, change)
            path_image = self.draw_path(global_map_image,path,location,navi_point)
            path_saver.save(path_image)
            self.move_along_path(path, navi_point, DC, records,capture_rate = 2, if_capture = False, sleep = 0)
            self.save_path(path,records)

        return global_2D_map, global_X, global_Y, global_Z, global_bgr    

import numpy as np   
from Others.DensityGuided.DensityGuided import DensityGuided
from Perception.GeneralizedLoss import gereralizedloss
from utils.saver import save_config, image_saver_plt
from Explore.path_3D import path_planning_3d

class LowDG:

    def __init__(self, config):
        self.GL = gereralizedloss()
        self.DG = DensityGuided()
        self.navi_saver = image_saver_plt(config['now'],config['Record_root'],'navi') 
        self.global_x, self.global_y, self.global_z = [], [], []
        self.location_list, self.global_bgr = [], []
        self.config = config

    def cut_point_cloud(self, landing_point, limit):
        count = 0
        tmp_x, tmp_y, tmp_z = [], [], []
        for i in range(len(self.global_x)):
            X, Y, Z = self.global_x[i], self.global_y[i], self.global_z[i]
            for key in X.keys():
                x, y, z = X[key].flatten(), Y[key].flatten(), Z[key].flatten()
                idx1 = ~np.isnan(x)
                idx2 = ~np.isnan(y)
                idx3 = ~np.isnan(z)
                idx = idx1 & idx2 & idx3
                x_c = x[idx]; y_c = y[idx]; z_c = z[idx]
                idx1 = np.where((np.abs(x_c - int(landing_point[0])) > limit))
                idx2 = np.where((np.abs(y_c - int(landing_point[1])) > limit))
                idx4 = np.union1d(idx1, idx2)
                x_c[idx4] = np.nan; y_c[idx4] = np.nan; z_c[idx4] = np.nan
                x[idx] = x_c; y[idx] = y_c; z[idx] = z_c
                X[key] = x.reshape(X[key].shape)
                Y[key] = y.reshape(Y[key].shape)
                Z[key] = z.reshape(Z[key].shape)
            tmp_x.append(X)
            tmp_y.append(Y)
            tmp_z.append(Z)
        return tmp_x, tmp_y, tmp_z
        
    
    def update(self, X, Y, Z, bgr, location):
        self.global_x.append(X)
        self.global_y.append(Y)
        self.global_z.append(Z)
        self.location_list.append(location)
        self.global_bgr.append(bgr)

    def clear(self):
        self.global_x, self.global_y, self.global_z = [], [], []
        self.location_list, self.global_bgr = [], []
    
    def save_path(self, path, record):

        if path.shape[-1] == 3:
            record['path'].append(path)
        else:
            location = record['location'][-1]
            height = location[-1]
            height_vector = np.ones((path.shape[0],1)) * height
            path = np.concatenate((path, height_vector),axis = -1)
            record['path'].append(path)

    def low_density_guide(self, DC, PT, camera, records, landing_location, limit, global_explore, current_explore):

        self.global_x, self.global_y, self.global_z = self.cut_point_cloud(landing_location, limit)

        global_point_cloud = self.DG.get_global_point_cloud(self.global_x, self.global_y, self.global_z)
        path_3d = path_planning_3d(global_point_cloud)
        voxelized_cloud = path_3d.voxelized_cloud

        # pcd = open3d.geometry.PointCloud()
        # pcd.points = open3d.utility.Vector3dVector(voxelized_cloud)
        # open3d.visualization.draw_geometries([pcd])      

        density_map_threshold = self.config['LowDG']['density_map_threshold']
        height_range = self.config['LowDG']['height_range']
        targets, see_target_locations, axs = self.DG.get_targets_from_density_maps(self.GL,density_map_threshold,camera, self.config, self.location_list, self.global_x, self.global_y, self.global_z, self.global_bgr, height_range = None)

        '''筛选掉只有在新探索过区域的目标点'''
        global_explore_set = set(map(tuple, global_explore))
        mask = np.array([tuple(loc) not in global_explore_set for loc in current_explore])
        current_explore = current_explore[mask]

        '''筛选出在新探索区域的targets'''
        idx_total = (np.array([], dtype=np.int64),)
        for loc in current_explore:
            idx_stay = np.where((np.abs(targets[:, 0] - loc[0]) <= 1) & (np.abs(targets[:, 1] - loc[1]) <= 1))
            idx_total = np.union1d(idx_stay, idx_total)
        targets = targets[idx_total]; see_target_locations = see_target_locations[idx_total] 

        if targets.shape[0] == 0:
            print('No target')
            return 

        voxel_size = self.config['LowDG']['voxel_size']
        voxels, voxels_idx, voxel_size, min_point = self.DG.voxelize_point_cloud(targets, voxel_size)
        
        # gaussian cluster targets
        GM_cluster_size = self.config['DensityGuide']['GM_cluster_size']
        labels, centers, targets, see_target_vectors, image = self.DG.GaussianMixture_voxel(targets, GM_cluster_size, voxels, voxels_idx, voxel_size, min_point, see_target_locations,global_point_cloud,voxelized_cloud)
        self.navi_saver.save(image)

        # get all norm vectors  
        norm_vectors, targets = self.DG.get_norm_vectors(targets)    

        norm_vectors_mean = self.DG.get_normal_vectors_mean(centers, targets, norm_vectors, labels, global_point_cloud)
        centers = self.DG.move_center_to_surface(centers,labels,targets,norm_vectors_mean,global_point_cloud)

        navi_vector_degree = self.config['LowDG']['navi_vector_degree']
        centers_navi, navi_vectors, idxs, references, see_target_vectors = self.DG.get_potential_navi_vectors(targets, centers, norm_vectors_mean, 30, navi_vector_degree, global_point_cloud, see_target_vectors)

        navi_point_range = self.config['LowDG']['navi_point_range']
        navi_points, centers = self.DG.get_navi_points_3(navi_vectors, see_target_vectors, centers_navi, idxs,navi_point_range,centers,voxelized_cloud)

        if navi_points is not None:
            navi_points = self.DG.array_to_dict(navi_points)

            error = 0
            camera = [0,1,4,2]
            while len(navi_points) != 0:
                navi_point = self.DG.reach_targets_greedy(navi_points, DC)
                path = path_3d.path_planning_3d(navi_point, DC)
                if path.shape[0] != 0:
                    self.DG.move_along_path_only_capture_when_reach(path,navi_point,camera,DC,records, sleep = 0)
                    # EX.move_along_path_and_trun_to_center(path,navi_point,center,camera,DC,records, sleep = 0)
                    self.save_path(path,records)
                    tmp_loc, tmp_pose = DC.get_world_location_pose()
                    records['low_location'].append(tmp_loc)
                    records['low_pose'].append(tmp_pose)
                else:
                    error += 1
            print('error:',error)
        else:
            print('No navi point')

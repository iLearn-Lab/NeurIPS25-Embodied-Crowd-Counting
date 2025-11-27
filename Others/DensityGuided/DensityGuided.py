import numpy as np
import math
import matplotlib.pyplot as plt 
from Drone.Control import drone_controller
from Perception.GeneralizedLoss import gereralizedloss
from sklearn.mixture import GaussianMixture
import open3d
import matplotlib.pyplot as plt  
from numpy.linalg import norm
from utils.saver import save_config, image_saver_plt


class DensityGuided:
    
    def run_fbe_method(self, config, global_X, global_Y, global_Z, global_bgr, records, DC, GL, EX, camera):
        navi_saver = image_saver_plt(config['now'],config['Record_root'],'navi')  
        global_point_cloud = self.get_global_point_cloud(global_X, global_Y, global_Z) 
        from Explore.path_3D import path_planning_3d
        path_3d = path_planning_3d(global_point_cloud)
        voxelized_cloud = path_3d.voxelized_cloud

        density_map_threshold = config['DensityGuide']['density_map_threshold']
        targets, see_target_locations, axs = self.get_targets_from_density_maps(GL,density_map_threshold,camera, config, records['location'],global_X, global_Y, global_Z, global_bgr)

        voxel_size = config['DensityGuide']['voxel_size']
        voxels, voxels_idx, voxel_size, min_point = self.voxelize_point_cloud(targets, voxel_size)

        # gaussian cluster targets
        GM_cluster_size = config['DensityGuide']['GM_cluster_size']
        labels, centers, targets, see_target_vectors, image = self.GaussianMixture_voxel(targets, GM_cluster_size, voxels, voxels_idx, voxel_size, min_point, see_target_locations,global_point_cloud,voxelized_cloud)
        navi_saver.save(image)

        # get all norm vectors  
        norm_vectors, targets = self.get_norm_vectors(targets)    

        norm_vectors_mean = self.get_normal_vectors_mean(centers, targets, norm_vectors, labels, global_point_cloud)
        centers = self.move_center_to_surface(centers,labels,targets,norm_vectors_mean,global_point_cloud)

        navi_vector_degree = config['DensityGuide']['navi_vector_degree']
        centers_navi, navi_vectors, idxs, references, see_target_vectors = self.get_potential_navi_vectors(targets, centers, norm_vectors_mean, 30, navi_vector_degree, global_point_cloud, see_target_vectors)

        navi_point_range = config['DensityGuide']['navi_point_range']
        navi_points, centers = self.get_navi_points_3(navi_vectors, see_target_vectors, centers_navi, idxs,navi_point_range,centers,voxelized_cloud)

        navi_points = self.array_to_dict(navi_points)

        error = 0
        camera = [0,1,4,2]
        while len(navi_points) != 0:
            navi_point = self.reach_targets_greedy(navi_points, DC)
            path = path_3d.path_planning_3d(navi_point, DC)
            if path.shape[0] != 0:
                self.move_along_path_only_capture_when_reach(path,navi_point,camera,DC,records, sleep = 0)
                EX.save_path(path,records)
            else:
                error += 1
        return records
    

    def run_our_method(self, config, global_X, global_Y, global_Z, global_bgr, up_global_x, up_global_y, up_global_z,records, DC, GL, EX, camera):
        navi_saver = image_saver_plt(config['now'],config['Record_root'],'navi')  
        global_point_cloud = self.get_global_point_cloud(global_X, global_Y, global_Z) 
        up_global_point_cloud = self.get_global_point_cloud(up_global_x, up_global_y, up_global_z)
        global_point_cloud = np.concatenate((global_point_cloud, up_global_point_cloud), axis = 0)
        from Explore.path_3D import path_planning_3d
        path_3d = path_planning_3d(global_point_cloud)
        voxelized_cloud = path_3d.voxelized_cloud

        density_map_threshold = config['DensityGuide']['density_map_threshold']
        targets, see_target_locations, axs = self.get_targets_from_density_maps(GL,density_map_threshold,camera, config, records['location'],global_X, global_Y, global_Z, global_bgr)

        voxel_size = config['DensityGuide']['voxel_size']
        voxels, voxels_idx, voxel_size, min_point = self.voxelize_point_cloud(targets, voxel_size)

        # gaussian cluster targets
        GM_cluster_size = config['DensityGuide']['GM_cluster_size']
        labels, centers, targets, see_target_vectors, image = self.GaussianMixture_voxel(targets, GM_cluster_size, voxels, voxels_idx, voxel_size, min_point, see_target_locations,global_point_cloud,voxelized_cloud)
        navi_saver.save(image)

        # get all norm vectors  
        norm_vectors, targets = self.get_norm_vectors(targets)    

        norm_vectors_mean = self.get_normal_vectors_mean(centers, targets, norm_vectors, labels, global_point_cloud)
        centers = self.move_center_to_surface(centers,labels,targets,norm_vectors_mean,global_point_cloud)

        navi_vector_degree = config['DensityGuide']['navi_vector_degree']
        centers_navi, navi_vectors, idxs, references, see_target_vectors = self.get_potential_navi_vectors(targets, centers, norm_vectors_mean, 30, navi_vector_degree, global_point_cloud, see_target_vectors)

        navi_point_range = config['DensityGuide']['navi_point_range']
        navi_points, centers = self.get_navi_points_3(navi_vectors, see_target_vectors, centers_navi, idxs,navi_point_range,centers,voxelized_cloud)

        navi_points = self.array_to_dict(navi_points)

        error = 0
        camera = [0,1,4,2]
        while len(navi_points) != 0:
            navi_point = self.reach_targets_greedy(navi_points, DC)
            path = path_3d.path_planning_3d(navi_point, DC)
            if path.shape[0] != 0:
                self.move_along_path_only_capture_when_reach(path,navi_point,camera,DC,records, sleep = 0)
                EX.save_path(path,records)
            else:
                error += 1
        return records

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
        if len(selected_navi_points) != 0:
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
        else:
            return None, None

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
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(targets) 
        open3d.visualization.draw_geometries([pcd])        
            
        return targets, np.concatenate(see_target_location,axis = 0),axs
    
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
        line_set2 = open3d.geometry.LineSet()  
        line_set2.points = open3d.utility.Vector3dVector(points)    
        line_set2.lines = open3d.utility.Vector2iVector(np.array(lines))      

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
            idx = np.where(dist == min(dist))[0][0]
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
        line_set2 = open3d.geometry.LineSet()  
        line_set2.points = open3d.utility.Vector3dVector(points)    
        line_set2.lines = open3d.utility.Vector2iVector(np.array(lines))
        
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(end_points)
        pcd.colors = open3d.utility.Vector3dVector(np.zeros_like(end_points))

        pcd2 = open3d.geometry.PointCloud()
        pcd2.points = open3d.utility.Vector3dVector(voxelized_cloud)

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
        
        # 将数据转换为Open3D的LineSet对象  
        line_set = open3d.geometry.LineSet()  
        line_set.points = open3d.utility.Vector3dVector(points)  
        # 注意：LineSet的lines需要是(2, N)形状的数组，其中N是线段的数量  
        # 每一行表示一个线段，包含两个点的索引  
        line_set.lines = open3d.utility.Vector2iVector(np.array(lines))
        # open3d.visualization.draw_geometries([pcd,pcd2,pcd3,line_set])
        return new_centers  
    
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
        line_set = open3d.geometry.LineSet()  
        line_set.points = open3d.utility.Vector3dVector(points)  
        # 注意：LineSet的lines需要是(2, N)形状的数组，其中N是线段的数量  
        # 每一行表示一个线段，包含两个点的索引  
        line_set.lines = open3d.utility.Vector2iVector(np.array(lines))
        # open3d.visualization.draw_geometries([pcd,pcd2,pcd3,line_set])
       
        return start_points, navi_vectors, idxs, references, new_see_target_vectors

    def array_to_dict(self,array):

        dict_array = dict()
        for item in array:
            dict_array[tuple(item.tolist())] = []
        return dict_array 
    
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

    def move_along_path_only_capture_when_reach(self, path, navi_point, camera, DC : drone_controller, records,capture_rate = 1, sleep = 0):

        for i in range(1, path.shape[0]-1):
            cord = path[i,:]
            DC.get_to_location_and_capture_vector(cord, camera, records, capture_rate=capture_rate, if_capture = False, sleep = sleep)    
        DC.get_to_location_only_reach_capture_vector(navi_point, camera, records, capture_rate=capture_rate, if_capture = True, sleep = sleep)
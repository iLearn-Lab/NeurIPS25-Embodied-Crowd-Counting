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
        
        # pcd = open3d.geometry.PointCloud()
        # pcd.points = open3d.utility.Vector3dVector(global_point_cloud)
        # open3d.visualization.draw_geometries([pcd])        
        
        from Explore.path_3D import path_planning_3d
        path_3d = path_planning_3d(global_point_cloud, EX.boundary)
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
        
        # pcd = open3d.geometry.PointCloud()
        # pcd.points = open3d.utility.Vector3dVector(global_point_cloud)
        # pcd.visualize.draw_geometries([pcd])
        
        up_global_point_cloud = self.get_global_point_cloud(up_global_x, up_global_y, up_global_z)
        global_point_cloud = np.concatenate((global_point_cloud, up_global_point_cloud), axis = 0)
        from Explore.path_3D import path_planning_3d
        path_3d = path_planning_3d(global_point_cloud, EX.boundary)
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

            # visualize
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
            # open3d.visualization.draw_geometries([pcd2,pcd3, line_set])
            
            return selected_navi_points, show_centers
        else:
            return None, None

    def generate_vertical_vector(self, normal_vector, theta):  

        n = norm(normal_vector, axis=-1).reshape(-1,1)
        n = np.repeat(n,normal_vector.shape[1],axis=-1) 
        normal_vector = normal_vector / n  
        
        a = np.array([0, 1, 0]) 
        u = np.dot(normal_vector, a.reshape(-1,1))
        u = np.repeat(u,normal_vector.shape[1],axis=-1) * normal_vector
        u = a - u
        n = norm(u, axis=-1).reshape(-1,1)
        n = np.repeat(n,u.shape[1],axis=-1)
        u = u / n
        
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

            U = u[i]
            v = np.dot(rotation_matrix, U)
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
        # idx4 = temp_Z < -2
        idx = idx1 & idx2 & idx3
        temp_X = temp_X[idx].reshape(-1,1)
        temp_Y = temp_Y[idx].reshape(-1,1)
        temp_Z = temp_Z[idx].reshape(-1,1)
        point_cloud = np.concatenate((temp_X,temp_Y,temp_Z),axis = -1)
        return point_cloud

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
            
            idx = np.where(density_map >= density_threshold)
            idx_0 = (idx[0] / density_map.shape[0] * X[camera].shape[0]).astype(int)
            idx_1 = (idx[1] / density_map.shape[1] * X[camera].shape[1]).astype(int) 

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
        # open3d.visualization.draw_geometries([pcd])        
            
        return targets, np.concatenate(see_target_location,axis = 0),axs
    
    def voxelize_point_cloud(self, points, voxel_size):  

        min_point = np.min(points, axis=0)  
        max_point = np.max(points, axis=0)  

        for i, item in enumerate(voxel_size):
            if item == -1:
                voxel_size[i] = (max_point - min_point + 1)[i]
        voxel_size = tuple(voxel_size)
        voxel_grid_dims = np.ceil((max_point - min_point) / voxel_size).astype(int)  

        voxels = {} 
        voxels_idx = {} 
    
        for i, point in enumerate(points):
            voxel_index = tuple(((point - min_point) / voxel_size).astype(int))

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

        pcd2 = open3d.geometry.PointCloud()
        pcd2.points = open3d.utility.Vector3dVector(targets)
        pcd2.colors = open3d.utility.Vector3dVector(np.zeros_like(targets))  

        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(voxelized_cloud)

        pcd3 = open3d.geometry.PointCloud()
        pcd3.points = open3d.utility.Vector3dVector(see_target_locations)
        pcd3.colors = open3d.utility.Vector3dVector(np.zeros_like(see_target_locations))  
 
        start_points = targets
        end_points = see_target_locations                 
        points = np.vstack((start_points, end_points))  
        lines = []  
        for i in range(len(start_points)):  
            lines.append([i, i + len(start_points)])
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
 
        start_points = centers 
        end_points = see_target_vectors * 5 + centers                 
        points = np.vstack((start_points, end_points))  
        lines = []  
        for i in range(len(start_points)):  
            lines.append([i, i + len(start_points)])
        line_set2 = open3d.geometry.LineSet()  
        line_set2.points = open3d.utility.Vector3dVector(points)    
        line_set2.lines = open3d.utility.Vector2iVector(np.array(lines))
        
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(end_points)
        pcd.colors = open3d.utility.Vector3dVector(np.zeros_like(end_points))

        pcd2 = open3d.geometry.PointCloud()
        pcd2.points = open3d.utility.Vector3dVector(voxelized_cloud)
        # open3d.visualization.draw_geometries([pcd,pcd2,line_set2])        

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
        max_idx = norm_vectors.shape[0]
        # see_target_vectors = ()
        for label in np.unique(labels):
            idx = np.where(labels == label)[0]
            idx = idx[idx < max_idx]
            norm_vector = norm_vectors[idx]
            mean_vector = np.mean(norm_vector, axis = 0)
            mean_vector = mean_vector / norm(mean_vector)
            mean_vectors += (mean_vector.reshape(1,-1), )

        mean_vectors = np.concatenate(mean_vectors,axis = 0) 
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
            lines.append([i, i + len(start_points)])
         
        line_set = open3d.geometry.LineSet()  
        line_set.points = open3d.utility.Vector3dVector(points)   
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
            lines.append([i, i + len(start_points)])
        line_set = open3d.geometry.LineSet()  
        line_set.points = open3d.utility.Vector3dVector(points)   
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
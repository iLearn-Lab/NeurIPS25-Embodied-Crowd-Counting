from numpy import ndarray
import numpy as np
import plotly.graph_objects as go
from PIL import Image, ImageDraw
import open3d
import cv2
import open3d
import matplotlib.pyplot as plt
from Point_cloud.Map_element import (
    UNKNOWN,
    OBSTACLE,
    EXPLORED,
    FRONTIER)

class point_cloud:

    def __init__(self):
        pass

    def single_depth_to_3D(self, intrinsic_matrix, pose_matrix,
                        depth: ndarray):
        
        fx = intrinsic_matrix[0, 0]
        fy = intrinsic_matrix[1, 1]
        cx = intrinsic_matrix[0, 2]
        cy = intrinsic_matrix[1, 2]

        h = depth.shape[0]
        w = depth.shape[1]

        Y = np.arange(w)
        Y = np.expand_dims(Y, axis = 0)
        Y = np.repeat(Y, h, axis = 0)
        Y = (Y - cx) * depth / fx

        X = depth

        Z = np.arange(h)
        Z = np.expand_dims(Z, axis = 1)
        Z = np.repeat(Z, w, axis = 1)
        Z = (Z - cy) * depth / fy

        Y_fin = Y * pose_matrix[0, 0] + \
                X * pose_matrix[0, 1] + \
                Z * pose_matrix[0, 2]
        
        X_fin = Y * pose_matrix[1, 0] + \
                X * pose_matrix[1, 1] + \
                Z * pose_matrix[1, 2]
        
        Z_fin = Y * pose_matrix[2, 0] + \
                X * pose_matrix[2, 1] + \
                Z * pose_matrix[2, 2]

        X = X_fin
        Y = Y_fin
        Z = Z_fin

        return X, Y, Z

    def get_point_clouds_from_depth(self, 
                                    intrinsic_matrix,
                                    pose_matrix,
                                    depth : np.ndarray,
                                    camera):
        '''
            x > 0 : forward
            y > 0 : right
            z > 0 : down
            point cloud order [0,1,4,2,3]: front -> left -> back -> right -> down
        '''

        X = dict()
        Y = dict()
        Z = dict()

        x, y, z = self.single_depth_to_3D(
            intrinsic_matrix, 
            pose_matrix,
            depth)
                
        X[str(camera)] = x
        Y[str(camera)] = y
        Z[str(camera)] = z

        return X,Y,Z
    
    def squeeze_point_cloud(self, 
                             X:np.ndarray,Y:np.ndarray,Z:np.ndarray):

        x = ()
        y = ()
        z = ()
        
        for key in X:
            x = x + (X[key],)
            y = y + (Y[key],)
            z = z + (Z[key],) 

        x = np.concatenate(x,axis = 1)
        y = np.concatenate(y,axis = 1)
        z = np.concatenate(z,axis = 1)

        return x,y,z
    
    def show_point_cloud_go(self, X, Y, Z):

        marker_data = go.Scatter3d(
            x=X.reshape(-1), 
            y=Y.reshape(-1), 
            z=-Z.reshape(-1),
            marker=go.scatter3d.Marker(
                size=1,                
                colorscale='piyg',
                opacity=0.8, 
                ),
            mode='markers'
        )

        fig=go.Figure(data=marker_data)
        fig.show()

    def show_point_cloud(self, X, Y, Z, bgr):

        rgb = cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
        rgb = rgb.reshape(-1,3)
        rgb = rgb / 255
        X, Y, Z = self.squeeze_point_cloud(X, Y, Z)
        X = X.reshape(-1,1)
        Y = Y.reshape(-1,1)
        Z = Z.reshape(-1,1)
        xyz = np.concatenate((X,Y,Z), axis=-1)
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(xyz)
        pcd.colors = open3d.utility.Vector3dVector(rgb)
        open3d.visualization.draw_geometries([pcd])
        print('ok')

    def create_2D_map(self, X, Y, threshold : int):
        '''
            change : location - standard_location
        '''

        x_max = threshold[0]
        x_min = threshold[1]
        y_max = threshold[2]
        y_min = threshold[3]

        map = np.zeros(
            (x_max - x_min + 1, y_max - y_min + 1)
        )  

        X = X - x_min
        Y = Y - y_min
        
        # 保留X和Y大于0的点
        idx = np.where((X >= 0) & (Y >= 0) & (X < map.shape[0]) & (Y < map.shape[1]))
        X = X[idx]
        Y = Y[idx]

        map[X.reshape(1,-1),
            Y.reshape(1,-1)] = OBSTACLE         

        return map, (x_min, y_min)

    def draw_point(self, map, point : tuple):
        '''
            x: index for height
            y: index for width
        '''
        map_draw = map * 255
        image = Image.fromarray(map_draw)
        image = image.convert('RGB')
        
        draw = ImageDraw.Draw(image)
        draw.point((point[1], point[0]), fill="red")

        return image.transpose(Image.FLIP_TOP_BOTTOM)
    
    def get_map_at_current_height(self, x : dict, y : dict, z : dict, 
                                  location : tuple, threshold_draw : tuple = None):

        x,y,z = self.squeeze_point_cloud(x,y,z)
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
            map, change = self.create_2D_map(x,y,threshold)
        else:
            map, change = self.create_2D_map(x,y,threshold_draw)
        # draw_location = tuple(a - b for a, b in zip(location, change))
        # image = self.draw_point(map, draw_location)
        
        return map.astype(int), change
        
    def get_view_in_range(self, x : dict, distance : tuple, image : np.ndarray):
        '''
            image should be in bgr format
        '''

        point_clouds = ()
        for key in x.keys():
            point_clouds += (x[key],)
        point_clouds = np.concatenate(point_clouds,axis = 1)
        idx = np.where((point_clouds>=distance[0]) & (point_clouds<=distance[1]))

        projected = np.zeros_like(image)
        projected[
            np.expand_dims(idx[0], axis = 0),
            np.expand_dims(idx[1], axis = 0),
            :
            ] = \
        image[
            np.expand_dims(idx[0], axis = 0),
            np.expand_dims(idx[1], axis = 0),
            :]
        
        return projected
    
    def get_global_point_cloud(self, 
                        current_location : tuple,
                        X : dict,
                        Y : dict,
                        Z : dict):

        X_global = dict()
        Y_global = dict()
        Z_global = dict()
        for key in X.keys():
            X_global[key] = X[key] + current_location[0]
        for key in Y.keys():
            Y_global[key] = Y[key] + current_location[1]
        for key in Z.keys():
            Z_global[key] = Z[key] + current_location[2]

        return X_global, Y_global, Z_global
    
    def get_point_cloud_from_mask(self,
                                  X : np.ndarray, 
                                  Y : np.ndarray, 
                                  Z : np.ndarray,
                                  mask : np.ndarray):
        
        idx = np.where(mask != 0)
        
        x = X[np.expand_dims(idx[0], axis = 0),
              np.expand_dims(idx[1], axis = 0)]
        y = Y[np.expand_dims(idx[0], axis = 0),
              np.expand_dims(idx[1], axis = 0)]
        z = Z[np.expand_dims(idx[0], axis = 0),
              np.expand_dims(idx[1], axis = 0)]
               
        return x,y,z
    
    def convert_to_open3d(self, X,Y,Z):

        x,y,z = self.squeeze_point_cloud(X, Y, Z)
        idx1 = ~np.isnan(x)
        idx2 = ~np.isnan(y)
        idx3 = ~np.isnan(z)
        idx = idx1 & idx2 & idx3
        x = x[idx].reshape(-1,1)
        y = y[idx].reshape(-1,1)  
        z = z[idx].reshape(-1,1)          
        point_cloud = np.concatenate((x,y,z),axis = 1)
        return point_cloud
    
    def convert_to_open3d_dict(self, X:dict,Y:dict,Z:dict,bgr:np.array):

        x,y,z = self.squeeze_point_cloud(X, Y, Z)
        rgb = cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
        rgb = rgb.reshape(-1,3)
        rgb = rgb / 255

        return self.convert_to_open3d(x,y,z), rgb
    
    def combine_point_cloud(self,point_cloud1,point_cloud2,color1,color2,voxel_size = 0.1):

        point_cloud = np.concatenate((point_cloud1,point_cloud2),axis = 0) 
        color = np.concatenate((color1,color2),axis = 0)
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(point_cloud)  
        pcd.colors = open3d.utility.Vector3dVector(color)
        downsampled = pcd.voxel_down_sample(voxel_size)  
        points = np.asarray(downsampled.points) 
        colors = np.asarray(downsampled.colors) 
        return points, colors

    def show_point_cloud_numpy(self, point_cloud, rgb):

        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(point_cloud)
        pcd.colors = open3d.utility.Vector3dVector(rgb)
        open3d.visualization.draw_geometries([pcd])    

    def cut_point_cloud_threshold(self, 
                                  depth : np.ndarray, threshold = 100):

        depth = depth.reshape((1,-1)) 
        depth = depth[abs(depth)<threshold]       

        return depth
    
    def draw_map(self, map, location, change):

        idx = np.where(map == OBSTACLE)
        x = idx[0]; y = idx[1]
        x = x + change[0]
        y = y + change[1]        
        plt.figure() 
        plt.scatter(y, x, 
                    c='black',s=1)
        plt.scatter(location[1], location[0], 
            c='red',s=3)        
        return plt.gca()        
        





        


        



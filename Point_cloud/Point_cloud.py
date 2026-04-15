from numpy import ndarray
import numpy as np
from Point_cloud.Map_element import (
    UNKNOWN,
    OBSTACLE,
    EXPLORED,
    FRONTIER)

class point_cloud:

    def __init__(self):
        pass

    def single_depth_to_3D(
        self, 
        intrinsic_matrix, 
        pose_matrix,
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

    def get_point_clouds_from_depth(
        self, 
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
    
    def squeeze_point_cloud(
        self, X:np.ndarray,Y:np.ndarray,Z:np.ndarray):

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
        
        idx = np.where((X >= 0) & (Y >= 0) & (X < map.shape[0]) & (Y < map.shape[1]))
        X = X[idx]
        Y = Y[idx]

        map[X.reshape(1,-1),
            Y.reshape(1,-1)] = OBSTACLE         

        return map, (x_min, y_min)
    
    def get_global_point_cloud(
        self, 
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
    
    def get_point_cloud_from_mask(
        self,
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
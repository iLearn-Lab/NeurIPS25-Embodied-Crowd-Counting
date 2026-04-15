from pathfinding3d.core.diagonal_movement import DiagonalMovement
from pathfinding3d.core.grid import Grid
from pathfinding3d.finder.a_star import AStarFinder
import numpy as np
from Drone.Control import drone_controller
import math

class path_planning_3d:

    def __init__(self, global_point_cloud, boundary):
        if boundary is None:
            raise ValueError("boundary is required for path_planning_3d")

        min_point = np.min(global_point_cloud, axis=0).astype(float)
        min_point[0] = min(min_point[0], boundary['x_min'])
        min_point[1] = min(min_point[1], boundary['y_min'])
        min_point = np.floor(min_point).astype(int)
        max_point = np.max(global_point_cloud, axis=0).astype(float)
        max_point[0] = max(max_point[0], boundary['x_max'])
        max_point[1] = max(max_point[1], boundary['y_max'])
        max_point = np.ceil(max_point).astype(int)
        map_point = global_point_cloud.astype(int) - np.repeat(min_point.reshape(1,-1), global_point_cloud.shape[0], axis=0)
        len = (max_point - min_point + 1).astype(int)
        matrix = np.ones((len[0], len[1], len[2]), dtype=np.int8)
        matrix[
            map_point[:,0].reshape(1,-1),
            map_point[:,1].reshape(1,-1),
            map_point[:,2].reshape(1,-1),
        ] = 0

        for w in range(0,0):
            add_items = ()
            map_point = np.where(matrix == 0)
            map_point = np.array(map_point).T
            for i in range(-1,2):
                for j in range(-1,2):
                    for k in range(-1,2):
                        item = map_point + np.array([[i,j,k]])
                        add_items += (item, )
            add_items = np.concatenate(add_items, axis = 0)
            for i in range(add_items.shape[-1]):
                add_items = add_items[(add_items[:, i] >= 0) & (add_items[:, i] < matrix.shape[i])]
            map_point = np.concatenate((map_point, add_items), axis = 0)
            matrix[
                map_point[:,0].reshape(1,-1),
                map_point[:,1].reshape(1,-1),
                map_point[:,2].reshape(1,-1),
            ] = 0         
        
        self.grid = Grid(matrix=matrix) 
        self.min_point = min_point
        self.max_point = max_point
        self.voxelized_map = np.array(np.where(matrix == 0)).T
        self.voxelized_cloud = np.array(np.where(matrix == 0)).T + self.min_point

    def path_planning_3d(self, end, DC : drone_controller):
        
        location, _ = DC.get_world_location_pose()
        location = np.array(location)
        start = np.array(location.astype(int)) - self.min_point
        end = np.array(end.astype(int)) - self.min_point
        self.grid.cleanup()
        finder = AStarFinder(diagonal_movement=DiagonalMovement.never)
        start_node = self.grid.node(start[0],start[1],start[2])
        end_node = self.grid.node(end[0],end[1],end[2])
        if start_node != None and end_node != None:
            path, runs = finder.find_path(start_node, end_node, self.grid)
        else:
            path = [] 

        path = [(p.x, p.y, p.z) for p in path]
        path = np.array(path)

        if path.shape[0] != 0:
            path = path + np.repeat(self.min_point.reshape(1,-1), path.shape[0], axis=0)
        return path
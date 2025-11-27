from pathfinding3d.core.diagonal_movement import DiagonalMovement
from pathfinding3d.core.grid import Grid
from pathfinding3d.finder.a_star import AStarFinder
import numpy as np
from Drone.Control import drone_controller
import open3d

class path_planning_3d:

    def __init__(self, global_point_cloud):

        # pcd = open3d.geometry.PointCloud()
        # pcd.points = open3d.utility.Vector3dVector(global_point_cloud)
        # downsampled = pcd.voxel_down_sample(2)
        # self.global_point_cloud = np.asarray(downsampled.points) 
        
        # self.global_point_cloud = global_point_cloud

        min_point = np.min(global_point_cloud, axis=0).astype(int)  
        max_point = np.max(global_point_cloud, axis=0).astype(int)
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
        
        # pcd = open3d.geometry.PointCloud()
        # map_point = np.where(matrix == 0)
        # map_point = np.array(map_point).T        
        # pcd.points = open3d.utility.Vector3dVector(map_point)
        # open3d.visualization.draw_geometries([pcd])         
        
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

        # if path.shape[0] == 0:

        #     pcd = open3d.geometry.PointCloud()
        #     pcd.points = open3d.utility.Vector3dVector(self.voxelized_map)
        #     # 定义起点和终点（向量）  
        #     start_points = start.reshape(1,-1)  
        #     end_points = end.reshape(1,-1)
        #     points = np.vstack((start_points, end_points))  
        #     lines = []  
        #     for i in range(len(start_points)):  
        #         lines.append([i, i + len(start_points)])
        #     # 将数据转换为Open3D的LineSet对象  
        #     line_set = open3d.geometry.LineSet()  
        #     line_set.points = open3d.utility.Vector3dVector(points)  
        #     # 注意：LineSet的lines需要是(2, N)形状的数组，其中N是线段的数量  
        #     # 每一行表示一个线段，包含两个点的索引  
        #     line_set.lines = open3d.utility.Vector2iVector(np.array(lines))

        #     pcd2 = open3d.geometry.PointCloud()
        #     pcd2.points = open3d.utility.Vector3dVector(start_points)   
        #     pcd2.colors = open3d.utility.Vector3dVector(np.zeros_like(start_points)) 
        #     pcd3 = open3d.geometry.PointCloud()
        #     pcd3.points = open3d.utility.Vector3dVector(end_points)   
        #     pcd3.colors = open3d.utility.Vector3dVector(np.zeros_like(end_points)) 
        #     open3d.visualization.draw_geometries([pcd,pcd2,pcd3,line_set])            

        if path.shape[0] != 0:
            path = path + np.repeat(self.min_point.reshape(1,-1), path.shape[0], axis=0)
        return path
    
    def path_planning_3d_cord(self, start, end):
        
        start = np.array(start.astype(int)) - self.min_point
        end = np.array(end.astype(int)) - self.min_point
        self.grid.cleanup()
        finder = AStarFinder(diagonal_movement=DiagonalMovement.never)
        start = self.grid.node(start[0],start[1],start[2])
        end = self.grid.node(end[0],end[1],end[2])
        path, runs = finder.find_path(start, end, self.grid) 

        path = [(p.x, p.y, p.z) for p in path]
        path = np.array(path)
        path = path + np.repeat(self.min_point.reshape(1,-1), path.shape[0], axis=0)

        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(self.global_point_cloud)
        colors = np.repeat(np.array([1,0,0]).reshape(1,-1),self.global_point_cloud.shape[0],axis = 0)
        pcd.colors = open3d.utility.Vector3dVector(colors)
        pcd2 = open3d.geometry.PointCloud()
        pcd2.points = open3d.utility.Vector3dVector(path)
        pcd2.colors = open3d.utility.Vector3dVector(np.zeros_like(path))
        open3d.visualization.draw_geometries([pcd,pcd2])          
        return path    
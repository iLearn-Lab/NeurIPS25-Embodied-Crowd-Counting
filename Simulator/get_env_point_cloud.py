import csv
import numpy as np

def read_obj_vertices(file_path):
    vertices = []
 
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('v '):
                vertex = line.strip().split(' ')[1:]
                vertex = [float(coord) for coord in vertex]
                vertices.append(vertex)
 
    return vertices

def get_env_point_cloud(path):

    obj_file_path = path + '/OBJ.obj'
    vertices = read_obj_vertices(obj_file_path)
    import open3d as o3d
    import random
    textured_mesh= o3d.io.read_triangle_mesh(obj_file_path)
    rgb_ = [128 / 255, 128 / 255, 128 / 255]
    textured_mesh.paint_uniform_color(rgb_)
    textured_mesh.compute_triangle_normals()
    textured_mesh.compute_vertex_normals()  
    # # 均匀采样5000个点转换为点云
    # pcd = textured_mesh.sample_points_uniformly(number_of_points=10000000)     
    return textured_mesh

if __name__=='__main__':

    name = 'stadium2'

    obj_file_path = name + '/OBJ.obj'
    vertices = read_obj_vertices(obj_file_path)

    import open3d as o3d
    import random
    textured_mesh= o3d.io.read_triangle_mesh(obj_file_path)
    rgb_ = [128 / 255, 128 / 255, 128 / 255]
    textured_mesh.paint_uniform_color(rgb_)
    textured_mesh.compute_triangle_normals()
    textured_mesh.compute_vertex_normals()
    # o3d.visualization.draw_geometries([textured_mesh])

    # # 均匀采样5000个点转换为点云
    # pcd = textured_mesh.sample_points_uniformly(number_of_points=10000000)
    
    # # 可视化点云
    # o3d.visualization.draw_geometries([pcd])    

    # path = 'parking/route.csv'
    # route = ()
    # with open(path, encoding='utf-8')as f:
    #     reader = csv.reader(f)
    #     for row in reader:
    #         cord = [float(x) / 100 for x in row]
    #         cord = np.array([cord])
    #         route += (cord, )
    # route = np.concatenate(route, axis = 0)
    # change = route[0,:]

    path = name + '/location.csv'
    ground_truth = ()
    with open(path, encoding='utf-8')as f:
        reader = csv.reader(f)
        for row in reader:
            cord = [float(x) / 100 for x in row]
            cord = np.array([cord])
            ground_truth += (cord, )
    ground_truth = np.concatenate(ground_truth, axis = 0)
    # ground_truth = ground_truth - change 
    # ground_truth[:,-1] = -ground_truth[:,-1]  
    ground_truth = ground_truth * 100
    x = ground_truth[:,0]
    y = ground_truth[:,1]
    z = ground_truth[:,2] + 200
    if name == 'city' or name == 'parking':
        ground_truth = np.stack((-x,z,-y), axis = 1)
    else:
        ground_truth = np.stack((x,z,y), axis = 1)

    import matplotlib.cm as cm
    if np.max(z) - np.min(z) == 0:
       colors = np.array([1,0,0]).reshape(1,-1).repeat(ground_truth.shape[0], axis = 0)
    else:
       normalized = (z - np.min(z)) / (np.max(z) - np.min(z))
       colors = cm.jet(normalized)[:, :3]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(ground_truth)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([textured_mesh, pcd])
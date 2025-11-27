import csv
import numpy as np
import matplotlib.pyplot as plt
import os

class Simulator:

    def __init__(self, config):

        self.config = config
    
    def get_route(self):

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

        return change, route[1:,:]
    
    def get_ground_truth(self, change):

        path = self.config['ground_truth_path']
        ground_truth = ()
        with open(path, encoding='utf-8')as f:
            reader = csv.reader(f)
            for row in reader:
                cord = [float(x) / 100 for x in row]
                cord = np.array([cord])
                ground_truth += (cord, )
        ground_truth = np.concatenate(ground_truth, axis = 0)
        ground_truth = ground_truth - change
        ground_truth[:,-1] = -ground_truth[:,-1]

        return ground_truth
    
    def draw_ground_truth(self, ground_truth, record_path):

        path = record_path + 'ground_truth/'
        if not os.path.exists(path):
            os.makedirs(path)
        plt.figure()
        plt.scatter(ground_truth[:, 1], ground_truth[:, 0], 
                    c='black',s=3) 
        plt.savefig(path + 'gd.png', dpi = 300)
        return plt.gca()
    
    def save_simulator(self, change, route, ground_truth, record_root, now):

        path = record_root + now + 'simulator/'
        if not os.path.exists(path):
            os.makedirs(path)

        np.save(path + 'change.npy', change)
        np.save(path + 'route.npy', route)
        np.save(path + 'ground_truth.npy', ground_truth)

# pose = []
# move_degree = []
# move_x = []
# move_z = []

# for i in range(cords.shape[0]-1):

#     current_cord = cords[i,:]
#     next_cord = cords[i+1,:]
#     if i == 0:
#         current_pose = 0
#     else:
#         current_pose = pose[-1]
#     residual = next_cord - current_cord
#     new_pose = math.atan2(residual[1], residual[0]) * 180 / math.pi
#     pose.append(new_pose)
#     move_degree1 = new_pose-current_pose
#     if move_degree1 > 0:
#         move_degree2 = move_degree1 - 360
#     else:
#         move_degree2 = move_degree1 + 360
#     if abs(move_degree1) < abs(move_degree2):
#         move_degree.append(move_degree1)
#     else:
#         move_degree.append(move_degree2)
#     move_x.append(np.linalg.norm([residual[0], residual[1]]))
#     move_z.append(-residual[2]) # in ue4, z>0 -> up
    
# return pose, move_degree, move_x, move_z

                

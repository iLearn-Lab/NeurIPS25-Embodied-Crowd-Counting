from Drone.Control import drone_controller
from numpy import ndarray
import numpy as np
import math
from PIL import Image, ImageDraw

def obstacle_analysis_x(drone_controller : drone_controller, 
                        scene : ndarray, depth : ndarray):

    # get fov
    h = scene.shape[0]
    w = scene.shape[1]
    w_fov = drone_controller.client.simGetCameraInfo(camera_name=0).fov
    h_fov = w_fov / w *h

    distance = math.floor(depth[int(h / 2)][int(w / 2)])
    
    return "There is obstacle " + str(distance) + " units away in front of you."

def draw_point_on_depth(depth,h,w):

    image = Image.fromarray(depth)
    image = image.convert('RGB')
    draw = ImageDraw.Draw(image)
    draw.ellipse((w-1, h-1, w+1, h+1), fill="red")
    image.save('test.png')

def target_location_with_distance(drone_controller : drone_controller, 
                             target_location):
    
    distance = np.linalg.norm([target_location[0], 
                               target_location[1]])
    direction_vector = np.array([target_location[0], 
                                 target_location[1]]) / distance
    move_distance  = distance-drone_controller.config['target_distance_xy']
    new_location = tuple(a * move_distance for a in direction_vector)
    new_location = new_location.__add__((target_location[-1] - drone_controller.config['target_distance_z'],))
    return new_location
    
from Perception.GroundingDINO import GroundingDINO_detector
from utils.saver import image_saver, image_saver_plt
import matplotlib.pyplot as plt
from Point_cloud.Map_element import UNKNOWN, OBSTACLE, EXPLORED
import numpy as np
from sklearn.neighbors import KDTree
import seaborn as sns
from collections import defaultdict
import copy
import warnings
warnings.filterwarnings("ignore")

class ValueMap:
    def __init__(self, config):
        self.config = config
        self.perception = GroundingDINO_detector(config['GroundingDINO'])
        self.prompt = config['ValueMap']['prompt']

        self.camera = ['0', '1', '4', '2']
        self.camera_pose = ['front', 'left', 'back', 'right']
        self.savers = [image_saver(config['now'],config['Record_root'], f'ValueMap/camera_{pose}') for pose in self.camera_pose]
        self.map_saver = image_saver_plt(config['now'],config['Record_root'], 'ValueMap/global_target_map')
        self.class_map_saver = image_saver_plt(config['now'],config['Record_root'], 'ValueMap/global_class_map')
        self.heat_map_saver = image_saver(config['now'],config['Record_root'], 'ValueMap/global_heat_map')
        
        self.global_target_2D_map = set()
        self.global_target_2D_map_with_count = defaultdict(lambda:0)
        self.current_target_2D_map = set()
        self.current_target_2D_map_with_count = defaultdict(lambda:0)

        self.data = {
            "global_explore_map": np.array([]), 
            "global_explore_value_map": np.array([]), 
            "class": np.array([]), 
            "global_explore_dists": np.array([])}
        self.data_valid = False
    
    def update_current_target_map(self, target_loc_2D):
        for loc in target_loc_2D:
            if (loc[0],loc[1]) not in self.current_target_2D_map:
                self.current_target_2D_map.add((loc[0],loc[1]))
            self.current_target_2D_map_with_count[(loc[0],loc[1])] += 1
        
    def clear_global_target_map(self):
        self.global_target_2D_map = set()
        self.global_target_2D_map_with_count = defaultdict(lambda:0)
    
    def clear_current_target_map(self):
        self.current_target_2D_map = set()
        self.current_target_2D_map_with_count = defaultdict(lambda:0)
    
    def meger_current_to_global(self, expolred_map):
        expolred_map = set(map(tuple, expolred_map))
        for loc in self.current_target_2D_map:
            if loc not in expolred_map: continue
            if loc not in self.global_target_2D_map:
                self.global_target_2D_map.add(loc)
            self.global_target_2D_map_with_count[loc] += self.current_target_2D_map_with_count[loc]

    def get_global_target_loc(self, bgr_imgs:list, X:dict, Y:dict, Z:dict):
        target_loc = [] 
        boxes_info = self.get_target_boxes(bgr_imgs)
        for c in self.camera:
            x_c, y_c, z_c = X[c], Y[c], Z[c]
            for box_info in boxes_info[c]:
                center_x, center_y = box_info['center_h'], box_info['center_w']
                x, y, z = x_c[center_x][center_y], y_c[center_x][center_y], z_c[center_x][center_y]
                if np.isnan(x) or np.isnan(y) or np.isnan(z):
                    continue
                target_loc.append((x,y,z))
        return target_loc
    
    def draw_current_map_and_frontier_and_target(self, frontiers,global_2D_map,history_location, map_saver=None):
        
        # current_target_2D_map = np.array(list(self.current_target_2D_map))
        current_target_2D_map = np.array(list(self.global_target_2D_map))
        
        obstacle = global_2D_map[OBSTACLE]
        explored = global_2D_map[EXPLORED]
        plt.figure() 
        plt.scatter(explored[:, 1], explored[:, 0], 
                    c='gray',s=1)
        plt.scatter(obstacle[:, 1], obstacle[:, 0], 
                    c='black',s=1)                
        history_location = np.array(history_location)
        plt.scatter(history_location[:, 1], history_location[:, 0], 
                    c='red',s=3)
        
        if len(frontiers) != 0:
            frontiers_squeeze = ()
            labels = ()
            for i in range(len(frontiers)):
                frontiers_squeeze += (frontiers[i]['frontier'],)
                labels += (i * np.ones(frontiers[i]['frontier'].shape[0]),)
            frontiers_squeeze = np.concatenate(frontiers_squeeze,axis=0)
            labels = np.concatenate(labels)
            plt.scatter(frontiers_squeeze[:, 1], frontiers_squeeze[:, 0], 
            c=labels, cmap=plt.cm.gist_rainbow, s=1)
        
        if len(current_target_2D_map) != 0:
            plt.scatter(current_target_2D_map[:, 1], current_target_2D_map[:, 0], 
                        c='blue',s=1)

        if map_saver is None: self.map_saver.save(plt.gca())
        else: map_saver.save(plt.gca())  
    
    def get_target_boxes(self, bgr_imgs:list):
        items = {}
        for bgr_img, key, saver in zip(bgr_imgs, self.camera, self.savers):
            item = self.perception.inference_single_image(bgr_img, self.prompt, low=True)
            saver.save(item['annotated_frame'])
            items[key] = self.perception.phrase_GD_boxes(item['boxes'], bgr_img)
        return items
    
    def check_frontier_around_target(self, frontier, check_range:int):
        ret = 0
        counted = dict()
        for point in frontier:
            for tar in self.global_target_2D_map:
                if tar in counted: continue
                dist = np.linalg.norm(np.array(point) - np.array(tar))
                if dist < check_range:
                    ret += self.global_target_2D_map_with_count[tar]
                    counted[tar] = True
        return ret    

    def select_navi_point_by_dist(self, frontiers):
        if len(self.global_target_2D_map) == 0: 
            for frontier in frontiers:
                frontier['point_scores'] = np.zeros(frontier['frontier'].shape[0])
                frontier['frontier_scores'] = np.zeros(frontier['centers'].shape[0])
            return None
        
        sum = 0
        for frontier in frontiers:
            scores = np.zeros(frontier['frontier'].shape[0])
            for i, point in enumerate(frontier['frontier']):
                scores[i] = self.get_point_score(point)
                sum += np.exp(scores[i])
            frontier['point_scores'] = scores
        
        '''softmax'''
        max_score = -1e9
        for frontier in frontiers:
            frontier['point_scores'] = np.exp(frontier['point_scores']) / sum
            scores = np.zeros(frontier['centers'].shape[0])
            for i in range(frontier['centers'].shape[0]):
                idx = np.where(frontier['labels'] == i)[0]
                scores[i] = frontier['point_scores'][idx].sum()
                if max_score < scores[i]:
                    max_score = scores[i]
                    best_frontier_center = frontier['centers'][i]
            frontier['frontier_scores'] = scores
        return best_frontier_center
    
    def get_point_score(self, point):
        score = 0
        for target in self.global_target_2D_map:
            dist = np.linalg.norm(np.array(point) - np.array(target))
            assert self.global_target_2D_map_with_count[target] > 0
            score += self.global_target_2D_map_with_count[target] / dist if dist != 0 else 1
        return score
          
    def select_navi_point_by_dist_and_LLM(self, frontiers,chosen_direction, high=False):
        self.select_navi_point_by_dist(frontiers)
        max_score = -1e9; idx = 0; best_frontier_center = None
        for fidx, frontier in enumerate(frontiers):
            scores = frontier['frontier_scores']
            for i in range(frontier['centers'].shape[0]):
                # if 'directions' in frontier.keys() and frontier['directions'][i] == chosen_direction: 
                #     if high: scores[i] += self.config["ValueMap"]["high_LLM_score"]
                #     else: scores[i] += self.config["ValueMap"]["LLM_score"]
                if max_score < scores[i]:
                    max_score = scores[i]
                    idx = fidx
                    best_frontier_center = frontier['centers'][i]
        return best_frontier_center, idx
                    
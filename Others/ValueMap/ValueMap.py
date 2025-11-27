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

'''question 1: 由于前面处理障碍物时有填充操作，会导致人群所在位置是障碍物的一部分；solution：忽略'''
'''question 2: 人群数量可能会很多，需要快速查找最近邻点对，solution：用kdtree加速'''
'''question 3: 需要筛选重复的点，solution：用set去重'''
'''question 4: 没有人群怎么办，solution: 特判，没有人群时采用原来的方法选择frontier'''
'''question 5: 对于同一个人，不同角度DINO会有不同的Box，solution: 不同Box的中心点散布应该在1米内，因此可以忽略不记'''
'''question 6: 现实中的人群是流动的，如何描述这种流动性？'''
'''question 7: DINO识别的范围可能会小于设定的探索范围'''
class ValueMap:
    def __init__(self, config):
        self.config = config
        self.perception = GroundingDINO_detector(config['GroundingDINO'])
        self.prompt = config['ValueMap']['prompt']
        '''设置相机位置'''
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
        '''前面是全局坐标，后面是对应的value'''
        self.data = {"global_explore_map": np.array([]), "global_explore_value_map": np.array([]), 
                     "class": np.array([]), "global_explore_dists": np.array([])}
        self.data_valid = False
    
    def update_global_target_map(self, global_target_loc_2D):
        '''过滤点，如果在explored中，就加入到global_target_map_2D中'''
        for loc in global_target_loc_2D:
            self.global_target_2D_map.add((loc[0],loc[1]))
            self.global_target_2D_map_with_count[(loc[0],loc[1])] += 1
        if len(self.global_target_2D_map) != 0: 
            self.data_valid = True
            self.kd_tree = KDTree(list(self.global_target_2D_map))
    
    def update_current_target_map(self, global_target_loc_2D):
        for loc in global_target_loc_2D:
            if (loc[0],loc[1]) in self.global_target_2D_map:
                continue
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
            self.global_target_2D_map.add(loc)
            self.global_target_2D_map_with_count[loc] += self.current_target_2D_map_with_count[loc]
        self.current_target_2D_map = set()
        self.current_target_2D_map_with_count = defaultdict(lambda:0)
                
    '''暂时废弃'''
    def update_value_map(self, global_2D_map):
        self.data["global_explore_map"] = global_2D_map[EXPLORED]
        self.data["global_explore_value_map"] = np.zeros(global_2D_map[EXPLORED].shape[0])
        self.data["class"] = np.zeros(global_2D_map[EXPLORED].shape[0])
        assert self.data["global_explore_map"].shape[0] == self.data["global_explore_value_map"].shape[0]
        '''判断有没有人群'''
        if len(self.global_target_2D_map) == 0: return
        
        dists, idxs = self.kd_tree.query(global_2D_map[EXPLORED], k=1)
        dists, idxs = dists.squeeze(), idxs.squeeze()
        C_sem = 1. - (dists - dists.min()) / (dists.max() - dists.min())
        self.data["global_explore_value_map"] = C_sem
        self.data["class"] = idxs
        self.data['global_explore_dists'] = dists    
    
    def get_global_target_loc(self, bgr_imgs:list, X:dict, Y:dict, Z:dict):
        target_loc = []
        
        boxes_info = self.get_target_boxes(bgr_imgs)
        for c in self.camera:
            '''每一个相机的点云'''
            x_c, y_c, z_c = X[c], Y[c], Z[c]
            for box_info in boxes_info[c]:
                center_x, center_y = box_info['center_h'], box_info['center_w']
                x, y, z = x_c[center_x][center_y], y_c[center_x][center_y], z_c[center_x][center_y]
                if np.isnan(x) or np.isnan(y) or np.isnan(z):
                    continue
                target_loc.append((x,y,z))
        return target_loc
    
    def draw_current_map_and_frontier_and_target(self, frontiers,global_2D_map,history_location, map_saver=None):
        
        current_target_2D_map = np.array(list(self.current_target_2D_map))
        
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
        
        '''绘制目标点'''
        if len(current_target_2D_map) != 0:
            plt.scatter(current_target_2D_map[:, 1], current_target_2D_map[:, 0], 
                        c='blue',s=1)

        if map_saver is None: self.map_saver.save(plt.gca())
        else: map_saver.save(plt.gca())
        
        # if self.data_valid:
        #     plt.figure()    
        #     plt.scatter(self.data["global_explore_map"][:, 1], self.data["global_explore_map"][:, 0], c=self.data["class"], s=1)
        #     self.class_map_saver.save(plt.gca())    
    
    def draw_global_map_and_frontier_and_target(self, frontiers,global_2D_map,history_location, map_saver=None):
        
        global_target_2D_map = np.array(list(self.global_target_2D_map))
        
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
        
        '''绘制目标点'''
        if len(global_target_2D_map) != 0:
            plt.scatter(global_target_2D_map[:, 1], global_target_2D_map[:, 0], 
                        c='blue',s=1)

        if map_saver is None: self.map_saver.save(plt.gca())
        else: map_saver.save(plt.gca())
        
        # if self.data_valid:
        #     plt.figure()    
        #     plt.scatter(self.data["global_explore_map"][:, 1], self.data["global_explore_map"][:, 0], c=self.data["class"], s=1)
        #     self.class_map_saver.save(plt.gca())
    
    def draw_heat_map(self, global_2D_map, location):
        plt.figure()
        explored = global_2D_map[EXPLORED]
        obstacles = global_2D_map[OBSTACLE]
        x_all = np.concatenate((explored[:,1], obstacles[:,1]),axis = 0)
        y_all = np.concatenate((explored[:,0], obstacles[:,0]),axis = 0)
        threshold = (int(max(location[1], x_all.max())), 
                     int(min(location[1], x_all.min())), 
                     int(max(location[0], y_all.max())), 
                     int(min(location[0], y_all.min())))            
        x_max = threshold[0]
        x_min = threshold[1]
        y_max = threshold[2]
        y_min = threshold[3]
        map = np.zeros(
            (x_max - x_min + 1, y_max - y_min + 1)
        ).astype(np.float32)  
        map[obstacles[:,1].reshape(1,-1) - x_min,
            obstacles[:,0].reshape(1,-1) - y_min] = -0.2
        map[explored[:,1].reshape(1,-1) - x_min,
            explored[:,0].reshape(1,-1) - y_min] = self.data["global_explore_value_map"]        
        ax = sns.heatmap(map)
        ax.get_figure().savefig(self.heat_map_saver.get_save_path())
        self.heat_map_saver.next()
        # self.heat_map_saver.save(ax.get_figure())     
    
    def get_target_boxes(self, bgr_imgs:list):
        items = {}
        for bgr_img, key, saver in zip(bgr_imgs, self.camera, self.savers):
            item = self.perception.inference_single_image(bgr_img, self.prompt, low=True)
            saver.save(item['annotated_frame'])
            '''phrase_boxes返回一个list，每个dict中的center_h和center_w标注了目标的中心点'''
            items[key] = self.perception.phrase_GD_boxes(item['boxes'], bgr_img)
        return items
    
    def select_navi_point_by_value(self, frontiers):
        if len(self.global_target_2D_map) == 0: return None
        
        dists = []
        for frontier in frontiers:
            dist, _ = self.kd_tree.query(frontier['frontier'], k=1)
            dists.extend(dist.squeeze().tolist())
        
        dists = np.array(dists)
        max_mean_values = -1e9
        best_frontier_center = None
        
        for frontier in frontiers:
            '''检查被分割了多少个小的frontier，计算每一个frontier的mean-value'''
            mean_values = np.zeros(frontier['centers'].shape[0])
            for i in range(frontier['centers'].shape[0]):
                assert frontier['labels'].shape[0] == frontier['frontier'].shape[0] 
                idx = np.where(frontier['labels'] == i)[0]
                points = frontier['frontier'][idx]
                dist, _ = self.kd_tree.query(points, k=1)
                dist = dist.squeeze()
                values = 1 - (dist - dists.min()) / (dists.max() - dists.min())
                mean_values[i] = values.mean()
                if max_mean_values < values.mean():
                    max_mean_values = values.mean()
                    best_frontier_center = frontier['centers'][i]                
            frontier['mean_values'] = mean_values
            
        return best_frontier_center
    
    def select_navi_point_by_dist(self, frontiers):
        if len(self.global_target_2D_map) == 0: 
            for frontier in frontiers:
                frontier['point_scores'] = np.zeros(frontier['frontier'].shape[0])
                frontier['frontier_scores'] = np.zeros(frontier['centers'].shape[0])
            return None
        '''每一个target对一个frontier点的贡献是dist的倒数， 
        最后将所有target的贡献相加，再取softmax，那么每个frontier的得分就是这个
        frontier上的点的得分之和'''
        
        '''sum是exp之和'''
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
    
    def check_frontier_around_target(self, frontier, check_range:int):
        vis = dict()
        ret = 0
        for point in frontier:
            for i in range(-check_range, check_range+1):
                for j in range(-check_range, check_range+1):
                    if (point[0] + i, point[1] + j) not in vis: 
                        vis[(point[0] + i, point[1] + j)] = True
                        if (point[0] + i, point[1] + j) in self.current_target_2D_map: 
                            ret += self.current_target_2D_map_with_count[(point[0] + i, point[1] + j)]
        return ret
                    
    def select_navi_point_by_dist_and_LLM(self, frontiers,chosen_direction, high=False):
        self.select_navi_point_by_dist(frontiers)
        max_score = -1e9; idx = 0; best_frontier_center = None
        for fidx, frontier in enumerate(frontiers):
            scores = frontier['frontier_scores']
            for i in range(frontier['centers'].shape[0]):
                if 'directions' in frontier.keys() and frontier['directions'][i] == chosen_direction: 
                    if high: scores[i] += self.config["ValueMap"]["high_LLM_score"]
                    else: scores[i] += self.config["ValueMap"]["LLM_score"]
                if max_score < scores[i]:
                    max_score = scores[i]
                    idx = fidx
                    best_frontier_center = frontier['centers'][i]
        return best_frontier_center, idx
                    
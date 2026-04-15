from numpy import ndarray
import numpy as np
from Point_cloud.Map_element import (
    UNKNOWN,
    OBSTACLE,
    EXPLORED,
    FRONTIER)

class frontier:

    def __init__(self, config):
        self.config = config
        self.obstacle_thick = config['obstacle_thick']

    def get_line(self, start : tuple, current : tuple, length : int):
        start = np.array(start)
        current = np.array(current)
        delta = current - start

        def build_points_along_axis(axis_index, fixed_axis_index):
            axis_delta = delta[axis_index]
            if axis_delta == 0:
                return np.array([], dtype=np.int32), np.array([], dtype=np.int32)

            axis_values = np.arange(axis_delta, 1, 1) if axis_delta < 0 else np.arange(0, axis_delta + 1, 1)
            slope = delta[fixed_axis_index] / axis_delta if axis_delta != 0 else 0
            fixed_values = (slope * axis_values).astype(np.int32)

            if axis_index == 0:
                line_i = axis_values + start[0]
                line_j = fixed_values + start[1]
            else:
                line_i = fixed_values + start[0]
                line_j = axis_values + start[1]

            return line_i, line_j

        i_line1, j_line1 = build_points_along_axis(axis_index=0, fixed_axis_index=1)
        i_line2, j_line2 = build_points_along_axis(axis_index=1, fixed_axis_index=0)

        i_line = np.concatenate((i_line1, i_line2))
        j_line = np.concatenate((j_line1, j_line2))

        line = np.concatenate((i_line.reshape(-1, 1), j_line.reshape(-1, 1)), axis=1)

        distance_from_start = np.linalg.norm(line - start.reshape(1, -1), axis=1)
        line = line[np.where(distance_from_start <= length)[0]]

        return line

    def get_neighbours_eight(self, cord : tuple, map : np.ndarray):

        results = []
        for I in range(cord[0]-1,cord[0]+1+1):
            for J in range(cord[1]-1,cord[1]+1+1):
                if I >= 0 and I < map.shape[0] and \
                   J >= 0 and J < map.shape[1]:
                    if I == cord[0] and J == cord[1]:
                        pass
                    else:
                        results.append(np.array([I,J]))
        return results

    def get_neighbours_four(self, cord : tuple, map : np.ndarray):

        results = []
        for I in range(cord[0]-1,cord[0]+1+1):
            if I >= 0 and I < map.shape[0]:
                if I == cord[0]:
                    pass
                else:
                    results.append(np.array([I,cord[1]]))

        for J in range(cord[1]-1,cord[1]+1+1):
            if J >= 0 and J < map.shape[1]:
                if J == cord[1]:
                    pass
                else:
                    results.append(np.array([cord[0],J]))
        return results

    def mark_explored(self, map : ndarray, start : tuple, explore_range):
        
        explored_map = np.copy(map).astype(int)

        for _ in range(0,self.obstacle_thick):
            obstacle_idxs = np.where(explored_map == OBSTACLE)
            for i in range(obstacle_idxs[0].shape[0]):
                obstacle_idx = (obstacle_idxs[0][i],obstacle_idxs[1][i])
                neighbours_four  = self.get_neighbours_eight(obstacle_idx ,explored_map)
                neighbours_four = np.array(neighbours_four)
                explored_map[neighbours_four[:,0].reshape(1,-1),
                    neighbours_four[:,1].reshape(1,-1)] = OBSTACLE
            
        obstacle_idxs = np.where(explored_map == OBSTACLE)
        for i in range(obstacle_idxs[0].shape[0]):
            obstacle_idx = (obstacle_idxs[0][i],obstacle_idxs[1][i])
            neighbours_four  = self.get_neighbours_eight(obstacle_idx ,explored_map)
            for neighbour in neighbours_four:
                neighbour = (neighbour[0], neighbour[1])
                if explored_map[neighbour] != OBSTACLE:
                    line = self.get_line(start, neighbour, explore_range + 10)
                    idx = np.where(
                        explored_map[line[:,0].reshape(1,-1),
                            line[:,1].reshape(1,-1)] == OBSTACLE
                        )
                    if len(idx[0]) == 0:
                        explored_map[line[:,0].reshape(1,-1),
                            line[:,1].reshape(1,-1)] = EXPLORED

        # get explored
        check = []
        for i in range(0,explored_map.shape[0]):
            check.append((i,0))
            check.append((i,explored_map.shape[1]-1))
        for i in range(0,explored_map.shape[1]):
            check.append((0,i))
            check.append((explored_map.shape[0]-1,i))

        for item in check:
            if abs(start[0]-item[0]) == abs(start[1]-item[1]):
                continue
            line = self.get_line(start, item, explore_range + 10)
            idx1 = np.where(
                explored_map[line[:,0].reshape(1,-1),
                    line[:,1].reshape(1,-1)] == OBSTACLE
                )             
            if len(idx1[0]) == 0:                
                explored_map[line[:,0].reshape(1,-1),
                    line[:,1].reshape(1,-1)] = EXPLORED 
                
        explored_idx = np.where(explored_map == EXPLORED)
        explored_idx = np.concatenate((explored_idx[0].reshape(-1,1),explored_idx[1].reshape(-1,1)),axis=-1)                            
        return explored_map    
    
    def get_explored(self, map, location, change, explore_range):

        location = tuple(a - b for a, b in zip(location, change))
        location = (int(location[0]),
                    int(location[1]))
        explored_map = self.mark_explored(map,location,explore_range)
        return explored_map
       
    def get_frontiers(self, global_explored_map, change):

        idx = np.where(global_explored_map == EXPLORED)
        explored = np.concatenate((idx[0].reshape(-1,1),
                                   idx[1].reshape(-1,1)),axis = -1)
        map = np.copy(global_explored_map)

        # get frontiers
        frontier_idx = []
        checked = np.zeros_like(map)
        for idx in explored:
            if checked[idx[0],idx[1]] == 0:
                checked[idx[0],idx[1]] = 1
                idx = (idx[0],idx[1])
                
                neighbours_four  = self.get_neighbours_four(idx ,map)
                for neighbour in neighbours_four:
                    neighbour = (neighbour[0], neighbour[1])
                    if map[neighbour] == UNKNOWN:
                        map[idx] = FRONTIER
                        frontier_idx.append(idx)
                        break                  

        # mark different frontiers
        num = FRONTIER
        for idx in frontier_idx:
            stack = []
            if map[idx] == FRONTIER:
                stack.append(idx) 
                num += 1
                while len(stack) != 0: 
                    item  = stack.pop() 
                    map[item] = num                    
                    neighbours_eight = self.get_neighbours_eight(item,map)
                    for neighbour in neighbours_eight:
                        neighbour = (neighbour[0], neighbour[1])
                        if map[neighbour] == FRONTIER:
                            stack.append(neighbour)

        map = map.astype(np.uint8)
        frontiers = []
        for num in range(FRONTIER+1, map.max()+1):
            temp = ()
            idx = np.where(map == num)
            for i in range(len(idx[0])):
                temp += (np.array([[idx[0][i] + change[0],
                                    idx[1][i] + change[1]]]),)
            temp = np.concatenate(temp, axis = 0)
            if temp.shape[0] >= self.config['frontier_size']:
                frontiers_info = dict()
                frontiers_info['frontier'] = temp
                frontiers_info['size'] = np.array([temp.shape[0]])
                frontiers.append(frontiers_info)
        return frontiers, map


               



    

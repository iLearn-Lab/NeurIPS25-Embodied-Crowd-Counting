import cv2
import os
import numpy as np
from PIL import Image
from matplotlib.axes import Axes
import json
from tqdm import tqdm
import shutil

def save_config(config):

    path = config['Record_root'] + config['now'] + '/config/'
    if not os.path.exists(path):  
        os.makedirs(path)
    shutil.copy2('Config.yml', path + 'Config.yml')

class anything_saver:

    def __init__(self,now,record_root,name):

        self.root = record_root + now + '/' + name
        self.save_count = 0
        self.format = '.png'

        if not os.path.exists(self.root):
            os.makedirs(self.root)

    def next(self):
        self.save_count += 1

    def get_save_count(self):
        return self.save_count
    
    def get_save_path(self, mark = None):
        path = self.root + '/' + str(self.save_count)
        if mark != None:
            path = path + '/'
            if not os.path.exists(path):
                os.makedirs(path)
        return path
    
    def save_func(self):
        raise NotImplementedError()
    
    def fin_func(self):
        pass
        
    def save(self, item, mark = None, if_next = True):
        """
            The image should be in bgr format.
        """
        if mark != None:
            filename = mark + self.format
        else:
            filename = self.format

        path = self.get_save_path(mark)
        self.save_func(path + filename, item)

        if if_next:
            self.next()
        self.fin_func()
    
    def save_list(self, 
                  item_list : list, 
                  mark : str = None, 
                  if_next = True):
        
        for i in tqdm(range(len(item_list)), desc="Saving list: "):
            filename = str(i) + self.format
            path = self.get_save_path(str(i))
            if mark != None:
                path = path + mark + '/'
                if not os.path.exists(path):
                    os.makedirs(path)
            self.save_func(path + filename, item_list[i])
        if if_next:
            self.next()
        self.fin_func()

class image_saver_plt(anything_saver):

    def __init__(self,now,record_root,name):

        super().__init__(now,record_root,name)
        self.save_func = self.save_ax
        
    def save_ax(self, path, ax: Axes):
        ax.figure.savefig(path,dpi=300)

    def fin_func(self):
        import matplotlib
        # matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.close('all')
        return super().fin_func()
  
class image_saver_PIL(anything_saver):

    def __init__(self,now,record_root,name):

        super().__init__(now,record_root,name)
        self.save_func = self.save_image

    def save_image(self, path, image : Image):
        image.save(path)

class json_saver(anything_saver):

    def __init__(self,now,record_root,name):

        super().__init__(now,record_root,name)
        self.save_func = self.save_json
        self.format = '.json'

    def save_json(self, path, json_data):

        # 打开文件并写入JSON数据
        with open(path, "w") as file:
            file.write(json_data)

class image_saver(anything_saver):

    def __init__(self,now,record_root,name):

        super().__init__(now,record_root,name)
        self.save_func = cv2.imwrite

class numpy_saver(anything_saver):

    def __init__(self,now,record_root,name):

        super().__init__(now,record_root,name)
        self.save_func = np.save
        self.format = '.npy'

